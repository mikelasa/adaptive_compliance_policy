import sys
import os
import argparse
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import zarr
import yaml
import spatialmath as sm
from collections import deque


from PyriteEnvSuites.envs.task.manip_server_env import ManipServerEnv
from PyriteEnvSuites.utils.env_utils import ts_to_js_traj, pose9pose9s1_to_traj

from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.planning_control.mpc import ModelPredictiveControllerHybrid
from PyriteUtility.planning_control.trajectory import LinearTransformationInterpolator
from PyriteUtility.pytorch_utils.model_io import load_policy
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.umi_utils.usb_util import reset_all_elgato_devices
from PyriteUtility.common import GracefulKiller

if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )
if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
hardware_config_folder_path = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")


def _cosine_alpha(window):
    """Raised-cosine crossfade weights for the OLD chunk, one per blended step:
    alpha[j] = 0.5 * (1 + cos(pi * (j+1) / (window+1))).

    Goes from ~1 (old chunk) to ~0 (new chunk) with ~zero slope at both ends,
    so velocity stays continuous across the seam (a linear ramp would kink it).
    """
    j = np.arange(window)
    return 0.5 * (1.0 + np.cos(np.pi * (j + 1) / (window + 1)))


def _slerp_safe(q_old, q_new, t):
    """Slerp between two wxyz unit quaternions, weight t in [0, 1] toward
    q_new. Takes the shortest arc and falls back to nlerp when the two are
    nearly identical, where slerp is singular (spatialmath's
    UnitQuaternion.interp() divides by zero there, so it is not used).
    """
    dot = float(np.dot(q_old, q_new))
    if dot < 0:
        q_new = -q_new
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:  # nearly parallel: nlerp
        out = (1.0 - t) * q_old + t * q_new
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return s0 * q_old + s1 * q_new


# False sends each predicted chunk unblended.
ENABLE_CHUNK_BOUNDARY_BLEND = True


def _blend_chunk_boundary(prev_chunk, blend_window, nominal, virtual, stiffness):
    """Crossfade the start of a new action chunk against the unexecuted tail
    of the previously sent chunk, removing the position/stiffness jump at
    each replanning boundary.

    prev_chunk: dict {"nominal", "virtual", "stiffness", "exec_h"} with the
        previous cycle's sent chunk, or None on an episode's first horizon
        (the new chunk then passes through unchanged).
    blend_window: leading steps of the new chunk to blend
        (sparse_action_horizon - sparse_execution_horizon).
    nominal, virtual: (N, 7) pos3 + quat4 wxyz.
    stiffness: (6, 6*N).

    Returns the blended (nominal, virtual, stiffness), same shapes. The caller
    stores the result as the next prev_chunk.
    """
    if prev_chunk is None or not ENABLE_CHUNK_BOUNDARY_BLEND:
        return nominal, virtual, stiffness

    exec_h = prev_chunk["exec_h"]
    window = min(blend_window, prev_chunk["virtual"].shape[0] - exec_h, virtual.shape[0])
    if window <= 0:
        return nominal, virtual, stiffness

    alpha = _cosine_alpha(window)  # (window,), weight on the OLD chunk

    nominal = nominal.copy()
    virtual = virtual.copy()
    stiffness = stiffness.copy()

    # prev steps [exec_h : exec_h+window] were never executed and cover the
    # same timestamps as the new steps [0:window]
    prev_nominal_tail = prev_chunk["nominal"][exec_h : exec_h + window]
    prev_virtual_tail = prev_chunk["virtual"][exec_h : exec_h + window]
    prev_stiffness_tail = prev_chunk["stiffness"][:, 6 * exec_h : 6 * (exec_h + window)]

    for j in range(window):
        a = float(alpha[j])

        # position: linear blend
        nominal[j, :3] = a * prev_nominal_tail[j, :3] + (1.0 - a) * nominal[j, :3]
        virtual[j, :3] = a * prev_virtual_tail[j, :3] + (1.0 - a) * virtual[j, :3]

        # orientation: slerp old -> new with weight (1-a)
        nominal[j, 3:7] = _slerp_safe(prev_nominal_tail[j, 3:7], nominal[j, 3:7], 1.0 - a)
        virtual[j, 3:7] = _slerp_safe(prev_virtual_tail[j, 3:7], virtual[j, 3:7], 1.0 - a)

        # stiffness: linear elementwise blend of the 6x6 matrices
        stiffness[:, 6 * j : 6 * j + 6] = (
            a * prev_stiffness_tail[:, 6 * j : 6 * j + 6]
            + (1.0 - a) * stiffness[:, 6 * j : 6 * j + 6]
        )

    return nominal, virtual, stiffness


def main():
    #   python inference_ACP_RACP.py --ckpt <run folder or .ckpt>
    # relative to $PYRITE_CHECKPOINT_FOLDERS (or absolute under it). A run
    # folder means its checkpoints/latest.ckpt. Without --ckpt,
    # pipeline_para["ckpt_path"] is used.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="checkpoint to run")
    args = parser.parse_args()

    control_para = {
        "raw_time_step_s": 0.001,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
        "slow_down_factor": 2,  # 3 for flipup, 1.5 for wiping
        "sparse_execution_horizon": 12,  # 12 for flipup, 8/24 for wiping
        "max_duration_s": 3500,
        "pausing_mode": False,
        "device": "cuda",
    }
    pipeline_para = {
        "save_low_dim_every_N_frame": 1,
        "save_visual_every_N_frame": 1,
        "ckpt_path": "/2026.11.19_08.35.44_switch_50_demos_RACP_SWITCH_$_RACP_full/checkpoints/latest.ckpt",
        "hardware_config_path": hardware_config_folder_path
        + "/single_arm_data_collection_franka.yaml",
        # one <date>_<run name>/ subfolder per launch, one episode_<id>.zarr
        # per episode
        "control_log_path": control_log_folder_path + "/temp_worldframe/",
        # also log, per horizon, the camera frames the policy received and the
        # full robot pose / raw wrench buffers, with hardware timestamps
        "log_observations": True,
    }
    if args.ckpt is not None:
        ckpt = args.ckpt
        root = checkpoint_folder_path.rstrip("/")
        if os.path.isabs(ckpt) and ckpt.startswith(root + "/"):
            ckpt = ckpt[len(root):]
        if not ckpt.endswith(".ckpt"):
            ckpt = ckpt.rstrip("/") + "/checkpoints/latest.ckpt"
        pipeline_para["ckpt_path"] = "/" + ckpt.lstrip("/")
    if not os.path.isfile(checkpoint_folder_path + pipeline_para["ckpt_path"]):
        raise FileNotFoundError(
            "Checkpoint not found: " + checkpoint_folder_path + pipeline_para["ckpt_path"]
        )
    force_filtering_para = {
        "sampling_freq": 1000,
        "cutoff_freq": 5,
        "order": 5,
    }
    verbose = 1

    episode_id = 0

    def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
        out_res = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            shape = attr.get("shape")
            if type == "rgb":
                co, ho, wo = shape
                if out_res is None:
                    out_res = (wo, ho)
                assert out_res == (wo, ho)
        return out_res

    def printOrNot(verbose, *args):
        if verbose >= 0:
            print(f"[Episode {episode_id}] ", *args)

    vbs_h1 = verbose + 1  # verbosity for header
    vbs_h2 = verbose  # verbosity for sub-header
    vbs_p = verbose - 1  # verbosity for paragraph

    reset_all_elgato_devices()

    # load policy
    print("Loading policy: ", checkpoint_folder_path + pipeline_para["ckpt_path"])
    device = torch.device(control_para["device"])
    policy, shape_meta = load_policy(
        checkpoint_folder_path + pipeline_para["ckpt_path"], device
    )

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

    # zarr img buffer size estimation: 20s, 1000hz step rate
    img_buffer_size_estimated = int(
        20 * 1000 / pipeline_para["save_visual_every_N_frame"]
    )
    rgb_buffer_shape_nhwc = (
        img_buffer_size_estimated, # number of images
        image_height, 
        image_width,
        3, # rgb channels
    )

    # create query sizes based on observation shape meta data (horizon and downsample steps)
    # determine how many historical data points to query from the env
    rgb_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["down_sample_steps"] + 1
    ts_pose_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1
    wrench_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"][
        "down_sample_steps"
    ] + 1
    query_sizes = {
        "rgb": rgb_query_size,
        "ts_pose_fb": ts_pose_query_size,
        "wrench": wrench_query_size,
    }

    # camera IDs from checkpoint shape_meta (may differ from robot id_list)
    env_camera_id_list = shape_meta.get("camera_id_list", None)

    # create the env
    env = ManipServerEnv(
        camera_res_hw=(image_height, image_width),
        hardware_config_path=pipeline_para["hardware_config_path"],
        filter_params=force_filtering_para,
        query_sizes=query_sizes,
        compliant_dimensionality=3,
        camera_id_list=env_camera_id_list,
    )

    env.reset()

    # policy doesnt need to control robot each ms, controlling every 10ms is enough
    # time steps are computed so that control is applied at the right time
    p_timestep_s = control_para["raw_time_step_s"]
    # how many timesteps between each action step
    sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    # action horizon: model predicted actions (Tp)
    sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    sparse_action_horizon_s = (
        sparse_action_horizon * sparse_action_down_sample_steps * p_timestep_s
    )
    # execution horizon: actual executed actions (Ta)
    sparse_execution_horizon = (
        sparse_action_down_sample_steps * control_para["sparse_execution_horizon"]
    )
    # time stamps for executed actions in execution horizon
    sparse_action_timesteps_s = (
        np.arange(0, sparse_action_horizon)
        * sparse_action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )

    # prediction vector format: ref pose (9D) + virtual target (9D) + stiffness (1D)
    action_type = "pose9"  # "pose9" or "pose9pose9s1"
    id_list = [0]
    if shape_meta["action"]["shape"][0] == 9:
        action_type = "pose9"
    elif shape_meta["action"]["shape"][0] == 19:
        action_type = "pose9pose9s1"
    elif shape_meta["action"]["shape"][0] == 38:
        action_type = "pose9pose9s1"
        id_list = [0, 1]
    else:
        raise RuntimeError("unsupported")

    # assigning function to convert action to trajectory
    if action_type == "pose9pose9s1":
        action_to_trajectory = pose9pose9s1_to_traj
    else:
        raise RuntimeError("unsupported")

    camera_id_list = shape_meta.get("camera_id_list", id_list)
    image_height = 480
    image_width = 640

    printOrNot(vbs_h2, "Creating MPC.")
    # create MPC controller: combines policy with interpolation
    controller = ModelPredictiveControllerHybrid(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        sparse_execution_horizon=sparse_execution_horizon,
    )
    # sets a time offset between env time and controller time to align
    controller.set_time_offset(env)

    # gets time from env (robot time)
    episode_initial_time_s = env.current_hardware_time_s

    # computes the time needed to execute one horizon by the MPC
    execution_duration_s = (
        sparse_execution_horizon * p_timestep_s * control_para["slow_down_factor"]
    )
    printOrNot(vbs_h2, "Starting main loop.")

    if control_para["pausing_mode"]:
        plt.ion()  # Enable interactive mode
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Trajectory Visualization")
        plt.show(block=False)
        plt.pause(0.001)

    # live force/torque norm plot, rolling window of the last 50 horizons
    _force_window = 50
    _force_norm_history = deque(maxlen=_force_window)
    _torque_norm_history = deque(maxlen=_force_window)
    plt.ion()
    _fig_f, (_ax_f, _ax_t) = plt.subplots(2, 1, figsize=(8, 4), tight_layout=True)
    _ax_f.set_title("||Force|| [N]")
    _ax_f.set_ylim(0, 40)
    _ax_t.set_title("||Torque|| [Nm]")
    _ax_t.set_ylim(0, 8)
    (_line_f,) = _ax_f.plot([], [], color="tab:blue")
    (_line_t,) = _ax_t.plot([], [], color="tab:orange")
    plt.show(block=False)
    plt.pause(0.001)

    def _update_force_plot(wrench_6d):
        _force_norm_history.append(np.linalg.norm(wrench_6d[:3]))
        _torque_norm_history.append(np.linalg.norm(wrench_6d[3:]))
        xs_f = np.arange(len(_force_norm_history))
        xs_t = np.arange(len(_torque_norm_history))
        _line_f.set_data(xs_f, list(_force_norm_history))
        _line_t.set_data(xs_t, list(_torque_norm_history))
        _ax_f.set_xlim(0, max(_force_window, len(_force_norm_history)))
        _ax_t.set_xlim(0, max(_force_window, len(_torque_norm_history)))
        _fig_f.canvas.flush_events()

    # log: one folder per launch, named after the checkpoint's training run,
    # holding one episode_<id>.zarr per episode (opened in the episode loop),
    # so earlier launches are never overwritten.
    ckpt_run_name = pipeline_para["ckpt_path"].split("/checkpoints/")[0].strip("/")
    run_log_dir = os.path.join(
        pipeline_para["control_log_path"],
        time.strftime("%Y.%m.%d_%H.%M.%S_") + ckpt_run_name.replace("/", "__"),
    )
    os.makedirs(run_log_dir, exist_ok=True)
    print("Logging episodes to: ", run_log_dir)

    # history lengths kept by the manip server: the whole buffer is logged
    # every horizon so consecutive horizons overlap with no gap in between
    with open(pipeline_para["hardware_config_path"]) as f:
        hardware_config = yaml.safe_load(f)
    robot_log_size = hardware_config["robot_buffer_size"]
    wrench_log_size = hardware_config["wrench_buffer_size"]

    horizon_count = 0
    print("test plotting RGB. Press q to continue.")

    # test camera and plot rgb
    while True:
        # gets observation from env
        obs_raw = env.get_observation_from_buffer()

        # plot the rgb image
        frames = [cv2.resize(cv2.cvtColor(obs_raw[f"rgb_{cam_id}"][-1], cv2.COLOR_RGB2BGR),
                             (image_width, image_height))
                  for cam_id in camera_id_list if f"rgb_{cam_id}" in obs_raw]
        bgr = np.vstack(frames) if len(frames) > 1 else frames[0]
        cv2.imshow("image", bgr)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    # get initial tool space poses
    ts_pose_initial = []
    for id in id_list:
        ts_pose_initial.append(obs_raw[f"ts_pose_fb_{id}"][-1])
    #########################################
    # main loop starts
    #########################################
    while True:
        printOrNot(vbs_h1, "New episode. Episode ID: ", episode_id)
        input("Press Enter to start the episode.")
        # killer handles graceful shutdown
        killer = GracefulKiller()

        # open the per-episode log
        log = zarr.open(
            os.path.join(run_log_dir, f"episode_{episode_id}.zarr"), mode="w"
        )
        log.attrs["ckpt_path"] = pipeline_para["ckpt_path"]
        log.attrs["hardware_config_path"] = pipeline_para["hardware_config_path"]
        log.attrs["control_para"] = control_para
        horizon_count = 0
        # previous sent chunk per id for _blend_chunk_boundary; reset every
        # episode so nothing is blended against the last episode
        prev_chunks = [None] * len(id_list)
        while not killer.kill_now:
            # get current hardware time
            horizon_initial_time_s = env.current_hardware_time_s
            printOrNot(vbs_h1, "Starting new horizon at ", horizon_initial_time_s)

            # get observation from env (rgb, ts poses, wrenches)
            obs_raw = env.get_observation_from_buffer()

            # plot the rgb image
            frames = [cv2.resize(cv2.cvtColor(obs_raw[f"rgb_{cam_id}"][-1], cv2.COLOR_RGB2BGR),
                                 (image_width, image_height))
                      for cam_id in camera_id_list if f"rgb_{cam_id}" in obs_raw]
            bgr = np.vstack(frames) if len(frames) > 1 else frames[0]
            cv2.imshow("image", bgr)
            cv2.waitKey(10)

            # update force plot with latest filtered wrench (robot 0)
            _update_force_plot(obs_raw["wrench_0"][-1])

            # convert raw obs to the policy's obs format
            obs_task = dict()
            raw_to_obs(obs_raw, obs_task, shape_meta)

            assert action_type == "pose9pose9s1"

            # set observation applies downsampling according to shape_meta
            controller.set_observation(obs_task["obs"])

            # inference
            (action_sparse_target_mats, action_sparse_vt_mats,
             action_stiffnesses, ) = controller.compute_sparse_control(device)

            # decode stiffness matrix
            outputs_ts_nominal_targets = [np.array] * len(id_list)
            outputs_ts_targets = [np.array] * len(id_list)
            outputs_ts_stiffnesses = [np.array] * len(id_list)
            for id in id_list:
                # transform from SE3 to pose7
                ts_targets_nominal = su.SE3_to_pose7(
                    action_sparse_target_mats[id].reshape([-1, 4, 4])
                )
                ts_targets_virtual = su.SE3_to_pose7(
                    action_sparse_vt_mats[id].reshape([-1, 4, 4])
                )

                ts_stiffnesses = np.zeros([6, 6 * ts_targets_virtual.shape[0]])
                # for each target in the horizon
                for i in range(ts_targets_virtual.shape[0]):
                    # get target and virtual target in matrix form
                    SE3_target = action_sparse_target_mats[id][i].reshape([4, 4])
                    SE3_virtual_target = action_sparse_vt_mats[id][i].reshape([4, 4])
                    stiffness = action_stiffnesses[id][i]

                    # compliance direction in the WORLD frame: virtual target
                    # labels are world poses and the impedance controller
                    # multiplies K by a world-frame pose error (the original
                    # tool-frame version was only correct for the UR5
                    # admittance controller).
                    compliance_direction = (
                        SE3_virtual_target[:3, 3] - SE3_target[:3, 3]
                    ).reshape(3)

                    # if the two are too close, set X direction
                    if np.linalg.norm(compliance_direction) < 0.001:
                        compliance_direction = np.array([1.0, 0.0, 0.0])

                    # normalize
                    compliance_direction /= np.linalg.norm(compliance_direction)

                    # orthonormal basis with the compliance direction as X.
                    # K depends on X alone, so the reference axis is free: use
                    # the world axis least aligned with X so cross(X, ref)
                    # never vanishes (a fixed Z would give NaNs when pressing
                    # straight down).
                    X = compliance_direction
                    ref = np.zeros(3)
                    ref[np.argmin(np.abs(X))] = 1.0
                    Y = np.cross(X, ref)
                    Y /= np.linalg.norm(Y)
                    Z = np.cross(X, Y)

                    default_stiffness = 1300
                    default_stiffness_rot = 400
                    target_stiffness = stiffness

                    # defines K_0 matrix with predicted stiffnes on first position
                    M = np.diag(
                        [target_stiffness, default_stiffness, default_stiffness]
                    )
                    # similarity matrix S
                    S = np.array([X, Y, Z]).T
                    #computes final K eq 6
                    stiffness_matrix = S @ M @ np.linalg.inv(S)
                    stiffness_matrix_full = np.eye(6) * default_stiffness_rot
                    stiffness_matrix_full[:3, :3] = stiffness_matrix
                    ts_stiffnesses[:, 6 * i : 6 * i + 6] = stiffness_matrix_full

                # final action (pose_ref, virtual_target and K)
                outputs_ts_nominal_targets[id] = ts_targets_nominal
                outputs_ts_targets[id] = ts_targets_virtual
                outputs_ts_stiffnesses[id] = ts_stiffnesses

                # smooth the replanning boundary (see _blend_chunk_boundary)
                (
                    outputs_ts_nominal_targets[id],
                    outputs_ts_targets[id],
                    outputs_ts_stiffnesses[id],
                ) = _blend_chunk_boundary(
                    prev_chunks[id],
                    sparse_action_horizon - control_para["sparse_execution_horizon"],
                    outputs_ts_nominal_targets[id],
                    outputs_ts_targets[id],
                    outputs_ts_stiffnesses[id],
                )
                # stash this cycle's actually-sent chunk for next cycle's blend.
                prev_chunks[id] = {
                    "nominal": outputs_ts_nominal_targets[id],
                    "virtual": outputs_ts_targets[id],
                    "stiffness": outputs_ts_stiffnesses[id],
                    "exec_h": control_para["sparse_execution_horizon"],
                }

            # the "now" when the observation is taken
            action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
            timestamps = sparse_action_timesteps_s

            if control_para["pausing_mode"]:
                # plot the actions for this horizon using matplotlib
                ax.cla()
                for id in id_list:
                    ax.plot3D(
                        action_sparse_target_mats[id][..., 0, 3],
                        action_sparse_target_mats[id][..., 1, 3],
                        action_sparse_target_mats[id][..., 2, 3],
                        color="red",
                        marker="o",
                        markersize=3,
                        label="Target (all)"
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][..., 0, 3],
                        action_sparse_vt_mats[id][..., 1, 3],
                        action_sparse_vt_mats[id][..., 2, 3],
                        color="blue",
                        marker="o",
                        markersize=3,
                        label="Virtual Target (all)"
                    )

                    ax.plot3D(
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="yellow",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                        label="Target (exec horizon)"
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="green",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                        label="VT (exec horizon)"
                    )

                    ax.plot3D(
                        obs_raw[f"ts_pose_fb_{id}"][-1][0],
                        obs_raw[f"ts_pose_fb_{id}"][-1][1],
                        obs_raw[f"ts_pose_fb_{id}"][-1][2],
                        color="black",
                        marker="o",
                        markersize=8,
                        label="Current pose"
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Horizon {horizon_count}")
                ax.legend()

                set_axes_equal(ax)

                # Force the figure to update and display
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.1)  # Give time for rendering
                
                input("Press Enter to start executing the plotted actions.")

                action_start_time_s = env.current_hardware_time_s

            # log the whole action (always)
            horizon_log = log.create_group(f"horizon_{horizon_count}")
            for id in id_list:
                horizon_log.create_dataset(
                    f"ts_virtual_targets_{id}", data=outputs_ts_targets[id]
                )
                horizon_log.create_dataset(
                    f"ts_nominal_targets_{id}", data=outputs_ts_nominal_targets[id]
                )
                horizon_log.create_dataset(
                    f"ts_stiffnesses_{id}", data=outputs_ts_stiffnesses[id]
                )
                horizon_log.create_dataset(
                    f"stiffness_scalars_{id}", data=action_stiffnesses[id]
                )
            horizon_log.create_dataset(
                "timestamps_s", data=timestamps + action_start_time_s
            )

            if control_para["pausing_mode"]:
                # cut the action and only keep the execution horizon
                for id in id_list:
                    outputs_ts_targets[id] = outputs_ts_targets[id][
                        : control_para["sparse_execution_horizon"], :
                    ]
                    outputs_ts_stiffnesses[id] = outputs_ts_stiffnesses[id][
                        :, : 6 * control_para["sparse_execution_horizon"]
                    ]
                timestamps = timestamps[: control_para["sparse_execution_horizon"]]
                action_start_time_s = env.current_hardware_time_s

            if len(id_list) == 1:
                outputs_ts_targets = outputs_ts_targets[0].T  # N x 7 to 7 x N
                outputs_ts_stiffnesses = outputs_ts_stiffnesses[0]
            else:
                outputs_ts_targets = np.hstack(
                    outputs_ts_targets
                ).T  # 2 x N x 7 to 14 x N
                outputs_ts_stiffnesses = np.vstack(
                    outputs_ts_stiffnesses
                )  # 6 x (6xN) to 12 x (6xN)

            # send to env for execution
            env.schedule_controls(
                pose7_cmd=outputs_ts_targets,
                stiffness_matrices_6x6=outputs_ts_stiffnesses,
                timestamps=(timestamps + action_start_time_s) * 1000,
            )

            # log what the policy saw (after scheduling, so it adds no delay to
            # the commands). obs_raw rgb is the env's reused output buffer, so
            # it is copied. get_pose/get_wrench overwrite the server's stored
            # timestamps, which is why they are read only now, after the
            # policy's observation has been taken.
            if pipeline_para["log_observations"]:
                horizon_log.create_dataset(
                    "obs_hardware_time_s", data=horizon_initial_time_s
                )
                for cam_id in camera_id_list:
                    horizon_log.create_dataset(
                        f"rgb_{cam_id}", data=np.array(obs_raw[f"rgb_{cam_id}"]),
                        chunks=(1, None, None, None),
                    )
                    horizon_log.create_dataset(
                        f"rgb_time_stamps_{cam_id}",
                        data=np.array(obs_raw[f"rgb_time_stamps_{cam_id}"]),
                    )
                for id in id_list:
                    horizon_log.create_dataset(
                        f"ts_pose_fb_stream_{id}",
                        data=env.server.get_pose(robot_log_size, id).transpose(),
                    )
                    horizon_log.create_dataset(
                        f"robot_time_stamps_stream_{id}",
                        data=env.server.get_pose_timestamps_ms(id) / 1000,
                    )
                    horizon_log.create_dataset(
                        f"wrench_raw_stream_{id}",
                        data=env.server.get_wrench(wrench_log_size, id).transpose(),
                    )
                    horizon_log.create_dataset(
                        f"wrench_time_stamps_stream_{id}",
                        data=env.server.get_wrench_timestamps_ms(id) / 1000,
                    )

            horizon_count += 1

            if control_para["pausing_mode"]:
                c = input("Press Enter to start the next horizon. q to quit.")
                if c == "q":
                    break

            # checks if need to sleep to wait for the execution to finish for this horizon
            time_s = env.current_hardware_time_s
            sleep_duration_s = horizon_initial_time_s + execution_duration_s - time_s

            printOrNot(vbs_h1, "sleep_duration_s", sleep_duration_s)
            time.sleep(max(0, sleep_duration_s))

            if not control_para["pausing_mode"]:
                # only check duration when not in pausing mode
                if time_s - episode_initial_time_s > control_para["max_duration_s"]:
                    break

        printOrNot(vbs_h1, "End of episode.")
        episode_id += 1

        print("Options:")
        print("     c: continue to next episode.")
        print("     r: reset to default pose, then continue.")
        print("     b: reset to default pose, then quit the program.")
        print("     others: quit the program.")
        c = input("Please select an option: ")
        if c == "r" or c == "b":
            print("Resetting to default pose.")
            obs_raw = env.get_observation_from_buffer()
            N = 100
            duration_s = 5
            timestamps = np.linspace(0, 1, N) * duration_s
            homing_ts_targets = np.zeros([7 * len(id_list), N])
            for id in id_list:
                ts_pose_fb = obs_raw[f"ts_pose_fb_{id}"][-1]
                SE3_waypoints = [
                    su.pose7_to_SE3(ts_pose_fb),
                    su.pose7_to_SE3(ts_pose_initial[id]),
                ]

                SE3_interpolator = LinearTransformationInterpolator(
                    x_wp=np.array([0, duration_s]),
                    y_wp=np.array(SE3_waypoints),
                )
                SE3_waypoints = SE3_interpolator(timestamps)

                for i in range(N):
                    wpi = SE3_waypoints[i]
                    pose7 = su.SE3_to_pose7(wpi)
                    homing_ts_targets[0 + id * 7 : 7 + id * 7, i] = pose7

            time_now_s = env.current_hardware_time_s
            env.schedule_controls(
                pose7_cmd=homing_ts_targets,
                timestamps=(timestamps + time_now_s) * 1000,
            )
        elif c == "c":
            pass
        else:
            print("Quitting the program.")
            break

        if c == "b":
            input("Press Enter to quit program.")
            break

        print("Continuing to execution.")
    env.cleanup()


if __name__ == "__main__":
    main()