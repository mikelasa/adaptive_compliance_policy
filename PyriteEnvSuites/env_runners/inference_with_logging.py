import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import zarr
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
    """Raised-cosine crossfade weights for the OLD chunk, one per blended
    step (index 0 = right at the replanning boundary, index window-1 = the
    far end of the blend region).

    alpha[j] = 0.5 * (1 + cos(pi * (j+1) / (window+1)))

    Goes smoothly from near-1 (trust the old, already-executing chunk) to
    near-0 (trust the freshly predicted chunk) with ~zero slope at both
    ends, so the blended trajectory's velocity/stiffness-rate stays
    continuous across the seam too, not just its position -- unlike a plain
    linear ramp, which kinks the derivative at both window edges.
    """
    j = np.arange(window)
    return 0.5 * (1.0 + np.cos(np.pi * (j + 1) / (window + 1)))


def _slerp_safe(q_old, q_new, t):
    """Slerp between two wxyz (scalar-first) unit quaternions, weight t in
    [0, 1] toward q_new. Always takes the shortest arc (flips q_new's sign
    if the dot product is negative) and falls back to nlerp+normalize when
    the two quaternions are (nearly) identical -- the two consecutively
    predicted chunks agreeing closely on orientation at a given step is a
    completely normal case here, not a rare edge case, and true slerp is
    singular there (sin(theta_0) -> 0): spatialmath's UnitQuaternion.interp()
    divides by zero in exactly this situation, so it is not used directly.
    """
    dot = float(np.dot(q_old, q_new))
    if dot < 0:
        q_new = -q_new
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:  # nearly parallel -- slerp is numerically unstable here,
        # and indistinguishable from nlerp at this closeness anyway.
        out = (1.0 - t) * q_old + t * q_new
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return s0 * q_old + s1 * q_new


# A/B test toggle: left-side box approaches seem to under-rotate, and the
# crossfade below is the newest thing in the replanning loop (absent from
# the older virtual_target_real_env_runner.py, which sends each chunk raw).
# Flip to False to send each freshly-diffused chunk through unblended, as
# that older script did, and see whether reach/rotation comes back.
ENABLE_CHUNK_BOUNDARY_BLEND = True


def _blend_chunk_boundary(prev_chunk, blend_window, nominal, virtual, stiffness):
    """Crossfade the start of a freshly-predicted action chunk against the
    tail of the previously *sent* chunk, to remove the periodic
    position/stiffness discontinuity caused by replanning the whole horizon
    from fresh diffusion noise every cycle (pausing_mode=False sends the
    full sparse_action_horizon chunk every execution_duration_s, but only
    the first sparse_execution_horizon steps were meant to actually run
    before the next chunk arrives and overwrites the rest -- the remaining
    steps are free, already-computed margin, reused here as the blend
    window instead of being silently discarded).

    prev_chunk: dict {"nominal", "virtual", "stiffness"} holding the
        previous cycle's post-blend (i.e. actually-sent) chunk for one id,
        or None on the first horizon of an episode -- nothing to blend
        against yet, so the new chunk passes through unchanged.
    blend_window: number of leading steps of the new chunk to blend
        (sparse_action_horizon - sparse_execution_horizon steps -- the
        margin that was predicted-but-discarded last cycle).
    nominal, virtual: (N, 7) pos3(0:3) + quat4 wxyz(3:7) arrays for this
        cycle (nominal = target, virtual = virtual target).
    stiffness: (6, 6*N) array for this cycle.

    Returns the blended (nominal, virtual, stiffness) with the same shapes
    as the inputs. Caller is responsible for stashing the return value as
    the new prev_chunk for next cycle.
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

    # The previous chunk's steps [exec_h : exec_h+window] were predicted but
    # never executed (replanning happened first) -- they cover the same
    # real-world timestamps as this new chunk's steps [0:window], so that's
    # what we blend against.
    prev_nominal_tail = prev_chunk["nominal"][exec_h : exec_h + window]
    prev_virtual_tail = prev_chunk["virtual"][exec_h : exec_h + window]
    prev_stiffness_tail = prev_chunk["stiffness"][:, 6 * exec_h : 6 * (exec_h + window)]

    for j in range(window):
        a = float(alpha[j])

        # position: linear blend
        nominal[j, :3] = a * prev_nominal_tail[j, :3] + (1.0 - a) * nominal[j, :3]
        virtual[j, :3] = a * prev_virtual_tail[j, :3] + (1.0 - a) * virtual[j, :3]

        # orientation (pose7 quat is wxyz, scalar-first): slerp from old to
        # new, weighted by (1-a) so a=0 (fully new) recovers the new
        # quaternion exactly and a=1 (fully old, at the boundary) recovers
        # the old one.
        nominal[j, 3:7] = _slerp_safe(prev_nominal_tail[j, 3:7], nominal[j, 3:7], 1.0 - a)
        virtual[j, 3:7] = _slerp_safe(prev_virtual_tail[j, 3:7], virtual[j, 3:7], 1.0 - a)

        # stiffness: linear elementwise blend of the two 6x6 matrices (a
        # convex combination of two PSD-ish matrices stays well-behaved,
        # even though it doesn't re-derive the compliance direction from a
        # blended virtual target).
        stiffness[:, 6 * j : 6 * j + 6] = (
            a * prev_stiffness_tail[:, 6 * j : 6 * j + 6]
            + (1.0 - a) * stiffness[:, 6 * j : 6 * j + 6]
        )

    return nominal, virtual, stiffness


def main():
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
        #"ckpt_path": "/switch/200/2026.09.29-12.41.41_RACP_full_200_demos/checkpoints/latest.ckpt",
        "ckpt_path": "/tilt/final/2026.09.09-18.02.49_RACP_curriculum/checkpoints/latest.ckpt",
        # "hardware_config_path": hardware_config_folder_path + "/manip_server_config_left_arm.yaml", episode_1778754577
        "hardware_config_path": hardware_config_folder_path
        + "/single_arm_data_collection_franka.yaml",
        "control_log_path": control_log_folder_path + "/temp/",
    }
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
    # Manip server makes the communication with the real hardware posible through manip server
    # and pybind11
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
    # set timestep
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

    # timestep_idx = 0
    # stiffness = None
    # gets time from env (robot time)
    episode_initial_time_s = env.current_hardware_time_s

    #computes the time needed to execute one horizon by the MPC (180 ms)
    execution_duration_s = (
        sparse_execution_horizon * p_timestep_s * control_para["slow_down_factor"]
    )
    printOrNot(vbs_h2, "Starting main loop.")

    if control_para["pausing_mode"]:
        plt.ion()  # Enable interactive mode
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Don't plot dummy data or call plt.show() here
        # Just set up the axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Trajectory Visualization")

        # Show non-blocking
        plt.show(block=False)
        plt.pause(0.001)

    # real-time force norm plot (~10s rolling window at ~5Hz update rate)
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

    # log
    log_store = zarr.DirectoryStore(path=pipeline_para["control_log_path"])
    log = zarr.open(store=log_store, mode="w")  # w: overrite if exists

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
        # Boundary-blend state (see _blend_chunk_boundary): the previous
        # cycle's post-blend (actually-sent) chunk, per id. Scoped to this
        # episode's while-loop so it's automatically None again at the start
        # of every new episode -- a fresh episode must never be blended
        # against the previous episode's leftover trajectory.
        prev_chunks = [None] * len(id_list)
        while not killer.kill_now:
            # get current hardware time
            horizon_initial_time_s = env.current_hardware_time_s
            printOrNot(vbs_h1, "Starting new horizon at ", horizon_initial_time_s)

            # get observation from env
            # takes rgb images, ts poses, wrenches anda processes them into a dictionary
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

            # save low dim obs named as raw in new dict called obs_task
            # function change format and names of obs
            obs_task = dict()
            raw_to_obs(obs_raw, obs_task, shape_meta)

            assert action_type == "pose9pose9s1"

            # set observation applies downsampling according to shape_meta
            controller.set_observation(obs_task["obs"])

            # inference: add batch size 
            (action_sparse_target_mats, action_sparse_vt_mats, 
             action_stiffnesses, ) = controller.compute_sparse_control(device)

            # for id in id_list:
            #     print(f"Stiffness {id}: ", action_stiffnesses[id])

            # decode stiffness matrix
            outputs_ts_nominal_targets = [np.array] * len(id_list)
            outputs_ts_targets = [np.array] * len(id_list)
            outputs_ts_stiffnesses = [np.array] * len(id_list)
            for id in id_list:
                # get tool to world transform
                SE3_TW = su.SE3_inv(su.pose7_to_SE3(obs_raw[f"ts_pose_fb_{id}"][-1]))
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

                    # stiffness: 1. convert vt to tool frame
                    SE3_TVt = SE3_TW @ SE3_virtual_target
                    SE3_Ttarget = SE3_TW @ SE3_target

                    # stiffness: 2. compute stiffness matrix in the tool frame
                    #compute direction from target to virtual target
                    compliance_direction_tool = (
                        SE3_TVt[:3, 3] - SE3_Ttarget[:3, 3]
                    ).reshape(3)

                    # if the two are too close, set X direction
                    if np.linalg.norm(compliance_direction_tool) < 0.001:  #
                        compliance_direction_tool = np.array([1.0, 0.0, 0.0])

                    # normalize
                    compliance_direction_tool /= np.linalg.norm(
                        compliance_direction_tool
                    )

                    # build orthonormal basis
                    # sets compliance direction as X axis
                    X = compliance_direction_tool
                    Y = np.cross(X, np.array([0, 0, 1]))
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

                # crossfade the start of this freshly-diffused chunk against
                # the tail of the previously *sent* one, to remove the
                # replanning-boundary bump (see _blend_chunk_boundary). No
                # effect on the first horizon of an episode (prev_chunks[id]
                # is None then).
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

            # # log the truncated action
            # horizon_log = log.create_group(f"horizon_{horizon_count}")
            # horizon_log.create_dataset("ts_targets", data=outputs_ts_targets)
            # horizon_log.create_dataset("timestamps", data=timestamps * 1000)
            # horizon_log.create_dataset("ts_stiffnesses", data=outputs_ts_stiffnesses)
            # horizon_log.create_dataset(
            #     "timestamps_s", data=timestamps + action_start_time_s
            # )
            horizon_count += 1

            if control_para["pausing_mode"]:
                c = input("Press Enter to start the next horizon. q to quit.")
                if c == "q":
                    break

            # # log
            # if warming_up_done:
            #     if timestep_idx % pipeline_para["save_low_dim_every_N_frame"] == 0:
            #         env.add_low_dim_observation(None, ts_target, time_s)
            #         # env.add_optional_observation(stiffness)
            #     if timestep_idx % pipeline_para["save_visual_every_N_frame"] == 0:
            #         env.add_visual_observation(time_s)

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
    # end of episode


if __name__ == "__main__":
    main()