"""Inference runner for the vision-only Diffusion Policy baseline.

Counterpart of inference_ACP_RACP.py. The ACP runner decodes a 19-dim action
into a nominal pose, a virtual target and a stiffness scalar, and schedules
poses plus a 6x6 stiffness per waypoint. DP has a 9-dim pose action only, so
it schedules the poses with a CONSTANT stiffness diag(1300,1300,1300,
400,400,400) -- ACP's stiff-axis values -- through the same
IMPEDANCE_CONTROLLER. DP and ACP thus share one controller and one set of
gains; ACP differs only by softening one axis and shifting its virtual target.
(A POSITION_CONTROLLER variant was tried and dropped: it went into a
self-sustained ~1 Hz oscillation.)

Other differences from the ACP runner:

  - No chunk-boundary crossfade. The nominal-pose half of
    inference_ACP_RACP._blend_chunk_boundary ports over directly if needed.

  - ModelPredictiveControllerPosition, not ...Hybrid: flat obs dict, wrench and
    timestamp keys dropped, each obs key truncated to its last n_obs_steps.

Pick the checkpoint with --ckpt (see main), or set pipeline_para["ckpt_path"].

Logging matches inference_ACP_RACP.py so the two can be compared with one
analysis script: one <date>_<run name>/ folder per launch, one
episode_<id>.zarr per episode, and per horizon the full 1 kHz raw wrench and
pose buffers plus the camera frames the policy saw, all with hardware
timestamps.
"""

import sys
import argparse
import yaml
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

from PyriteEnvSuites.envs.task.manip_server_env import ManipServerEnv

from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.planning_control.mpc import ModelPredictiveControllerPosition
from PyriteUtility.planning_control.trajectory import LinearTransformationInterpolator
from PyriteUtility.pytorch_utils.model_io import load_policy_and_cfg
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


def main():
    #   python inference_DP.py --ckpt <run folder or .ckpt>
    # relative to $PYRITE_CHECKPOINT_FOLDERS (or absolute under it). A run
    # folder means its checkpoints/latest.ckpt. Without --ckpt,
    # pipeline_para["ckpt_path"] is used.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="checkpoint to run")
    args = parser.parse_args()

    control_para = {
        "raw_time_step_s": 0.001,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
        # upstream eval_real_robot.py replans every 6 steps at demo speed;
        # slow_down_factor > 1 makes the 2-frame image history show less
        # motion than in training
        "slow_down_factor": 3,
        "sparse_execution_horizon": 6,  # steps executed before replanning
        # upstream trains with 100 DDIM steps and uses 16 at eval
        "num_inference_steps": 16,
        "max_duration_s": 3500,
        "pausing_mode": False,
        "device": "cuda",
        # constant stiffness [x y z rx ry rz] (N/m, Nm/rad) = ACP's stiff axes
        "stiffness_diag": [1300, 1300, 1300, 400, 400, 400],
    }
    pipeline_para = {
        "save_low_dim_every_N_frame": 1,
        "save_visual_every_N_frame": 1,
        # relative to $PYRITE_CHECKPOINT_FOLDERS
        "ckpt_path": "/CAP6/E3-DPVSACP/2026.11.15_00.42.02_wipe_gopro177_single_DP_Baseline/checkpoints/latest.ckpt",
        "hardware_config_path": hardware_config_folder_path
        + "/single_arm_DP_franka.yaml",
        # one <date>_<run name>/ subfolder per launch, one episode_<id>.zarr
        # per episode
        "control_log_path": control_log_folder_path + "/temp_DP/",
        # also log, per horizon, the camera frames the policy received and the
        # full robot pose / raw wrench buffers (same keys as ACP)
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
    policy, cfg = load_policy_and_cfg(
        checkpoint_folder_path + pipeline_para["ckpt_path"], device
    )
    policy.num_inference_steps = control_para["num_inference_steps"]
    print(f"DDIM inference steps: {policy.num_inference_steps}")
    # Two metas, and the distinction matters:
    #   task.shape_meta  - the full ACP meta. Drives the observation horizons,
    #                      downsampling, and the relative-pose conversion.
    #   shape_meta       - the reduced meta the policy network sees: vision +
    #                      proprioception in, pose9 out.
    shape_meta = cfg.task.shape_meta
    policy_shape_meta = cfg.shape_meta

    action_dim = policy_shape_meta["action"]["shape"][0]
    if action_dim % 9 != 0:
        raise RuntimeError(
            f"This runner expects a pose9 action, got {action_dim} dims. "
            "A 19- or 38-dim action is an ACP/RACP policy -- use "
            "inference_ACP_RACP.py for that."
        )
    id_list = list(range(action_dim // 9))
    print(f"Vision-only DP policy: action dim {action_dim}, arms {id_list}")
    if "robot0_eef_wrench" in policy_shape_meta["obs"]:
        raise RuntimeError(
            "policy shape_meta contains robot0_eef_wrench; this is not a "
            "vision-only policy."
        )

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

    # create query sizes based on observation shape meta data (horizon and downsample steps)
    # determine how many historical data points to query from the env.
    # The wrench query stays even though the policy never sees a wrench:
    # sparse_obs_to_obs_sample reads it while solving the relative wrench, and
    # the MPC drops it afterwards.
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

    camera_id_list = shape_meta.get("camera_id_list", id_list)
    image_height = 480
    image_width = 640

    printOrNot(vbs_h2, "Creating MPC.")
    controller = ModelPredictiveControllerPosition(
        shape_meta=shape_meta,
        policy_shape_meta=policy_shape_meta,
        id_list=id_list,
        policy=policy,
        n_obs_steps=cfg.n_obs_steps,
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
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Trajectory Visualization")

    print("test plotting RGB. Press q to continue.")

    # test camera and plot rgb
    while True:
        obs_raw = env.get_observation_from_buffer()

        frames = [
            cv2.resize(
                cv2.cvtColor(obs_raw[f"rgb_{cam_id}"][-1], cv2.COLOR_RGB2BGR),
                (image_width, image_height),
            )
            for cam_id in camera_id_list
            if f"rgb_{cam_id}" in obs_raw
        ]
        bgr = np.vstack(frames) if len(frames) > 1 else frames[0]
        cv2.imshow("image", bgr)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

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

    # get initial tool space poses, for homing between episodes
    ts_pose_initial = []
    for id in id_list:
        ts_pose_initial.append(obs_raw[f"ts_pose_fb_{id}"][-1])

    #########################################
    # main loop starts
    #########################################
    while True:
        printOrNot(vbs_h1, "New episode. Episode ID: ", episode_id)
        input("Press Enter to start the episode.")
        killer = GracefulKiller()

        # open the per-episode log
        log = zarr.open(
            os.path.join(run_log_dir, f"episode_{episode_id}.zarr"), mode="w"
        )
        log.attrs["ckpt_path"] = pipeline_para["ckpt_path"]
        log.attrs["hardware_config_path"] = pipeline_para["hardware_config_path"]
        log.attrs["control_para"] = control_para
        horizon_count = 0

        while not killer.kill_now:
            # get current hardware time
            horizon_initial_time_s = env.current_hardware_time_s
            printOrNot(vbs_h1, "Starting new horizon at ", horizon_initial_time_s)

            # get observation from env
            obs_raw = env.get_observation_from_buffer()

            # plot the rgb image
            frames = [
                cv2.resize(
                    cv2.cvtColor(obs_raw[f"rgb_{cam_id}"][-1], cv2.COLOR_RGB2BGR),
                    (image_width, image_height),
                )
                for cam_id in camera_id_list
                if f"rgb_{cam_id}" in obs_raw
            ]
            bgr = np.vstack(frames) if len(frames) > 1 else frames[0]
            cv2.imshow("image", bgr)
            cv2.waitKey(10)

            # save low dim obs named as raw in new dict called obs_task
            obs_task = dict()
            raw_to_obs(obs_raw, obs_task, shape_meta)

            # set observation applies downsampling according to shape_meta
            controller.set_observation(obs_task["obs"])

            # inference. Returns one (T, 4, 4) array of absolute poses per arm.
            action_sparse_target_mats = controller.compute_sparse_control(device)

            # convert SE3 waypoints to the pose7 the server schedules
            outputs_ts_targets = [np.array] * len(id_list)
            for id in id_list:
                outputs_ts_targets[id] = su.SE3_to_pose7(
                    action_sparse_target_mats[id].reshape([-1, 4, 4])
                )

            # the "now" when the observation is taken
            action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
            timestamps = sparse_action_timesteps_s

            if control_para["pausing_mode"]:
                ax.cla()
                for id in id_list:
                    ax.plot3D(
                        action_sparse_target_mats[id][..., 0, 3],
                        action_sparse_target_mats[id][..., 1, 3],
                        action_sparse_target_mats[id][..., 2, 3],
                        color="red",
                        marker="o",
                        markersize=3,
                        label="Target (all)",
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
                        label="Target (exec horizon)",
                    )
                    ax.plot3D(
                        obs_raw[f"ts_pose_fb_{id}"][-1][0],
                        obs_raw[f"ts_pose_fb_{id}"][-1][1],
                        obs_raw[f"ts_pose_fb_{id}"][-1][2],
                        color="black",
                        marker="o",
                        markersize=8,
                        label="Current pose",
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Horizon {horizon_count}")
                ax.legend()
                set_axes_equal(ax)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.1)

                input("Press Enter to start executing the plotted actions.")
                action_start_time_s = env.current_hardware_time_s

            # log the whole action (no virtual target, unlike ACP logs)
            horizon_log = log.create_group(f"horizon_{horizon_count}")
            for id in id_list:
                horizon_log.create_dataset(
                    f"ts_nominal_targets_{id}", data=outputs_ts_targets[id]
                )
                horizon_log.create_dataset(
                    f"ts_pose_fb_{id}", data=obs_raw[f"ts_pose_fb_{id}"][-1]
                )
                horizon_log.create_dataset(
                    f"wrench_{id}", data=obs_raw[f"wrench_{id}"][-1]
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
                timestamps = timestamps[: control_para["sparse_execution_horizon"]]
                action_start_time_s = env.current_hardware_time_s

            if len(id_list) == 1:
                outputs_ts_targets = outputs_ts_targets[0].T  # N x 7 to 7 x N
            else:
                outputs_ts_targets = np.hstack(outputs_ts_targets).T  # 2xNx7 to 14xN

            # the same constant 6x6 stiffness for every waypoint and arm:
            # 6 x (6N) for one arm, 12 x (6N) for two, as ACP schedules it
            n_wp = outputs_ts_targets.shape[1]
            outputs_ts_stiffnesses = np.vstack(
                [np.tile(np.diag(control_para["stiffness_diag"]), n_wp)] * len(id_list)
            )
            for id in id_list:
                horizon_log.create_dataset(
                    f"ts_stiffnesses_{id}",
                    data=outputs_ts_stiffnesses[6 * id : 6 * id + 6],
                )

            # send to env for execution, through ACP's impedance controller
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
                    if f"rgb_{cam_id}" not in obs_raw:
                        continue
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

        # stop the robot where it is, otherwise the rest of the last chunk
        # still plays out after Ctrl+C: one waypoint at the measured pose,
        # 50 ms ahead, replaces everything queued after it
        obs_raw = env.get_observation_from_buffer()
        hold_pose7 = np.concatenate(
            [obs_raw[f"ts_pose_fb_{id}"][-1] for id in id_list]
        ).reshape(-1, 1)  # 7x1, or 14x1 for two arms
        env.schedule_controls(
            pose7_cmd=hold_pose7,
            timestamps=np.array([env.current_hardware_time_s + 0.05]) * 1000,
        )
        print("Robot stopped: holding the current pose.")

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
