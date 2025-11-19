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


from PyriteEnvSuites.envs.task.manip_server_env import ManipServerEnv
from PyriteEnvSuites.utils.env_utils import ts_to_js_traj, pose9pose9s1_to_traj

from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.planning_control.mpc import ModelPredictiveControllerHybrid
from PyriteUtility.pytorch_utils.model_io import load_policy
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.umi_utils.usb_util import reset_all_elgato_devices
from PyriteUtility.common import GracefulKiller
from DemoPlayerEnv import DemoReplayEnv

if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

"""
Offline inference runner to test the virtual target policy on recorded demonstrations.
Loads recorded demo data, runs policy inference, and logs predictions without robot execution.
"""

def main():
    control_para = {
        "raw_time_step_s": 0.001,
        "slow_down_factor": 1.5,
        "sparse_execution_horizon": 12,
        "device": "cuda",
        "max_horizons": 10,  # Limit for testing
    }
    pipeline_para = {
        "ckpt_path": "/2025.11.04_10.05.05_Wipe_single_arm_Wipe_single_arm",
        "control_log_path": control_log_folder_path + "/offline_predictions_4/",
    }
    verbose = 1
    demo_zarr_path = "/home/robotlab/ACP/data/real/wipe_single_arm/episode_1761901305"

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
            print(*args)

    vbs_h1 = verbose + 1
    vbs_h2 = verbose
    vbs_p = verbose - 1

    # load policy
    print("Loading policy: ", checkpoint_folder_path + pipeline_para["ckpt_path"])
    device = torch.device(control_para["device"])
    policy, shape_meta = load_policy(
        checkpoint_folder_path + pipeline_para["ckpt_path"], device
    )

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

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

    # create the replay env - path to raw episode data
    env = DemoReplayEnv(
        demo_store_path=demo_zarr_path,
        query_sizes=query_sizes
    )
    env.reset()

    # set timestep
    p_timestep_s = control_para["raw_time_step_s"]
    sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    sparse_execution_horizon = (
        sparse_action_down_sample_steps * control_para["sparse_execution_horizon"]
    )
    sparse_action_timesteps_s = (
        np.arange(0, sparse_action_horizon)
        * sparse_action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )

    print("sparse_action_timesteps_s: ", sparse_action_timesteps_s)

    # determine action type
    action_type = "pose9"
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

    if action_type == "pose9pose9s1":
        action_to_trajectory = pose9pose9s1_to_traj
    else:
        raise RuntimeError("unsupported")

    printOrNot(vbs_h2, "Creating MPC.")
    controller = ModelPredictiveControllerHybrid(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        sparse_execution_horizon=sparse_execution_horizon,
    )
    controller.set_time_offset(env)

    # setup logging
    os.makedirs(pipeline_para["control_log_path"], exist_ok=True)
    log_store = zarr.DirectoryStore(path=pipeline_para["control_log_path"])
    log = zarr.open(store=log_store, mode="w")

    horizon_count = 0

    #########################################
    # main loop starts
    #########################################
    printOrNot(vbs_h1, "Starting offline inference on demo episode")

    killer = GracefulKiller()

    # Skip initial frames until we have enough history for the observation window
    max_query = max(rgb_query_size, ts_pose_query_size, wrench_query_size)
    printOrNot(vbs_h2, f"Skipping first {max_query} frames to build observation history...")
    for _ in range(max_query):
        if not env.step_time():
            printOrNot(vbs_h1, "Episode too short!")
            return

    while not killer.kill_now:
        horizon_initial_time_s = env.current_hardware_time_s
        printOrNot(vbs_p, f"Horizon {horizon_count} at time {horizon_initial_time_s:.3f}s")

        # get observation from replay buffer
        obs_raw = env.get_observation_from_buffer()

        # optional: display image
        if len(id_list) == 1:
            rgb = obs_raw["rgb_0"][-1]
        else:
            rgb = np.vstack([obs_raw[f"rgb_{i}"][-1] for i in id_list])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", bgr)
        cv2.waitKey(1)

        # convert to model format
        obs_task = dict()
        raw_to_obs(obs_raw, obs_task, shape_meta)

        assert action_type == "pose9pose9s1"

        # run policy inference and measure time
        inference_start_time = time.time()
        controller.set_observation(obs_task["obs"])
        (action_sparse_target_mats, action_sparse_vt_mats, 
         action_stiffnesses) = controller.compute_sparse_control(device)
        inference_time_ms = (time.time() - inference_start_time) * 1000.0
        
        printOrNot(vbs_h1, f"Inference time: {inference_time_ms:.2f} ms")
        print(action_stiffnesses)

        action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
        timestamps = sparse_action_timesteps_s
        print("timestamps: ", timestamps)

        # Log raw model predictions
        horizon_log = log.create_group(f"horizon_{horizon_count}")
        for id in id_list:
            horizon_log.create_dataset(
                f"nominal_targets_{id}", data=action_sparse_target_mats[id]
            )
            horizon_log.create_dataset(
                f"virtual_targets_{id}", data=action_sparse_vt_mats[id]
            )
            horizon_log.create_dataset(
                f"stiffnesses_{id}", data=action_stiffnesses[id]
            )
        horizon_log.create_dataset(
            "timestamps_s", data=timestamps + action_start_time_s
        )
        horizon_log.create_dataset(
            "inference_time_ms", data=np.array([inference_time_ms])
        )
        
        horizon_count += 1
        
        if horizon_count % 50 == 0:
            printOrNot(vbs_h1, f"Processed {horizon_count} horizons...")
        
        # Stop after max horizons for testing
        if horizon_count >= control_para["max_horizons"]:
            printOrNot(vbs_h1, f"Reached max horizons limit ({control_para['max_horizons']})")
            break
        
        # Advance demo by execution horizon to simulate robot motion time
        # This matches the real control loop where the robot executes sparse_execution_horizon 
        # waypoints before the next inference
        printOrNot(vbs_p, f"Advancing demo by {sparse_execution_horizon} frames...")
        for _ in range(sparse_execution_horizon):
            if not env.step_time():
                printOrNot(vbs_h1, "Reached end of demonstration data")
                return

    printOrNot(vbs_h1, f"Offline inference complete. Processed {horizon_count} horizons.")
    printOrNot(vbs_h1, f"Predictions saved to: {pipeline_para['control_log_path']}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
