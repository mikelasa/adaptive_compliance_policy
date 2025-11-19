import zarr
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
    EpisodeDataIncreImageBuffer,
)
from spatialmath.base import q2r, r2q
from spatialmath import SE3, SO3, UnitQuaternion
import concurrent.futures

from PyriteUtility.planning_control import compliance_helpers as ch
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

# Config for flip up (single robot)
dataset_path = dataset_folder_path + "/wipe_single_arm"
id_list = [0]

# # Config for vase wiping (bimanual)
# dataset_path = dataset_folder_path + "/vase_wiping_v6.3/"
# id_list = [0, 1]

wrench_moving_average_window_size = 1000  # should be around 1s of data
buffer = zarr.open(dataset_path, mode="r+")

num_of_process = 1
flag_plot = False
fin_every_n = 50

# struct that defines the parameters to simulate the penetration with a given stiffness
stiffness_estimation_para = {
    # penetration estimator
    "k_max": 5000,  # 1cm 50N maximum stiffness
    "k_min": 200,  # 1cm 2.5N minimum stiffness
    "f_low": 0.5, #lower bound of the force
    "f_high": 5,  #upper bound of the force
    "dim": 6, #3 or 6, 3 for translational, 6 for full 6D
    "characteristic_length": 0.02, #the characteristic length for rotational stiffness
    "vel_tol": 999.002,  # (not using) vel larger than this will trigger stiffness adjustment
}

flag_real = False
if "real" in dataset_path:
    flag_real = True

if flag_plot:
    assert num_of_process == 1, "Plotting is not supported for multi-process"


def process_episode(ep, ep_data, id_list):
    #for each episode, for robot in id_list:
    for id in id_list:
        print(f"Processing episode {ep}, id {id}: ")
        #extract robot pose and wrench
        ts_pose_fb = ep_data[f"ts_pose_fb_{id}"]
        wrench = ep_data[f"wrench_{id}"]

        # pre-allocate moving average array
        wrench_moving_average = np.zeros_like(wrench)

        # remove wrench measurement offset
        Noffset = 200
        wrench_offset = np.mean(wrench[:Noffset], axis=0)
        print("wrench offset: ", wrench_offset)

        # # FT300 only: flip the sign of the wrench
        # for i in range(6):
        #     wrench[:, i] = -wrench[:, i]

        # filter wrench using moving average
        N = wrench_moving_average_window_size
        print("Computing moving average")
        # fmt: off
        wrench_moving_average[:, 0] = np.convolve(wrench[:, 0], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 1] = np.convolve(wrench[:, 1], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 2] = np.convolve(wrench[:, 2], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 3] = np.convolve(wrench[:, 3], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 4] = np.convolve(wrench[:, 4], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 5] = np.convolve(wrench[:, 5], np.ones(N) / N, mode="same")
        # fmt: on
        wrench_time_stamps = ep_data[f"wrench_time_stamps_{id}"]
        robot_time_stamps = ep_data[f"robot_time_stamps_{id}"]

        if not flag_real:  # for simulation data
            ft_sensor_pose_fb = ep_data["ft_sensor_pose_fb"]

        num_robot_time_steps = len(robot_time_stamps)

        print("creating virtual target estimator")

        pe = ch.VirtualTargetEstimator(
            stiffness_estimation_para["k_max"],
            stiffness_estimation_para["k_min"],
            stiffness_estimation_para["f_low"],
            stiffness_estimation_para["f_high"],
            stiffness_estimation_para["dim"],
            stiffness_estimation_para["characteristic_length"],
            stiffness_estimation_para["vel_tol"],
        )

        ts_pose_virtual_target = np.zeros((num_robot_time_steps, 7))
        stiffness = np.zeros(num_robot_time_steps)
        mask_adjusted = [False] * num_robot_time_steps
        print("Running virtual target estimator")

        for t in range(num_robot_time_steps):
            pose7_WT = ts_pose_fb[t]
            SE3_WT = SE3.Rt(q2r(pose7_WT[3:7]), pose7_WT[0:3], check=False)

            # find the id in wrench_time_stamps where the time is closest to robot_time_stamps[t]
            t_wrench = np.argmin(np.abs(wrench_time_stamps - robot_time_stamps[t]))

            #apply moving average 
            if flag_real:
                wrench_T = wrench_moving_average[t_wrench]
            else:
                pose7_WS = ft_sensor_pose_fb[t]
                wrench_S = wrench_moving_average[t]
                SE3_WS = SE3.Rt(q2r(pose7_WS[3:7]), pose7_WS[0:3], check=False)
                SE3_ST = SE3_WS.inv() * SE3_WT
                wrench_T = SE3_ST.Ad().T @ wrench_S

            # compute velocity twist with a window of 20 time steps (10 before and 10 after)
            half_window_size = 10
            id_start = max(0, t - half_window_size)
            id_end = min(num_robot_time_steps - 1, t + half_window_size)
            window_size = id_end - id_start

            # compute twist rel_pose =  inv(SE3_start) * SE3_end
            SE3_start = su.pose7_to_SE3(ts_pose_fb[id_start])
            SE3_end = su.pose7_to_SE3(ts_pose_fb[id_end])
            twist_diff = su.SE3_to_spt(su.SE3_inv(SE3_start) @ SE3_end)

            # compute stiffness
            if stiffness_estimation_para["dim"] == 6:
                k, mat_TC, flag_adjusted = pe.update(wrench_T, twist_diff)
                SE3_TC = SE3(mat_TC)
            else:
                k, pos_TC, flag_adjusted = pe.update(wrench_T, twist_diff)
                SE3_TC = SE3.Rt(np.eye(3), pos_TC)
            SE3_WC = SE3_WT * SE3_TC

            ts_pose_virtual_target[t] = np.concatenate([SE3_WC.t, r2q(SE3_WC.R)])
            stiffness[t] = k
            mask_adjusted[t] = flag_adjusted

        ep_data[f"ts_pose_virtual_target_{id}"] = ts_pose_virtual_target
        ep_data[f"stiffness_{id}"] = stiffness
        print("Done")

        if flag_plot:
            print("Plotting...")
            plt.ion()  # to run GUI event loop
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            x = np.linspace(-0.02, 0.2, 20)
            y = np.linspace(-0.1, 0.1, 20)
            z = np.linspace(-0.1, 0.1, 20)
            ax.plot3D(x, y, z, color="blue", marker="o", markersize=3)
            ax.plot3D(x, y, z, color="red", marker="o", markersize=3)
            ax.set_title("Target and virtual target")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()

            ax.cla()
            ax.plot3D(
                ts_pose_fb[..., 0],
                ts_pose_fb[..., 1],
                ts_pose_fb[..., 2],
                color="red",
                marker="o",
                markersize=2,
            )

            ax.plot3D(
                ts_pose_virtual_target[..., 0],
                ts_pose_virtual_target[..., 1],
                ts_pose_virtual_target[..., 2],
                color="blue",
                marker="o",
                markersize=2,
            )
            # adjusted points
            ts_pose_fb_adjusted = ts_pose_fb[mask_adjusted]
            ts_pose_virtual_target_adjusted = ts_pose_virtual_target[mask_adjusted]
            ax.plot3D(
                ts_pose_fb_adjusted[..., 0],
                ts_pose_fb_adjusted[..., 1],
                ts_pose_fb_adjusted[..., 2],
                color="yellow",
                marker="o",
                markersize=3,
            )

            ax.plot3D(
                ts_pose_virtual_target_adjusted[..., 0],
                ts_pose_virtual_target_adjusted[..., 1],
                ts_pose_virtual_target_adjusted[..., 2],
                color="green",
                marker="o",
                markersize=3,
            )
            # starting point
            ax.plot3D(
                ts_pose_fb[0][0],
                ts_pose_fb[0][1],
                ts_pose_fb[0][2],
                color="black",
                marker="o",
                markersize=8,
            )

            # fin
            for i in np.arange(0, num_robot_time_steps, fin_every_n):
                ax.plot3D(
                    [ts_pose_fb[i][0], ts_pose_virtual_target[i][0]],
                    [ts_pose_fb[i][1], ts_pose_virtual_target[i][1]],
                    [ts_pose_fb[i][2], ts_pose_virtual_target[i][2]],
                    color="black",
                    marker="o",
                    markersize=2,
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            set_axes_equal(ax)

            plt.draw()
            input("Press Enter to continue...")
            return True

# dependign the number of processes, 
if num_of_process == 1:
    for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes"):
        process_episode(ep, ep_data, id_list)
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_process) as executor:
        futures = [
            executor.submit(
                process_episode,
                ep,
                ep_data,
                id_list,
            )
            for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes")
        ]
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                raise RuntimeError("Multi-processing failed!")


print("Done!")
