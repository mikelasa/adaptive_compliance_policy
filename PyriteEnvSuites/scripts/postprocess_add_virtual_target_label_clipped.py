import zarr
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt

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
dataset_path = dataset_folder_path + "/demonstration_impacts/E1_minimum_demos/200demos"
id_list = [0]

# # Config for vase wiping (bimanual)
# dataset_path = dataset_folder_path + "/vase_wiping_v6.3/"
# id_list = [0, 1]

# Butterworth low-pass filter matching inference-time force_filtering_para
wrench_filter_cutoff_hz = 5.0
wrench_filter_order = 5
wrench_filter_fs = 1000.0
buffer = zarr.open(dataset_path, mode="r+")

num_of_process = 32
flag_plot = False
fin_every_n = 50

stiffness_estimation_para = {
    "k_max": 2000,  # 1cm 50N maximum stiffness
    "k_min": 500,  # 1cm 2.5N minimum stiffness
    "f_low": 7, #lower bound of the force
    "f_high": 17,  #upper bound of the force
    "max_disp": 0.034,  #
    "dim": 3,
    "characteristic_length": 1,
    "vel_tol": 999.002,
}

flag_real = False
if "real" in dataset_path:
    flag_real = True

if flag_plot:
    assert num_of_process == 1, "Plotting is not supported for multi-process"


def cap_displacement(pos_TC, max_disp):
    """Cap virtual target displacement magnitude, preserving direction."""
    norm = np.linalg.norm(pos_TC)
    if norm > max_disp:
        return pos_TC / norm * max_disp
    return pos_TC


def process_episode(ep, ep_data, id_list):
    for id in id_list:
        print(f"Processing episode {ep}, id {id}: ")
        ts_pose_fb = ep_data[f"ts_pose_fb_{id}"]
        wrench = ep_data[f"wrench_{id}"]

        wrench_moving_average = np.zeros_like(wrench)

        Noffset = 200
        wrench_offset = np.mean(wrench[:Noffset], axis=0)
        print("wrench offset: ", wrench_offset)
        wrench = wrench - wrench_offset

        # filter wrench using zero-phase Butterworth (matches inference-time filter params)
        print("Computing Butterworth filter")
        sos = butter(wrench_filter_order, wrench_filter_cutoff_hz,
                     fs=wrench_filter_fs, btype="low", output="sos")
        for i in range(6):
            wrench_moving_average[:, i] = sosfiltfilt(sos, wrench[:, i])
        wrench_time_stamps = ep_data[f"wrench_time_stamps_{id}"]
        robot_time_stamps = ep_data[f"robot_time_stamps_{id}"]

        if not flag_real:
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

            t_wrench = np.argmin(np.abs(wrench_time_stamps - robot_time_stamps[t]))

            if flag_real:
                wrench_O = wrench_moving_average[t_wrench]
                R = SE3_WT.R  # rotation tool→world; R.T converts world→tool
                wrench_T = np.concatenate([R.T @ wrench_O[:3], R.T @ wrench_O[3:]])
            else:
                pose7_WS = ft_sensor_pose_fb[t]
                wrench_S = wrench_moving_average[t]
                SE3_WS = SE3.Rt(q2r(pose7_WS[3:7]), pose7_WS[0:3], check=False)
                SE3_ST = SE3_WS.inv() * SE3_WT
                wrench_T = SE3_ST.Ad().T @ wrench_S

            half_window_size = 10
            id_start = max(0, t - half_window_size)
            id_end = min(num_robot_time_steps - 1, t + half_window_size)

            SE3_start = su.pose7_to_SE3(ts_pose_fb[id_start])
            SE3_end = su.pose7_to_SE3(ts_pose_fb[id_end])
            twist_diff = su.SE3_to_spt(su.SE3_inv(SE3_start) @ SE3_end)

            if stiffness_estimation_para["dim"] == 6:
                k, mat_TC, flag_adjusted = pe.update(wrench_T, twist_diff)
                SE3_TC = SE3(mat_TC)
            else:
                k, pos_TC, flag_adjusted = pe.update(wrench_T, twist_diff)
                pos_TC = cap_displacement(pos_TC, stiffness_estimation_para["max_disp"])
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
        plt.ion()
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(121, projection="3d")
        ax_k = fig.add_subplot(122)

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
        ts_pose_fb_adjusted = np.array(ts_pose_fb)[mask_adjusted]
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
        ax.plot3D(
            ts_pose_fb[0][0],
            ts_pose_fb[0][1],
            ts_pose_fb[0][2],
            color="black",
            marker="o",
            markersize=8,
        )
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

        # stiffness over time
        ax_k.cla()
        t_axis = np.arange(num_robot_time_steps)
        ax_k.plot(t_axis, stiffness, color="purple", linewidth=1.5, label="stiffness k")
        ax_k.axhline(stiffness_estimation_para["k_min"], color="red",   linestyle="--", linewidth=1, label=f"k_min={stiffness_estimation_para['k_min']}")
        ax_k.axhline(stiffness_estimation_para["k_max"], color="green", linestyle="--", linewidth=1, label=f"k_max={stiffness_estimation_para['k_max']}")
        ax_k.set_xlabel("timestep")
        ax_k.set_ylabel("k [N/m]")
        ax_k.set_title(f"Stiffness (clipped) — episode {ep}")
        ax_k.set_ylim(stiffness_estimation_para["k_min"] - 100, stiffness_estimation_para["k_max"] + 100)
        ax_k.legend()

        plt.tight_layout()
        plt.draw()
        input("Press Enter to continue...")

    return True


if num_of_process == 1:
    for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes"):
        process_episode(ep, ep_data, id_list)
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_process) as executor:
        futures = [
            executor.submit(process_episode, ep, ep_data, id_list)
            for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes")
        ]
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                raise RuntimeError("Multi-processing failed!")

print("Done!")