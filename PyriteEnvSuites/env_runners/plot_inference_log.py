"""
Plot inference log data from a zarr directory.

Each horizon_N group contains:
  - timestamps_s          (16,)    predicted timestamps for the 16-step action horizon
  - ts_stiffnesses_0      (6, 96)  stiffness matrix diagonal, stored as (6 dof, 16 steps * 6 dof)
  - ts_virtual_targets_0  (16, 7)  predicted virtual target poses
  - ts_nominal_targets_0  (16, 7)  predicted nominal target poses

Usage:
    python plot_inference_log.py <folder_name>        # loads from /home/robotlab/data/resultados/<folder_name>
    python plot_inference_log.py                      # loads from /home/robotlab/data/inference_log/temp
"""

import sys
import numpy as np
import zarr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import os
script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, "../../"))
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal

DOF_LABELS = ["Kx", "Ky", "Kz", "Krx", "Kry", "Krz"]
WRENCH_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
WRENCH_UNITS = ["N", "N", "N", "Nm", "Nm", "Nm"]
FIN_EVERY_N = 5  # draw connecting lines every N steps
WRENCH_OFFSET_SAMPLES = 200    # same as Noffset in postprocess script
WRENCH_MA_WINDOW = 1000        # same as wrench_moving_average_window_size


TEST_PATH    = "/home/robotlab/data/resultados/temp"
COMPARE_PATH = "/home/robotlab/data/resultados/cajagopro"
TRAIN_PATH = "/home/robotlab/data/real_processed/flip_up_230_500/data/episode_1770284952"
FLAG_GROUND_TRUTH = False  # set to False to skip loading/plotting ground-truth episode
FLAG_COMPARE = False        # set to False to skip the comparison plot


def extract_stiffness_diag(ts_stiffnesses: np.ndarray) -> np.ndarray:
    """
    Extract diagonal stiffness values per timestep.

    Args:
        ts_stiffnesses: (6, n_steps * 6) array

    Returns:
        (6, n_steps) array of stiffness values per DOF per timestep
    """
    n_dof = 6
    n_steps = ts_stiffnesses.shape[1] // n_dof
    stiff = ts_stiffnesses.reshape(n_dof, n_steps, n_dof)  # (6, n_steps, 6)
    return np.array([stiff[i, :, i] for i in range(n_dof)])  # (6, n_steps)


def load_log(dataset_path: str):
    """
    Load all horizon groups. For each horizon we take step index 0
    (the current timestep), giving one sample per inference call.

    Returns:
        timestamps:       (N,)    wall-clock timestamps normalized to 0
        stiffness:        (6, N)  stiffness per DOF
        nominal_targets:  (N, 3)  nominal target XYZ positions
        virtual_targets:  (N, 3)  virtual target XYZ positions
    """
    buffer = zarr.open(dataset_path, mode="r")
    horizon_keys = sorted(buffer.keys(), key=lambda k: int(k.split("_")[1]))

    timestamps = []
    stiffness_per_step = []
    nominal_pos = []
    virtual_pos = []

    for key in horizon_keys:
        h = buffer[key]
        timestamps.append(h["timestamps_s"][0])
        stiffness_per_step.append(extract_stiffness_diag(h["ts_stiffnesses_0"][:])[:, 0])
        nominal_pos.append(h["ts_nominal_targets_0"][0, :3])
        virtual_pos.append(h["ts_virtual_targets_0"][0, :3])

    timestamps = np.array(timestamps)
    timestamps -= timestamps[0]

    return (
        timestamps,
        np.stack(stiffness_per_step, axis=1),   # (6, N)
        np.array(nominal_pos),                   # (N, 3)
        np.array(virtual_pos),                   # (N, 3)
    )


def load_wrench_log(dataset_path: str):
    """
    Load impedance_controller.log from dataset_path.
    Format: timestamp  Fx  Fy  Fz  Tx  Ty  Tz  (space-separated, one row per control step)

    Returns:
        timestamps: (N,)   normalized to 0
        wrench:     (N, 6) cartesian wrench [Fx Fy Fz Tx Ty Tz]
    """
    log_path = os.path.join(dataset_path, "impedance_controller.log")
    data = np.loadtxt(log_path)
    timestamps = data[:, 0]
    timestamps = np.cumsum(timestamps)   # dt column → cumulative time
    timestamps -= timestamps[0]
    wrench = data[:, 1:]                 # (N, 6)
    return timestamps, wrench


def process_wrench(wrench: np.ndarray,
                   offset_samples: int = WRENCH_OFFSET_SAMPLES,
                   ma_window: int = WRENCH_MA_WINDOW) -> np.ndarray:
    """
    Apply the same preprocessing used in postprocess_add_virtual_target_label.py:
      1. subtract mean of the first `offset_samples` rows (static offset removal)
      2. apply a causal moving-average filter with `ma_window` samples per channel

    Args:
        wrench: (N, 6) raw wrench

    Returns:
        (N, 6) filtered wrench
    """
    offset = np.mean(wrench[:offset_samples], axis=0)
    w = wrench - offset
    kernel = np.ones(ma_window) / ma_window
    filtered = np.stack(
        [np.convolve(w[:, i], kernel, mode="same") for i in range(w.shape[1])],
        axis=1,
    )
    return filtered


def load_episode(episode_path: str, robot_id: int = 0):
    """
    Load ground-truth stiffness and poses from a processed demonstration episode.

    Args:
        episode_path: path to the zarr episode directory
        robot_id:     robot index (matches the _{id} suffix in the zarr keys)

    Returns:
        timestamps:  (N,)   wall-clock timestamps normalized to 0
        stiffness:   (N,)   scalar stiffness estimated by VirtualTargetEstimator
        nominal_pos: (N, 3) robot end-effector XYZ (ts_pose_fb)
        virtual_pos: (N, 3) virtual target XYZ (ts_pose_virtual_target)
    """
    ep = zarr.open(episode_path, mode="r")
    timestamps = ep[f"robot_time_stamps_{robot_id}"][:]
    timestamps -= timestamps[0]
    stiffness = ep[f"stiffness_{robot_id}"][:]
    nominal_pos = ep[f"ts_pose_fb_{robot_id}"][:, :3]
    virtual_pos = ep[f"ts_pose_virtual_target_{robot_id}"][:, :3]
    return timestamps, stiffness, nominal_pos, virtual_pos


def plot(dataset_path: str, episode_path: str = None,
         plot_ground_truth: bool = FLAG_GROUND_TRUTH,
         compare_path: str = None):
    print(f"Loading: {dataset_path}")
    timestamps, stiffness, nominal_pos, virtual_pos = load_log(dataset_path)
    N = stiffness.shape[1]
    print(f"  {N} inference steps, duration ~{timestamps[-1]:.1f}s")

    wrench_data = None
    wrench_log = os.path.join(dataset_path, "impedance_controller.log")
    if os.path.exists(wrench_log):
        print(f"Loading wrench log: {wrench_log}")
        wrench_ts, wrench = load_wrench_log(dataset_path)
        wrench_data = (wrench_ts, wrench)
    else:
        print(f"  No impedance_controller.log found in {dataset_path}, skipping wrench plot")

    ep_data = None
    if plot_ground_truth and episode_path is not None:
        print(f"Loading episode: {episode_path}")
        ep_ts, ep_stiffness, ep_nominal_pos, ep_virtual_pos = load_episode(episode_path)
        ep_data = (ep_ts, ep_stiffness, ep_nominal_pos, ep_virtual_pos)

    # load compare data if requested
    cmp_data = None
    cmp_wrench_data = None
    cmp_label = ""
    if compare_path is not None:
        print(f"Loading compare: {compare_path}")
        try:
            cmp_ts, cmp_stiffness, cmp_nominal_pos, cmp_virtual_pos = load_log(compare_path)
            cmp_data = (cmp_ts, cmp_stiffness, cmp_nominal_pos, cmp_virtual_pos)
            cmp_label = os.path.basename(compare_path)
            cmp_wrench_log = os.path.join(compare_path, "impedance_controller.log")
            if os.path.exists(cmp_wrench_log):
                cmp_wts, cmp_w = load_wrench_log(compare_path)
                cmp_wrench_data = (cmp_wts, cmp_w)
        except Exception as e:
            print(f"  WARNING: could not load compare path — {e}. Skipping comparison.")

    test_label = os.path.basename(dataset_path)

    # --- Figure 1: stiffness ---
    has_ep = ep_data is not None
    ncols = 2 if has_ep else 1
    fig1, axes_grid = plt.subplots(6, ncols, figsize=(12 if has_ep else 8, 10),
                                   squeeze=False)
    fig1.suptitle("Stiffness: inference (left) vs ground truth (right)" if has_ep
                  else f"Stiffness\n{dataset_path}")

    for i, label in enumerate(DOF_LABELS):
        unit = "N/m" if i < 3 else "Nm/rad"
        ax_inf = axes_grid[i][0]
        ax_inf.plot(timestamps, stiffness[i], color="steelblue",
                    marker=".", markersize=4, label=test_label)
        if cmp_data is not None:
            ax_inf.plot(cmp_data[0], cmp_data[1][i], color="darkorange",
                        linewidth=0.9, label=cmp_label)
            ax_inf.legend(fontsize=6, loc="upper right")
        ax_inf.set_ylabel(unit)
        ax_inf.set_title(f"{label} — inference")
        ax_inf.grid(True)

        if has_ep:
            ax_gt = axes_grid[i][1]
            if i < 3:
                ax_gt.plot(ep_data[0], ep_data[1], color="orange", linewidth=0.8)
                ax_gt.set_title(f"{label} — ground truth")
            else:
                ax_gt.set_title(f"{label} — ground truth (n/a)")
                ax_gt.text(0.5, 0.5, "not estimated (dim=3)",
                           ha="center", va="center", transform=ax_gt.transAxes,
                           color="gray", fontsize=9)
            ax_gt.set_ylabel(unit)
            ax_gt.grid(True)

    for col in range(ncols):
        axes_grid[-1][col].set_xlabel("Time (s)")
    fig1.tight_layout()

    # --- Figure 2: 3D trajectory ---
    fig2 = plt.figure(figsize=(10, 8))
    ax3d = fig2.add_subplot(111, projection="3d")
    ax3d.set_title(f"Nominal vs virtual target trajectory\n{dataset_path}")

    ax3d.plot(nominal_pos[:, 0], nominal_pos[:, 1], nominal_pos[:, 2],
              color="red", marker="o", markersize=2, label=f"nominal: {test_label}")
    ax3d.plot(virtual_pos[:, 0], virtual_pos[:, 1], virtual_pos[:, 2],
              color="blue", marker="o", markersize=2, label=f"virtual: {test_label}")
    ax3d.plot([nominal_pos[0, 0]], [nominal_pos[0, 1]], [nominal_pos[0, 2]],
              color="black", marker="o", markersize=8, label="start")
    for i in range(0, N, FIN_EVERY_N):
        ax3d.plot(
            [nominal_pos[i, 0], virtual_pos[i, 0]],
            [nominal_pos[i, 1], virtual_pos[i, 1]],
            [nominal_pos[i, 2], virtual_pos[i, 2]],
            color="gray", linewidth=0.8,
        )

    if cmp_data is not None:
        cmp_N = len(cmp_data[2])
        ax3d.plot(cmp_data[2][:, 0], cmp_data[2][:, 1], cmp_data[2][:, 2],
                  color="salmon", linestyle="--", marker="o", markersize=1,
                  label=f"nominal: {cmp_label}")
        ax3d.plot(cmp_data[3][:, 0], cmp_data[3][:, 1], cmp_data[3][:, 2],
                  color="cornflowerblue", linestyle="--", marker="o", markersize=1,
                  label=f"virtual: {cmp_label}")
        for i in range(0, cmp_N, FIN_EVERY_N):
            ax3d.plot(
                [cmp_data[2][i, 0], cmp_data[3][i, 0]],
                [cmp_data[2][i, 1], cmp_data[3][i, 1]],
                [cmp_data[2][i, 2], cmp_data[3][i, 2]],
                color="lightyellow", linewidth=0.6,
            )

    if ep_data is not None:
        ep_ts, ep_stiffness, ep_nominal_pos, ep_virtual_pos = ep_data
        ep_N = len(ep_nominal_pos)
        ax3d.plot(ep_nominal_pos[:, 0], ep_nominal_pos[:, 1], ep_nominal_pos[:, 2],
                  color="salmon", linestyle="--", marker="o", markersize=1,
                  label="gt: robot pose")
        ax3d.plot(ep_virtual_pos[:, 0], ep_virtual_pos[:, 1], ep_virtual_pos[:, 2],
                  color="cornflowerblue", linestyle="--", marker="o", markersize=1,
                  label="gt: virtual target")
        for i in range(0, ep_N, FIN_EVERY_N):
            ax3d.plot(
                [ep_nominal_pos[i, 0], ep_virtual_pos[i, 0]],
                [ep_nominal_pos[i, 1], ep_virtual_pos[i, 1]],
                [ep_nominal_pos[i, 2], ep_virtual_pos[i, 2]],
                color="lightgray", linewidth=0.6,
            )

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(fontsize=7)
    set_axes_equal(ax3d)

    # --- Figure 3: filtered wrench + force norm ---
    if wrench_data is not None:
        w_ts, w = wrench_data
        w_filt = process_wrench(w)
        force_norm = np.linalg.norm(w_filt[:, :3], axis=1)

        cmp_filt_data = None
        if cmp_wrench_data is not None:
            cmp_wts, cmp_w = cmp_wrench_data
            cmp_w_filt = process_wrench(cmp_w)
            cmp_force_norm = np.linalg.norm(cmp_w_filt[:, :3], axis=1)
            cmp_filt_data = (cmp_wts, cmp_w_filt, cmp_force_norm)

        fig3, f_axes = plt.subplots(7, 1, figsize=(12, 12), sharex=False)
        fig3.suptitle(
            f"Cartesian wrench — filtered (offset={WRENCH_OFFSET_SAMPLES} samples, "
            f"MA={WRENCH_MA_WINDOW} samples)\n{dataset_path}"
        )
        for i, (ax, label, unit) in enumerate(zip(f_axes[:6], WRENCH_LABELS, WRENCH_UNITS)):
            ax.plot(w_ts, w[:, i], color="lightgray", linewidth=0.6, label="raw")
            ax.plot(w_ts, w_filt[:, i], color="steelblue", linewidth=0.9, label=f"filtered: {test_label}")
            if cmp_filt_data is not None:
                ax.plot(cmp_filt_data[0], cmp_filt_data[1][:, i],
                        color="darkorange", linewidth=0.9, linestyle="--", label=f"filtered: {cmp_label}")
            ax.set_ylabel(unit)
            ax.set_title(label)
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(True)
        f_axes[6].plot(w_ts, force_norm, color="steelblue", linewidth=1.0, label=test_label)
        if cmp_filt_data is not None:
            f_axes[6].plot(cmp_filt_data[0], cmp_filt_data[2],
                           color="darkorange", linewidth=1.0, linestyle="--", label=cmp_label)
            f_axes[6].legend(fontsize=6, loc="upper right")
        f_axes[6].set_ylabel("N")
        f_axes[6].set_title("||F|| (filtered)")
        f_axes[6].grid(True)
        f_axes[-1].set_xlabel("Time (s)")
        fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot(TEST_PATH, episode_path=TRAIN_PATH,
         compare_path=COMPARE_PATH if FLAG_COMPARE else None)
