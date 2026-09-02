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
SPARSE_EXECUTION_HORIZON = 12  # executed steps per horizon (control_para["sparse_execution_horizon"])


TEST_PATH    = "/home/robotlab/data/resultados-tests/singleCamera/preprocesado35f/ACP/caja11"
#TEST_PATH    = "/home/robotlab/data/resultados-tests/temp"
COMPARE_PATH = "/home/robotlab/data/resultados-tests/singleCamera/preprocesado35f/bicrossDAT/caja11"
TRAIN_PATH = "/home/robotlab/data/real_processed/V3/flip_up_V3_distractors_500K_30F_0045c/data/episode_1778746728"
# caja1: episode_1778746728
# caja2: episode_1778749386
FLAG_GROUND_TRUTH = True      # set to False to skip loading/plotting ground-truth episode
FLAG_COMPARE = True        # set to False to skip the comparison plot episode_1778754577 episode_1778746728


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
    Load ground-truth stiffness, poses, and wrench from a processed demonstration episode.

    Args:
        episode_path: path to the zarr episode directory
        robot_id:     robot index (matches the _{id} suffix in the zarr keys)

    Returns:
        timestamps:    (N,)   wall-clock timestamps normalized to 0
        stiffness:     (N,)   scalar stiffness estimated by VirtualTargetEstimator
        nominal_pos:   (N, 3) robot end-effector XYZ (ts_pose_fb)
        virtual_pos:   (N, 3) virtual target XYZ (ts_pose_virtual_target)
        wrench_ts:     (M,)   wrench timestamps in seconds, normalized to 0
        wrench_raw:    (M, 6) raw wrench [Fx Fy Fz Tx Ty Tz]
        wrench_filt:   (M, 6) filtered wrench (same preprocessing as postprocess script)
    """
    ep = zarr.open(episode_path, mode="r")
    timestamps = ep[f"robot_time_stamps_{robot_id}"][:]
    timestamps -= timestamps[0]
    stiffness = ep[f"stiffness_{robot_id}"][:]
    nominal_pos = ep[f"ts_pose_fb_{robot_id}"][:, :3]
    virtual_pos = ep[f"ts_pose_virtual_target_{robot_id}"][:, :3]
    wrench_ts = ep[f"wrench_time_stamps_{robot_id}"][:]
    wrench_ts = (wrench_ts - wrench_ts[0]) / 1000.0   # ms → s, normalized to 0
    wrench_raw = ep[f"wrench_{robot_id}"][:]
    wrench_filt = ep[f"wrench_filtered_{robot_id}"][:]
    return timestamps, stiffness, nominal_pos, virtual_pos, wrench_ts, wrench_raw, wrench_filt


def load_stiffness_sequences(dataset_path: str):
    """
    Load full per-step stiffness diagonal for every horizon.

    Returns:
        sequences: list of (6, N_steps) arrays, one entry per horizon.
    """
    buffer = zarr.open(dataset_path, mode="r")
    horizon_keys = sorted(buffer.keys(), key=lambda k: int(k.split("_")[1]))
    sequences = []
    for key in horizon_keys:
        h = buffer[key]
        sequences.append(extract_stiffness_diag(h["ts_stiffnesses_0"][:]))  # (6, N_steps)
    return sequences


def load_scalar_stiffness_sequences(dataset_path: str):
    """
    Load the raw scalar stiffness sequences logged by the runner for each horizon.

    Returns:
        scalar_stiffnesses: list of (N_steps,) arrays, one per horizon.
                            None entries mean the key was absent (old log format).
        has_scalars:        True if at least one horizon had the key.
    """
    buffer = zarr.open(dataset_path, mode="r")
    horizon_keys = sorted(buffer.keys(), key=lambda k: int(k.split("_")[1]))

    scalar_stiffnesses = []
    has_scalars = False
    for key in horizon_keys:
        h = buffer[key]
        if "stiffness_scalars_0" in h:
            scalar_stiffnesses.append(h["stiffness_scalars_0"][:])
            has_scalars = True
        else:
            scalar_stiffnesses.append(None)

    return scalar_stiffnesses, has_scalars


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
    ep_wrench_data = None
    if plot_ground_truth and episode_path is not None:
        print(f"Loading episode: {episode_path}")
        ep_ts, ep_stiffness, ep_nominal_pos, ep_virtual_pos, ep_wts, ep_wraw, ep_wfilt = load_episode(episode_path)
        ep_data = (ep_ts, ep_stiffness, ep_nominal_pos, ep_virtual_pos)
        ep_wrench_data = (ep_wts, ep_wraw, ep_wfilt)

    # load compare data if requested
    cmp_data = None
    cmp_wrench_data = None
    cmp_label = ""
    if compare_path is not None:
        print(f"Loading compare: {compare_path}")
        try:
            cmp_ts, cmp_stiffness, cmp_nominal_pos, cmp_virtual_pos = load_log(compare_path)
            cmp_data = (cmp_ts, cmp_stiffness, cmp_nominal_pos, cmp_virtual_pos)
            cmp_label = os.path.basename(os.path.dirname(os.path.normpath(compare_path)))
            cmp_wrench_log = os.path.join(compare_path, "impedance_controller.log")
            if os.path.exists(cmp_wrench_log):
                cmp_wts, cmp_w = load_wrench_log(compare_path)
                cmp_wrench_data = (cmp_wts, cmp_w)
        except Exception as e:
            print(f"  WARNING: could not load compare path — {e}. Skipping comparison.")

    test_label = os.path.basename(os.path.dirname(os.path.normpath(dataset_path)))

    # --- Figure 1: translational stiffness (Kx, Ky, Kz) with horizon X axis ---
    def _norm_ts(ts):
        d = ts[-1] - ts[0]
        return (ts - ts[0]) / d if d > 0 else ts - ts[0]

    stiff_seqs = load_stiffness_sequences(dataset_path)
    cmp_stiff_seqs = None
    if compare_path is not None and cmp_data is not None:
        try:
            cmp_stiff_seqs = load_stiffness_sequences(compare_path)
        except Exception as e:
            print(f"  WARNING: could not load compare stiffness sequences — {e}")

    _TRANS_COLORS = ["red", "green", "blue"]
    _TRANS_AXIS_LABELS = ["x-axis stiffness", "y-axis stiffness", "z-axis stiffness"]

    def _plot_stiff_dof(ax, sequences, dof_idx, color, series_label,
                        lw_exec=1.5, lw_noexec=0.8, alpha=1.0):
        exec_end = min(SPARSE_EXECUTION_HORIZON, sequences[0].shape[1])
        for h_idx, seq in enumerate(sequences):
            x = h_idx + np.arange(seq.shape[1]) / SPARSE_EXECUTION_HORIZON
            vals = seq[dof_idx]
            lbl = series_label if h_idx == 0 else "_nolegend_"
            ax.plot(x[:exec_end], vals[:exec_end], color=color,
                    linewidth=lw_exec, alpha=alpha, label=lbl)
            if exec_end < seq.shape[1]:
                ax.plot(x[exec_end - 1:], vals[exec_end - 1:],
                        color=color, linewidth=lw_noexec, linestyle="--", alpha=alpha)

    from matplotlib.lines import Line2D
    _noexec_handle = Line2D([0], [0], color="gray", linewidth=1.0,
                            linestyle="--", label="not executed action")

    def _fill_ax(ax, sequences, subtitle):
        for i, (dof_label, color) in enumerate(zip(_TRANS_AXIS_LABELS, _TRANS_COLORS)):
            _plot_stiff_dof(ax, sequences, i, color=color, series_label=dof_label)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [_noexec_handle],
                  labels=labels + ["not executed action"],
                  fontsize=7, loc="upper right")
        ax.set_title(subtitle)
        ax.set_ylabel("N/m")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.set_xlim(-0.5, N + 0.5)
        for j in range(1, N):
            ax.axvline(j, color="gray", linewidth=0.4, linestyle=":", alpha=0.5)

    nrows = 2 if cmp_stiff_seqs is not None else 1
    fig_w1 = max(12, min(N * 1.0, 40))
    fig1, axes_f1 = plt.subplots(nrows, 1, figsize=(fig_w1, 4 * nrows),
                                  sharex=True, squeeze=False)
    fig1.suptitle("Predicted Stiffness Value along Each World Coordinate Axis")

    _fill_ax(axes_f1[0][0], stiff_seqs, test_label)
    if cmp_stiff_seqs is not None:
        _fill_ax(axes_f1[1][0], cmp_stiff_seqs, cmp_label)

    # X-axis labels only on bottom subplot
    axes_f1[-1][0].set_xticks(np.arange(N))
    axes_f1[-1][0].set_xticklabels(
        [f"Horizon {j + 1}" for j in range(N)],
        rotation=45, ha="right", fontsize=7,
    )
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
    if wrench_data is not None or ep_wrench_data is not None:
        w_filt = force_norm = w_ts = w = None
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

        ep_force_norm = None
        if ep_wrench_data is not None:
            ep_force_norm = np.linalg.norm(ep_wrench_data[2][:, :3], axis=1)

        w_ts_n       = _norm_ts(w_ts)                  if w_ts is not None        else None
        cmp_ts_n     = _norm_ts(cmp_filt_data[0])      if cmp_filt_data is not None else None
        ep_ts_n      = _norm_ts(ep_wrench_data[0])     if ep_wrench_data is not None else None

        fig3, f_axes = plt.subplots(7, 1, figsize=(12, 12), sharex=False)
        fig3.suptitle(
            f"Cartesian wrench — filtered (offset={WRENCH_OFFSET_SAMPLES} samples, "
            f"MA={WRENCH_MA_WINDOW} samples)\n{dataset_path}"
        )

        for i, (ax, label, unit) in enumerate(zip(f_axes[:6], WRENCH_LABELS, WRENCH_UNITS)):
            if w_ts_n is not None:
                 #ax.plot(w_ts_n, w[:, i], color="lightgray", linewidth=0.6, label="raw")
                ax.plot(w_ts_n, w_filt[:, i], color="black", linewidth=0.9, label=f"filtered: {test_label}")
            if cmp_filt_data is not None:
                ax.plot(cmp_ts_n, cmp_filt_data[1][:, i],
                        color="red", linewidth=0.9, label=f"filtered: {cmp_label}")
            if ep_wrench_data is not None:
                #ax.plot(ep_ts_n, ep_wrench_data[1][:, i],
                        #color="lightgreen", linewidth=0.6, label="gt raw")
                ax.plot(ep_ts_n, ep_wrench_data[2][:, i],
                        color="blue", linewidth=0.9, linestyle="--", label="gt filtered")
            ax.set_ylabel(unit)
            ax.set_title(label)
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(True)
        if w_ts_n is not None:
            f_axes[6].plot(w_ts_n, force_norm, color="black", linewidth=1.0, label=test_label)
        if cmp_filt_data is not None:
            f_axes[6].plot(cmp_ts_n, cmp_filt_data[2],
                           color="red", linewidth=1.0, linestyle="--", label=cmp_label)
        if ep_force_norm is not None:
            f_axes[6].plot(ep_ts_n, ep_force_norm,
                           color="blue", linewidth=1.0, linestyle="--", label="gt")
        f_axes[6].legend(fontsize=6, loc="upper right")
        f_axes[6].set_ylabel("N")
        f_axes[6].set_title("||F|| (filtered)")
        f_axes[6].grid(True)
        f_axes[-1].set_xlabel("Normalized time (0 = start, 1 = end)")
        fig3.tight_layout()

    # --- Figure 4: scalar stiffness per step (top) + ground truth (bottom) ---
    scalar_stiffnesses, has_scalars = load_scalar_stiffness_sequences(dataset_path)
    if not has_scalars:
        print("  No stiffness_scalars_0 found in log — skipping scalar stiffness figure.")
    else:
        n_horizons = len(scalar_stiffnesses)
        n_steps    = len(scalar_stiffnesses[0])  # steps per horizon (action horizon)

        # optionally load comparison scalar stiffnesses
        cmp_scalar_stiffnesses = None
        if compare_path is not None:
            try:
                cmp_ss, cmp_has = load_scalar_stiffness_sequences(compare_path)
                if cmp_has:
                    cmp_scalar_stiffnesses = cmp_ss
                else:
                    print("  WARNING: compare path has no stiffness_scalars_0, skipping compare in Fig 4.")
            except Exception as e:
                print(f"  WARNING: could not load compare scalar stiffness — {e}")

        fig_w4 = max(10, min(n_horizons * 1.2, 40))
        fig4, (ax4_top, ax4_bot) = plt.subplots(
            2, 1, figsize=(fig_w4, 6), sharex=True
        )
        fig4.suptitle("Scalar Stiffness: Inference vs Ground Truth")

        def _plot_scalar_series(ax, series, color, label):
            """Plot one inference scalar stiffness series, solid/dashed for exec/non-exec."""
            n_s = len(series[0]) if series[0] is not None else n_steps
            exec_end = min(SPARSE_EXECUTION_HORIZON, n_s)
            for h_idx, scalars in enumerate(series):
                if scalars is None:
                    continue
                x = h_idx + np.arange(len(scalars)) / SPARSE_EXECUTION_HORIZON
                ax.plot(x[:exec_end], scalars[:exec_end],
                        color=color, linewidth=1.5,
                        label=label if h_idx == 0 else "_nolegend_")
                if exec_end < len(scalars):
                    ax.plot(x[exec_end - 1:], scalars[exec_end - 1:],
                            color=color, linewidth=0.8, linestyle="--")

        # --- top: inference (test + optional compare) ---
        _plot_scalar_series(ax4_top, scalar_stiffnesses, color="black", label=test_label)
        if cmp_scalar_stiffnesses is not None:
            _plot_scalar_series(ax4_top, cmp_scalar_stiffnesses, color="red", label=cmp_label)

        for h_idx in range(1, n_horizons):
            ax4_top.axvline(h_idx, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
        ax4_top.set_ylabel("Stiffness (N/m)")
        ax4_top.set_title("Inference  (solid = executed, dashed = not executed)")
        ax4_top.legend(fontsize=8, loc="upper right")
        ax4_top.grid(True, axis="y", linestyle=":", alpha=0.5)

        # --- bottom: ground truth from TRAIN_PATH ---
        if ep_data is not None:
            ep_ts_gt, ep_stiffness_gt = ep_data[0], ep_data[1]
            ep_ts_gt_norm = _norm_ts(ep_ts_gt) * n_horizons
            ax4_bot.plot(ep_ts_gt_norm, ep_stiffness_gt,
                         color="royalblue", linewidth=1.0, label="ground truth")
            for h_idx in range(1, n_horizons):
                ax4_bot.axvline(h_idx, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
        else:
            ax4_bot.text(0.5, 0.5, "No ground truth loaded\n(set FLAG_GROUND_TRUTH=True)",
                         ha="center", va="center", transform=ax4_bot.transAxes,
                         color="gray", fontsize=9)
        ax4_bot.set_ylabel("Stiffness (N/m)")
        ax4_bot.set_title("Ground Truth (TRAIN_PATH)")
        ax4_bot.legend(fontsize=8, loc="upper right")
        ax4_bot.grid(True, axis="y", linestyle=":", alpha=0.5)

        # shared X axis labels on bottom subplot (sharex=True)
        ax4_bot.set_xticks(np.arange(n_horizons))
        ax4_bot.set_xticklabels([f"Horizon {i + 1}" for i in range(n_horizons)],
                                rotation=45, ha="right", fontsize=8)
        ax4_bot.set_xlim(0, n_horizons)
        fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot(TEST_PATH, episode_path=TRAIN_PATH,
         compare_path=COMPARE_PATH if FLAG_COMPARE else None)
