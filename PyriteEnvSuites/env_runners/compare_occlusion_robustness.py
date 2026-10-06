"""
Quantify occlusion robustness without relying on attention-weight
introspection (ACP's modality-attention UNet encoder has no per-modality
cross-attention maps to visualize, unlike RACP's bi-cross-attention DAT).

Two model-agnostic, input/output-level metrics, each computed from a matched
pair of real inference logs for one architecture (lens uncovered "normal" vs
lens covered "tapa", same task):

1. Motion-progress retention under occlusion
   How much of the virtual-target displacement achieved with vision
   available still happens once vision is covered, on a common normalized
   time axis. retention = mean(progress_tapa) / mean(progress_normal).
   Close to 1 -> motion is basically unaffected by occlusion. Close to 0 ->
   the arm stalls near its start pose once vision is covered (matches "ACP
   can't even start tilting the box").

2. Force-vs-stiffness responsiveness under occlusion
   Pearson correlation, over the covered ("tapa") run, between measured
   contact force (||F||, filtered) and predicted post-blend scalar
   stiffness (the frame-invariant magnitude actually sent to the robot --
   see extract_stiffness_scalar in plot_inference_log.py). A strong negative
   correlation means the model softens in response to real contact force
   even without vision, i.e. it is compensating with force feedback. Near
   zero means the commanded stiffness is decoupled from what the robot is
   actually feeling -- evidence it is not using force to compensate for the
   missing modality.

Neither metric needs internal attention weights: both are read directly off
the same inference logs plot_inference_log.py already parses.

Usage:
    python compare_occlusion_robustness.py
Fill in the four *_PATH constants below before running -- ACP's normal/tapa
log paths were not available when this script was written (see the
plot_inference_log.py history for how the RACP paths were confirmed).
"""

import numpy as np
import matplotlib.pyplot as plt

from plot_inference_log import (
    load_log,
    load_wrench_log,
    process_wrench,
    load_postblend_scalar_stiffness_sequences,
)
import os

ACP_NORMAL_PATH = "/home/robotlab/data/resultados-tests/pruebas-cortas/ACP+curr+gating/temp"  # TODO: fill in -- ACP+curriculum+gating, lens uncovered
ACP_TAPA_PATH = "/home/robotlab/data/resultados-tests/pruebas-cortas/ACP+curr+gating/temp"  # TODO: fill in -- ACP+curriculum+gating, lens covered
RACP_NORMAL_PATH = "/home/robotlab/data/resultados-tests/pruebas-cortas/RACP+curr+gating/normal/temp"
RACP_TAPA_PATH = "/home/robotlab/data/resultados-tests/pruebas-cortas/RACP+curr+gating/tapa/temp"


def _progress(virtual_pos: np.ndarray) -> np.ndarray:
    """Cumulative displacement of the virtual target from its start pose (m)."""
    return np.linalg.norm(virtual_pos - virtual_pos[0], axis=1)


def _norm_time(ts: np.ndarray) -> np.ndarray:
    span = ts[-1] - ts[0]
    return (ts - ts[0]) / span if span > 0 else ts - ts[0]


def motion_progress_retention(normal_path: str, tapa_path: str, n_grid: int = 200):
    """
    Resample virtual-target progress onto a common normalized-time grid for
    both conditions.

    Returns:
        grid:      (n_grid,) normalized time in [0, 1]
        prog_n:    (n_grid,) progress (m) for the normal-vision run
        prog_t:    (n_grid,) progress (m) for the lens-covered run
        retention: mean(prog_t) / mean(prog_n)
    """
    t_n, _, _, vpos_n = load_log(normal_path)
    t_t, _, _, vpos_t = load_log(tapa_path)

    prog_n = _progress(vpos_n)
    prog_t = _progress(vpos_t)

    grid = np.linspace(0.0, 1.0, n_grid)
    prog_n_i = np.interp(grid, _norm_time(t_n), prog_n)
    prog_t_i = np.interp(grid, _norm_time(t_t), prog_t)

    retention = prog_t_i.mean() / prog_n_i.mean() if prog_n_i.mean() > 0 else float("nan")
    return grid, prog_n_i, prog_t_i, retention


def force_stiffness_correlation(tapa_path: str):
    """
    Pearson correlation between measured ||F|| and predicted post-blend
    scalar stiffness over the covered run, aligned on the per-horizon
    inference timestamps (one stiffness sample per inference call).

    Returns None if the run has no impedance_controller.log.
    """
    wrench_log = os.path.join(tapa_path, "impedance_controller.log")
    if not os.path.exists(wrench_log):
        print(f"  WARNING: no impedance_controller.log in {tapa_path}, skipping force-stiffness correlation")
        return None

    t, _, _, _ = load_log(tapa_path)
    w_ts, w = load_wrench_log(tapa_path)
    w_filt = process_wrench(w)
    force_norm = np.linalg.norm(w_filt[:, :3], axis=1)

    postblend_seqs = load_postblend_scalar_stiffness_sequences(tapa_path)
    stiffness_first_step = np.array([seq[0] for seq in postblend_seqs])  # (N,)

    force_at_t = np.interp(t, w_ts, force_norm)
    r = float(np.corrcoef(force_at_t, stiffness_first_step)[0, 1])
    return t, force_at_t, stiffness_first_step, r


def report(name: str, normal_path: str, tapa_path: str):
    print(f"\n=== {name} ===")
    if not normal_path or not tapa_path:
        print("  paths not set, skipping")
        return None

    grid, prog_n, prog_t, retention = motion_progress_retention(normal_path, tapa_path)
    print(
        f"  motion-progress retention under occlusion: {retention:.2f}"
        f"  (mean progress normal={prog_n.mean() * 1000:.1f}mm,"
        f" tapa={prog_t.mean() * 1000:.1f}mm)"
    )

    corr_result = force_stiffness_correlation(tapa_path)
    r = corr_result[3] if corr_result is not None else float("nan")
    if corr_result is not None:
        verdict = "compensating with force" if r < -0.3 else "weak/no force-driven response"
        print(f"  force-vs-stiffness correlation (tapa run): r={r:.2f}  ({verdict})")

    return dict(grid=grid, prog_n=prog_n, prog_t=prog_t, retention=retention, corr=r)


if __name__ == "__main__":
    acp = report("ACP + curriculum + gating", ACP_NORMAL_PATH, ACP_TAPA_PATH)
    racp = report("RACP + curriculum + gating", RACP_NORMAL_PATH, RACP_TAPA_PATH)

    fig, ax = plt.subplots(figsize=(8, 5))
    if acp is not None:
        ax.plot(acp["grid"], acp["prog_n"] * 1000, color="black", label="ACP normal")
        ax.plot(acp["grid"], acp["prog_t"] * 1000, color="black", linestyle="--", label="ACP tapa")
    if racp is not None:
        ax.plot(racp["grid"], racp["prog_n"] * 1000, color="red", label="RACP normal")
        ax.plot(racp["grid"], racp["prog_t"] * 1000, color="red", linestyle="--", label="RACP tapa")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Virtual-target displacement from start (mm)")
    ax.set_title("Motion progress: normal vision vs lens covered")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    plt.show()
