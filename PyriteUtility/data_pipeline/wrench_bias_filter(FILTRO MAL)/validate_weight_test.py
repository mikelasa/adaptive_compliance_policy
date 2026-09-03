"""
validate_weight_test.py

Validates the wrench bias model against episodes recorded with a known weight
attached to the tool tip (no other contact). In free motion the corrected
wrench should equal the weight's gravitational force.

O_F_ext_hat_K uses a reaction convention: the robot must push UP to support a
hanging weight, so the corrected Fz reads POSITIVE:
    Fx ≈ 0,  Fy ≈ 0,  Fz ≈ +mass * 9.81 N
The torque components (Tx,Ty,Tz) depend on the weight's offset from the K frame
and are NOT expected to be zero — they are real physical torque from the moment
arm, which the model correctly preserves (does not subtract).

Usage:
    python validate_weight_test.py \\
        --data   /home/robotlab/data/real/wrench_calibration_test \\
        --model  /home/robotlab/data/wrench_bias_models/wrench_bias_tau_J.npz \\
        --mass   0.7
"""

import argparse
import pathlib
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "../../../"))

from PyriteUtility.data_pipeline.wrench_bias_filter.model import numpy_forward

AXES  = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
UNITS = ["N",  "N",  "N",  "Nm", "Nm", "Nm"]
G     = 9.81


def _middle_window(arr, frac=0.6):
    n = len(arr)
    half = int(n * frac / 2)
    mid  = n // 2
    return arr[max(0, mid - half):min(n, mid + half)]


def load_episode(ep_dir, robot_id=0, window_frac=0.6):
    joint_path  = ep_dir / f"joint_data_{robot_id}.json"
    wrench_path = ep_dir / f"wrench_data_{robot_id}.json"
    torque_path = ep_dir / f"torque_data_{robot_id}.json"

    if not joint_path.exists() or not wrench_path.exists():
        return None
    try:
        df_j = pd.read_json(joint_path)
        df_w = pd.read_json(wrench_path)
        df_t = pd.read_json(torque_path) if torque_path.exists() else None
    except ValueError:
        return None
    if len(df_j) == 0 or len(df_w) == 0:
        return None

    q      = _middle_window(np.stack(df_j["q"].values),      window_frac).mean(axis=0)
    wrench = _middle_window(np.stack(df_w["wrench"].values),  window_frac).mean(axis=0)

    has_tau = df_t is not None and len(df_t) > 0
    if has_tau:
        tau      = _middle_window(np.stack(df_t["tau_J"].values), window_frac).mean(axis=0)
        features = np.concatenate([q, tau])
    else:
        features = q

    return dict(features=features, wrench=wrench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     required=True, help="folder with weight-test episodes")
    ap.add_argument("--model",    required=True, help=".npz weights file")
    ap.add_argument("--mass",     type=float, default=0.7, help="attached weight [kg]")
    ap.add_argument("--out",      default=None,  help="output PNG (default: next to --model)")
    ap.add_argument("--robot_id", type=int,   default=0)
    args = ap.parse_args()

    weights     = dict(np.load(args.model, allow_pickle=True))
    data_folder = pathlib.Path(args.data)
    expected_fz = +args.mass * G  # positive: robot pushes up to support the weight

    episode_dirs = sorted(
        d for d in data_folder.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )

    raw_list, corr_list = [], []
    skipped = 0
    for ep in episode_dirs:
        res = load_episode(ep, args.robot_id)
        if res is None:
            skipped += 1
            continue
        bias      = numpy_forward(weights, res["features"])
        raw_list.append(res["wrench"])
        corr_list.append(res["wrench"] - bias)

    if not raw_list:
        print("[validate] No valid episodes found — check --data path.")
        return

    raw  = np.array(raw_list)   # (N, 6)
    corr = np.array(corr_list)  # (N, 6)
    n    = len(corr)

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\n[validate] episodes loaded : {n}  (skipped {skipped})")
    print(f"[validate] weight mass     : {args.mass} kg")
    print(f"[validate] expected Fz     : {expected_fz:.3f} N\n")

    col = f"{'Axis':<5}  {'Raw mean':>9}  {'Raw std':>7}  "  \
          f"{'Corr mean':>10}  {'Corr std':>8}  {'Expected':>9}"
    print(col)
    print("-" * len(col))
    for k, (ax, unit) in enumerate(zip(AXES, UNITS)):
        exp_str = f"{expected_fz:+.3f}" if k == 2 else "   —   "
        print(f"{ax:<5}  {raw[:,k].mean():>+9.3f}  {raw[:,k].std():>7.3f}  "
              f"{corr[:,k].mean():>+10.3f}  {corr[:,k].std():>8.3f}  "
              f"{exp_str:>9}  {unit}")

    fz_err = corr[:, 2].mean() - expected_fz
    print(f"\n[validate] Fz residual (corrected mean − expected): {fz_err:+.3f} N")
    fn_raw  = np.linalg.norm(raw[:,  :3], axis=1)
    fn_corr = np.linalg.norm(corr[:, :3], axis=1)
    print(f"[validate] force-norm before: mean={fn_raw.mean():.3f} N")
    print(f"[validate] force-norm after : mean={fn_corr.mean():.3f} N  "
          f"(vs expected Fz ≈ {expected_fz:.3f} N, Fx/Fy ≈ 0)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        f"Weight validation — {args.mass} kg attached, n={n} poses\n"
        f"Corrected Fz should be ≈ {expected_fz:.2f} N",
        fontsize=12,
    )

    ep_idx = np.arange(n)
    for k in range(6):
        ax = axes[k // 3][k % 3]
        ax.plot(ep_idx, raw[:, k],  "o-", alpha=0.5, ms=4, label="raw")
        ax.plot(ep_idx, corr[:, k], "s-", alpha=0.9, ms=4, label="corrected")
        if k == 2:  # Fz — draw expected line
            ax.axhline(expected_fz, color="red", ls="--", lw=1.5,
                       label=f"expected {expected_fz:.2f} N")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{AXES[k]} [{UNITS[k]}]")
        ax.set_xlabel("episode index")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = args.out or (
        str(pathlib.Path(args.model).with_suffix("")) + "_weight_validation.png"
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n[validate] saved {out_path}")


if __name__ == "__main__":
    main()
