"""
apply_wrench_bias.py

Post-processes recorded demonstration episodes by applying the trained wrench
bias correction model to every timestep. Writes wrench_data_filtered_0.json
alongside the existing wrench_data_0.json in each episode folder.

The EMA filter on tau_J (alpha=0.0609, 10 Hz at 1 kHz) mirrors the C++
wrench_bias_corrector so the correction is identical to what ManipServer applies.
"""

import json
import os
import pathlib
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "../../../"))

from PyriteUtility.data_pipeline.wrench_bias_filter.model import numpy_forward

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_FOLDER = pathlib.Path("/home/robotlab/data/real/test/test_filtro")
MODEL_PATH  = "/home/robotlab/data/wrench_bias_models/wrench_bias_tau_J_filter_relu.npz"
ROBOT_ID    = 0
# ──────────────────────────────────────────────────────────────────────────────

# 10 Hz EMA at 1 kHz — matches wrench_bias_corrector.h
_TAU_EMA_ALPHA = 0.0609


def _load_json(path):
    """Load JSON that may have trailing commas (ManipServer format)."""
    text = pathlib.Path(path).read_text()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def process_episode(ep_dir: pathlib.Path, weights: dict) -> bool:
    joint_path  = ep_dir / f"joint_data_{ROBOT_ID}.json"
    wrench_path = ep_dir / f"wrench_data_{ROBOT_ID}.json"
    torque_path = ep_dir / f"torque_data_{ROBOT_ID}.json"
    out_path    = ep_dir / f"wrench_data_filtered_{ROBOT_ID}.json"

    if not joint_path.exists() or not wrench_path.exists():
        return False

    joint_records  = _load_json(joint_path)
    wrench_records = _load_json(wrench_path)

    has_tau = torque_path.exists()
    if has_tau:
        torque_records = _load_json(torque_path)

    n = min(len(joint_records), len(wrench_records))
    if has_tau:
        n = min(n, len(torque_records))

    tau_ema = np.zeros(7)
    out_records = []

    for i in range(n):
        q      = np.array(joint_records[i]["q"],       dtype=np.float64)
        wrench = np.array(wrench_records[i]["wrench"], dtype=np.float64)

        if has_tau:
            tau = np.array(torque_records[i]["tau_J"], dtype=np.float64)
            tau_ema = _TAU_EMA_ALPHA * tau + (1.0 - _TAU_EMA_ALPHA) * tau_ema
            features = np.concatenate([q, tau_ema])
        else:
            features = q

        bias = numpy_forward(weights, features)

        rec = dict(wrench_records[i])   # preserve timestamp and any other fields
        rec["wrench"] = (wrench - bias).tolist()
        out_records.append(rec)

    with open(out_path, "w") as f:
        json.dump(out_records, f)

    return True


if __name__ == "__main__":
    weights = dict(np.load(MODEL_PATH, allow_pickle=True))

    act_id   = int(weights.get("activation", np.array(0)))
    act_name = {0: "relu", 1: "gelu", 2: "tanh", 3: "elu"}.get(act_id, "relu")
    x_dim    = len(weights["x_mean"])
    h1       = weights["W1"].shape[0]
    h2       = weights["W2"].shape[0]
    print(f"[model] arch {x_dim}→{h1}→{h2}→6  act={act_name}")

    episode_dirs = sorted(
        d for d in DATA_FOLDER.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )

    if not episode_dirs:
        print(f"[error] no episode_* directories found in {DATA_FOLDER}")
        sys.exit(1)

    ok, skipped = 0, 0
    for ep in episode_dirs:
        if process_episode(ep, weights):
            ok += 1
        else:
            skipped += 1
            print(f"  [skip] {ep.name}  (missing joint or wrench data)")

    print(f"\nDone: {ok} episodes processed, {skipped} skipped.")
    print(f"Output: wrench_data_filtered_{ROBOT_ID}.json written in each episode folder.")
