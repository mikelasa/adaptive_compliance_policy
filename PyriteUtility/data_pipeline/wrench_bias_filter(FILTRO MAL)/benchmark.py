"""
Benchmark wrench-bias MLP inference time.

Two modes:
  1. Benchmark a trained model (.npz) — confirms the deployed model is safe.
  2. Architecture sweep with random weights — lets you pick the right size
     BEFORE training, by showing the speed/capacity trade-off.

The NumPy forward pass is timed because it mirrors the C++ Eigen implementation
exactly. Eigen on the same CPU is typically equal or faster.

Usage:
    # Benchmark a trained model
    python benchmark.py --model /home/robotlab/data/wrench_bias_models/wrench_bias_V1.npz

    # Architecture sweep only (no trained model needed)
    python benchmark.py --sweep

    # Both
    python benchmark.py --model wrench_bias_V1.npz --sweep
"""

import argparse
import time
import numpy as np
import sys, os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))
from PyriteUtility.data_pipeline.wrench_bias_filter.model import numpy_forward

INPUT_DIM  = 7   # joint angles q[0..6]
OUTPUT_DIM = 6
FRANKA_BUDGET_US = 1000.0  # 1 ms in microseconds


# ─── Timing helpers ───────────────────────────────────────────────────────────

def _time_forward(weights: dict, n: int, warmup: int) -> np.ndarray:
    xyz = np.array([-0.10, 0.55, 0.01, -1.81, -0.02, 2.99, 0.78])  # representative q[7]
    for _ in range(warmup):
        numpy_forward(weights, xyz)
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        numpy_forward(weights, xyz)
        times[i] = (time.perf_counter() - t0) * 1e6  # µs
    return times


def _random_weights(h1: int, h2: int) -> dict:
    """Build a weights dict with random values for timing purposes."""
    rng = np.random.default_rng(0)
    return dict(
        W1=rng.standard_normal((h1, INPUT_DIM)),
        b1=rng.standard_normal(h1),
        W2=rng.standard_normal((h2, h1)),
        b2=rng.standard_normal(h2),
        W3=rng.standard_normal((OUTPUT_DIM, h2)),
        b3=rng.standard_normal(OUTPUT_DIM),
        x_mean=np.zeros(INPUT_DIM),
        x_std=np.ones(INPUT_DIM),
        y_mean=np.zeros(OUTPUT_DIM),
        y_std=np.ones(OUTPUT_DIM),
    )


def _n_params(h1: int, h2: int) -> int:
    return (INPUT_DIM * h1 + h1) + (h1 * h2 + h2) + (h2 * OUTPUT_DIM + OUTPUT_DIM)


def _print_row(label, times, n_params=None):
    mean   = times.mean()
    median = np.median(times)
    p99    = np.percentile(times, 99)
    budget = mean / FRANKA_BUDGET_US * 100
    params = f"{n_params:>6}" if n_params is not None else "     —"
    ok = "✓" if p99 < FRANKA_BUDGET_US * 0.10 else ("~" if p99 < FRANKA_BUDGET_US * 0.50 else "✗")
    print(
        f"  {label:<22}  params={params}  "
        f"mean={mean:6.2f} µs  median={median:6.2f} µs  "
        f"p99={p99:6.2f} µs  budget={budget:5.2f}%  {ok}"
    )


# ─── Benchmark a trained model ────────────────────────────────────────────────

def benchmark_model(model_path: str, n: int, warmup: int):
    weights = dict(np.load(model_path))
    h1 = weights["W1"].shape[0]
    h2 = weights["W2"].shape[0]
    n_params = _n_params(h1, h2)

    print(f"\n── Trained model: {os.path.basename(model_path)}")
    print(f"   Architecture: {INPUT_DIM} → {h1} → {h2} → {OUTPUT_DIM}   ({n_params} params)")
    print(f"   Iterations: {n:,}  Warm-up: {warmup:,}\n")
    print(f"  {'label':<22}  {'params':>12}  {'mean':>12}  {'median':>14}  {'p99':>12}  {'budget':>10}  ok?")
    print("  " + "─" * 100)

    times = _time_forward(weights, n, warmup)
    _print_row(f"{INPUT_DIM}→{h1}→{h2}→{OUTPUT_DIM} (trained)", times, n_params)

    print(f"\n  Budget: {FRANKA_BUDGET_US:.0f} µs  ✓ = p99 < 10%   ~ = p99 < 50%   ✗ = p99 > 50%")


# ─── Architecture sweep ───────────────────────────────────────────────────────

SWEEP_ARCHITECTURES = [
    (8,  4),
    (16, 8),
    (32, 16),   # current default
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
]

def benchmark_sweep(n: int, warmup: int):
    print(f"\n── Architecture sweep  ({n:,} iterations, random weights)")
    print(f"   Input={INPUT_DIM}  Output={OUTPUT_DIM}  Budget=1 ms\n")
    print(f"  {'architecture':<22}  {'params':>12}  {'mean':>12}  {'median':>14}  {'p99':>12}  {'budget':>10}  ok?")
    print("  " + "─" * 100)

    for h1, h2 in SWEEP_ARCHITECTURES:
        weights = _random_weights(h1, h2)
        n_params = _n_params(h1, h2)
        times = _time_forward(weights, n, warmup)
        label = f"{INPUT_DIM}→{h1}→{h2}→{OUTPUT_DIM}"
        _print_row(label, times, n_params)

    print(f"\n  Budget: {FRANKA_BUDGET_US:.0f} µs  ✓ = p99 < 10%   ~ = p99 < 50%   ✗ = p99 > 50%")
    print("  Note: C++ Eigen is typically equal or faster than NumPy on the same CPU.")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="path to trained .npz — benchmark that specific model")
    p.add_argument("--sweep", action="store_true",
                   help="sweep standard architectures with random weights")
    p.add_argument("--n",      type=int, default=50000)
    p.add_argument("--warmup", type=int, default=1000)
    args = p.parse_args()

    if args.model is None and not args.sweep:
        # Default: sweep if no model given
        args.sweep = True

    if args.model:
        benchmark_model(args.model, args.n, args.warmup)
    if args.sweep:
        benchmark_sweep(args.n, args.warmup)
