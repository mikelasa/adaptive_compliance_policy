#!/usr/bin/env python3
"""Compute a representative initial joint configuration from demonstration data.

Loads the first valid (mask==0) frame's `q` from every joint_data_0.json under
a demos root, then reports mean, median, per-joint std, and the medoid demo
(the actual recorded initial pose closest to the mean in joint space).
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load_lenient_json(path):
    """These files use trailing commas (not valid JSON) - strip them before parsing."""
    with open(path) as f:
        s = f.read()
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return json.loads(s)


def get_initial_q(episode_dir):
    path = os.path.join(episode_dir, "joint_data_0.json")
    if not os.path.exists(path):
        return None
    data = load_lenient_json(path)
    for frame in data:
        if frame.get("mask", 0) == 0:
            return np.array(frame["q"], dtype=float)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/home/robotlab/data/real/CAP6/E1-camaras/realsense86/flipup",
        help="Directory containing episode_* demo folders",
    )
    args = parser.parse_args()

    episode_dirs = sorted(
        d for d in glob.glob(os.path.join(args.root, "episode_*")) if os.path.isdir(d)
    )
    print(f"Found {len(episode_dirs)} episode folders under {args.root}")

    names, qs = [], []
    for d in episode_dirs:
        q = get_initial_q(d)
        if q is None:
            print(f"  [skip] {os.path.basename(d)}: no valid initial q found")
            continue
        names.append(os.path.basename(d))
        qs.append(q)

    qs = np.stack(qs)  # (N, 7)
    n = len(qs)
    print(f"Loaded initial q from {n} demos\n")

    mean_q = qs.mean(axis=0)
    median_q = np.median(qs, axis=0)
    std_q = qs.std(axis=0)
    min_q = qs.min(axis=0)
    max_q = qs.max(axis=0)
    range_q = max_q - min_q

    # Medoid: actual demo initial q closest (Euclidean) to the mean
    dists_to_mean = np.linalg.norm(qs - mean_q, axis=1)
    medoid_idx = int(np.argmin(dists_to_mean))
    medoid_q = qs[medoid_idx]
    medoid_name = names[medoid_idx]

    joint_names = [f"q{i}" for i in range(qs.shape[1])]

    def fmt(vec):
        return "[" + ", ".join(f"{v: .4f}" for v in vec) + "]"

    print("Per-joint statistics across all demos' initial pose:")
    header = f"{'joint':>6} {'mean':>9} {'median':>9} {'std':>9} {'min':>9} {'max':>9} {'range':>9}"
    print(header)
    for i, jn in enumerate(joint_names):
        print(
            f"{jn:>6} {mean_q[i]:>9.4f} {median_q[i]:>9.4f} {std_q[i]:>9.4f} "
            f"{min_q[i]:>9.4f} {max_q[i]:>9.4f} {range_q[i]:>9.4f}"
        )

    print()
    print(f"Mean q:   {fmt(mean_q)}")
    print(f"Median q: {fmt(median_q)}")
    print(f"Medoid q: {fmt(medoid_q)}  (from {medoid_name}, dist to mean = {dists_to_mean[medoid_idx]:.5f})")
    print()

    # Flag joints where spread is large relative to a rough "meaningful motion" scale
    # (0.05 rad ~ 3 deg is a reasonable per-joint precision threshold to look at, not a hard rule)
    flagged = [i for i in range(len(std_q)) if std_q[i] > 0.05]
    if flagged:
        print(f"NOTE: joints {[joint_names[i] for i in flagged]} have std > 0.05 rad "
              f"across initial poses - check for multimodal start positions before trusting the mean.")
    else:
        print("Initial poses are tightly clustered (all joint stds <= 0.05 rad) - mean/medoid should agree closely.")

    print()
    print("Distances (Euclidean, rad) from mean, sorted, for outlier sanity check:")
    order = np.argsort(dists_to_mean)
    for idx in order[-5:]:
        print(f"  {names[idx]}: {dists_to_mean[idx]:.5f}")

    # Quick multimodality check: does splitting into 2 clusters explain meaningfully
    # more variance than treating the initial poses as one blob?
    try:
        from sklearn.cluster import KMeans

        km1 = KMeans(n_clusters=1, n_init=1, random_state=0).fit(qs)
        km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(qs)
        inertia_ratio = km2.inertia_ / km1.inertia_
        print()
        print(f"K-means check: 2-cluster inertia is {inertia_ratio:.2%} of 1-cluster inertia "
              f"(cluster sizes: {np.bincount(km2.labels_)})")
        if inertia_ratio < 0.5:
            print("  -> Strong sign of two distinct start regions (multimodal). "
                  "A single mean/medoid across both would sit between them, in a pose "
                  "neither cluster actually starts from - consider a per-cluster launch pose instead.")
        else:
            print("  -> No strong bimodal split; spread looks like continuous variation "
                  "around one region, so a single mean/medoid is reasonable.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
