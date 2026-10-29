"""Repair robot/wrench timestamps in an already-processed zarr, in place.

The processing pipeline now repairs these timestamps as it reads the raw JSON
(see PyriteUtility/data_pipeline/real_data_processing.py), so datasets built
from here on need no patching. This script exists for datasets processed before
that fix: it rewrites only the `robot_time_stamps_*` and `wrench_time_stamps_*`
arrays, leaving the images, poses and wrenches untouched. Reprocessing from raw
would re-encode every video for no benefit, since the stall corrupts only the
timestamps and never the sample values.

The repaired times are re-derived from each stream's hardware `seq_id` in the
raw JSON, then shifted by the same per-episode offset the pipeline originally
applied (it zeroes every stream against the earliest start time across
streams). That offset is recovered exactly from the data and is checked to be a
single constant before anything is written.

Originals are saved as .npz alongside the dataset before being overwritten, and
their presence marks an episode as already repaired so re-running is safe.

Usage:
    python repair_timestamps_in_processed_zarr.py --raw RAW_DIR --processed ZARR_DIR
    python repair_timestamps_in_processed_zarr.py ... --dry-run    # report only
    python repair_timestamps_in_processed_zarr.py ... --force      # redo from backup
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import zarr

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../.."))

from PyriteUtility.data_pipeline.timestamp_repair import (  # noqa: E402
    PHASE_CONCENTRATION_MIN,
    repair_timestamps_from_seq_id,
)

# Streams in the processed zarr that carry a seq_id in the raw JSON, mapped to
# the raw file that holds them. Only `wrench` is repaired by default: the F/T
# stream is hardware-clocked, so its stall artifacts are reconstructable, while
# the Franka robot stream is stamped on arrival and its scatter is real. Each
# stream is checked against PHASE_CONCENTRATION_MIN regardless of this list.
STREAMS = {"robot": "robot_data_{id}.json", "wrench": "wrench_data_{id}.json"}
DEFAULT_STREAMS = ("wrench",)

# The pipeline subtracts one offset from every stream of an episode, so
# `raw - processed` must be constant. Allow only floating-point noise.
OFFSET_TOL_MS = 1e-6

# A repair larger than this is not the stall artifact this script is for, and
# is left alone rather than written blindly.
MAX_PLAUSIBLE_CORRECTION_MS = 500.0


def discover_streams(episode_group, wanted):
    """Yield (stream, id, zarr_key) for each requested stream in an episode."""
    for key in sorted(episode_group.keys()):
        match = re.fullmatch(r"(robot|wrench)_time_stamps_(\d+)", key)
        if match and match.group(1) in wanted:
            yield match.group(1), int(match.group(2)), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="raw episode directory")
    parser.add_argument("--processed", required=True, help="processed zarr directory")
    parser.add_argument("--backup-dir", default=None, help="default: <processed>.timestamp_backup")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--force", action="store_true", help="re-repair from the backup")
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=sorted(STREAMS),
        default=list(DEFAULT_STREAMS),
        help=f"streams to repair (default: {' '.join(DEFAULT_STREAMS)})",
    )
    args = parser.parse_args()

    backup_dir = args.backup_dir or (args.processed.rstrip("/") + ".timestamp_backup")
    root = zarr.open(args.processed, mode="r" if args.dry_run else "r+")
    episodes = sorted(root["data"].keys())
    print(f"{len(episodes)} episodes in {args.processed}")
    if not args.dry_run:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"backups -> {backup_dir}")
    print()

    n_written = 0
    n_skipped = 0
    totals = {}
    for episode in episodes:
        group = root["data"][episode]
        backup_path = os.path.join(backup_dir, f"{episode}.npz")
        already = os.path.exists(backup_path)
        if already and not args.force:
            n_skipped += 1
            continue
        backup = np.load(backup_path) if (already and args.force) else None

        repairs = {}
        notes = []
        for stream, stream_id, key in discover_streams(group, set(args.streams)):
            raw_path = os.path.join(
                args.raw, episode, STREAMS[stream].format(id=stream_id)
            )
            if not os.path.exists(raw_path):
                notes.append(f"{key}: no raw file, skipped")
                continue
            frame = pd.read_json(raw_path)
            if "seq_id" not in frame.columns:
                notes.append(f"{key}: no seq_id in raw, skipped")
                continue

            raw_ts = frame[f"{stream}_time_stamps"].to_numpy().astype(np.float64)
            # Compare against the pre-repair values, so --force recomputes the
            # offset from the same baseline the first run used.
            current = (
                backup[key].astype(np.float64)
                if backup is not None and key in backup
                else np.asarray(group[key][:]).astype(np.float64).ravel()
            )
            if raw_ts.size != current.size:
                notes.append(
                    f"{key}: raw has {raw_ts.size} samples, processed has "
                    f"{current.size}, skipped"
                )
                continue

            offsets = raw_ts - current
            spread = float(offsets.max() - offsets.min())
            if spread > OFFSET_TOL_MS:
                notes.append(
                    f"{key}: raw-to-processed offset is not constant "
                    f"(spread {spread:.3e} ms), skipped"
                )
                continue

            repaired, info = repair_timestamps_from_seq_id(
                frame["seq_id"].to_numpy(), raw_ts
            )
            if info["phase_concentration"] < PHASE_CONCENTRATION_MIN:
                notes.append(
                    f"{key}: stamped on arrival, not hardware-clocked "
                    f"(phase concentration {info['phase_concentration']:.3f} < "
                    f"{PHASE_CONCENTRATION_MIN}), skipped"
                )
                continue
            if info["max_correction"] > MAX_PLAUSIBLE_CORRECTION_MS:
                notes.append(
                    f"{key}: correction of {info['max_correction']:.1f} ms is too "
                    f"large to be a logging stall, skipped"
                )
                continue

            new = repaired - float(np.median(offsets))
            if np.any(np.diff(new) <= 0):
                notes.append(f"{key}: repair is not strictly increasing, skipped")
                continue

            repairs[key] = (new, current, info)
            totals.setdefault(key, []).append(info)

        for note in notes:
            print(f"  {episode} {note}")
        if not repairs:
            continue

        summary = "  ".join(
            f"{key}: {info['n_stall_intervals']} stalls, "
            f"max {info['max_correction']:.2f} ms"
            for key, (_, _, info) in sorted(repairs.items())
        )
        print(f"  {episode}  {summary}")

        if args.dry_run:
            continue
        if backup is None:
            np.savez(
                backup_path, **{key: old for key, (_, old, _) in repairs.items()}
            )
        for key, (new, _, _) in repairs.items():
            group[key][:] = new.astype(group[key].dtype)
        n_written += 1

    print()
    print(f"episodes repaired: {n_written}; already repaired (skipped): {n_skipped}")
    for key, infos in sorted(totals.items()):
        stalls = sum(i["n_stall_intervals"] for i in infos)
        worst = max(i["max_correction"] for i in infos)
        residual = max(i["clean_residual_max"] for i in infos)
        fallbacks = sum(1 for i in infos if i["method"] != "lstsq_clean")
        print(
            f"  {key}: {stalls} stall intervals over {len(infos)} episodes, "
            f"worst correction {worst:.2f} ms, worst clean-sample residual "
            f"{residual:.3f} ms, {fallbacks} fit fallback(s)"
        )
    if args.dry_run:
        print("\n(dry run -- nothing was written)")


if __name__ == "__main__":
    main()
