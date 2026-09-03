"""
Dataset extraction for the wrench-bias correction model.

Each calibration episode is a static hold at one workspace pose with no external
contact, recorded by `wrench_calib_recorder`. The Franka momentum-observer wrench
(`O_F_ext_hat_K`) during that hold is the *bias* we want to predict and subtract:
the true external wrench is zero, so whatever the sensor reports is error.

We reduce each episode to a single training pair:
    input  x = mean EE position [x, y, z]        (from robot_data_<id>.json)
    target y = mean wrench [fx, fy, fz, tx, ty, tz]  (from wrench_data_<id>.json)

Because recording only starts after the travel+settle phase, the entire episode
file is already a settled hold (pose std ~1e-6 m). We still average over a middle
window and reject high-variance episodes as a safety net against contact/motion.
"""

import pathlib
import numpy as np
import pandas as pd


def _middle_window(arr: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Return the central `frac` fraction of rows, skipping start/end transients."""
    n = len(arr)
    if n == 0:
        return arr
    half = int(n * frac / 2.0)
    mid = n // 2
    lo = max(0, mid - half)
    hi = min(n, mid + half)
    return arr[lo:hi]


def extract_episode(
    episode_dir: pathlib.Path,
    robot_id: int = 0,
    window_frac: float = 0.6,
    max_wrench_std: float = 0.5,
    max_pos_std_m: float = 1e-3,
):
    """
    Reduce one episode folder to a single (xyz, wrench) training pair.

    Returns
    -------
    dict with keys {xyz(3,), wrench(6,), wrench_std(6,), pos_std(3,), ok(bool)}
    or None if the required JSON files are missing.
    """
    joint_path  = episode_dir / f"joint_data_{robot_id}.json"
    wrench_path = episode_dir / f"wrench_data_{robot_id}.json"
    torque_path = episode_dir / f"torque_data_{robot_id}.json"
    if not joint_path.exists() or not wrench_path.exists():
        return None

    # Tolerate half-written / malformed JSON (e.g. an episode still being
    # recorded, or one interrupted by a crash) — skip rather than abort.
    try:
        df_joint  = pd.read_json(joint_path)
        df_wrench = pd.read_json(wrench_path)
        df_torque = pd.read_json(torque_path) if torque_path.exists() else None
    except ValueError:
        return None
    if len(df_joint) == 0 or len(df_wrench) == 0:
        return None

    q      = np.stack(df_joint["q"].values)              # (N, 7) joint angles [rad]
    wrench = np.stack(df_wrench["wrench"].values)         # (M, 6)

    # tau_J lives in torque_data_0.json (same convention as demonstrations).
    has_tau = df_torque is not None and len(df_torque) > 0
    if has_tau:
        tau = np.stack(df_torque["tau_J"].values)         # (N, 7) joint torques [Nm]

    q_win      = _middle_window(q, window_frac)
    wrench_win = _middle_window(wrench, window_frac)

    q_mean    = q_win.mean(axis=0)
    q_std     = q_win.std(axis=0)
    wrench_mean = wrench_win.mean(axis=0)
    wrench_std  = wrench_win.std(axis=0)

    if has_tau:
        tau_win  = _middle_window(tau, window_frac)
        tau_mean = tau_win.mean(axis=0)
        features = np.concatenate([q_mean, tau_mean])     # (14,) q + tau_J
    else:
        features = q_mean                                 # (7,) legacy fallback

    # Quality gate: joints and wrench must be settled and contact-free.
    settled = bool(
        np.all(q_std < max_pos_std_m) and np.all(wrench_std < max_wrench_std)
    )

    return dict(
        features=features,
        q=q_mean,
        wrench=wrench_mean,
        wrench_std=wrench_std,
        q_std=q_std,
        ok=settled,
    )


def build_dataset(
    data_folder,
    robot_id: int = 0,
    window_frac: float = 0.6,
    max_wrench_std: float = 0.5,
    max_pos_std_m: float = 1e-3,
    verbose: bool = True,
):
    """
    Walk every `episode_*` folder under `data_folder` and build (X, Y) arrays.

    Returns
    -------
    X : (K, 3) float64   mean EE position per accepted episode
    Y : (K, 6) float64   mean wrench bias per accepted episode
    info : dict          diagnostics (counts, rejected episode names)
    """
    data_folder = pathlib.Path(data_folder)
    episode_dirs = sorted(
        d for d in data_folder.iterdir() if d.is_dir() and d.name.startswith("episode_")
    )

    X, Y = [], []
    rejected, missing = [], []
    for ep in episode_dirs:
        res = extract_episode(
            ep, robot_id, window_frac, max_wrench_std, max_pos_std_m
        )
        if res is None:
            missing.append(ep.name)
            continue
        if not res["ok"]:
            rejected.append(ep.name)
            continue
        X.append(res["features"])
        Y.append(res["wrench"])

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    info = dict(
        n_total=len(episode_dirs),
        n_accepted=len(X),
        n_rejected=len(rejected),
        n_missing=len(missing),
        rejected=rejected,
        missing=missing,
    )

    if verbose:
        print(f"[dataset] episodes found    : {info['n_total']}")
        print(f"[dataset] accepted          : {info['n_accepted']}")
        print(f"[dataset] rejected (motion) : {info['n_rejected']}")
        print(f"[dataset] missing files     : {info['n_missing']}")
        if len(X) > 0:
            fn = np.linalg.norm(Y[:, :3], axis=1)
            print(
                f"[dataset] bias force-norm   : mean={fn.mean():.3f} N  "
                f"max={fn.max():.3f} N  min={fn.min():.3f} N"
            )
            input_dim = X[0].shape[0] if len(X) > 0 else "?"
            label = "q[7] + tau_J[7]" if input_dim == 14 else "q[7] (no tau_J in data)"
            print(f"[dataset] input ({input_dim:>2} features) : {label}")

    return X, Y, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a wrench-calibration dataset")
    parser.add_argument(
        "data_folder",
        nargs="?",
        default="/home/robotlab/data/real/wrench_calibration",
    )
    args = parser.parse_args()

    X, Y, info = build_dataset(args.data_folder)
    print("\nX shape:", X.shape, "  Y shape:", Y.shape)
