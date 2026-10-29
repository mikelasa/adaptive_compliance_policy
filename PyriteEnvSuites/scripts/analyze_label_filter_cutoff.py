"""Does the force oscillation in a demo survive the label filter?

The stiffness label is a static function of the *filtered* force norm, so any
force wiggle that gets through the lowpass in real_data_processing.py lands
directly in the training target. This script replays that chain over raw
recorded episodes and reports, per filter design:

  - how much of the dominant oscillation survives,
  - how fast the resulting k label moves (the chatter a policy would copy),
  - how often the label slams onto the k_min rail,
  - how late the label reacts to contact onset.

It reads the raw wrench json only. Nothing is written and no episode is
modified, so it is safe to point at a live dataset.

Usage
-----
    python analyze_label_filter_cutoff.py <episode_dir> [<episode_dir> ...]
    python analyze_label_filter_cutoff.py <dataset_dir> --all
    python analyze_label_filter_cutoff.py <episode_dir> --plot out.png

The defaults for k_max/k_min/f_low/f_high mirror stiffness_estimation_para in
postprocess_add_virtual_target_label.py; override them if you change that dict.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal

# Mirrors stiffness_estimation_para in postprocess_add_virtual_target_label.py.
K_MAX, K_MIN, F_LOW, F_HIGH = 1500.0, 500.0, 7.0, 15.0

# What real_data_processing.py builds today: LiveLPFilter(fs=500, cutoff=5,
# order=5), applied sample by sample. The declared rate is what matters for the
# coefficients, not the rate the data actually arrives at -- see FilterSpec.
CURRENT_ORDER = 5
CURRENT_CUTOFF = 5.0
CURRENT_DECLARED_FS = 500.0


class FilterSpec:
    """A lowpass design, plus the sampling rate it was *declared* with.

    scipy.signal.butter places the corner at cutoff/declared_fs in normalized
    frequency. Feeding the resulting coefficients a stream sampled at some
    other rate rescales the corner by true_fs/declared_fs -- which is exactly
    what happens today, since the wrench logs at 1 kHz while the filter is
    built for 500 Hz. effective_cutoff() reports the corner that the data
    actually sees.
    """

    def __init__(self, label, cutoff, declared_fs, zero_phase, order=5):
        self.label = label
        self.cutoff = cutoff
        self.declared_fs = declared_fs
        self.zero_phase = zero_phase
        self.order = order

    def effective_cutoff(self, true_fs):
        return self.cutoff * true_fs / self.declared_fs

    def apply(self, x):
        b, a = scipy.signal.butter(self.order, self.cutoff, fs=self.declared_fs)
        # filtfilt runs the filter forwards and backwards, which cancels the
        # phase but squares the magnitude response. That is legitimate here:
        # the label pipeline is offline over an already-recorded array, so
        # there is no causality requirement -- LiveLPFilter is simply being
        # reused out of habit from the online path.
        if self.zero_phase:
            return scipy.signal.filtfilt(b, a, x, axis=0)
        return scipy.signal.lfilter(b, a, x, axis=0)

    def survives(self, freq_hz, true_fs):
        """Analytic Butterworth magnitude at freq_hz, as a fraction."""
        mag = 1.0 / np.sqrt(
            1.0 + (freq_hz / self.effective_cutoff(true_fs)) ** (2 * self.order)
        )
        return mag**2 if self.zero_phase else mag


VARIANTS = [
    FilterSpec("current (fs=500, fc=5, causal)", 5.0, 500.0, False),
    FilterSpec("fs corrected (fs=1000, fc=5)", 5.0, 1000.0, False),
    FilterSpec("fc=2, causal", 2.0, 1000.0, False),
    FilterSpec("fc=2, zero-phase", 2.0, 1000.0, True),
]


def stiffness_schedule(force_norm):
    """The same piecewise map VirtualTargetEstimator applies to |f|."""
    k = np.full_like(force_norm, K_MAX)
    k[force_norm > F_HIGH] = K_MIN
    ramp = (force_norm >= F_LOW) & (force_norm <= F_HIGH)
    k[ramp] = K_MAX - (K_MAX - K_MIN) * (force_norm[ramp] - F_LOW) / (F_HIGH - F_LOW)
    return k


def load_episode(episode_dir, arm_id=0):
    """Returns (t_seconds, force_xyz). Timestamps are logged in milliseconds."""
    df = pd.read_json(Path(episode_dir) / f"wrench_data_{arm_id}.json")
    t = df["wrench_time_stamps"].to_numpy().astype(float)
    t = (t - t[0]) / 1000.0
    force = np.vstack(df["wrench"]).astype(float)[:, :3]
    return t, force


def dominant_frequency(signal, fs, f_min=0.5):
    """Welch peak above f_min, with the mean removed so DC cannot win."""
    if len(signal) < 64:
        return float("nan")
    nperseg = min(512, len(signal))
    freqs, power = scipy.signal.welch(signal - signal.mean(), fs=fs, nperseg=nperseg)
    above = freqs > f_min
    if not above.any():
        return float("nan")
    return float(freqs[above][np.argmax(power[above])])


def contact_window(t, force_norm, pad_s=0.0):
    """Span from first to last sample above f_low.

    The oscillation we care about only exists during contact, and the phase
    boundaries differ between episodes, so hardcoding a window would compare
    different parts of different demos.
    """
    in_contact = np.flatnonzero(force_norm > F_LOW)
    if in_contact.size < 2:
        return None
    lo = max(t[0], t[in_contact[0]] - pad_s)
    hi = min(t[-1], t[in_contact[-1]] + pad_s)
    return (t >= lo) & (t <= hi)


def analyze(episode_dir, arm_id=0, span=None):
    t, force = load_episode(episode_dir, arm_id)
    true_fs = (len(t) - 1) / (t[-1] - t[0])
    raw_norm = np.linalg.norm(force, axis=1)

    # Default to the whole contact phase. That is the right default for a
    # survey, but it averages a brief unstable transition in with the steady
    # push either side of it, and the dominant frequency of the two differs --
    # pass --window to zoom on the part that misbehaves.
    if span is not None:
        window = (t >= span[0]) & (t < span[1])
        if window.sum() < 64:
            return None
    else:
        window = contact_window(t, raw_norm)
        if window is None:
            return None

    f_osc = dominant_frequency(raw_norm[window], true_fs)
    # Reference onset: when the unfiltered force first crosses f_low. Every
    # filter's onset is reported relative to this, so the lag column is the
    # delay the label would inherit.
    raw_onset = t[np.argmax(raw_norm > F_LOW)]

    rows = []
    for spec in VARIANTS:
        norm = np.linalg.norm(spec.apply(force), axis=1)
        k = stiffness_schedule(norm)
        dt = np.diff(t[window])
        rate = np.percentile(np.abs(np.diff(k[window])) / np.maximum(dt, 1e-9), 95)
        rail = int(np.abs(np.diff((norm[window] > F_HIGH).astype(int))).sum())
        onset_lag_ms = 1000.0 * (t[np.argmax(norm > F_LOW)] - raw_onset)
        rows.append(
            {
                "label": spec.label,
                "cutoff": spec.effective_cutoff(true_fs),
                "survives": spec.survives(f_osc, true_fs),
                "rate": rate,
                "rail": rail,
                "lag_ms": onset_lag_ms,
                "norm": norm,
            }
        )

    return {
        "name": Path(episode_dir).name,
        "t": t,
        "raw_norm": raw_norm,
        "window": window,
        "window_label": (
            f"{span[0]:.1f}-{span[1]:.1f} s" if span is not None else "contact"
        ),
        "true_fs": true_fs,
        "f_osc": f_osc,
        "duration": t[-1],
        "rows": rows,
    }


def report(result):
    print(f"\n=== {result['name']} ===")
    print(
        f"  {result['duration']:.1f} s at {result['true_fs']:.0f} Hz, "
        f"{result['window_label']} window spans "
        f"{result['window'].sum() / result['true_fs']:.1f} s, "
        f"dominant oscillation {result['f_osc']:.2f} Hz"
    )
    if abs(result["true_fs"] - CURRENT_DECLARED_FS) > 1.0:
        print(
            f"  NOTE: the label filter is declared fs={CURRENT_DECLARED_FS:.0f} Hz "
            f"but this data is {result['true_fs']:.0f} Hz, so its real corner is "
            f"{CURRENT_CUTOFF * result['true_fs'] / CURRENT_DECLARED_FS:.1f} Hz."
        )
    print(
        f"\n  {'filter':<32s} {'cutoff':>8s} {'osc kept':>9s} "
        f"{'dk/dt p95':>12s} {'k_min rail':>11s} {'onset lag':>10s}"
    )
    print("  " + "-" * 86)
    for row in result["rows"]:
        print(
            f"  {row['label']:<32s} {row['cutoff']:6.1f}Hz "
            f"{100 * row['survives']:8.1f}% {row['rate']:9.0f} N/m/s "
            f"{row['rail']:9d}x {row['lag_ms']:8.0f} ms"
        )


def plot(results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(results), 1, figsize=(14, 3.2 * len(results)), squeeze=False
    )
    for ax, result in zip(axes[:, 0], results):
        t = result["t"]
        ax.plot(t, result["raw_norm"], color="0.75", lw=0.8, label="raw")
        for row, color in zip(result["rows"], ["C3", "C1", "C0", "C2"]):
            ax.plot(t, row["norm"], color=color, lw=1.1, label=row["label"])
        ax.axhline(F_HIGH, color="g", ls="--", lw=1)
        ax.axhline(F_LOW, color="r", ls="--", lw=1)
        ax.set_title(f"{result['name']}  (oscillation {result['f_osc']:.2f} Hz)")
        ax.set_ylabel("|Fxyz| [N]")
        ax.legend(fontsize=7, ncol=5, loc="upper right")
    axes[-1, 0].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nwrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="episode dir(s), or a dataset dir")
    parser.add_argument(
        "--all",
        action="store_true",
        help="treat each path as a dataset dir and analyze every episode_* in it",
    )
    parser.add_argument("--arm-id", type=int, default=0)
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        help="restrict the metrics to this interval instead of the whole "
        "contact phase, e.g. --window 10.2 12.5 for an unstable handover",
    )
    parser.add_argument("--plot", metavar="OUT.png", help="also save an overlay plot")
    args = parser.parse_args()

    episodes = []
    for path in args.paths:
        path = Path(path)
        if args.all:
            episodes.extend(sorted(p for p in path.glob("episode_*") if p.is_dir()))
        else:
            episodes.append(path)

    results = []
    for episode in episodes:
        try:
            result = analyze(episode, args.arm_id, args.window)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"\n=== {episode.name} ===\n  skipped: {exc}")
            continue
        if result is None:
            reason = "window too short" if args.window else "never crossed f_low"
            print(f"\n=== {episode.name} ===\n  skipped: {reason}")
            continue
        report(result)
        results.append(result)

    if args.plot and results:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
