"""Re-derive sample times from a hardware sequence counter.

The robot and F/T logging threads stamp each sample with the host time at which
they *processed* it, not the time the hardware *sampled* it. When a logging
thread stalls (GIL contention, a disk flush, a camera write), the samples that
queue up behind it are all stamped with the single host time at which the
backlog finally drained. The recorded stream then shows a hole as wide as the
stall, followed by a burst of samples sharing one timestamp, after which the
cadence resumes on its original sub-ms phase.

No data is lost in such a stall -- `seq_id` stays contiguous across it, and the
cadence afterwards picks up exactly where an uninterrupted stream would have
been. The sample *values* are therefore fine; only their timestamps are wrong.
That matters because downstream consumers align streams by timestamp, so a
query landing inside one of these fictitious holes finds its nearest force
sample tens of milliseconds away and is rejected (see
SequenceSampler.sample_sequence).

Because the sensor is hardware-clocked, the true sample times are exactly
linear in `seq_id`, so they are recoverable: fit time against `seq_id` over the
samples that were not caught in a stall, and evaluate the fit everywhere. The
fit is deliberately *not* anchored on the first and last samples -- those carry
the same sub-ms jitter as any other sample, and in a few episodes the stream
stalls within its final samples, which would tilt an endpoint-anchored line.

This only reconstructs anything for a stream whose timestamps really do come
from a hardware clock, and that is a property of the stream, not an assumption
to make blindly. A hardware-clocked stream lands on one sub-ms phase over and
over (the F/T stream sits on a single phase for >99% of its samples), whereas a
stream stamped purely on arrival spreads its phase uniformly (the Franka robot
stream does, so its scatter is real receive jitter and forcing it onto a line
would invent precision rather than restore it). `info["phase_concentration"]`
reports this so callers can refuse to repair the wrong kind of stream; see
PHASE_CONCENTRATION_MIN.
"""

import numpy as np

# An interval this far from the stream's nominal period (as a fraction of it) is
# taken to be stall-contaminated rather than ordinary jitter, and the samples on
# either side of it are left out of the fit.
DEFAULT_JITTER_TOL = 0.25

# Fitting needs a usable majority of the stream to be stall-free. Below this,
# the linear model is not trustworthy and we fall back to anchoring on the first
# sample at the nominal period.
MIN_CLEAN_FRACTION = 0.5
MIN_CLEAN_SAMPLES = 64

# Fraction of samples sharing one sub-ms phase, above which a stream is taken to
# be hardware-clocked and therefore reconstructable. Measured over this project's
# wipe dataset: the F/T stream scores 0.69-1.00 (median 0.99), the Franka robot
# stream 0.05-0.16, so anything in between is ambiguous and worth a look rather
# than a silent repair.
PHASE_CONCENTRATION_MIN = 0.5
_PHASE_BINS = 100


def repair_timestamps_from_seq_id(
    seq_id,
    time_stamps,
    jitter_tol=DEFAULT_JITTER_TOL,
    min_clean_fraction=MIN_CLEAN_FRACTION,
    min_clean_samples=MIN_CLEAN_SAMPLES,
):
    """Return (repaired_time_stamps, info).

    `seq_id` is the hardware sample counter and must be strictly increasing; it
    need not be contiguous, since a genuine dropout shows up as a jump and the
    linear model spaces the surviving samples correctly across it.

    `time_stamps` are the recorded host times, in the same units as the output.
    `info` reports what was found, for logging and for after-the-fact auditing.
    """
    seq = np.asarray(seq_id).astype(np.int64).ravel()
    ts = np.asarray(time_stamps).astype(np.float64).ravel()
    if seq.shape != ts.shape:
        raise ValueError(
            f"seq_id and time_stamps must have equal length, got {seq.shape} and {ts.shape}"
        )

    n = ts.size
    if n < 2:
        return ts.copy(), {
            "n": n,
            "method": "too_short",
            "dt_nominal": float("nan"),
            "slope": float("nan"),
            "n_clean": n,
            "n_stall_intervals": 0,
            "max_correction": 0.0,
            "clean_residual_max": 0.0,
            "seq_contiguous": True,
            "phase_concentration": float("nan"),
        }

    dseq = np.diff(seq)
    if np.any(dseq <= 0):
        raise ValueError("seq_id must be strictly increasing")

    # Host time elapsed per hardware tick, which is immune to real dropouts.
    period = np.diff(ts) / dseq
    dt_nominal = float(np.median(period))
    if not np.isfinite(dt_nominal) or dt_nominal <= 0:
        raise ValueError(f"could not establish a sampling period (median={dt_nominal})")

    # Does this stream behave like a hardware clock? A hardware-clocked stream
    # revisits the same sub-ms phase; an arrival-stamped one does not.
    phase_hist, _ = np.histogram(
        np.mod(ts / dt_nominal, 1.0), bins=_PHASE_BINS, range=(0.0, 1.0)
    )
    phase_concentration = float(phase_hist.max()) / n

    stalled = np.abs(period - dt_nominal) > jitter_tol * dt_nominal
    # A sample is usable for the fit only if both intervals touching it look
    # nominal. A stall's burst is a run of consecutive anomalous intervals, so
    # every sample inside it is excluded by this alone.
    clean = np.ones(n, dtype=bool)
    clean[:-1] &= ~stalled
    clean[1:] &= ~stalled

    n_clean = int(clean.sum())
    if n_clean >= max(min_clean_samples, min_clean_fraction * n):
        design = np.column_stack([seq[clean].astype(np.float64), np.ones(n_clean)])
        slope, intercept = np.linalg.lstsq(design, ts[clean], rcond=None)[0]
        method = "lstsq_clean"
    else:
        # Not enough uncontaminated samples to fit a line; hold the nominal rate
        # from the first sample so at least the cadence is right.
        slope = dt_nominal
        intercept = ts[0] - dt_nominal * float(seq[0])
        method = "anchor_first"

    repaired = slope * seq.astype(np.float64) + intercept

    info = {
        "n": n,
        "method": method,
        "dt_nominal": dt_nominal,
        "slope": float(slope),
        "n_clean": n_clean,
        "n_stall_intervals": int(stalled.sum()),
        "max_correction": float(np.abs(repaired - ts).max()),
        "clean_residual_max": (
            float(np.abs(repaired[clean] - ts[clean]).max()) if n_clean else 0.0
        ),
        "seq_contiguous": bool(np.all(dseq == 1)),
        "phase_concentration": phase_concentration,
    }
    return repaired, info
