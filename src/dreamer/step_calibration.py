"""Maps joint_pos_trace.bin's "update" field (Dreamer._update_count) back to
approximate real env steps.

This mapping isn't trivial: _update_count is a plain Python int that resets
to 0 every time train.py is launched (crash, manual restart, etc.) - it's
NOT restored from the checkpoint. So across a run with N restarts, the trace
file's update values go 0..k1, 0..k2, 0..k3, ... one increasing-then-resetting
segment ("era") per process launch, not one global monotonic sequence.

What IS recoverable: metrics.jsonl logs full metrics (including update_count)
every log_every steps, from the SAME Dreamer instances - so it has the same
per-era reset pattern. By matching up corresponding eras and fitting a line
(step ~= a + b*update) per era from those calibration points, we can assign
each trace record an approximate env step - exact at the calibration points,
linearly interpolated/extrapolated elsewhere within that era.
"""

import json
import pathlib

import numpy as np


def _split_into_eras(values):
    """Split a sequence into segments at each point it decreases (a reset) -
    returns a list of (start_idx, end_idx) exclusive-end ranges."""
    values = np.asarray(values)
    reset_points = np.where(np.diff(values) < 0)[0] + 1
    bounds = [0, *reset_points.tolist(), len(values)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def load_calibration_points(metrics_path: pathlib.Path):
    """Returns a list of (step, update_count) pairs, one list of (start,end)
    era ranges matching _split_into_eras, from metrics.jsonl's full-metric
    lines (the ones that include update_count)."""
    steps, updates = [], []
    with open(metrics_path) as f:
        for line in f:
            d = json.loads(line)
            if "update_count" in d:
                steps.append(d["step"])
                updates.append(d["update_count"])
    return np.array(steps), np.array(updates)


def build_update_to_step_fn(logdir: pathlib.Path, trace_updates: np.ndarray):
    """Returns a function approx_step(update_array, era_index_array) that
    maps each trace record's update value to an approximate env step, using
    a separate linear fit per era."""
    cal_steps, cal_updates = load_calibration_points(logdir / "metrics.jsonl")
    cal_eras = _split_into_eras(cal_updates)
    trace_eras = _split_into_eras(trace_updates)

    if len(cal_eras) != len(trace_eras):
        print(
            f"[step_calibration] WARNING: metrics.jsonl has {len(cal_eras)} era(s) but "
            f"joint_pos_trace.bin has {len(trace_eras)} - eras may not line up 1:1 "
            f"(e.g. a restart with no full-metrics log yet). Using min of the two."
        )
    n_eras = min(len(cal_eras), len(trace_eras))

    fits = []  # (slope, intercept) per era, step = intercept + slope*update
    for era_i in range(n_eras):
        c_start, c_end = cal_eras[era_i]
        era_steps = cal_steps[c_start:c_end]
        era_updates = cal_updates[c_start:c_end]
        if len(era_updates) >= 2:
            slope, intercept = np.polyfit(era_updates, era_steps, 1)
        elif len(era_updates) == 1:
            # Only one calibration point in this era - assume the previous
            # era's rate (steps/update), or 1:1 if this is the first era.
            slope = fits[-1][0] if fits else 1.0
            intercept = era_steps[0] - slope * era_updates[0]
        else:
            slope, intercept = (fits[-1] if fits else (1.0, 0.0))
        fits.append((slope, intercept))

    def approx_step(update_value, era_index):
        if era_index >= len(fits):
            era_index = len(fits) - 1
        slope, intercept = fits[era_index]
        return intercept + slope * update_value

    trace_era_ranges = trace_eras
    return approx_step, trace_era_ranges
