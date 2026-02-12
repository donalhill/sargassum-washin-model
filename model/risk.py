"""Coastal segment washup probability rollup.

Converts ensemble hit data into a (n_segments, n_days) probability array,
smoothed along the coastline for visual coherence.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from config import DT_FORCING_HOURS, N_DAYS, RISK_SMOOTH_SIGMA
from data.coastline import SEGMENT_MIDPOINTS
from model.ensemble import EnsembleResult


def compute_risk(result: EnsembleResult) -> np.ndarray:
    """Compute cumulative washup probability per coastal segment per day.

    Parameters
    ----------
    result : EnsembleResult
        Output from ensemble.run_ensemble().

    Returns
    -------
    risk : ndarray (n_segments, n_days)
        risk[s, d] = fraction of total particles that hit segment s
        by the end of day d (cumulative).
    """
    n_segments = len(SEGMENT_MIDPOINTS)
    n_total = result.n_total
    steps_per_day = 24 // DT_FORCING_HOURS  # 4 steps per day

    risk = np.zeros((n_segments, N_DAYS))

    hit_mask = result.hit_segment >= 0

    for day in range(N_DAYS):
        max_step = (day + 1) * steps_per_day - 1
        # Particles that hit by this day
        day_mask = hit_mask & (result.hit_step <= max_step)

        if day_mask.any():
            segments_hit = result.hit_segment[day_mask]
            counts = np.bincount(segments_hit, minlength=n_segments)
            risk[:, day] = counts[:n_segments] / n_total

    # Smooth along coastline for visual coherence
    for day in range(N_DAYS):
        risk[:, day] = gaussian_filter1d(risk[:, day], sigma=RISK_SMOOTH_SIGMA)

    return risk
