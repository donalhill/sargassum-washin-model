"""Pre-calculate ensemble tracks and risk for fast app loading.

Run once:  python precompute.py
Outputs:   data/precomputed.npz  (~550 KB, committed to repo)
"""

import numpy as np

from config import ALPHA_DEFAULT
from model.ensemble import run_ensemble
from model.risk import compute_risk

N_ANIM = 10_000

print(f"Running ensemble ({N_ANIM:,} particles, α={ALPHA_DEFAULT}) ...")
result = run_ensemble(alpha=ALPHA_DEFAULT, n_particles=N_ANIM)
risk = compute_risk(result)

hit_mask = result.hit_segment >= 0
print(f"  {result.n_total:,} seeded, {int(hit_mask.sum()):,} arrivals "
      f"({100 * hit_mask.sum() / result.n_total:.1f}%)")

out = "data/precomputed.npz"
np.savez_compressed(
    out,
    traj_lon=result.traj_lon.astype(np.float32),
    traj_lat=result.traj_lat.astype(np.float32),
    hit_segment=result.hit_segment,
    risk=risk.astype(np.float32),
)

import os
print(f"Saved {out}  ({os.path.getsize(out) / 1024:.0f} KB)")
