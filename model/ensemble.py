"""Stochastic ensemble runner — uniform random Monte Carlo prior.

Seeds N particles uniformly at random in a circular buffer zone around
Barbados.  Each particle is an independent sample from the uniform prior.
Stores ALL trajectories at coarse temporal resolution for full visualisation.
"""

from dataclasses import dataclass
import numpy as np

from config import (
    N_PARTICLES, BUFFER_KM,
    ISLAND_CENTER_LON, ISLAND_CENTER_LAT,
    ALPHA_HALF_RANGE,
    CURRENT_NOISE_STD, WIND_NOISE_STD,
    AR1_DECAY, GLOBAL_SEED,
    ALPHA_DEFAULT, N_STEPS,
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
)
from data.coastline import nearest_segment, _GADM, points_in_polygon
from data import synthetic_forcing as sf
from model.advection import rk4_step, build_interpolators


# ── Land exclusion ─────────────────────────────────────────────────────



# ── Random circular seeding ────────────────────────────────────────────

_KM_PER_DEG_LAT = 111.32
_KM_PER_DEG_LON = 111.32 * np.cos(np.radians(ISLAND_CENTER_LAT))


def _seed_uniform_circle(n, buffer_km, rng):
    n_gen = int(n * 1.05) + 100
    r_km = buffer_km * np.sqrt(rng.random(n_gen))
    theta = rng.uniform(0, 2 * np.pi, n_gen)
    lon = ISLAND_CENTER_LON + (r_km * np.cos(theta)) / _KM_PER_DEG_LON
    lat = ISLAND_CENTER_LAT + (r_km * np.sin(theta)) / _KM_PER_DEG_LAT
    on_land = points_in_polygon(lon, lat, _GADM)
    lon, lat = lon[~on_land], lat[~on_land]
    return lon[:n], lat[:n]


@dataclass
class EnsembleResult:
    # ALL trajectories at coarse resolution (every TRAJ_STEP forcing steps)
    traj_lon: np.ndarray          # (n_total, n_snapshots) — NaN after deactivation
    traj_lat: np.ndarray
    hit_segment: np.ndarray       # (n_total,) -1 if no hit
    hit_step: np.ndarray          # (n_total,) -1 if no hit
    n_total: int

# Snapshot interval (forcing steps): every 8 steps = every 2 days
TRAJ_STEP = 1


def run_ensemble(alpha: float = None,
                 n_particles: int = None,
                 seed: int = None) -> EnsembleResult:
    alpha = alpha if alpha is not None else ALPHA_DEFAULT
    n_particles = n_particles if n_particles is not None else N_PARTICLES
    seed = seed if seed is not None else GLOBAL_SEED
    n_steps = N_STEPS

    rng = np.random.default_rng(seed)

    start_lon, start_lat = _seed_uniform_circle(n_particles, BUFFER_KM, rng)
    n_actual = len(start_lon)

    alphas = rng.uniform(alpha - ALPHA_HALF_RANGE,
                         alpha + ALPHA_HALF_RANGE, n_actual)

    # Trajectory storage — ALL particles, coarse time resolution
    n_snapshots = n_steps // TRAJ_STEP + 1
    traj_lon = np.full((n_actual, n_snapshots), np.nan)
    traj_lat = np.full((n_actual, n_snapshots), np.nan)
    traj_lon[:, 0] = start_lon
    traj_lat[:, 0] = start_lat

    hit_segment = np.full(n_actual, -1, dtype=np.int32)
    hit_step = np.full(n_actual, -1, dtype=np.int32)
    active = np.ones(n_actual, dtype=bool)

    noise_cu = np.zeros(n_actual)
    noise_cv = np.zeros(n_actual)
    noise_wu = np.zeros(n_actual)
    noise_wv = np.zeros(n_actual)

    lon = start_lon.copy()
    lat = start_lat.copy()

    for t in range(n_steps):
        noise_cu = AR1_DECAY * noise_cu + rng.normal(0, CURRENT_NOISE_STD, n_actual)
        noise_cv = AR1_DECAY * noise_cv + rng.normal(0, CURRENT_NOISE_STD, n_actual)
        noise_wu = AR1_DECAY * noise_wu + rng.normal(0, WIND_NOISE_STD, n_actual)
        noise_wv = AR1_DECAY * noise_wv + rng.normal(0, WIND_NOISE_STD, n_actual)

        if not active.any():
            break

        interps = build_interpolators(
            sf.u_current[t], sf.v_current[t],
            sf.u_wind[t], sf.v_wind[t],
            sf.lon_grid, sf.lat_grid,
        )

        idx = np.where(active)[0]
        lo_new, la_new, hit, left = rk4_step(
            lon[idx], lat[idx], alphas[idx], interps,
            noise_cu=noise_cu[idx], noise_cv=noise_cv[idx],
            noise_wu=noise_wu[idx], noise_wv=noise_wv[idx],
        )

        lon[idx] = lo_new
        lat[idx] = la_new

        # Snapshot storage
        if (t + 1) % TRAJ_STEP == 0:
            col = (t + 1) // TRAJ_STEP
            if col < n_snapshots:
                traj_lon[idx, col] = lo_new
                traj_lat[idx, col] = la_new

        # Record hits
        hit_idx = idx[hit]
        if len(hit_idx) > 0:
            segs = nearest_segment(lon[hit_idx], lat[hit_idx])
            hit_segment[hit_idx] = segs
            hit_step[hit_idx] = t
            active[hit_idx] = False
            # Write coast-arrival position into trajectory
            col = t // TRAJ_STEP + 1
            if col < n_snapshots:
                traj_lon[hit_idx, col] = lon[hit_idx]
                traj_lat[hit_idx, col] = lat[hit_idx]

        left_idx = idx[left & ~hit]
        active[left_idx] = False

    return EnsembleResult(
        traj_lon=traj_lon,
        traj_lat=traj_lat,
        hit_segment=hit_segment,
        hit_step=hit_step,
        n_total=n_actual,
    )
