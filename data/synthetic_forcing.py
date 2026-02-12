"""Generate synthetic CMEMS-like ocean current and wind forcing fields.

All fields live on a regular lon/lat grid and are generated deterministically
from a seeded RNG at import time.  A simple land mask (NaN over Barbados grid
cells) is applied.

Public API
----------
lon_grid, lat_grid : 1-D arrays (grid axes)
LON, LAT           : 2-D meshgrid arrays
u_current, v_current : (nt, nlat, nlon) current components (m/s)
u_wind, v_wind       : (nt, nlat, nlon) 10-m wind components (m/s)
land_mask            : (nlat, nlon) bool — True on land
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from config import (
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
    DLON, DLAT,
    DT_FORCING_HOURS, N_STEPS,
    GLOBAL_SEED,
)

# ── Grid construction ───────────────────────────────────────────────────

lon_grid = np.arange(LON_MIN, LON_MAX + DLON / 2, DLON)
lat_grid = np.arange(LAT_MIN, LAT_MAX + DLAT / 2, DLAT)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

_nlat, _nlon = LON.shape
_nt = N_STEPS

# ── Land mask (ray-casting point-in-polygon from GADM coastline) ────────

from data.coastline import _GADM as _coast_poly, points_in_polygon

land_mask = points_in_polygon(LON.ravel(), LAT.ravel(), _coast_poly).reshape(LON.shape)

# ── Helpers ─────────────────────────────────────────────────────────────

def _stream_function_eddy(lon0, lat0, radius, strength, lon2d, lat2d):
    """Divergence-free velocity from a Gaussian stream-function eddy.

    Returns (u, v) arrays on the grid, with u = -dψ/dy, v = dψ/dx.
    """
    dx = lon2d - lon0
    dy = lat2d - lat0
    r2 = dx ** 2 + dy ** 2
    psi = strength * np.exp(-r2 / (2 * radius ** 2))
    # Finite-difference gradient (central, same units as grid)
    u = -strength * (-dy / radius ** 2) * psi  # -dψ/dy
    v = strength * (-dx / radius ** 2) * psi   #  dψ/dx
    # Scale to reasonable m/s
    scale = 1.0 / (radius * 111_000)  # degrees → metres
    return u * scale, v * scale


def _apply_land_mask(field):
    """Set land cells to NaN in-place for a 3-D (nt, nlat, nlon) field."""
    field[:, land_mask] = np.nan
    return field

# ── Generate currents ───────────────────────────────────────────────────

rng = np.random.default_rng(GLOBAL_SEED)

# Base westward flow (North Equatorial Current analogue)
u_current = np.full((_nt, _nlat, _nlon), -0.30)  # m/s westward
v_current = np.full((_nt, _nlat, _nlon), 0.02)    # slight northward

# Add 3 mesoscale eddies (divergence-free via stream function)
_eddy_params = [
    # (lon0, lat0, radius_deg, strength, rotation_sign)
    (-59.6, 13.4, 0.35, 0.15, +1),
    (-59.9, 13.0, 0.40, -0.12, -1),
    (-59.2, 13.2, 0.30, 0.10, +1),
]

for lon0, lat0, rad, amp, sign in _eddy_params:
    ue, ve = _stream_function_eddy(lon0, lat0, rad, sign * amp, LON, LAT)
    # Let eddies drift slowly and pulse over time
    for t in range(_nt):
        phase = 2 * np.pi * t / _nt
        modulation = 0.7 + 0.3 * np.sin(phase)
        drift = t * 0.001  # slow westward drift in degrees
        ue_t, ve_t = _stream_function_eddy(
            lon0 - drift, lat0, rad, sign * amp * modulation, LON, LAT
        )
        u_current[t] += ue_t
        v_current[t] += ve_t

# Gaussian-filtered noise for small-scale variability
_noise_u = rng.normal(0, 0.04, (_nt, _nlat, _nlon))
_noise_v = rng.normal(0, 0.04, (_nt, _nlat, _nlon))
for t in range(_nt):
    _noise_u[t] = gaussian_filter(_noise_u[t], sigma=2)
    _noise_v[t] = gaussian_filter(_noise_v[t], sigma=2)

u_current += _noise_u
v_current += _noise_v

# ── Generate wind ───────────────────────────────────────────────────────

# Easterly trades: ~7 m/s from ENE (so u < 0, v slightly positive)
_base_u_wind = -6.5
_base_v_wind = 2.5

u_wind = np.full((_nt, _nlat, _nlon), _base_u_wind)
v_wind = np.full((_nt, _nlat, _nlon), _base_v_wind)

# 3–5 day synoptic oscillation
_hours = np.arange(_nt) * DT_FORCING_HOURS
_synoptic_period_hours = 4 * 24  # 4-day period
_synoptic_u = 2.0 * np.sin(2 * np.pi * _hours / _synoptic_period_hours)
_synoptic_v = 1.0 * np.cos(2 * np.pi * _hours / _synoptic_period_hours)

for t in range(_nt):
    u_wind[t] += _synoptic_u[t]
    v_wind[t] += _synoptic_v[t]

# Spatial + temporal noise
_wnoise_u = rng.normal(0, 0.8, (_nt, _nlat, _nlon))
_wnoise_v = rng.normal(0, 0.8, (_nt, _nlat, _nlon))
for t in range(_nt):
    _wnoise_u[t] = gaussian_filter(_wnoise_u[t], sigma=1.5)
    _wnoise_v[t] = gaussian_filter(_wnoise_v[t], sigma=1.5)

u_wind += _wnoise_u
v_wind += _wnoise_v

# NOTE: Land mask is NOT applied to forcing fields — particles near the
# coast need valid interpolated velocities.  Coast/land contact is handled
# by the advection step's per-substep geometry checks.
