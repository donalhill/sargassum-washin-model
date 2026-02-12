"""Vectorised RK4 Lagrangian particle advection.

Optimised for large particle counts (100k+): bounding-box pre-filter on
coastal hit detection, per-substep coast/land checking to prevent tunnelling.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from config import (
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
    DT_FORCING_HOURS, DT_SUBSTEP_HOURS,
    COAST_BUFFER_DEG,
)
from data.coastline import COORDS as COAST_COORDS, _GADM, points_in_polygon

_coast_lon = COAST_COORDS[:, 0]
_coast_lat = COAST_COORDS[:, 1]

# Bounding box of coastline + buffer (for fast pre-filter)
_BBOX_LO = _coast_lon.min() - COAST_BUFFER_DEG * 3
_BBOX_HI = _coast_lon.max() + COAST_BUFFER_DEG * 3
_BBOX_LA_LO = _coast_lat.min() - COAST_BUFFER_DEG * 3
_BBOX_LA_HI = _coast_lat.max() + COAST_BUFFER_DEG * 3

# Degrees-per-metre at ~13°N
_DEG_PER_M_LAT = 1.0 / 111_320
_DEG_PER_M_LON = 1.0 / (111_320 * np.cos(np.radians(13.15)))

_DT_SUB = DT_SUBSTEP_HOURS * 3600.0
_N_SUB = DT_FORCING_HOURS // DT_SUBSTEP_HOURS


def build_interpolators(u_c, v_c, u_w, v_w, lon_grid, lat_grid):
    """Pre-build interpolators for one forcing timestep (reused across RK4)."""
    def _bi(field):
        return RegularGridInterpolator(
            (lat_grid, lon_grid), field,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
    return _bi(u_c), _bi(v_c), _bi(u_w), _bi(v_w)


def _effective_velocity(lon, lat, alpha, interps,
                        noise_cu, noise_cv, noise_wu, noise_wv):
    """Compute effective velocity (deg/s) using cached interpolators."""
    iuc, ivc, iuw, ivw = interps
    pts = np.column_stack([lat, lon])

    uc = np.nan_to_num(iuc(pts), nan=0.0)
    vc = np.nan_to_num(ivc(pts), nan=0.0)
    uw = np.nan_to_num(iuw(pts), nan=0.0)
    vw = np.nan_to_num(ivw(pts), nan=0.0)

    u_eff = uc + alpha * uw
    v_eff = vc + alpha * vw

    if noise_cu is not None:
        u_eff += noise_cu + alpha * noise_wu
        v_eff += noise_cv + alpha * noise_wv

    dlon_dt = u_eff * _DEG_PER_M_LON
    dlat_dt = v_eff * _DEG_PER_M_LAT
    return dlon_dt, dlat_dt


def _near_coast(lon, lat):
    """Boolean mask: True if particle within COAST_BUFFER_DEG of coast.

    Uses bounding-box pre-filter so only particles near the island
    pay the full distance-matrix cost.
    """
    result = np.zeros(len(lon), dtype=bool)
    near_bbox = ((lon >= _BBOX_LO) & (lon <= _BBOX_HI) &
                 (lat >= _BBOX_LA_LO) & (lat <= _BBOX_LA_HI))
    if not near_bbox.any():
        return result

    idx = np.where(near_bbox)[0]
    dlon = lon[idx, None] - _coast_lon[None, :]
    dlat = lat[idx, None] - _coast_lat[None, :]
    min_dist = np.sqrt((dlon ** 2 + dlat ** 2).min(axis=1))
    result[idx] = min_dist < COAST_BUFFER_DEG
    return result


def _on_land(lon, lat):
    """Boolean mask: True if particle inside the Barbados polygon.

    Uses same bounding-box pre-filter as _near_coast.
    """
    result = np.zeros(len(lon), dtype=bool)
    near_bbox = ((lon >= _BBOX_LO) & (lon <= _BBOX_HI) &
                 (lat >= _BBOX_LA_LO) & (lat <= _BBOX_LA_HI))
    if not near_bbox.any():
        return result

    idx = np.where(near_bbox)[0]
    result[idx] = points_in_polygon(lon[idx], lat[idx], _GADM)
    return result


def _outside_domain(lon, lat):
    return (lon < LON_MIN) | (lon > LON_MAX) | (lat < LAT_MIN) | (lat > LAT_MAX)


def rk4_step(lon, lat, alpha, interps,
             noise_cu=None, noise_cv=None,
             noise_wu=None, noise_wv=None):
    """Advance particles by one forcing timestep using RK4 with substeps.

    Coast and land checks run after each substep to prevent tunnelling.
    Particles that hit coast or cross land are frozen at their hit position.

    Returns
    -------
    lon_new, lat_new, hit_coast, left_domain : ndarrays
    """
    lo = lon.copy()
    la = lat.copy()
    n = len(lo)
    frozen = np.zeros(n, dtype=bool)

    def _vel(lo_, la_):
        return _effective_velocity(lo_, la_, alpha, interps,
                                   noise_cu, noise_cv, noise_wu, noise_wv)

    for _ in range(_N_SUB):
        move = ~frozen
        if not move.any():
            break

        k1_lo, k1_la = _vel(lo, la)
        k2_lo, k2_la = _vel(lo + 0.5 * _DT_SUB * k1_lo,
                            la + 0.5 * _DT_SUB * k1_la)
        k3_lo, k3_la = _vel(lo + 0.5 * _DT_SUB * k2_lo,
                            la + 0.5 * _DT_SUB * k2_la)
        k4_lo, k4_la = _vel(lo + _DT_SUB * k3_lo,
                            la + _DT_SUB * k3_la)

        dlo = (_DT_SUB / 6) * (k1_lo + 2 * k2_lo + 2 * k3_lo + k4_lo)
        dla = (_DT_SUB / 6) * (k1_la + 2 * k2_la + 2 * k3_la + k4_la)

        lo[move] += dlo[move]
        la[move] += dla[move]

        # Freeze particles that reach coast or cross onto land
        new_hit = (~frozen) & (_near_coast(lo, la) | _on_land(lo, la))
        frozen |= new_hit

    hit = frozen
    left = _outside_domain(lo, la) & ~hit
    return lo, la, hit, left
