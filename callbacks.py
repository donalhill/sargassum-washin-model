"""Dash callbacks: compute ensemble on click, display results.

Single callback: click Simulate → run 10k-particle ensemble with random seed,
build map + pizza chart, return figures.
"""

import time
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback

from config import (
    N_DAYS, ALPHA_DEFAULT, BUFFER_KM,
    ISLAND_CENTER_LON, ISLAND_CENTER_LAT,
)
from data.coastline import (
    COORDS, SEGMENT_MIDPOINTS, CUMULATIVE_DISTANCE_KM, SEGMENT_LABELS,
)
from model.ensemble import run_ensemble
from model.risk import compute_risk

# ── Constants ──────────────────────────────────────────────────────────

N_ANIM = 10_000
DAY_IDX = N_DAYS - 1

# ── Map style ──────────────────────────────────────────────────────────

_SAT_STYLE = {
    "version": 8,
    "sources": {
        "satellite": {
            "type": "raster",
            "tiles": [
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ],
            "tileSize": 256,
            "attribution": "Esri, Maxar, Earthstar Geographics",
        }
    },
    "layers": [{"id": "satellite", "type": "raster", "source": "satellite"}],
}

_MAP_CENTER = dict(lon=-59.55, lat=13.17)
_MAP_ZOOM = 8

# ── Colour helpers ─────────────────────────────────────────────────────

_MAGMA = [
    (0.00, (0, 0, 4)),
    (0.10, (20, 14, 54)),
    (0.20, (59, 15, 112)),
    (0.30, (100, 26, 128)),
    (0.40, (140, 41, 129)),
    (0.50, (183, 55, 121)),
    (0.60, (222, 73, 104)),
    (0.70, (247, 115, 92)),
    (0.80, (252, 176, 118)),
    (0.90, (252, 224, 158)),
    (1.00, (252, 253, 191)),
]
_N_COLOR_BINS = 32


def _magma_color(t):
    t = max(0.0, min(1.0, t))
    for i in range(len(_MAGMA) - 1):
        s0, c0 = _MAGMA[i]
        s1, c1 = _MAGMA[i + 1]
        if t <= s1:
            f = (t - s0) / (s1 - s0) if s1 > s0 else 0
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f"rgb({r},{g},{b})"
    return f"rgb({_MAGMA[-1][1][0]},{_MAGMA[-1][1][1]},{_MAGMA[-1][1][2]})"


def _map_layout():
    return dict(
        map=dict(style=_SAT_STYLE, center=_MAP_CENTER, zoom=_MAP_ZOOM),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )


# ── Segment builder (vectorised per timestep) ────────────────────────

def _segments_for_step(traj_lon, traj_lat, t, mask):
    """Build None-separated line segments for timestep t→t+1."""
    lo0 = traj_lon[mask, t];  la0 = traj_lat[mask, t]
    lo1 = traj_lon[mask, t + 1]; la1 = traj_lat[mask, t + 1]
    valid = (~np.isnan(lo0) & ~np.isnan(lo1)
             & (np.abs(lo1 - lo0) < 0.2) & (np.abs(la1 - la0) < 0.2))
    lo0, la0 = lo0[valid], la0[valid]
    lo1, la1 = lo1[valid], la1[valid]
    n = len(lo0)
    if n == 0:
        return [], []
    seg_lon = np.empty(n * 3)
    seg_lon[0::3] = np.round(lo0, 3)
    seg_lon[1::3] = np.round(lo1, 3)
    seg_lon[2::3] = np.nan
    seg_lat = np.empty(n * 3)
    seg_lat[0::3] = np.round(la0, 3)
    seg_lat[1::3] = np.round(la1, 3)
    seg_lat[2::3] = np.nan
    lons = [None if x != x else x for x in seg_lon.tolist()]
    lats = [None if x != x else x for x in seg_lat.tolist()]
    return lons, lats


# ── Figure builders ──────────────────────────────────────────────────

def _build_map(traj_lon, traj_lat, hit_mask, risk):
    fig = go.Figure()
    n_steps = traj_lon.shape[1] - 1

    # Miss tracks — graduated opacity by timestep
    for t in range(n_steps):
        lons, lats = _segments_for_step(traj_lon, traj_lat, t, ~hit_mask)
        if not lons:
            continue
        frac = t / max(n_steps - 1, 1)
        alpha = 0.03 + 0.09 * frac
        width = 0.3 + 0.2 * frac
        fig.add_trace(go.Scattermap(
            lon=lons, lat=lats, mode="lines",
            line=dict(width=width, color=f"rgba(180,200,255,{alpha:.3f})"),
            hoverinfo="skip", showlegend=False,
        ))

    # Hit tracks — graduated opacity by timestep
    for t in range(n_steps):
        lons, lats = _segments_for_step(traj_lon, traj_lat, t, hit_mask)
        if not lons:
            continue
        frac = t / max(n_steps - 1, 1)
        alpha = 0.03 + 0.17 * frac
        width = 0.5 + 0.8 * frac
        fig.add_trace(go.Scattermap(
            lon=lons, lat=lats, mode="lines",
            line=dict(width=width, color=f"rgba(255,200,50,{alpha:.3f})"),
            hoverinfo="skip", showlegend=False,
        ))

    # Seed point markers — miss (faint) then hit (orange) on top
    start_lo = np.round(traj_lon[:, 0], 3)
    start_la = np.round(traj_lat[:, 0], 3)
    fig.add_trace(go.Scattermap(
        lon=start_lo[~hit_mask].tolist(), lat=start_la[~hit_mask].tolist(),
        mode="markers",
        marker=dict(size=1.5, color="rgba(180,200,255,0.25)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattermap(
        lon=start_lo[hit_mask].tolist(), lat=start_la[hit_mask].tolist(),
        mode="markers",
        marker=dict(size=3, color="rgba(255,200,50,0.5)"),
        hoverinfo="skip", showlegend=False,
    ))

    # Seed zone circle (100 km radius)
    _km_per_deg_lat = 111.32
    _km_per_deg_lon = 111.32 * np.cos(np.radians(ISLAND_CENTER_LAT))
    theta = np.linspace(0, 2 * np.pi, 120)
    circ_lon = ISLAND_CENTER_LON + (BUFFER_KM * np.cos(theta)) / _km_per_deg_lon
    circ_lat = ISLAND_CENTER_LAT + (BUFFER_KM * np.sin(theta)) / _km_per_deg_lat
    fig.add_trace(go.Scattermap(
        lon=np.round(circ_lon, 3).tolist(),
        lat=np.round(circ_lat, 3).tolist(),
        mode="lines",
        line=dict(width=1, color="rgba(100,160,255,0.4)"),
        hoverinfo="skip", showlegend=False,
    ))

    # Risk-colored coastline (magma gradient, grouped by color bin)
    prob = risk[:, DAY_IDX]
    pmax = max(prob.max(), 1e-8)
    normalized = prob / pmax
    bin_idx = np.clip((normalized * _N_COLOR_BINS).astype(int), 0, _N_COLOR_BINS - 1)

    for b in range(_N_COLOR_BINS):
        segs = np.where(bin_idx == b)[0]
        if len(segs) == 0:
            continue
        lons, lats = [], []
        for s in segs:
            lons.extend([float(COORDS[s, 0]), float(COORDS[s + 1, 0]), None])
            lats.extend([float(COORDS[s, 1]), float(COORDS[s + 1, 1]), None])
        t = b / max(_N_COLOR_BINS - 1, 1)
        fig.add_trace(go.Scattermap(
            lon=lons, lat=lats, mode="lines",
            line=dict(width=5, color=_magma_color(t)),
            hoverinfo="skip", showlegend=False,
        ))

    # Invisible hover markers at segment midpoints
    hover = [f"{SEGMENT_LABELS.get(i, f'Seg {i}')}<br>"
             f"P = {100*p:.3f}%" for i, p in enumerate(prob)]
    fig.add_trace(go.Scattermap(
        lon=SEGMENT_MIDPOINTS[:, 0].tolist(),
        lat=SEGMENT_MIDPOINTS[:, 1].tolist(),
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)", opacity=0),
        text=hover, hoverinfo="text",
        name="Washup risk",
    ))

    fig.update_layout(**_map_layout())
    return fig


_N_PIZZA = 20


def _build_profile(risk):
    prob = risk[:, DAY_IDX]
    prob_pct = prob * 100
    n_segs = len(prob)
    total_dist = CUMULATIVE_DISTANCE_KM[-1]

    # Bin segments into pizza slices
    bin_edges = np.linspace(0, total_dist, _N_PIZZA + 1)
    mid_dist = (CUMULATIVE_DISTANCE_KM[:-1] + CUMULATIVE_DISTANCE_KM[1:]) / 2
    bin_idx = np.digitize(mid_dist, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, _N_PIZZA - 1)

    # Normalise so slices sum to 100%
    bin_sum = np.zeros(_N_PIZZA)
    for b in range(_N_PIZZA):
        mask = bin_idx == b
        if mask.any():
            bin_sum[b] = prob[mask].sum()
    total = bin_sum.sum()
    bin_pct = (bin_sum / total * 100) if total > 0 else bin_sum
    bin_angles = np.linspace(0, 360, _N_PIZZA, endpoint=False) + 360 / _N_PIZZA / 2

    # Nearest landmark for each bin
    bin_labels = []
    for b in range(_N_PIZZA):
        mid = (bin_edges[b] + bin_edges[b + 1]) / 2
        best = None
        for seg_idx, label in SEGMENT_LABELS.items():
            if seg_idx < n_segs:
                d = abs(mid_dist[seg_idx] - mid)
                if best is None or d < best[0]:
                    best = (d, label)
        bin_labels.append(best[1] if best else f"Bin {b}")

    hover = [f"{bin_labels[i]}<br>P = {bin_pct[i]:.1f}%" for i in range(_N_PIZZA)]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=bin_pct,
        theta=bin_angles,
        width=360 / _N_PIZZA,
        marker_color="rgba(255,200,50,0.7)",
        marker_line_color="rgba(255,200,50,1)",
        marker_line_width=1,
        text=hover,
        hoverinfo="text",
    ))

    # Landmark labels
    placed = set()
    for seg_idx, label in SEGMENT_LABELS.items():
        if seg_idx < n_segs:
            b = int(bin_idx[seg_idx])
            if b in placed:
                continue
            placed.add(b)
            a = float(bin_angles[b])
            fig.add_annotation(
                x=0.5 + 0.47 * np.cos(np.radians(90 - a)),
                y=0.5 + 0.47 * np.sin(np.radians(90 - a)),
                xref="paper", yref="paper",
                text=label, showarrow=False,
                font=dict(size=8, color="#555"),
            )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True,
                            ticksuffix="%", tickfont=dict(size=9)),
            angularaxis=dict(visible=False),
        ),
        margin=dict(l=30, r=30, t=20, b=20),
        showlegend=False,
    )
    return fig


# ── Callback: Simulate button ────────────────────────────────────────

@callback(
    Output("map-figure", "figure"),
    Output("profile-figure", "figure"),
    Output("stats-text", "children"),
    Output("profile-container", "style"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True,
)
def run_model(n_clicks):
    seed = int(time.time() * 1000) % (2**31)
    result = run_ensemble(alpha=ALPHA_DEFAULT, n_particles=N_ANIM, seed=seed)
    risk = compute_risk(result)

    hit_mask = result.hit_segment >= 0
    n_hits = int(hit_mask.sum())
    hit_pct = 100 * n_hits / result.n_total

    map_fig = _build_map(result.traj_lon, result.traj_lat, hit_mask, risk)
    profile_fig = _build_profile(risk)
    stats = f"{result.n_total:,} particles \u00b7 {n_hits:,} arrivals ({hit_pct:.1f}%)"

    return (map_fig, profile_fig, stats,
            {"flex": "1", "minWidth": "300px", "visibility": "visible"})
