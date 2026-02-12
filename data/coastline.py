"""Embedded Barbados coastline from GADM 4.1 boundary data.

Resampled to uniform arc-length spacing via linear interpolation.
Stored directly in source to avoid file-I/O or path issues on Render.

Public API
----------
COORDS : ndarray (N, 2)            – (lon, lat) around the island
SEGMENT_MIDPOINTS : ndarray (N-1,2) – midpoint of each consecutive pair
CUMULATIVE_DISTANCE_KM : ndarray (N,) – arc-length from first point (km)
SEGMENT_LABELS : dict               – {index: name} for notable locations
nearest_segment(lon, lat) -> int    – index of closest coastal segment
"""

import numpy as np

# ── GADM 4.1 boundary coordinates ──────────────────────────────────────
# 139 vertices from gadm41_BRB_0.json, counterclockwise from SE coast.
# Reordered below to start from North Point for intuitive labelling.
_GADM = np.array([
    [-59.4743, 13.0765], [-59.489, 13.066], [-59.4954, 13.0665],
    [-59.5037, 13.0521], [-59.5057, 13.0532], [-59.519, 13.0454],
    [-59.5218, 13.0465], [-59.5282, 13.0446], [-59.534, 13.0471],
    [-59.5382, 13.059], [-59.5426, 13.0629], [-59.5443, 13.0621],
    [-59.5535, 13.0657], [-59.5651, 13.0637], [-59.5685, 13.0665],
    [-59.5726, 13.0662], [-59.5893, 13.0735], [-59.5979, 13.0732],
    [-59.6126, 13.0779], [-59.6132, 13.0793], [-59.6096, 13.0829],
    [-59.6101, 13.0907], [-59.6157, 13.0951], [-59.6218, 13.0954],
    [-59.6237, 13.0979], [-59.6332, 13.1021], [-59.6329, 13.1054],
    [-59.6315, 13.1021], [-59.6299, 13.1021], [-59.6307, 13.1065],
    [-59.6282, 13.1079], [-59.626, 13.1071], [-59.6263, 13.1104],
    [-59.6307, 13.1099], [-59.6307, 13.1121], [-59.6282, 13.1129],
    [-59.6274, 13.1151], [-59.6343, 13.1257], [-59.6387, 13.1437],
    [-59.6371, 13.1671], [-59.6387, 13.169], [-59.6376, 13.1749],
    [-59.6396, 13.1771], [-59.6379, 13.1885], [-59.6413, 13.191],
    [-59.6426, 13.1979], [-59.6404, 13.2029], [-59.6401, 13.2101],
    [-59.6424, 13.2182], [-59.6415, 13.2215], [-59.6435, 13.2254],
    [-59.6424, 13.234], [-59.6449, 13.2399], [-59.644, 13.2499],
    [-59.6451, 13.2535], [-59.6432, 13.2596], [-59.6454, 13.2643],
    [-59.6435, 13.2657], [-59.6449, 13.2663], [-59.6479, 13.2796],
    [-59.6504, 13.2843], [-59.649, 13.2915], [-59.6507, 13.3065],
    [-59.6387, 13.3218], [-59.6374, 13.3268], [-59.631, 13.3287],
    [-59.6296, 13.3307], [-59.6232, 13.3315], [-59.6201, 13.3346],
    [-59.6176, 13.3335], [-59.6146, 13.3351], [-59.6071, 13.3326],
    [-59.6015, 13.3268], [-59.599, 13.3279], [-59.599, 13.3257],
    [-59.5937, 13.3229], [-59.5868, 13.3149], [-59.5821, 13.3143],
    [-59.5762, 13.3074], [-59.5776, 13.3043], [-59.5749, 13.3029],
    [-59.5735, 13.2999], [-59.5771, 13.2977], [-59.5724, 13.2938],
    [-59.5715, 13.2899], [-59.5674, 13.2846], [-59.5671, 13.2818],
    [-59.5685, 13.2813], [-59.5676, 13.2754], [-59.5635, 13.2665],
    [-59.5607, 13.266], [-59.5599, 13.2568], [-59.5571, 13.2557],
    [-59.5554, 13.2526], [-59.5576, 13.2499], [-59.5537, 13.2499],
    [-59.5482, 13.2449], [-59.5393, 13.2296], [-59.5324, 13.2246],
    [-59.5337, 13.2235], [-59.5299, 13.219], [-59.5226, 13.214],
    [-59.5079, 13.2099], [-59.5015, 13.2024], [-59.4943, 13.1974],
    [-59.4879, 13.1979], [-59.4857, 13.1912], [-59.4779, 13.1896],
    [-59.4774, 13.1876], [-59.4676, 13.1799], [-59.4635, 13.1801],
    [-59.4621, 13.1824], [-59.4579, 13.1835], [-59.451, 13.1743],
    [-59.4487, 13.1754], [-59.4499, 13.1726], [-59.4476, 13.1685],
    [-59.4449, 13.1682], [-59.4435, 13.166], [-59.4399, 13.166],
    [-59.4404, 13.1646], [-59.4385, 13.164], [-59.4313, 13.166],
    [-59.4307, 13.1624], [-59.4276, 13.1632], [-59.4254, 13.1565],
    [-59.4193, 13.156], [-59.4221, 13.1529], [-59.4221, 13.1513],
    [-59.4199, 13.1501], [-59.4221, 13.1404], [-59.4301, 13.1235],
    [-59.4399, 13.1132], [-59.446, 13.1018], [-59.4504, 13.1007],
    [-59.4535, 13.0915], [-59.4632, 13.0832], [-59.4685, 13.0813],
    [-59.4743, 13.0765],
])

# Rotate so index 0 = North Point (northernmost vertex, idx 70 in GADM)
_north_idx = int(np.argmax(_GADM[:, 1]))
_RAW = np.roll(_GADM, -_north_idx, axis=0)
# Ensure the ring closes
if not np.allclose(_RAW[0], _RAW[-1]):
    _RAW = np.vstack([_RAW, _RAW[:1]])


# ── Resample to uniform spacing (~400 pts) ─────────────────────────────

def _resample(raw: np.ndarray, n_points: int = 400) -> np.ndarray:
    """Linearly resample to *n_points* uniformly spaced in arc-length."""
    diffs = np.diff(raw, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_length[-1]

    targets = np.linspace(0, total, n_points, endpoint=False)
    lon_new = np.interp(targets, cum_length, raw[:, 0])
    lat_new = np.interp(targets, cum_length, raw[:, 1])
    return np.column_stack([lon_new, lat_new])


COORDS = _resample(_RAW, n_points=400)

# ── Derived quantities ──────────────────────────────────────────────────

def _haversine_km(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in km between paired coordinate rows."""
    lon1, lat1 = np.radians(c1[:, 0]), np.radians(c1[:, 1])
    lon2, lat2 = np.radians(c2[:, 0]), np.radians(c2[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


_seg_distances = _haversine_km(COORDS[:-1], COORDS[1:])

CUMULATIVE_DISTANCE_KM = np.zeros(len(COORDS))
CUMULATIVE_DISTANCE_KM[1:] = np.cumsum(_seg_distances)

SEGMENT_MIDPOINTS = 0.5 * (COORDS[:-1] + COORDS[1:])

# ── Landmark labels ─────────────────────────────────────────────────────
# Assign by finding the segment nearest to each known location.
_LANDMARKS = {
    "North Point":   (-59.6146, 13.3351),
    "Speightstown":  (-59.6440, 13.2499),
    "Holetown":      (-59.6387, 13.1671),
    "Bridgetown":    (-59.6126, 13.0779),
    "Oistins":       (-59.5382, 13.0590),
    "South Point":   (-59.5190, 13.0454),
    "Bathsheba":     (-59.5226, 13.2140),
    "Belleplaine":   (-59.5676, 13.2754),
}


def _find_landmark_indices():
    labels = {}
    for name, (lo, la) in _LANDMARKS.items():
        d2 = (SEGMENT_MIDPOINTS[:, 0] - lo) ** 2 + (SEGMENT_MIDPOINTS[:, 1] - la) ** 2
        labels[int(np.argmin(d2))] = name
    return labels


SEGMENT_LABELS = _find_landmark_indices()


def points_in_polygon(px, py, poly):
    """Ray-casting point-in-polygon for 1-D arrays px, py."""
    n = len(poly)
    inside = np.zeros(px.shape, dtype=bool)
    x1, y1 = poly[0]
    for i in range(1, n):
        x2, y2 = poly[i]
        mask = ((py > min(y1, y2)) & (py <= max(y1, y2)) &
                (px <= max(x1, x2)))
        if y1 != y2:
            xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
            mask = mask & ((x1 == x2) | (px <= xinters))
        inside[mask] = ~inside[mask]
        x1, y1 = x2, y2
    return inside


def nearest_segment(lon: float | np.ndarray, lat: float | np.ndarray) -> np.ndarray:
    """Return index of closest coastal segment midpoint for each (lon, lat).

    Parameters
    ----------
    lon, lat : float or 1-D array

    Returns
    -------
    indices : int ndarray (same shape as input)
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    d = (SEGMENT_MIDPOINTS[:, 0][None, :] - lon[:, None]) ** 2 + \
        (SEGMENT_MIDPOINTS[:, 1][None, :] - lat[:, None]) ** 2
    return np.argmin(d, axis=1)
