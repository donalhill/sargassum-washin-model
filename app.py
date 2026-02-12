"""Dash application: layout and server entry point."""

import dash
import plotly.graph_objects as go
from dash import dcc, html

from data.coastline import COORDS

app = dash.Dash(
    __name__,
    title="Sargassum Coastal Exposure — Barbados",
    update_title="Computing...",
)
server = app.server  # for gunicorn

# ── Satellite base map (shown on load) ────────────────────────────────

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

_initial_map = go.Figure(data=[
    go.Scattermap(
        lon=COORDS[:, 0].tolist(), lat=COORDS[:, 1].tolist(),
        mode="lines", line=dict(width=1.5, color="rgba(255,255,255,0.5)"),
        name="Coastline", hoverinfo="skip",
    ),
])
_initial_map.update_layout(
    map=dict(style=_SAT_STYLE, center=dict(lon=-59.55, lat=13.17), zoom=8),
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
)

# ── Info card helper ──────────────────────────────────────────────────

_CARD = {
    "background": "#fff", "borderRadius": "8px",
    "border": "1px solid #e0e0e0", "padding": "14px 16px",
    "borderLeft": "4px solid {color}",
}


def _card(title, body, color="#1a73e8"):
    style = {**_CARD, "borderLeft": f"4px solid {color}"}
    return html.Div(style=style, children=[
        html.Div(title, style={"fontWeight": "700", "fontSize": "13px",
                                "marginBottom": "6px", "color": "#333"}),
        html.Div(body, style={"fontSize": "12px", "color": "#555",
                               "lineHeight": "1.55"}),
    ])


# ── Layout ────────────────────────────────────────────────────────────

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, sans-serif",
           "margin": "0 auto", "maxWidth": "1500px", "padding": "16px"},
    children=[
        html.H2("Sargassum Coastal Exposure Forecast",
                style={"marginBottom": "2px", "letterSpacing": "-0.5px"}),
        html.P("Barbados \u2014 2-day integrated washup risk from uniform ocean prior",
               style={"color": "#888", "marginTop": 0, "fontSize": "13px",
                      "marginBottom": "14px"}),

        # Info cards
        html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                   "gap": "10px", "marginBottom": "14px"},
            children=[
                _card("Particle Transport",
                      "10,000 virtual sargassum rafts seeded uniformly within "
                      "100 km of Barbados. Each is propagated forward for 2 days "
                      "using RK4 Lagrangian advection with 1-hour substeps. "
                      "The blue circle indicates the 100 km seeding zone. "
                      "Gold tracks reach the coast; faint tracks do not.",
                      "#1a73e8"),
                _card("Ocean Forcing Model",
                      "Surface currents: synthetic North Equatorial Current "
                      "(\u22120.3 m/s westward) with mesoscale eddies. "
                      "Wind: synthetic easterly trades (~7 m/s) with "
                      "\u03b1 = 0.02 leeway. Real deployment would ingest "
                      "CMEMS reanalysis + GFS/ECMWF wind forecasts.",
                      "#e8a21a"),
                _card("Integrated Risk Map",
                      "The coastal heatmap shows the probability that "
                      "sargassum arrives at each segment, integrated over "
                      "2 days of exposure from a uniform prior on initial "
                      "positions. Brighter = higher washup probability. "
                      "Gold tracks are rafts that wash up on the coast; "
                      "faint blue tracks are those that miss.",
                      "#e84040"),
                _card("Coastal Risk Profile",
                      "The polar chart divides the coastline into 20 equal "
                      "segments. Each slice shows that segment's share of "
                      "the total washup probability, normalised to 100%. "
                      "Taller slices receive a larger fraction of incoming "
                      "sargassum. Click Simulate again for a fresh ensemble.",
                      "#7c3aed"),
            ],
        ),

        # Simulate button + stats
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "16px",
                   "marginBottom": "10px"},
            children=[
                html.Button(
                    "Simulate", id="run-button", n_clicks=0,
                    style={"padding": "9px 32px", "fontSize": "14px",
                           "cursor": "pointer", "background": "#1a73e8",
                           "color": "white", "border": "none",
                           "borderRadius": "6px", "fontWeight": "600",
                           "letterSpacing": "0.3px"},
                ),
                html.Div(
                    id="stats-text",
                    style={"fontSize": "12px", "color": "#666",
                           "minHeight": "20px"},
                ),
            ],
        ),

        # Map + profile
        dcc.Loading(
            type="circle",
            children=html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
                children=[
                    html.Div(
                        dcc.Graph(id="map-figure", figure=_initial_map,
                                  style={"height": "420px"},
                                  config={"scrollZoom": True}),
                        style={"flex": "1.3", "minWidth": "360px",
                               "borderRadius": "8px", "overflow": "hidden"},
                    ),
                    html.Div(
                        id="profile-container",
                        children=dcc.Graph(id="profile-figure",
                                           style={"height": "420px"}),
                        style={"flex": "1", "minWidth": "300px",
                               "visibility": "hidden"},
                    ),
                ],
            ),
        ),
    ],
)

# Register callbacks
import callbacks  # noqa: F401, E402

if __name__ == "__main__":
    app.run(debug=True, port=8050)
