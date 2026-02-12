# Sargassum Coastal Exposure Forecast — Barbados

Lagrangian particle model forecasting sargassum washup probability around Barbados. 10,000 virtual rafts are advected with RK4 integration through synthetic ocean currents and wind fields, and coastal arrival probability is computed per shoreline segment.

Interactive Dash app with satellite imagery, magma-colored coastal risk heatmap, and polar washup profile.

![App screenshot](https://raw.githubusercontent.com/donalhill/sargassum-washin-model/main/docs/screenshot.png)

## How it works

1. **Seeding** — 10,000 particles are scattered uniformly within 100 km of Barbados
2. **Advection** — Each particle is propagated for 2 days using 4th-order Runge-Kutta integration (1-hour substeps) under surface currents + wind leeway
3. **Coastal detection** — Particles that reach within ~450 m of the GADM 4.1 coastline are registered as arrivals
4. **Risk rollup** — Arrival counts are normalised per coastline segment to give washup probability, then smoothed along-shore

Every click of **Simulate** draws a fresh random seed, so the ensemble varies each time.

## Ocean forcing (synthetic)

This demo uses synthetic fields as a stand-in for real ocean data:

| Field | Description |
|-------|-------------|
| **Currents** | Westward base flow (−0.3 m/s NEC analogue) + 3 divergence-free mesoscale eddies + Gaussian-filtered noise |
| **Wind** | Easterly trades (~7 m/s from ENE) + 4-day synoptic oscillation + spatiotemporal noise |

A real deployment would ingest CMEMS reanalysis currents and GFS/ECMWF wind forecasts.

## Project structure

```
├── app.py                    # Dash layout and server entry point
├── callbacks.py              # Simulate button callback, figure builders
├── config.py                 # Domain bounds, grid spec, model parameters
├── data/
│   ├── coastline.py          # GADM 4.1 Barbados coastline (~400 points)
│   └── synthetic_forcing.py  # Synthetic current and wind fields
├── model/
│   ├── advection.py          # Vectorised RK4 Lagrangian advection
│   ├── ensemble.py           # Stochastic ensemble runner
│   └── risk.py               # Coastal segment probability rollup
├── precompute.py             # Optional: pre-generate ensemble to .npz
├── requirements.txt
└── render.yaml               # One-click deploy to Render
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Opens at [http://localhost:8050](http://localhost:8050). Click **Simulate** to run the ensemble.

## Deploy to Render

The included `render.yaml` is ready for Render's free tier:

```
Build command:  pip install -r requirements.txt
Start command:  gunicorn app:server --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

No environment variables or API keys required.

## Dependencies

Five direct dependencies — no cartopy, no geopandas, no netCDF4:

- dash
- plotly
- numpy
- scipy
- gunicorn
