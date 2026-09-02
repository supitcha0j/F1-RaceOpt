# RaceOpt

RaceOpt analyzes and simulates Formula 1 race strategy from real lap telemetry. It predicts lap times with a `RandomForestRegressor` trained on [FastF1](https://github.com/theOehrly/Fast-F1) data, then uses that model to grid-search for faster pit strategies and to simulate full-grid races for user-defined strategies.

## Features

- **Strategy Analysis** — pick a real race/driver combo; the model calibrates against real lap times, grid-searches tyre/pit-lap combinations, and classifies the results as undercut/overcut
- **Strategy Lab** — build your own tyre strategy and simulate it against the full real grid, with an animated lap-by-lap position replay
- **Drivers / Teams** — rosters for the current season with real photos, team colors, and car numbers (sourced from OpenF1 and Wikimedia Commons)
- **Model Report** — validation numbers (MAE/RMSE, real vs. simulated) computed live from cached telemetry, not hardcoded

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

`model.pkl` and the `cache/` directory are committed to the repo, so the app works out of the box on a fresh clone with no network access — every page can be exercised from cached data alone.

To retrain the model or refresh driver/circuit media:

```bash
python train_model_advanced.py    # retrain model.pkl
python fetch_media.py             # refresh driver photos, circuit photos, team data
```

## Deploying

A `Dockerfile` and `render.yaml` are included for deploying with [Render](https://render.com) (Blueprint) or [Railway](https://railway.app) (auto-detects the Dockerfile). See `CLAUDE.md` for details.

## Tech stack

Flask · scikit-learn · pandas/numpy · FastF1 · Chart.js
