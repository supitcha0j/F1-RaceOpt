# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RaceOpt is a Flask app that analyzes and simulates Formula 1 race strategy from real lap telemetry (via the `fastf1` library). It predicts lap times with a `RandomForestRegressor`, then uses that model to grid-search for faster pit strategies and to simulate full-grid races for user-defined strategies. All Thai-language comments in the code explain *why*, not *what* — read them, they carry real constraints (e.g. why a function avoids a live fetch).

## Commands

```bash
pip install -r requirements.txt   # fastf1, pandas, numpy, scikit-learn, flask, matplotlib, requests
python app.py                     # dev server on :5000, debug=True (auto-reload)
python train_model_advanced.py    # retrain model.pkl from TRAIN_COMBINATIONS in that file
python fetch_media.py             # one-off: cache real driver/circuit photos to static/img/ (see media.py)
```

`model.pkl` and the `cache/` directory are committed to the repo, so `python app.py` works out of the box on a fresh clone with no network access — every route can be exercised from cached data alone. There is no test suite, linter, or build step configured.

## Deploying

`Dockerfile` runs the app with `gunicorn` (never the `debug=True` dev server). `requirements.txt` is pinned to the exact versions `model.pkl` was trained/tested against — notably `scikit-learn==1.7.2`; bumping it silently risks an `InconsistentVersionWarning` or, on a bigger jump, a broken unpickle. `.dockerignore` excludes `cache/2023_*` (old, unused season data kept in git per a deliberate choice, but not worth shipping to production — see the file's comment) from the built image. `render.yaml` is a Render Blueprint (`docker` env) for one-click deploy; Railway auto-detects the `Dockerfile` with no extra config. Build/run were verified locally with `docker build` + `docker run` before this was written.

## Architecture

**Request flow**: `app.py` is the only Flask entrypoint (all routes). It loads `model.pkl` once at import time and builds a module-level `COMBOS` list at startup (see `_build_combos`) by scanning `cache/*.pkl` for driver/race combos that already have cached lap data, then re-running the model against them — this is the app's source of truth for "real vs. simulated" accuracy numbers shown on Model Report, and it deliberately never fetches live data at startup.

**The three-layer pipeline**, each in its own module:
- `data_pipeline.py` — all FastF1 access goes through here. Nearly every function follows the same shape: check a disk-cached `.pkl` under `cache/` first, only call `fastf1.get_session(...).load()` on a miss. `try_load_cached_race_laps` and `estimate_pit_loss` are disk-only and never fetch; `get_race_weather`/`get_race_grid`/`get_race_incidents`/`load_race_laps` will fetch-and-cache on a miss. When adding code that iterates many races (e.g. a new "all circuits" view), prefer the disk-only helpers or data already in `COMBOS` — iterating the never-fetch-live functions over an uncached season is slow/network-dependent.
- `strategy_optimizer.py` — used by `/analysis`. Builds per-lap feature rows for a candidate strategy (`build_strategy_laps`), calibrates a per-driver `pace_offset` against their real laps (`calibrate_pace_offset`), then grid-searches compound/pit-lap combinations (`grid_search_strategies`) and classifies results as undercut/overcut (`classify_pit_tactics`).
- `race_simulator.py` — used by `/play` (Strategy Lab). Simulates an entire grid at once (`simulate_full_race`/`simulate_driver`) using real grid positions and per-driver pace offsets (real if cached, deterministic-synthetic fallback otherwise via `build_field_strategies`). `compute_lap_positions` derives lap-by-lap rank (not just final result) from cumulative lap time, which drives the Strategy Lab's animated position replay.

**Media**: `media.py` reads real driver headshots / circuit photos back off disk (`static/img/drivers/`, `static/img/circuits/`) and driver metadata (team, team colour, car number — `static/data/drivers.json`), and is registered as Jinja globals (`driver_photo`, `circuit_photo`) so templates can call them directly; it never fetches, only reads from disk, keeping app startup offline like everything else. `fetch_media.py` is the one-off script that actually populates those files (OpenF1 `headshot_url`/team data for drivers, Wikimedia Commons search for circuit photos) — rerun it after a season changes or to backfill new drivers/circuits; it's safe to rerun since it skips image files already on disk (metadata is small and always re-fetched). No real team logos are used anywhere (trademark risk) — team identity is shown via `team_colour` instead.

**Templates**: `templates/base.html` is a single monolithic file holding the *entire* site's design system — all CSS custom properties/tokens, every shared component class (`.panel`, `.metrics`, `.tbl`, `.specbar`, `.pill`, replay/ladder/confetti styles, etc.), the nav/footer, and the shared `data-countup` animation script. Every other template `{% extends "base.html" %}` and only fills `content`/`scripts` blocks — there are no per-page stylesheets or JS files. When styling something new, check base.html for an existing class before adding one.

**Pages and their data source**:
- `/` (Home) and `/analysis` (Strategy Analysis) both call `_run_strategy_analysis` (in app.py) for one driver/race: real stints, calibration, grid-search results, undercut/overcut tactics.
- `/play` (Strategy Lab) — user-defined strategy simulated against the full real grid; renders an animated lap-by-lap position replay (Chart.js + a live standings ladder + commentary feed + synthesized Web Audio sound effects, all client-side in `play_strategy.html`).
- `/model-report` — validation numbers straight from `COMBOS`.
- `/drivers`, `/teams` — roster pages built from `media.driver_meta_list()` (cached OpenF1 data). Cards link to detail pages (`/drivers/<code>`, `/teams/<slug>` — `slug` from `_team_slug`) that are deliberately not in the nav bar, reachable only from these roster cards, so clicking a driver/team never silently jumps you to an unrelated tool page. `/drivers/<code>` lists every race that driver appears in this season (from `DRIVER_RACES`, built once at startup, offline, from the season schedule) as its own card into Strategy Analysis for that specific race; `/teams/<slug>` lists that team's drivers, each linking to their `/drivers/<code>` page (`_build_teams()` is the shared grouping helper both team routes use).
- `/about` — static content page.
- `/analysis` and `/play` read `race_key`/`driver` via `request.values` (not just `request.form`), and `/analysis` auto-runs on `GET` when `?race_key=...` is present — this lets links from elsewhere (e.g. Home) deep-link straight into a live result.

**Cache key conventions** (all under `cache/`, `.pkl`): `{year}_{gp_round}_{driver}.pkl` (per-driver laps), `{year}_{gp_round}_weather.pkl`, `_grid.pkl`, `_incidents.pkl` (per-race, no driver), and `schedule_{year}.pkl` (season calendar + driver lists). `gp_round` is the numeric round, not the circuit name.
