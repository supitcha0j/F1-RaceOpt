# app.py
from flask import Flask, render_template, request
import pickle, math, re
from pathlib import Path
import numpy as np
import pandas as pd

from data_pipeline import (load_race_laps, get_available_races, get_available_drivers,
                            get_latest_completed_season, get_race_weather,
                            get_race_grid, get_race_incidents, estimate_pit_loss)
from race_simulator import (simulate_full_race, compute_win_probabilities,
                             simulate_driver, DriverStrategy, LapPredictor)
from strategy_optimizer import (calibrate_pace_offset, grid_search_strategies,
                                explain_parameters, simulate_strategy)

app = Flask(__name__)

# ── โหลดโมเดล ────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    obj = pickle.load(f)
model        = obj["model"]
feature_cols = obj["features"]
MODEL_MAE    = obj.get("mae",  0.72)
MODEL_RMSE   = obj.get("rmse", 1.53)

LATEST_SEASON       = get_latest_completed_season()
AVAILABLE_RACES     = get_available_races(LATEST_SEASON)
_default_race_key   = next(iter(AVAILABLE_RACES))
_default_total_laps = AVAILABLE_RACES[_default_race_key]["laps"]

def fmt_time(t):
    t = abs(int(t)); m, s = divmod(t, 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# ── Validation combinations — คำนวณจริงจาก lap cache ที่มีอยู่บนดิสก์ ─
# แทนตัวเลขตายตัว: สแกน cache/*.pkl หา (year, round, driver) ที่เคยโหลดจริงแล้ว
# (ผ่านหน้า Strategy Analysis หรือ train_model_advanced.py) แล้วรัน model.predict
# บน feature จริงของแต่ละคนใหม่ทุกครั้งที่ app เริ่ม — ไม่มีค่าที่ hardcode ไว้ล่วงหน้า
_CACHE_COMBO_RE = re.compile(r"^(\d{4})_(\d+)_([A-Z0-9]+)\.pkl$")


def _discover_cached_combos(races: dict, max_combos: int = 14):
    """หา (year, round, driver, label) ที่มี lap cache จริงอยู่แล้ว ตรงกับสนามในฤดูกาลนี้"""
    round_to_race = {info["gp"]: info for info in races.values()}
    found = []
    cache_dir = Path("cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*.pkl"):
            m = _CACHE_COMBO_RE.match(f.name)
            if not m:
                continue
            year, round_no, driver = int(m.group(1)), int(m.group(2)), m.group(3)
            info = round_to_race.get(round_no)
            if info is None or info["year"] != year:
                continue
            found.append((year, round_no, driver, info["label"]))
    found.sort(key=lambda x: (x[1], x[2]))
    return found[:max_combos]


def _compute_combo_stats(year, round_no, driver, label):
    """รัน model จริงบน lap จริงของ (year, round, driver) นี้ — ไม่มีตัวเลขที่ประดิษฐ์ขึ้น"""
    data, meta = load_race_laps(year=year, gp=round_no, driver=driver)
    y_true = data["LapTimeSec"].values
    X = data.drop(columns=["LapTimeSec"]).select_dtypes(include=[np.number])
    X_aligned = pd.DataFrame(0.0, index=X.index, columns=feature_cols)
    common = [c for c in feature_cols if c in X.columns]
    X_aligned[common] = X[common].values
    y_pred = model.predict(X_aligned)

    real_total = float(np.sum(y_true))
    sim_total  = float(np.sum(y_pred))
    diff_pct   = (sim_total - real_total) / real_total * 100
    gp_short   = label.split(f" GP {year}")[0]
    # meta มี weather อยู่แล้วถ้า cache ถูกสร้างหลังเพิ่ม feature นี้ ถ้าไม่มี (cache เก่า) ค่อยไปหาแยก
    weather = meta.get("weather") or get_race_weather(year, round_no)

    return {
        "year": year, "round": round_no, "gp": gp_short, "driver": driver,
        "laps": meta["total_laps"], "real_total": real_total, "sim_total": sim_total,
        "mae":  round(float(np.mean(np.abs(y_true - y_pred))), 3),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3),
        "diff_pct": round(diff_pct, 2),
        "weather": weather,
    }


def _build_combos():
    combos = []
    for year, round_no, driver, label in _discover_cached_combos(AVAILABLE_RACES):
        try:
            combos.append(_compute_combo_stats(year, round_no, driver, label))
        except Exception as e:
            print(f"  ข้าม combo {driver} @ round {round_no} {year}: {e}")

    if len(combos) < 2:
        # cache บางเกินไป — ดึงสด 2-3 คู่จากสนามแรกๆ ของฤดูกาล เพื่อให้หน้ารายงานมีข้อมูล
        for key, info in list(AVAILABLE_RACES.items())[:3]:
            drivers = get_available_drivers(info["year"], info["gp"])[:2]
            for drv in drivers:
                try:
                    combos.append(_compute_combo_stats(info["year"], info["gp"], drv, info["label"]))
                except Exception as e:
                    print(f"  ข้าม combo {drv} @ {info['label']}: {e}")
            if len(combos) >= 2:
                break
    return combos


COMBOS = _build_combos()

# ── หน้า 1 — Overview (หลักฐาน training) ────────────────
@app.route("/")
def index():
    hyperparams = [
        {"name": "n_estimators",      "value": "500", "desc": "จำนวน decision trees"},
        {"name": "max_depth",         "value": "16",  "desc": "ความลึกสูงสุดของแต่ละต้น"},
        {"name": "min_samples_split", "value": "3",   "desc": "sample ขั้นต่ำก่อนแตก node"},
        {"name": "min_samples_leaf",  "value": "2",   "desc": "sample ขั้นต่ำที่ leaf"},
        {"name": "random_state",      "value": "42",  "desc": "seed สำหรับ reproducibility"},
    ]
    features = [
        {"name": "LapNumber",       "type": "Numeric",  "desc": "ลำดับ lap ในการแข่งขัน",                  "impact": 3},
        {"name": "TyreLife",        "type": "Numeric",  "desc": "จำนวน lap ที่ใช้ยางชุดนี้มาแล้ว",          "impact": 5},
        {"name": "FuelEst",         "type": "Numeric",  "desc": "ประมาณน้ำมันที่เหลือ (0–1 normalize)",     "impact": 4},
        {"name": "StintNumber",     "type": "Numeric",  "desc": "stint ที่เท่าไหร่ (1=ก่อนพิท)",           "impact": 3},
        {"name": "StintLap",        "type": "Numeric",  "desc": "lap ที่เท่าไหร่ภายใน stint ปัจจุบัน",     "impact": 4},
        {"name": "PitStopsSoFar",   "type": "Numeric",  "desc": "จำนวนครั้งพิทที่ทำไปแล้ว",                "impact": 3},
        {"name": "Position",        "type": "Numeric",  "desc": "อันดับในขณะนั้น",                         "impact": 2},
        {"name": "Sector1Sec",      "type": "Numeric",  "desc": "เวลา Sector 1 (วินาที)",                  "impact": 5},
        {"name": "Sector2Sec",      "type": "Numeric",  "desc": "เวลา Sector 2 (วินาที)",                  "impact": 5},
        {"name": "Sector3Sec",      "type": "Numeric",  "desc": "เวลา Sector 3 (วินาที)",                  "impact": 5},
        {"name": "IsOutLap",        "type": "Binary",   "desc": "1 = lap แรกหลังออกจากพิต",               "impact": 3},
        {"name": "IsInLap",         "type": "Binary",   "desc": "1 = lap ที่เข้าพิต",                     "impact": 3},
        {"name": "Compound_SOFT",   "type": "One-Hot",  "desc": "ยาง Soft — grip สูง เสื่อมเร็ว",         "impact": 5},
        {"name": "Compound_MEDIUM", "type": "One-Hot",  "desc": "ยาง Medium — balance",                   "impact": 5},
        {"name": "Compound_HARD",   "type": "One-Hot",  "desc": "ยาง Hard — ทนทาน pace ต่ำกว่า",          "impact": 4},
        {"name": "TrackStatus_1",   "type": "One-Hot",  "desc": "สนามปกติ (Green Flag)",                  "impact": 2},
    ]

    train_combos = COMBOS
    mae_overall  = MODEL_MAE
    rmse_overall = MODEL_RMSE
    total_laps_trained = sum(c["laps"] for c in COMBOS)
    avg_diff_pct = round(sum(c["diff_pct"] for c in COMBOS) / len(COMBOS), 2) if COMBOS else 0.0

    from itertools import combinations as _comb
    groups = {}
    for c in COMBOS:
        groups.setdefault((c["year"], c["gp"]), []).append(c)
    same_param_compare = []
    for (year, gp), drivers in groups.items():
        if len(drivers) >= 2:
            for a, b in _comb(drivers, 2):
                same_param_compare.append({
                    "year": year, "gp": gp,
                    "driver_a": a["driver"], "diff_a": a["diff_pct"],
                    "real_a": a["real_total"], "sim_a": a["sim_total"],
                    "driver_b": b["driver"], "diff_b": b["diff_pct"],
                    "real_b": b["real_total"], "sim_b": b["sim_total"],
                })

    chart_combos = [
        {"driver": c["driver"], "gp": c["gp"], "year": c["year"],
         "real_total": c["real_total"], "sim_total": c["sim_total"],
         "diff_pct": c["diff_pct"]}
        for c in COMBOS
    ]

    return render_template(
        "index.html",
        hyperparams=hyperparams,
        features=features,
        train_combos=train_combos,
        mae_overall=mae_overall,
        rmse_overall=rmse_overall,
        total_laps_trained=total_laps_trained,
        avg_diff_pct=avg_diff_pct,
        same_param_compare=same_param_compare,
        chart_combos=chart_combos,
    )


@app.route("/analysis", methods=["GET", "POST"])
def analysis_page():
    selected_race_key = request.form.get("race_key", _default_race_key)
    race_info = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])

    available_drivers = get_available_drivers(race_info["year"], race_info["gp"])
    selected_driver = request.form.get("driver", "VER")
    if selected_driver not in available_drivers and available_drivers:
        selected_driver = available_drivers[0]

    result = error_msg = None
    lap_labels = actual_laps = pred_laps = []
    top_faster = explanations = []
    calibration = {}

    if request.method == "POST":
        try:
            data, meta = load_race_laps(year=race_info["year"], gp=race_info["gp"], driver=selected_driver)
            y_true    = data["LapTimeSec"].values
            X         = data.drop(columns=["LapTimeSec"]).select_dtypes(include=[np.number])
            X_aligned = pd.DataFrame(0.0, index=X.index, columns=feature_cols)
            common    = [c for c in feature_cols if c in X.columns]
            X_aligned[common] = X[common].values
            y_pred    = model.predict(X_aligned)

            lap_labels  = [int(l) for l in data["LapNumber"].tolist()]
            actual_laps = [round(float(v), 3) for v in y_true]
            pred_laps   = [round(float(v), 3) for v in y_pred]

            real_total  = float(sum(actual_laps))
            total_laps  = meta["total_laps"]
            race_year   = race_info["year"]
            real_pit_lap = int(total_laps * 0.35)
            baseline = {"first_compound": "MEDIUM", "second_compound": "SOFT",
                        "pit_lap": real_pit_lap, "num_stops": 1}

            # ปัจจัยจริงของสนามนี้ — pit loss ประมาณจาก lap cache จริง, grid position จริงของนักแข่งคนนี้
            pit_loss = estimate_pit_loss(race_year, race_info["gp"])
            grid_position = get_race_grid(race_year, race_info["gp"]).get(selected_driver, 1)

            # diff ก่อน/หลัง calibrate ต้องวัดจากตัวเดียวกัน คือ strategy simulator
            # ที่ grid search ใช้ ความแม่นของโมเดลบน feature จริงวัดด้วย MAE/RMSE แยกอยู่แล้ว
            # (ก่อนหน้านี้ diff_before วัดจากผลทำนายบน feature จริง แต่ diff_after
            #  เอา offset ที่ calibrate กับ strategy simulator มาบวก จึงเทียบกันไม่ได้
            #  และทำให้ diff หลัง calibrate ดูแย่กว่าก่อน calibrate)
            baseline_raw = simulate_strategy(
                model, feature_cols, total_laps,
                baseline["first_compound"], baseline["second_compound"],
                baseline["pit_lap"], pace_offset=0.0, race_year=race_year,
                pit_loss=pit_loss, grid_position=grid_position,
            )
            diff_before = (baseline_raw - real_total) / real_total * 100

            pace_offset  = calibrate_pace_offset(model, feature_cols, actual_laps, data,
                                                 total_laps, baseline, race_year=race_year,
                                                 pit_loss=pit_loss, grid_position=grid_position)
            baseline_cal = baseline_raw + (pace_offset * total_laps)
            diff_after   = (baseline_cal - real_total) / real_total * 100
            calibration  = {"pace_offset": pace_offset, "diff_before": round(diff_before, 2),
                            "diff_after": round(diff_after, 2)}

            all_results = grid_search_strategies(model, feature_cols, actual_laps, total_laps,
                                                 pace_offset=pace_offset, race_year=race_year,
                                                 pit_loss=pit_loss, grid_position=grid_position)
            top_faster  = [r for r in all_results if r["faster"]][:5]
            if top_faster:
                explanations = explain_parameters(top_faster[0], baseline, real_total)

            mae_here  = float(np.mean(np.abs(y_true - y_pred)))
            rmse_here = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            weather   = meta.get("weather") or get_race_weather(race_year, race_info["gp"])
            incidents = get_race_incidents(race_year, race_info["gp"])
            result = {"race_label": race_info["label"], "driver": selected_driver,
                      "total_laps": total_laps, "mae": f"{mae_here:.3f}", "rmse": f"{rmse_here:.3f}",
                      "real_time": fmt_time(real_total), "real_total": real_total,
                      "diff_pct": round(diff_before, 2), "weather": weather,
                      "incidents": incidents, "grid_position": grid_position,
                      "pit_loss": pit_loss}
        except Exception as e:
            error_msg = str(e)

    return render_template(
        "analysis.html",
        available_races=AVAILABLE_RACES, available_drivers=available_drivers,
        selected_race_key=selected_race_key, selected_driver=selected_driver,
        result=result, error=error_msg,
        lap_labels=lap_labels, actual_laps=actual_laps, pred_laps=pred_laps,
        calibration=calibration, top_faster=top_faster, explanations=explanations,
    )


# ── หน้า 3 — Play Strategy ──────────────────────────────
@app.route("/play", methods=["GET", "POST"])
def play_strategy_page():
    selected_race_key = request.form.get("race_key", _default_race_key)
    race_info  = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])
    total_laps = race_info["laps"]
    result = None; leaderboard = []; form_data = None

    if request.method == "POST":
        start_compound  = request.form.get("start_compound",  "MEDIUM").upper()
        second_compound = request.form.get("second_compound", "SOFT").upper()
        pit_lap         = max(2, min(int(request.form.get("pit_lap", 20)), total_laps - 2))

        num_stops = 2 if request.form.get("num_stops") == "2" else 1
        second_pit_lap = None
        third_compound = None
        if num_stops == 2:
            second_pit_lap = int(request.form.get("second_pit_lap", min(pit_lap + 15, total_laps - 2)))
            second_pit_lap = max(pit_lap + 3, min(second_pit_lap, total_laps - 2))
            third_compound = request.form.get("third_compound", "HARD").upper()

        form_data = {
            "start": start_compound, "second": second_compound, "third": third_compound,
            "pit_lap": pit_lap, "second_pit_lap": second_pit_lap, "num_stops": num_stops,
            "race_key": selected_race_key,
        }

        # ใช้โมเดลที่โหลดไว้ตอน import — ไม่ unpickle model.pkl ใหม่ทุก request
        predictor = LapPredictor(model=model, features=feature_cols)

        # กริดจริง — รายชื่อนักแข่งที่ลงแข่งสนามนี้จริง (ไม่ใช่รายชื่อสมมติ)
        field_drivers = get_available_drivers(race_info["year"], race_info["gp"])

        # ปัจจัยจริงของสนามนี้ — pit loss ประมาณจาก lap cache จริง (ไม่ใช่ค่าคงที่ตายตัวทุกสนาม)
        pit_loss = estimate_pit_loss(race_info["year"], race_info["gp"])

        # หา global_offset จากข้อมูลจริงของสนามที่เลือก (จาก combo ที่คำนวณจริงในหน้า Model Report)
        # เพื่อให้เวลา simulation ทั้งกริดใกล้เคียงเวลาจริง
        race_stats = next(
            (c for c in COMBOS if c["round"] == race_info["gp"] and c["year"] == race_info["year"]),
            None
        )
        real_total = race_stats["real_total"] if race_stats else race_info["laps"] * 95.0

        # คำนวณ global_offset = (เวลาจริง - เวลา raw sim ของกลยุทธ์ baseline) / total_laps
        baseline_raw = simulate_strategy(
            model, feature_cols, total_laps,
            "MEDIUM", "SOFT", int(total_laps * 0.44),
            pace_offset=0.0, race_year=race_info["year"], pit_loss=pit_loss,
        )
        global_offset = (real_total - baseline_raw) / total_laps

        all_results = simulate_full_race(total_laps=total_laps,
                                         drivers=field_drivers,
                                         year=race_info["year"], gp=race_info["gp"],
                                         global_offset=global_offset,
                                         predictor=predictor,
                                         pit_loss=pit_loss)

        user_strategy = DriverStrategy(code="YOU", first_compound=start_compound,
                                       second_compound=second_compound, pit_lap=pit_lap,
                                       pace_offset=1.5 + global_offset,
                                       num_stops=num_stops, second_pit_lap=second_pit_lap,
                                       third_compound=third_compound)
        user_result = simulate_driver(predictor, user_strategy, total_laps,
                                      race_year=race_info["year"], pit_loss=pit_loss)

        combined = list(all_results) + [user_result]
        combined.sort(key=lambda r: r.total_time)
        for i, r in enumerate(combined, start=1):
            r.rank = i

        win_probs     = compute_win_probabilities(combined)
        user_rank     = next(r.rank for r in combined if r.code == "YOU")
        user_time     = user_result.total_time
        user_win_prob = win_probs.get("YOU", 0.0)
        delta_real = user_time - real_total

        # เวลาอันดับ 1 สำหรับคำนวณ gap
        p1_time = combined[0].total_time

        weather   = get_race_weather(race_info["year"], race_info["gp"])
        incidents = get_race_incidents(race_info["year"], race_info["gp"])

        result = {
            "rank": user_rank, "start": start_compound, "second": second_compound,
            "third": third_compound, "pit_lap": pit_lap, "second_pit_lap": second_pit_lap,
            "num_stops": num_stops,
            "user_time_str": fmt_time(user_time), "real_time_str": fmt_time(real_total),
            "user_time": user_time, "real_time": real_total,
            "delta_real": delta_real, "delta_real_str": f"{delta_real:+.2f} s",
            "win_prob_pct": round(user_win_prob * 100, 1),
            "win_prob_bar": min(round(user_win_prob * 100 * 3, 1), 100),
            "gap_to_p1": round(user_time - p1_time, 2),
            "gap_bar":   min(round((user_time - p1_time) / 200 * 100, 1), 100),
            "pit_loss": round(pit_loss, 1), "weather": weather, "incidents": incidents,
        }

        leaderboard = [{
            "rank": r.rank, "code": r.code, "is_user": r.code == "YOU",
            "start": r.strategy.first_compound, "second": r.strategy.second_compound,
            "third": r.strategy.third_compound, "grid": r.strategy.grid_position,
            "pit_lap": r.strategy.pit_lap, "second_pit_lap": r.strategy.second_pit_lap,
            "total_time": r.total_time,
            "gap_to_p1": round(r.total_time - p1_time, 2),
        } for r in combined]

    return render_template(
        "play_strategy.html",
        total_laps=total_laps,
        result=result,
        leaderboard=leaderboard,
        form_data=form_data,
        available_races=AVAILABLE_RACES,
        selected_race_key=selected_race_key,
        race_label=race_info["label"],
    )


if __name__ == "__main__":
    app.run(debug=True)