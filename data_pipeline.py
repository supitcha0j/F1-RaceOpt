# data_pipeline.py
"""
ดึงข้อมูล Lap จาก FastF1 และเตรียม Feature สำหรับใช้เทรน AI และ Simulation
รองรับหลาย Race / หลายนักแข่ง
"""

import os
import pickle
import datetime
import fastf1
import pandas as pd
import numpy as np


def get_latest_completed_season() -> int:
    """
    หาปี (ฤดูกาล) ล่าสุดที่แข่งจบครบทุกสนามแล้ว
    เทียบวันที่ของสนามสุดท้ายในตารางกับวันนี้ ถ้ายังไม่จบ ใช้ปีก่อนหน้าแทน
    """
    now = datetime.datetime.now()
    try:
        sched = fastf1.get_event_schedule(now.year, include_testing=False)
        if not sched.empty and sched["EventDate"].max().to_pydatetime() < now:
            return now.year
    except Exception:
        pass
    return now.year - 1


def _season_cache_path(year: int) -> str:
    return os.path.join("cache", f"season_{year}.pkl")


def _build_season_schedule(year: int):
    """
    ดึงตาราง GP ทุกสนามของ `year` จาก FastF1 พร้อมจำนวน lap และ
    รายชื่อนักแข่งจริงที่ลงแข่งแต่ละสนาม (โหลดเฉพาะ results ไม่โหลด telemetry
    เพื่อความเร็ว) แล้ว cache ผลลงดิสก์เพื่อไม่ต้องดึงซ้ำทุกครั้งที่ start app
    """
    cache_dir = os.path.join("cache", f"schedule_{year}")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    races = {}
    drivers = {}
    for _, event in schedule.iterrows():
        round_no = int(event["RoundNumber"])
        if round_no <= 0:
            continue  # testing / pre-season entries
        try:
            session = fastf1.get_session(year, round_no, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            if session.results is None or session.results.empty:
                continue
            total_laps = int(session.results["Laps"].max())
            grid = sorted(session.results["Abbreviation"].dropna().unique().tolist())
        except Exception as e:
            print(f"  ข้าม round {round_no} {event['EventName']} {year}: {e}")
            continue

        short_name = event["EventName"].replace(" Grand Prix", "")
        race_key = f"{year}_{short_name}".replace(" ", "_")
        races[race_key] = {
            "year": year, "gp": round_no,
            "label": f"{short_name} GP {year}", "laps": total_laps,
        }
        drivers[race_key] = grid

    return races, drivers


def _load_season_schedule(year: int):
    cache_path = _season_cache_path(year)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    races, drivers = _build_season_schedule(year)
    if not races:
        raise ValueError(f"ไม่พบข้อมูลสนามแข่งของฤดูกาล {year}")

    os.makedirs("cache", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((races, drivers), f)
    return races, drivers


def get_available_races(year: int = None) -> dict:
    """ทุกสนามของฤดูกาลล่าสุดที่แข่งจบแล้ว (หรือปีที่ระบุ) พร้อม lap count จริง"""
    year = year or get_latest_completed_season()
    races, _ = _load_season_schedule(year)
    return races


def get_available_drivers(year: int, gp) -> list:
    """รายชื่อนักแข่งจริงที่ลงแข่งสนามนี้ (round `gp`) ของ `year`"""
    races, drivers = _load_season_schedule(year)
    race_key = next((k for k, v in races.items() if v["gp"] == gp), None)
    return drivers.get(race_key, [])


def try_load_cached_race_laps(year, gp, driver):
    """
    โหลด lap cache จากดิสก์เท่านั้น — ไม่ยิง fetch ไป FastF1 ถ้ายังไม่มี cache
    คืนค่า None ถ้ายังไม่เคยโหลดคู่ (year, gp, driver) นี้มาก่อน
    ใช้ตอนต้องการข้อมูลจริงแบบเร็ว (เช่น จำลองทั้งกริด) โดยไม่ยอมให้ request ช้าเพราะรอ network
    """
    cache_key = f"{year}_{gp}_{driver}".replace(" ", "_")
    cache_path = os.path.join("cache", f"{cache_key}.pkl")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_race_weather(year, gp):
    """
    สรุปสภาพอากาศจริงของสนามนี้จาก FastF1 weather_data (เฉลี่ยทั้ง session)
    คืนค่า None ถ้าไม่มีข้อมูลสภาพอากาศ (บาง session เก่าไม่มี)

    สภาพอากาศเป็นข้อมูลระดับสนาม ไม่ใช่ต่อนักแข่ง จึง cache แยกต่างหาก
    (ไม่ผูกกับ driver เหมือน load_race_laps) เพื่อไม่ต้องคำนวณซ้ำต่อนักแข่ง
    """
    cache_key = f"{year}_{gp}_weather".replace(" ", "_")
    cache_path = os.path.join("cache", f"{cache_key}.pkl")
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache_dir = os.path.join("cache", f"{year}_{str(gp).replace(' ', '_')}")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, "R")
    session.load(laps=False, telemetry=False, messages=False)

    weather = None
    wd = session.weather_data
    if wd is not None and not wd.empty:
        weather = {
            "air_temp":   round(float(wd["AirTemp"].mean()), 1),
            "track_temp": round(float(wd["TrackTemp"].mean()), 1),
            "humidity":   round(float(wd["Humidity"].mean()), 1),
            "wet":        bool(wd["Rainfall"].any()),
        }

    with open(cache_path, "wb") as f:
        pickle.dump(weather, f)
    return weather


def get_race_grid(year, gp):
    """
    ตำแหน่งกริดสตาร์ทจริงของทุกนักแข่งในสนามนี้ {driver_code: grid_position}
    ข้อมูลระดับสนาม cache แยกจาก load_race_laps เหมือน get_race_weather
    """
    cache_key = f"{year}_{gp}_grid".replace(" ", "_")
    cache_path = os.path.join("cache", f"{cache_key}.pkl")
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache_dir = os.path.join("cache", f"{year}_{str(gp).replace(' ', '_')}")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    grid = {}
    res = session.results
    if res is not None and not res.empty:
        for _, row in res.iterrows():
            code = row.get("Abbreviation")
            pos = row.get("GridPosition")
            if code and pos == pos:  # ไม่ใช่ NaN
                grid[code] = int(pos)

    with open(cache_path, "wb") as f:
        pickle.dump(grid, f)
    return grid


def get_race_incidents(year, gp):
    """
    เหตุการณ์ระหว่างการแข่งของสนามนี้จริง (Safety Car / VSC / Red flag / Yellow)
    จาก FastF1 track_status_data — คืนค่า None ถ้าไม่มีข้อมูล
    """
    cache_key = f"{year}_{gp}_incidents".replace(" ", "_")
    cache_path = os.path.join("cache", f"{cache_key}.pkl")
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache_dir = os.path.join("cache", f"{year}_{str(gp).replace(' ', '_')}")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, "R")
    session.load(telemetry=False, weather=False)  # track_status ต้องมี laps=True ถึงจะโหลด

    incidents = None
    ts = session.track_status
    if ts is not None and not ts.empty:
        statuses = set(ts["Status"].astype(str))
        incidents = {
            "safety_car": "4" in statuses,
            "vsc":        "6" in statuses or "7" in statuses,
            "red_flag":   "5" in statuses,
            "yellow":     "2" in statuses,
        }

    with open(cache_path, "wb") as f:
        pickle.dump(incidents, f)
    return incidents


def estimate_pit_loss(year, gp, fallback=22.0):
    """
    ประมาณเวลาที่เสียตอนเข้าพิทจริงของสนามนี้ จาก lap cache ที่มีอยู่แล้วบนดิสก์
    (in-lap time เทียบกับ lap ปกติของนักแข่งคนเดียวกัน) — ไม่ fetch ใหม่
    คืนค่า fallback ถ้ายังไม่มี lap cache ของสนามนี้เลย (pit loss ต่างกันจริงตามสนาม
    แต่ยังไม่มีข้อมูลจริงให้ประมาณ)
    """
    import re
    pattern = re.compile(rf"^{year}_{gp}_([A-Z0-9]+)\.pkl$")
    cache_dir = "cache"
    if not os.path.isdir(cache_dir):
        return fallback

    for fname in os.listdir(cache_dir):
        if not pattern.match(fname):
            continue
        with open(os.path.join(cache_dir, fname), "rb") as f:
            data, _meta = pickle.load(f)
        if "IsInLap" not in data.columns or "LapTimeSec" not in data.columns:
            continue
        normal = data.loc[data["IsInLap"] == 0, "LapTimeSec"]
        in_lap = data.loc[data["IsInLap"] == 1, "LapTimeSec"]
        if len(normal) < 3 or in_lap.empty:
            continue
        delta = float(in_lap.median() - normal.median())
        if 5 <= delta <= 45:  # ช่วงที่สมเหตุสมผลของ pit loss จริง
            return round(delta, 1)

    return fallback


def load_race_laps(year=2023, gp="Bahrain", driver="VER"):
    """
    โหลดข้อมูลจาก FastF1 พร้อม disk cache
    ครั้งแรก: ดึงจาก FastF1 แล้วบันทึก cache
    ครั้งต่อไป: โหลด cache ทันที (เร็วมาก)
    """
    # ── Disk cache ──────────────────────────────────
    cache_key  = f"{year}_{gp}_{driver}".replace(" ", "_")
    cache_path = os.path.join("cache", f"{cache_key}.pkl")
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)        # ← โหลด cache ทันที

    # ── ดึงจาก FastF1 (ครั้งแรก) ──────────────────
    cache_dir = os.path.join("cache", f"{year}_{str(gp).replace(' ', '_')}")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, "R")
    session.load()

    laps = session.laps.pick_driver(driver).reset_index(drop=True)

    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["Sector1Sec"] = laps["Sector1Time"].dt.total_seconds()
    laps["Sector2Sec"] = laps["Sector2Time"].dt.total_seconds()
    laps["Sector3Sec"] = laps["Sector3Time"].dt.total_seconds()

    laps = laps[~laps["LapTimeSec"].isna() & ~laps["LapNumber"].isna()].copy()
    laps = laps.reset_index(drop=True)

    if laps.empty:
        raise ValueError(f"ไม่พบ lap ของ {driver} ใน {gp} {year}")

    total_laps = int(laps["LapNumber"].max())

    laps["FuelEst"] = (total_laps - laps["LapNumber"]) / total_laps

    if "Stint" in laps.columns:
        laps["StintNumber"] = laps["Stint"].ffill().fillna(1).astype(int)
    else:
        laps["StintNumber"] = 1

    laps["StintLap"]      = laps.groupby("StintNumber").cumcount() + 1
    laps["PitStopsSoFar"] = laps["StintNumber"] - 1
    laps["IsOutLap"]      = laps["PitOutTime"].notna().astype(int)
    laps["IsInLap"]       = laps["PitInTime"].notna().astype(int)

    if "TrackStatus" not in laps.columns:
        laps["TrackStatus"] = 1
    laps["TrackStatus"] = laps["TrackStatus"].fillna(1).astype(int)

    laps["TyreLife"] = laps["TyreLife"].fillna(0) if "TyreLife" in laps.columns else 0

    if "Position" in laps.columns:
        laps["Position"] = laps["Position"].ffill().fillna(1)
    else:
        laps["Position"] = 1

    base_cols = [
        "LapNumber", "LapTimeSec", "Compound", "TyreLife", "TrackStatus",
        "Position", "Sector1Sec", "Sector2Sec", "Sector3Sec",
        "FuelEst", "StintNumber", "StintLap", "PitStopsSoFar",
        "IsOutLap", "IsInLap",
    ]
    base_cols = [c for c in base_cols if c in laps.columns]
    data = laps[base_cols].copy()

    data = pd.get_dummies(
        data,
        columns=[col for col in ["Compound", "TrackStatus"] if col in data.columns],
        drop_first=False,
    )

    weather = None
    wd = session.weather_data
    if wd is not None and not wd.empty:
        weather = {
            "air_temp":   round(float(wd["AirTemp"].mean()), 1),
            "track_temp": round(float(wd["TrackTemp"].mean()), 1),
            "humidity":   round(float(wd["Humidity"].mean()), 1),
            "wet":        bool(wd["Rainfall"].any()),
        }
    # เขียน weather cache แยกไว้เลย (race-level) กันไม่ต้องโหลด session ซ้ำใน get_race_weather
    weather_cache_path = os.path.join("cache", f"{year}_{gp}_weather".replace(" ", "_") + ".pkl")
    if not os.path.exists(weather_cache_path):
        with open(weather_cache_path, "wb") as f:
            pickle.dump(weather, f)

    meta = {"year": year, "gp": gp, "driver": driver, "total_laps": total_laps, "weather": weather}

    # ── บันทึก cache ────────────────────────────────
    with open(cache_path, "wb") as f:
        pickle.dump((data, meta), f)     # ← indent ถูกต้องแล้ว

    return data, meta


def load_multi_race_laps(race_driver_list):
    all_data = []
    all_meta = []

    for year, gp, driver in race_driver_list:
        try:
            print(f"  โหลด {driver} @ {gp} {year}...")
            data, meta = load_race_laps(year, gp, driver)
            data["DriverCode"] = driver
            data["RaceYear"]   = year
            data["GP_Label"]   = gp
            all_data.append(data)
            all_meta.append(meta)
            print(f"    ✓ {len(data)} laps")
        except Exception as e:
            print(f"    ✗ ข้าม {driver} @ {gp} {year}: {e}")

    if not all_data:
        raise ValueError("ไม่สามารถโหลดข้อมูลได้เลย")

    combined = pd.concat(all_data, ignore_index=True)
    combined = pd.get_dummies(combined, columns=["DriverCode", "GP_Label"], drop_first=False)

    return combined, all_meta