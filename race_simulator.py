# race_simulator.py
"""
Full race simulation ด้วยโมเดล RandomForest — ใช้กริดนักแข่งจริงของสนามที่เลือก
(จาก data_pipeline.get_available_drivers) พร้อม pace offset จริงจาก lap cache
ถ้ามี ไม่งั้น fallback เป็น synthetic offset ที่ deterministic ต่อคน
"""

import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from data_pipeline import try_load_cached_race_laps, get_race_grid

PIT_LOSS = 22.0  # ค่า fallback ถ้ายังไม่มี lap cache ให้ประมาณ pit loss จริงของสนามนั้น


@dataclass
class DriverStrategy:
    code: str
    first_compound: str
    second_compound: str
    pit_lap: int
    pace_offset: float
    num_stops: int = 1
    second_pit_lap: int = None
    third_compound: str = None
    grid_position: int = 1  # ตำแหน่งกริดสตาร์ทจริง (ถ้ามี) — ใช้เป็น Position feature


@dataclass
class DriverResult:
    code: str
    total_time: float
    laps: List[float]
    strategy: DriverStrategy
    rank: int = 0


class LapPredictor:
    def __init__(self, model_path: str = "model.pkl", model=None, features: List[str] = None):
        """
        ส่ง model/features ที่โหลดไว้แล้วเข้ามาได้ เพื่อไม่ต้อง unpickle
        model.pkl (8.3 MB, ~1.3 วินาที) ซ้ำในทุก request
        """
        if model is not None and features is not None:
            self.model = model
            self.features: List[str] = features
        else:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            self.model = obj["model"]
            self.features: List[str] = obj["features"]

    def _to_row(self, feature_dict: Dict) -> Dict:
        row = {f: 0.0 for f in self.features}
        for k, v in feature_dict.items():
            if k in row:
                row[k] = v
        return row

    def predict_lap(self, feature_dict: Dict) -> float:
        return float(self.predict_laps([feature_dict])[0])

    def predict_laps(self, feature_dicts: List[Dict]) -> np.ndarray:
        """
        predict หลาย lap ในครั้งเดียว — การเรียก predict() ทีละแถวมี overhead
        ~48 ms ต่อครั้ง ขณะที่ส่ง 1,140 แถวไปพร้อมกันใช้เวลา ~68 ms
        """
        X = pd.DataFrame([self._to_row(f) for f in feature_dicts], columns=self.features)
        return self.model.predict(X)


def _compound_one_hot(comp: str) -> Dict[str, int]:
    comp = comp.upper()
    return {
        "Compound_SOFT":   1 if comp == "SOFT"   else 0,
        "Compound_MEDIUM": 1 if comp == "MEDIUM" else 0,
        "Compound_HARD":   1 if comp == "HARD"   else 0,
    }


def build_driver_laps(
    strategy: DriverStrategy,
    total_laps: int,
    race_year: int = None,
    pit_loss: float = PIT_LOSS,
) -> tuple:
    """
    สร้าง feature ของทุก lap ให้นักแข่งหนึ่งคน รองรับทั้ง 1-stop และ 2-stop
    คืนค่า (feature_dicts, pit_penalties) เพื่อ predict แบบ batch
    """
    feats = []
    penalties = []
    stint_number = 1
    pit_stops = 0
    current_compound = strategy.first_compound

    for lap in range(1, total_laps + 1):
        if lap == strategy.pit_lap:
            pit_stops += 1
            stint_number = 2
            current_compound = strategy.second_compound
            tyre_life = 1
            is_in_lap = 1
            pit_penalty = pit_loss
        elif (strategy.num_stops == 2 and strategy.second_pit_lap
              and lap == strategy.second_pit_lap):
            pit_stops += 1
            stint_number = 3
            current_compound = strategy.third_compound or "HARD"
            tyre_life = 1
            is_in_lap = 1
            pit_penalty = pit_loss
        else:
            is_in_lap = 0
            pit_penalty = 0.0
            if stint_number == 1:
                tyre_life = lap
            elif stint_number == 2:
                tyre_life = lap - strategy.pit_lap + 1
            else:
                tyre_life = lap - strategy.second_pit_lap + 1

        feat = {
            "LapNumber":      lap,
            "TyreLife":       tyre_life,
            "FuelEst":        (total_laps - lap) / total_laps,
            "StintNumber":    stint_number,
            "StintLap":       tyre_life,
            "PitStopsSoFar":  pit_stops,
            "IsInLap":        is_in_lap,
            "IsOutLap":       0,
            "TrackStatus_1":  1,
            "Position":       strategy.grid_position,
        }
        if race_year is not None:
            feat["RaceYear"] = race_year
        feat.update(_compound_one_hot(current_compound))

        feats.append(feat)
        penalties.append(pit_penalty)

    return feats, penalties


def _assemble_result(strategy, base_times, penalties, total_laps) -> DriverResult:
    noise = np.random.normal(0, 0.12, total_laps)
    laps = (np.asarray(base_times) + strategy.pace_offset + noise
            + np.asarray(penalties)).tolist()
    return DriverResult(
        code=strategy.code,
        total_time=float(np.sum(laps)),
        laps=laps,
        strategy=strategy,
    )


def simulate_driver(
    predictor: LapPredictor,
    strategy: DriverStrategy,
    total_laps: int,
    race_year: int = None,
    pit_loss: float = PIT_LOSS,
) -> DriverResult:
    feats, penalties = build_driver_laps(strategy, total_laps, race_year, pit_loss)
    base_times = predictor.predict_laps(feats)
    return _assemble_result(strategy, base_times, penalties, total_laps)


def _real_driver_offset(predictor: LapPredictor, year: int, gp, driver: str):
    """
    ถ้ามี lap cache จริงของนักแข่งคนนี้ในสนามนี้อยู่แล้วบนดิสก์ (ไม่ fetch ใหม่)
    คำนวณ pace offset จริง = ค่าเฉลี่ย(เวลาจริง − เวลาที่โมเดลทำนาย) จาก lap จริงของเขา
    คืนค่า None ถ้าไม่มี cache — ให้ผู้เรียกใช้ synthetic offset แทน
    """
    cached = try_load_cached_race_laps(year, gp, driver)
    if cached is None:
        return None
    data, _meta = cached
    if "LapTimeSec" not in data.columns or data.empty:
        return None

    y_true = data["LapTimeSec"].values
    X = data.drop(columns=["LapTimeSec"]).select_dtypes(include=[np.number])
    rows = [{f: float(row[f]) for f in predictor.features if f in X.columns}
            for _, row in X.iterrows()]
    y_pred = predictor.predict_laps(rows)
    return float(np.mean(y_true - y_pred))


def _default_strategy_for(index: int, total_laps: int) -> tuple:
    """
    กลยุทธ์เริ่มต้นของรถที่คอมพิวเตอร์คุม (ไม่ใช่ผู้เล่น) — วนรูปแบบ compound
    ทั่วไปและกระจาย pit lap เล็กน้อยตามตำแหน่งกริด เพื่อไม่ให้ทั้งสนามเข้าพิทพร้อมกัน
    """
    compounds = [("MEDIUM", "SOFT"), ("SOFT", "MEDIUM"), ("SOFT", "HARD")]
    c1, c2 = compounds[index % len(compounds)]
    pit_lap = max(3, min(total_laps - 3, int(total_laps * (0.30 + 0.015 * (index % 7)))))
    return c1, c2, pit_lap


def build_field_strategies(
    predictor: LapPredictor,
    year: int,
    gp,
    drivers: List[str],
    total_laps: int,
    global_offset: float = 0.0,
) -> List[DriverStrategy]:
    """
    สร้างกลยุทธ์ของทุกคันในกริดจริงของสนามนี้ (รายชื่อนักแข่งจริงจาก data_pipeline)
    ใช้ pace offset จริงจาก lap cache ถ้ามี ไม่งั้น fallback เป็น synthetic offset
    ที่ deterministic ต่อคน (ไม่ใช่ค่าคงที่ตายตัวที่ผูกกับสนามใดสนามหนึ่ง)
    """
    grid = get_race_grid(year, gp)

    strategies = []
    for i, code in enumerate(drivers):
        grid_position = grid.get(code, i + 1)
        real_offset = _real_driver_offset(predictor, year, gp, code)
        if real_offset is not None:
            pace = real_offset + global_offset
        else:
            # ไม่มี lap cache จริง — ใช้ synthetic offset ที่กระจายตามตำแหน่งกริดจริง
            # (คนออกตัวหลังมักจบช้ากว่า) แทนลำดับ enumerate ที่ไม่มีความหมาย
            seed = abs(hash((year, gp, code))) % (2 ** 32)
            rng = np.random.default_rng(seed)
            pace = 1.0 + (grid_position - 1) * 0.09 + float(rng.uniform(-0.25, 0.25)) + global_offset

        c1, c2, pit = _default_strategy_for(i, total_laps)
        strategies.append(DriverStrategy(
            code=code, first_compound=c1, second_compound=c2,
            pit_lap=pit, pace_offset=pace, grid_position=grid_position,
        ))
    return strategies


def simulate_full_race(
    total_laps: int,
    drivers: List[str],
    year: int,
    gp,
    global_offset: float = 0.0,
    predictor: LapPredictor = None,
    model_path: str = "model.pkl",
    pit_loss: float = PIT_LOSS,
) -> List[DriverResult]:
    """
    จำลองทั้งกริดจริงของสนามที่เลือก — ทุกนักแข่งทุก lap ถูกรวมเป็น predict ครั้งเดียว
    """
    if predictor is None:
        predictor = LapPredictor(model_path)

    strategies = build_field_strategies(predictor, year, gp, drivers, total_laps, global_offset)

    all_feats = []
    all_penalties = []
    for st in strategies:
        feats, penalties = build_driver_laps(st, total_laps, year, pit_loss)
        all_feats.extend(feats)
        all_penalties.append(penalties)

    base_times = predictor.predict_laps(all_feats)

    results = []
    for i, st in enumerate(strategies):
        seg = base_times[i * total_laps:(i + 1) * total_laps]
        results.append(_assemble_result(st, seg, all_penalties[i], total_laps))

    results.sort(key=lambda r: r.total_time)
    for i, r in enumerate(results, start=1):
        r.rank = i

    return results


def compute_lap_positions(results: List[DriverResult], total_laps: int) -> Dict[str, List[int]]:
    """
    อันดับของทุกคนใน "แต่ละ" รอบ (ไม่ใช่แค่อันดับสุดท้าย) — คำนวณจากเวลาสะสม
    (cumulative time) ถึง lap นั้นๆ ใช้ทำกราฟ position-by-lap ที่เห็นการไล่แซง
    ขึ้น-ลงระหว่างแข่ง แทนที่จะเห็นแค่ผลลัพธ์ปลายทาง
    """
    cum = np.array([np.cumsum(r.laps[:total_laps]) for r in results])  # (n_drivers, total_laps)
    ranks = np.argsort(np.argsort(cum, axis=0), axis=0) + 1
    return {r.code: ranks[i].tolist() for i, r in enumerate(results)}


def compute_win_probabilities(results: List[DriverResult]) -> Dict[str, float]:
    times = np.array([r.total_time for r in results])
    scores = -times
    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()
    return {r.code: float(p) for r, p in zip(results, probs)}