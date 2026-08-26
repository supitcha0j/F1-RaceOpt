# strategy_optimizer.py
"""
Grid Search กลยุทธ์ที่ดีกว่าของจริง และอธิบายว่า parameter ไหนทำให้เร็วขึ้น
"""

import numpy as np
import pandas as pd
from itertools import product


PIT_LOSS = 22.0  # ค่า fallback ถ้ายังไม่มี lap cache ให้ประมาณ pit loss จริงของสนามนั้น


def _compound_one_hot(comp: str) -> dict:
    comp = comp.upper()
    return {
        "Compound_SOFT":   1 if comp == "SOFT"   else 0,
        "Compound_MEDIUM": 1 if comp == "MEDIUM" else 0,
        "Compound_HARD":   1 if comp == "HARD"   else 0,
    }


def build_strategy_laps(
    feature_cols: list,
    total_laps: int,
    first_compound: str,
    second_compound: str,
    pit_lap: int,
    num_stops: int = 1,
    second_pit_lap: int = None,
    third_compound: str = None,
    race_year: int = None,
    pit_loss: float = PIT_LOSS,
    grid_position: int = 1,
) -> tuple:
    """
    สร้าง feature ของทุก lap ในกลยุทธ์หนึ่งๆ เป็น list เดียว
    คืนค่า (rows, pit_penalties) เพื่อให้ผู้เรียกเอาไป predict แบบ batch ได้

    แยกออกมาจาก simulate_strategy เพื่อให้หลายกลยุทธ์รวม predict ครั้งเดียวได้
    การเรียก model.predict() ทีละแถวมี overhead ~48 ms ต่อครั้ง ขณะที่
    การส่ง 1,000+ แถวไปในครั้งเดียวใช้เวลาเกือบเท่ากัน
    """
    rows = []
    pit_penalties = []
    stint = 1
    pit_stops = 0
    compound = first_compound.upper()

    for lap in range(1, total_laps + 1):
        pit_penalty = 0.0

        # เช็ค pit stop
        if lap == pit_lap:
            pit_stops += 1
            stint = 2
            compound = second_compound.upper()
            tyre_life = 1
            pit_penalty = pit_loss
        elif num_stops == 2 and second_pit_lap and lap == second_pit_lap:
            pit_stops += 1
            stint = 3
            compound = (third_compound or "HARD").upper()
            tyre_life = 1
            pit_penalty = pit_loss
        else:
            if stint == 1:
                tyre_life = lap
            elif stint == 2:
                tyre_life = lap - pit_lap + 1
            else:
                tyre_life = lap - second_pit_lap + 1

        fuel_est = (total_laps - lap) / total_laps

        feat = {f: 0.0 for f in feature_cols}
        feat.update({
            "LapNumber":     lap,
            "TyreLife":      tyre_life,
            "FuelEst":       fuel_est,
            "StintNumber":   stint,
            "StintLap":      tyre_life,
            "PitStopsSoFar": pit_stops,
            "IsInLap":       1 if pit_penalty > 0 else 0,
            "IsOutLap":      0,
            "TrackStatus_1": 1,
            "Position":      grid_position,
        })
        if race_year is not None:
            feat["RaceYear"] = race_year
        feat.update(_compound_one_hot(compound))

        # เก็บเฉพาะ column ที่โมเดลรู้จัก
        rows.append({f: feat[f] for f in feature_cols})
        pit_penalties.append(pit_penalty)

    return rows, pit_penalties


def _predict_totals(model, feature_cols, rows, penalties_per_strategy, total_laps, pace_offset):
    """
    predict ทุกแถวของทุกกลยุทธ์ในครั้งเดียว แล้วหั่นผลกลับเป็นเวลารวมต่อกลยุทธ์
    pace_offset ถูกบวกเป็นค่าคงที่ต่อ lap จึงคูณ total_laps ได้เลย
    """
    preds = model.predict(pd.DataFrame(rows, columns=feature_cols))
    totals = []
    for i, penalties in enumerate(penalties_per_strategy):
        seg = preds[i * total_laps:(i + 1) * total_laps]
        totals.append(float(seg.sum() + sum(penalties) + pace_offset * total_laps))
    return totals


def simulate_strategy(
    model,
    feature_cols: list,
    total_laps: int,
    first_compound: str,
    second_compound: str,
    pit_lap: int,
    pace_offset: float = 0.0,
    num_stops: int = 1,
    second_pit_lap: int = None,
    third_compound: str = None,
    race_year: int = None,
    pit_loss: float = PIT_LOSS,
    grid_position: int = 1,
) -> float:
    """
    จำลอง race time สำหรับกลยุทธ์หนึ่งๆ
    รองรับ 1-stop และ 2-stop
    """
    rows, penalties = build_strategy_laps(
        feature_cols, total_laps, first_compound, second_compound, pit_lap,
        num_stops=num_stops, second_pit_lap=second_pit_lap,
        third_compound=third_compound, race_year=race_year,
        pit_loss=pit_loss, grid_position=grid_position,
    )
    return _predict_totals(model, feature_cols, rows, [penalties], total_laps, pace_offset)[0]


def calibrate_pace_offset(
    model,
    feature_cols: list,
    real_lap_times: list,
    actual_data: pd.DataFrame,
    total_laps: int,
    baseline_strategy: dict,
    race_year: int = None,
    pit_loss: float = PIT_LOSS,
    grid_position: int = 1,
) -> float:
    """
    หา pace_offset ที่ทำให้เวลาจำลองตรงกับเวลาจริง

    pace_offset ถูกบวกเป็นค่าคงที่ต่อ lap ดังนั้น
        sim_total(offset) = sim_total(0) + offset * total_laps
    เป็นสมการเชิงเส้น จึงแก้ได้ตรงๆ ด้วยการจำลองรอบเดียว:
        offset = (real_total - sim_total(0)) / total_laps

    เวอร์ชันก่อนหน้าไล่ค้น 101 ค่าในช่วง [-5, +5] ซึ่งทั้งช้า (จำลอง race
    ซ้ำ 101 รอบเพื่อหาค่าที่คำนวณตรงๆ ได้) และผิดเมื่อค่าที่ถูกต้องอยู่นอก
    ช่วงนั้น — ผลลัพธ์จะไปติดขอบที่ +5.00 s/lap แล้วทำให้ diff หลัง
    calibrate แย่กว่าก่อน calibrate
    """
    real_total = sum(real_lap_times)

    sim_raw = simulate_strategy(
        model, feature_cols, total_laps,
        baseline_strategy["first_compound"],
        baseline_strategy["second_compound"],
        baseline_strategy["pit_lap"],
        pace_offset=0.0,
        race_year=race_year,
        pit_loss=pit_loss,
        grid_position=grid_position,
    )

    return round((real_total - sim_raw) / total_laps, 4)


def grid_search_strategies(
    model,
    feature_cols: list,
    real_lap_times: list,
    total_laps: int,
    pace_offset: float = 0.0,
    race_year: int = None,
    pit_loss: float = PIT_LOSS,
    grid_position: int = 1,
) -> list:
    """
    ลอง compound + pit_lap ทุก combination
    คืนค่าผลลัพธ์ทุก combo เรียงจากเร็วสุด

    ทุกกลยุทธ์ถูกรวมเป็น DataFrame เดียวแล้ว predict ครั้งเดียว
    """
    real_total = sum(real_lap_times)

    compounds = ["SOFT", "MEDIUM", "HARD"]
    # pit_lap ช่วง 20%-55% ของ race
    pit_laps  = list(range(
        max(5, int(total_laps * 0.20)),
        min(total_laps - 5, int(total_laps * 0.55)) + 1,
        2,  # step 2 laps เพื่อประหยัดเวลา
    ))

    # ── รวบรวมกลยุทธ์ที่จะลองทั้งหมดก่อน ──────────────
    specs = []

    # 1-stop
    for c1, c2, pit in product(compounds, compounds, pit_laps):
        if c1 == c2:
            continue
        specs.append({
            "num_stops":       1,
            "first_compound":  c1,
            "second_compound": c2,
            "third_compound":  None,
            "pit_lap":         pit,
            "second_pit_lap":  None,
        })

    # 2-stop (เฉพาะ combo ที่น่าสนใจ)
    two_stop_combos = [
        ("SOFT", "MEDIUM", "HARD"),
        ("SOFT", "HARD",   "MEDIUM"),
        ("MEDIUM", "SOFT", "HARD"),
        ("MEDIUM", "HARD", "SOFT"),
    ]
    pit1_options = [int(total_laps * 0.25), int(total_laps * 0.30)]
    pit2_options = [int(total_laps * 0.55), int(total_laps * 0.60)]

    for (c1, c2, c3), p1, p2 in product(two_stop_combos, pit1_options, pit2_options):
        if p1 >= p2:
            continue
        specs.append({
            "num_stops":       2,
            "first_compound":  c1,
            "second_compound": c2,
            "third_compound":  c3,
            "pit_lap":         p1,
            "second_pit_lap":  p2,
        })

    # ── สร้าง feature ของทุกกลยุทธ์ แล้ว predict รอบเดียว ──
    all_rows = []
    all_penalties = []
    for s in specs:
        rows, penalties = build_strategy_laps(
            feature_cols, total_laps,
            s["first_compound"], s["second_compound"], s["pit_lap"],
            num_stops=s["num_stops"],
            second_pit_lap=s["second_pit_lap"],
            third_compound=s["third_compound"],
            race_year=race_year,
            pit_loss=pit_loss,
            grid_position=grid_position,
        )
        all_rows.extend(rows)
        all_penalties.append(penalties)

    totals = _predict_totals(
        model, feature_cols, all_rows, all_penalties, total_laps, pace_offset
    )

    results = []
    for s, sim in zip(specs, totals):
        delta = sim - real_total
        results.append({
            **s,
            "sim_total":  sim,
            "real_total": real_total,
            "delta_sec":  delta,
            "delta_pct":  delta / real_total * 100,
            "faster":     delta < 0,
        })

    results.sort(key=lambda r: r["sim_total"])
    return results


def explain_parameters(best: dict, baseline: dict, real_total: float) -> list:
    """
    อธิบายว่า parameter ไหนที่เปลี่ยนไปจาก baseline และทำให้เร็วขึ้น
    """
    explanations = []

    # เปรียบ compound
    if best["first_compound"] != baseline.get("first_compound"):
        explanations.append({
            "param":   "ยางเริ่มต้น (Start Compound)",
            "from":    baseline.get("first_compound", "?"),
            "to":      best["first_compound"],
            "reason":  _compound_reason(baseline.get("first_compound"), best["first_compound"], "start"),
            "impact":  "high",
        })

    if best["second_compound"] != baseline.get("second_compound"):
        explanations.append({
            "param":   "ยางหลังพิท (Second Compound)",
            "from":    baseline.get("second_compound", "?"),
            "to":      best["second_compound"],
            "reason":  _compound_reason(baseline.get("second_compound"), best["second_compound"], "second"),
            "impact":  "high",
        })

    if best["third_compound"] and best["third_compound"] != baseline.get("third_compound"):
        explanations.append({
            "param":   "ยาง Stint ที่ 3 (Third Compound)",
            "from":    baseline.get("third_compound", "ไม่มี"),
            "to":      best["third_compound"],
            "reason":  "เพิ่ม stint ที่ 3 เพื่อใช้ยางที่ทนทานกว่าในช่วงท้าย",
            "impact":  "medium",
        })

    # เปรียบ pit lap
    pit_diff = best["pit_lap"] - baseline.get("pit_lap", best["pit_lap"])
    if abs(pit_diff) >= 2:
        direction = "ช้าลง" if pit_diff > 0 else "เร็วขึ้น"
        tactic    = "Overcut" if pit_diff > 0 else "Undercut"
        explanations.append({
            "param":   f"รอบที่เข้าพิท (Pit Lap: {baseline.get('pit_lap','?')} → {best['pit_lap']})",
            "from":    f"Lap {baseline.get('pit_lap', '?')}",
            "to":      f"Lap {best['pit_lap']}",
            "reason":  f"เลื่อนพิท{direction} {abs(pit_diff)} laps → ใช้กลยุทธ์ {tactic} เพื่อใช้ยางที่มี grip ดีกว่าในช่วงวิกฤต",
            "impact":  "high",
        })

    # เปรียบจำนวนครั้งพิท
    base_stops = baseline.get("num_stops", 1)
    if best["num_stops"] != base_stops:
        explanations.append({
            "param":   f"จำนวนครั้งพิท ({base_stops}-stop → {best['num_stops']}-stop)",
            "from":    f"{base_stops}-stop",
            "to":      f"{best['num_stops']}-stop",
            "reason":  "การเพิ่มจำนวนพิทช่วยให้ใช้ยางที่มี grip สูงขึ้นได้ตลอด race แม้เสียเวลาพิทเพิ่ม",
            "impact":  "medium",
        })

    return explanations


def _compound_reason(from_c, to_c, position):
    mapping = {
        ("MEDIUM", "SOFT",   "start"):  "Soft มี grip สูงกว่าในช่วงต้น race ช่วยให้ได้เวลาที่ดีกว่าก่อนที่ยางจะเสื่อม",
        ("SOFT",   "MEDIUM", "start"):  "Medium ทนทานกว่าในช่วงต้น ช่วยให้ยืด stint ได้นานขึ้น",
        ("MEDIUM", "HARD",   "start"):  "Hard ทนทานมากที่สุด เหมาะกับการยืด stint ยาวในสนามที่ tyre deg สูง",
        ("SOFT",   "MEDIUM", "second"): "Medium ในช่วงหลังช่วยยืด stint ท้าย race ได้นานกว่า Soft ที่เสื่อมเร็ว",
        ("MEDIUM", "SOFT",   "second"): "Soft ในช่วงท้ายช่วยให้ push pace ได้สูงสุดในช่วงโค้งสุดท้าย",
        ("HARD",   "SOFT",   "second"): "Soft ในช่วงท้ายให้ grip สูงสุด เหมาะกับการ attack ในช่วงท้าย race",
    }
    return mapping.get((from_c, to_c, position),
                       f"เปลี่ยนจาก {from_c} → {to_c} เพื่อ balance ระหว่าง pace และ durability")