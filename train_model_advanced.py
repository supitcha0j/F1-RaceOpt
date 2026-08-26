# train_model_advanced.py
"""
เทรนโมเดล Random Forest จากข้อมูลหลาย Race / หลายนักแข่ง
บันทึก model.pkl เดียวที่ใช้ได้กับทุก race
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import load_multi_race_laps


# ============================================================
# ตั้งค่า: เลือก race + driver ที่จะใช้เทรน
# เพิ่ม/ลดได้เลย (ยิ่งมาก = โมเดลแม่นยำขึ้น แต่ใช้เวลาโหลดนานขึ้น)
# ============================================================
TRAIN_COMBINATIONS = [
    # (year, gp round number, driver) — ฤดูกาลล่าสุดที่แข่งจบแล้ว (2025)
    (2025, 1, "VER"),   # Australian GP
    (2025, 1, "NOR"),
    (2025, 4, "VER"),   # Bahrain GP
    (2025, 4, "HAM"),
    (2025, 5, "VER"),   # Saudi Arabian GP
    (2025, 5, "LEC"),
    (2025, 8, "NOR"),   # Monaco GP
    (2025, 8, "LEC"),
    (2025, 12, "VER"),  # British GP
    (2025, 12, "RUS"),
]


def train_advanced_model():
    print("=== โหลดข้อมูลจาก FastF1 (หลาย race) ===")
    combined_data, all_meta = load_multi_race_laps(TRAIN_COMBINATIONS)

    print(f"\nรวมข้อมูลทั้งหมด: {len(combined_data)} laps จาก {len(all_meta)} combinations")

    # แยก X, y
    y = combined_data["LapTimeSec"].values
    X = combined_data.drop(columns=["LapTimeSec"])

    # ลบ column ที่ไม่ใช่ตัวเลข (เผื่อมีหลงเหลือ)
    X = X.select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")

    print("\n=== เทรนโมเดล RandomForest ===")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=16,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAE  = {mae:.3f} วินาที")
    print(f"RMSE = {rmse:.3f} วินาที")

    print("\n=== บันทึกโมเดล → model.pkl ===")
    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model":    model,
            "features": feature_cols,
            "meta":     all_meta,
            "mae":      mae,
            "rmse":     rmse,
            "trained_on": TRAIN_COMBINATIONS,
        }, f)

    print("เสร็จสิ้น 🎉")


if __name__ == "__main__":
    train_advanced_model()