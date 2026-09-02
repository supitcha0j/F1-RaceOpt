# fetch_media.py
"""
สคริปต์ one-off: ดึงรูปนักแข่งจริงจาก OpenF1 (headshot_url) และรูปสนามจริงจาก Wikimedia Commons
(ภาพ license เปิด CC) มาแคชไว้ที่ static/img/ ให้ครอบคลุมทั้งฤดูกาลล่าสุด (ทุกสนาม + ทุกนักแข่ง
ที่เคยลงกริดจริง) ไม่ใช่แค่ combo ที่มี lap cache อยู่แล้ว — เพราะ Strategy Lab ให้เลือกสนามไหน
ของฤดูกาลก็ได้ และจำลองทั้งกริดจริง (ไม่ใช่แค่นักแข่งที่เคย analyze) รูปเลยต้องครบทั้งกริด

ดึง metadata นักแข่ง (ทีม, สีทีม, เบอร์รถ, ชื่อเต็ม) จาก OpenF1 มาเก็บที่ static/data/drivers.json
ด้วย — ใช้สร้างหน้า Drivers/Teams โดยไม่ต้องมีรูป logo ทีมจริง (ติด trademark) ใช้สีทีมจริงแทน

รันแยกต่างหาก ไม่ใช่ตอน Flask start เพราะ app.py ต้องบูตแบบ offline ได้เสมอ (ดู CLAUDE.md)
รันซ้ำได้ปลอดภัย: ข้ามไฟล์ที่ดาวน์โหลดไว้แล้วในดิสก์ (metadata เบา เลยดึงใหม่ทุกครั้งที่รันได้)

    python fetch_media.py
"""
import json
import os
import time
import requests

from data_pipeline import get_available_races, get_available_drivers, get_latest_completed_season

DRIVER_DIR = os.path.join("static", "img", "drivers")
CIRCUIT_DIR = os.path.join("static", "img", "circuits")
DATA_DIR = os.path.join("static", "data")
DRIVER_META_PATH = os.path.join(DATA_DIR, "drivers.json")
UA = {"User-Agent": "RaceOptMediaFetcher/1.0 (educational F1 strategy project)"}

# ชื่อสนามทางการ — ใช้ค้นหารูปที่ถูกต้องบน Wikimedia Commons เพราะชื่อสั้นในตารางแข่ง
# (เช่น "British") ค้นตรงๆ จะได้ผลลัพธ์กว้างเกินไปหรือไม่ตรงสนาม
CIRCUIT_SEARCH_NAMES = {
    "Australian": "Albert Park Circuit Melbourne",
    "Chinese": "Shanghai International Circuit",
    "Japanese": "Suzuka International Racing Course",
    "Bahrain": "Bahrain International Circuit",
    "Saudi_Arabian": "Saudi Arabia F1 GP UMBRA Jeddah",
    "Miami": "Miami Grand Prix startfinish",
    "Emilia_Romagna": "Autodromo Enzo e Dino Ferrari Imola",
    "Monaco": "Circuit de Monaco",
    "Canadian": "Circuit Gilles Villeneuve",
    "Spanish": "Circuit de Barcelona-Catalunya",
    "Austrian": "Red Bull Ring Spielberg",
    "British": "Silverstone Circuit",
    "Belgian": "Circuit de Spa-Francorchamps",
    "Hungarian": "Hungaroring",
    "Dutch": "Circuit Zandvoort",
    "Italian": "Autodromo Nazionale Monza",
    "Azerbaijan": "Baku City Circuit",
    "Singapore": "Marina Bay Street Circuit",
    "United_States": "Circuit of the Americas",
    "Mexico_City": "Autodromo Hermanos Rodriguez",
    "São_Paulo": "Autodromo Jose Carlos Pace Interlagos",
    "Las_Vegas": "Las Vegas Grand Prix Sphere Orbi 2024",
    "Qatar": "Qatar Grand Prix start",
    "Abu_Dhabi": "Yas Marina Circuit",
}


def _commons_get_json(params, retries=4):
    """Wikimedia Commons จำกัด rate ของ automated request บ่อยๆ จะได้หน้า error ที่ไม่ใช่ JSON
    กลับมา (ไม่ใช่ 4xx/5xx ให้ raise_for_status ดักได้) — retry แบบ backoff แทน"""
    for attempt in range(retries):
        r = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                          headers=UA, timeout=15)
        try:
            return r.json()
        except ValueError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


_PHOTO_EXT = (".jpg", ".jpeg", ".png")


def _commons_photo_url(query, extra_hint=""):
    """หา URL รูปแรกที่ค้นเจอบน Commons สำหรับ query (ลองแบบเจาะจงก่อน แล้วค่อย fallback กว้างขึ้น)
    ข้ามผลลัพธ์ที่ไม่ใช่รูปจริง (.svg แผนผังสนาม, .pdf เอกสารที่ค้นเจอโดยบังเอิญ) เลือกไฟล์รูปถ่ายจริงก่อน"""
    title = None
    candidates = (f"{query} {extra_hint}".strip(), query) if extra_hint else (query,)
    for candidate in candidates:
        search = _commons_get_json({
            "action": "query", "list": "search", "srsearch": candidate,
            "srnamespace": 6, "format": "json", "srlimit": 5,
        })
        results = search.get("query", {}).get("search", [])
        photo = next((x for x in results if x["title"].lower().endswith(_PHOTO_EXT)), None)
        if photo:
            title = photo["title"]
            break
        if results and title is None:
            title = results[0]["title"]  # เก็บผลลัพธ์แรกไว้เผื่อไม่เจอรูปจริงเลยสักคำค้น
    if not title:
        return None, None

    info = _commons_get_json({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url", "iiurlwidth": 1400, "format": "json",
    })
    pages = info.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    imageinfo = page.get("imageinfo")
    if not imageinfo:
        return None, title
    return (imageinfo[0].get("thumburl") or imageinfo[0]["url"]), title


def fetch_driver_photo(code, season=None):
    os.makedirs(DRIVER_DIR, exist_ok=True)
    path = os.path.join(DRIVER_DIR, f"{code}.jpg")
    if os.path.exists(path):
        return
    try:
        r = requests.get("https://api.openf1.org/v1/drivers",
                          params={"name_acronym": code}, headers=UA, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            print(f"  ไม่พบนักแข่ง {code} ใน OpenF1"); return
        last = rows[-1]  # แถวล่าสุด = ฤดูกาลล่าสุดที่มีข้อมูล
        img_url = last.get("headshot_url")

        if img_url:
            img = requests.get(img_url, headers=UA, timeout=15)
            img.raise_for_status()
            with open(path, "wb") as f:
                f.write(img.content)
            print(f"  driver {code} ok (OpenF1)")
            return

        # OpenF1 บางคนไม่มี headshot_url (เช่น เพิ่งเลื่อนขึ้นมาแข่งกลางฤดูกาล) — fallback ไป
        # หาภาพเหมือนจริงจาก Wikimedia Commons ด้วยชื่อเต็มแทน
        full_name = last.get("full_name", "").title()
        if not full_name:
            print(f"  ไม่มี headshot_url สำหรับ {code}"); return
        query = f"{full_name} {season}" if season else full_name
        img_url, title = _commons_photo_url(query)
        if not img_url:
            print(f"  ไม่มี headshot_url สำหรับ {code} (ลอง Commons แล้วก็ไม่พบ)"); return
        img = requests.get(img_url, headers=UA, timeout=15)
        img.raise_for_status()
        with open(path, "wb") as f:
            f.write(img.content)
        print(f"  driver {code} ok (Commons: {title})")
    except Exception as e:
        print(f"  ข้าม driver {code}: {e}")


def fetch_driver_meta(code):
    """เมทาดาต้านักแข่ง (ทีม, สีทีม, เบอร์รถ, ชื่อเต็ม) จาก OpenF1 — request เบา (JSON ไม่ใช่ไฟล์รูป)
    เลยไม่ skip แบบรูป ดึงใหม่ทุกครั้งที่รัน main() เพื่อให้ทันย้ายทีมกลางฤดูกาล"""
    try:
        r = requests.get("https://api.openf1.org/v1/drivers",
                          params={"name_acronym": code}, headers=UA, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        last = rows[-1]
        return {
            "code": code,
            "full_name": last.get("full_name", "").title(),
            "team_name": last.get("team_name") or "Unknown",
            "team_colour": last.get("team_colour") or "666666",
            "driver_number": last.get("driver_number"),
        }
    except Exception as e:
        print(f"  ข้าม metadata {code}: {e}")
        return None


def fetch_circuit_photo(slug):
    os.makedirs(CIRCUIT_DIR, exist_ok=True)
    path = os.path.join(CIRCUIT_DIR, f"{slug}.jpg")
    if os.path.exists(path):
        return
    query = CIRCUIT_SEARCH_NAMES.get(slug, slug.replace("_", " ") + " Grand Prix circuit")
    try:
        img_url, title = _commons_photo_url(query, extra_hint="aerial")
        if not img_url:
            print(f"  ไม่พบรูปสนาม {slug} บน Commons"); return
        img = requests.get(img_url, headers=UA, timeout=20)
        img.raise_for_status()
        with open(path, "wb") as f:
            f.write(img.content)
        print(f"  circuit {slug} ok ({title})")
    except Exception as e:
        print(f"  ข้าม circuit {slug}: {e}")


def main():
    season = get_latest_completed_season()
    races = get_available_races(season)
    if not races:
        print(f"ไม่มีตารางแข่งของฤดูกาล {season}")
        return

    drivers = set()
    circuits = set()
    for race_key, info in races.items():
        drivers.update(get_available_drivers(info["year"], info["gp"]))
        slug = race_key.split("_", 1)[1] if "_" in race_key else race_key
        circuits.add(slug)

    print(f"ฤดูกาล {season}: นักแข่ง {len(drivers)} คน, สนาม {len(circuits)} สนาม")

    meta_list = []
    for code in sorted(drivers):
        fetch_driver_photo(code, season=season)
        meta = fetch_driver_meta(code)
        if meta:
            meta_list.append(meta)
        time.sleep(0.3)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DRIVER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(meta_list, key=lambda d: d["code"]), f, ensure_ascii=False, indent=2)
    print(f"  บันทึก metadata {len(meta_list)} คนที่ {DRIVER_META_PATH}")

    for slug in sorted(circuits):
        fetch_circuit_photo(slug)
        time.sleep(1.5)

    print("เสร็จแล้ว")


if __name__ == "__main__":
    main()
