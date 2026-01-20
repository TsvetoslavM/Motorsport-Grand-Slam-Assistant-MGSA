import csv
from pathlib import Path
from typing import Any, Dict, List
from .config import DATA_DIR
from .models import GPSPoint

def lap_csv_path(lap_id: str) -> Path:
    return DATA_DIR / f"{lap_id}.csv"

def ensure_csv_header(path: Path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["timestamp", "latitude", "longitude", "altitude", "speed", "fix_quality", "hdop", "sats", "source"]
            )

def append_point_csv(lap_id: str, point: GPSPoint):
    path = lap_csv_path(lap_id)
    ensure_csv_header(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                point.timestamp,
                point.latitude,
                point.longitude,
                point.altitude,
                point.speed,
                point.fix_quality,
                point.hdop,
                point.sats,
                point.source,
            ]
        )

def load_lap_points(lap_id: str) -> List[Dict[str, Any]]:
    path = lap_csv_path(lap_id)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(row)
    return out

def load_xy(lap_id: str):
    pts = load_lap_points(lap_id)
    out = []
    for r in pts:
        try:
            out.append((float(r["latitude"]), float(r["longitude"])))
        except Exception:
            pass
    return out
