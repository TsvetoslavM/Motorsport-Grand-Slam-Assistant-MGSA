from __future__ import annotations

import json
import csv
from pathlib import Path
from .config import TRACK_DIR
from .runtime import now_iso

def track_path(track_id: str) -> Path:
    p = TRACK_DIR / track_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def boundaries_csv_path(track_id: str) -> Path:
    return track_path(track_id) / "boundaries.csv"

def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def safe_kind(kind: str) -> str:
    return "".join(c for c in kind if c.isalnum() or c in ("_", "-")) or "line"

def resample_polyline(points, n: int):
    if n <= 1 or len(points) < 2:
        return points[:]

    import math

    d = [0.0]
    for i in range(1, len(points)):
        lat1, lon1 = points[i - 1]
        lat2, lon2 = points[i]
        dd = math.hypot(lat2 - lat1, lon2 - lon1)
        d.append(d[-1] + dd)

    total = d[-1]
    if total <= 0:
        return [points[0]] * n

    out = []
    step = total / (n - 1)
    j = 1
    for k in range(n):
        target = k * step
        while j < len(d) and d[j] < target:
            j += 1
        if j >= len(d):
            out.append(points[-1])
            continue
        i0 = j - 1
        i1 = j
        t0 = d[i0]
        t1 = d[i1]
        if t1 <= t0:
            out.append(points[i1])
            continue
        a = (target - t0) / (t1 - t0)
        lat0, lon0 = points[i0]
        lat1, lon1 = points[i1]
        out.append((lat0 + a * (lat1 - lat0), lon0 + a * (lon1 - lon0)))
    return out

def write_boundaries_csv(track_id: str, outer_r, inner_r):
    out_path = boundaries_csv_path(track_id)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "outer_lat", "outer_lon", "inner_lat", "inner_lon"])
        for i in range(len(outer_r)):
            o_lat, o_lon = outer_r[i]
            in_lat, in_lon = inner_r[i]
            w.writerow([i, o_lat, o_lon, in_lat, in_lon])
    return out_path

def write_boundaries_meta(track_id: str, meta: dict):
    meta_path = track_path(track_id) / "boundaries.meta.json"
    meta = {"updated": now_iso(), "track_id": track_id, **meta}
    write_json(meta_path, meta)
    return meta_path


from pathlib import Path
import json, csv
from typing import Any, Dict, Iterable

def save_optimal_files(track_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
    root = track_path(track_id)
    root.mkdir(parents=True, exist_ok=True)

    p_json = root / "optimal.json"
    p_csv = root / "optimal_latlon.csv"  # временно XY вътре (виж т.3)

    p_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pts = payload.get("optimal") or []
    if not pts:
        raise RuntimeError("save_optimal_files: payload['optimal'] is empty/missing")

    with p_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "speed_mps", "time_s"])
        w.writeheader()
        for p in pts:
            w.writerow({
                "x": float(p["x"]),
                "y": float(p["y"]),
                "speed_mps": float(p.get("speed_mps", 0.0)),
                "time_s": float(p.get("time_s", 0.0)),
            })

    return {
        "optimal_json": str(p_json),
        "optimal_latlon_csv": str(p_csv),
    }
