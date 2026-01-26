from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .auth import verify_token
from .tracks import track_path, boundaries_csv_path, safe_kind
from .runtime import now_iso

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class CompareRequest(BaseModel):
    driver_kind: str = Field("driver")
    n_points: int = Field(250, ge=30, le=5000)
    segment_len_m: float = Field(20.0, gt=1.0, le=500.0)
    ipopt_max_iter: int = Field(2000, ge=50, le=20000)
    ipopt_print_level: int = Field(0, ge=0, le=12)


def _latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat
    return x, y


def _polyline_arclen(xy: List[Tuple[float, float]]) -> List[float]:
    if not xy:
        return []
    d = [0.0]
    for i in range(1, len(xy)):
        x0, y0 = xy[i - 1]
        x1, y1 = xy[i]
        d.append(d[-1] + math.hypot(x1 - x0, y1 - y0))
    return d


def _resample_polyline(xy: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    if n <= 1:
        return xy[:1]
    if len(xy) < 2:
        return (xy[:1] * n) if xy else [(0.0, 0.0)] * n

    d = _polyline_arclen(xy)
    total = d[-1]
    if total <= 1e-12:
        return [xy[0]] * n

    out = []
    step = total / (n - 1)
    j = 1
    for k in range(n):
        target = k * step
        while j < len(d) and d[j] < target:
            j += 1
        if j >= len(d):
            out.append(xy[-1])
            continue
        i0 = j - 1
        i1 = j
        t0 = d[i0]
        t1 = d[i1]
        if t1 <= t0 + 1e-12:
            out.append(xy[i1])
            continue
        a = (target - t0) / (t1 - t0)
        x0, y0 = xy[i0]
        x1, y1 = xy[i1]
        out.append((x0 + a * (x1 - x0), y0 + a * (y1 - y0)))
    return out


def _read_boundaries(track_id: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    p = boundaries_csv_path(track_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="boundaries.csv not found")
    outer = []
    inner = []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            outer.append((float(row["outer_lat"]), float(row["outer_lon"])))
            inner.append((float(row["inner_lat"]), float(row["inner_lon"])))
    if len(outer) < 3 or len(inner) < 3:
        raise HTTPException(status_code=400, detail="boundaries.csv has too few points")
    return outer, inner


def _read_racing_line(track_id: str, kind: str) -> Tuple[List[Tuple[float, float]], List[float]]:
    kind2 = safe_kind(kind)
    p = track_path(track_id) / f"racing_{kind2}.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"racing_{kind2}.csv not found")
    pts = []
    v_kmh = []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append((float(row["lat"]), float(row["lon"])))
            v_kmh.append(float(row.get("speed_kmh", 0.0) or 0.0))
    if len(pts) < 3:
        raise HTTPException(status_code=400, detail="racing line has too few points")
    return pts, v_kmh


def _stats(arr: List[float]) -> Dict[str, Any]:
    if not arr:
        return {"count": 0}
    s = sorted(arr)
    n = len(s)
    mean = sum(s) / n
    rmse = math.sqrt(sum(x * x for x in s) / n)
    p95 = s[int(0.95 * (n - 1))]
    return {"count": n, "mean": mean, "rmse": rmse, "p95": p95, "max": s[-1]}

def resample_scalar_by_arclen(values: List[float], xy: List[Tuple[float, float]], n: int) -> List[float]:
    if len(values) != len(xy) or len(values) < 2:
        return [0.0] * n
    d = _polyline_arclen(xy)
    total = d[-1]
    if total <= 1e-12:
        return [float(values[0])] * n
    out = []
    step = total / (n - 1)
    j = 1
    for k in range(n):
        target = k * step
        while j < len(d) and d[j] < target:
            j += 1
        if j >= len(d):
            out.append(float(values[-1]))
            continue
        i0 = j - 1
        i1 = j
        t0 = d[i0]
        t1 = d[i1]
        if t1 <= t0 + 1e-12:
            out.append(float(values[i1]))
            continue
        a = (target - t0) / (t1 - t0)
        v0 = float(values[i0])
        v1 = float(values[i1])
        out.append(v0 + a * (v1 - v0))
    return out

def ll_to_xy_list(latlon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [_latlon_to_xy_m(lat, lon, lat0, lon0) for (lat, lon) in latlon]

async def compare(track_id: str, req: CompareRequest, token: dict = Depends(verify_token)):
    try:
        from firmware.Optimal_Control.solver_api import OptimizeOptions, optimize_trajectory_from_two_lines
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"solver_api unavailable: {e}")

    outer_latlon, inner_latlon = _read_boundaries(track_id)
    driver_latlon, driver_v_kmh = _read_racing_line(track_id, req.driver_kind)

    lat0 = (sum(p[0] for p in outer_latlon) + sum(p[0] for p in inner_latlon)) / (len(outer_latlon) + len(inner_latlon))
    lon0 = (sum(p[1] for p in outer_latlon) + sum(p[1] for p in inner_latlon)) / (len(outer_latlon) + len(inner_latlon))

    outer_xy = ll_to_xy_list(outer_latlon)
    inner_xy = ll_to_xy_list(inner_latlon)

    N = int(req.n_points)
    outer_xy_r = _resample_polyline(outer_xy, N)
    inner_xy_r = _resample_polyline(inner_xy, N)

    left_line = [{"x": float(x), "y": float(y)} for (x, y) in inner_xy_r]
    right_line = [{"x": float(x), "y": float(y)} for (x, y) in outer_xy_r]

    opts = OptimizeOptions(
        n_points=N,
        ipopt_max_iter=int(req.ipopt_max_iter),
        ipopt_print_level=int(req.ipopt_print_level),
    )
    sol = optimize_trajectory_from_two_lines(left_line, right_line, options=opts)
    optimal = sol.get("optimal") or []
    if len(optimal) < 3:
        raise HTTPException(status_code=500, detail="solver returned empty optimal")

    opt_xy = [(float(p["x"]), float(p["y"])) for p in optimal]
    opt_v_mps = [max(0.1, float(p.get("speed_mps", 0.1))) for p in optimal]

    drv_xy = ll_to_xy_list(driver_latlon)
    drv_xy_r = _resample_polyline(drv_xy, N)

    drv_v_kmh_r = resample_scalar_by_arclen(driver_v_kmh, drv_xy, N)
    drv_v_mps = [max(0.1, v / 3.6) for v in drv_v_kmh_r]

    s_opt = _polyline_arclen(opt_xy)
    ds = [0.0] * N
    for i in range(1, N):
        ds[i] = max(1e-6, s_opt[i] - s_opt[i - 1])

    error_m = []
    time_loss_s = []
    for i in range(N):
        xo, yo = opt_xy[i]
        xd, yd = drv_xy_r[i]
        error_m.append(math.hypot(xd - xo, yd - yo))

        if i == 0:
            time_loss_s.append(0.0)
            continue
        t_opt = ds[i] / max(0.1, opt_v_mps[i])
        t_drv = ds[i] / max(0.1, drv_v_mps[i])
        time_loss_s.append(t_drv - t_opt)

    seg_len = float(req.segment_len_m)
    total_len = s_opt[-1] if s_opt else 0.0
    n_segs = max(1, int(math.ceil(total_len / seg_len)))

    segs = []
    for si in range(n_segs):
        s0 = si * seg_len
        s1 = min(total_len, (si + 1) * seg_len)
        idx = [i for i in range(N) if s0 <= s_opt[i] < s1]
        if not idx:
            continue
        seg_time_loss = float(sum(time_loss_s[i] for i in idx))
        seg_mean_err = float(sum(error_m[i] for i in idx) / len(idx))
        seg_max_err = float(max(error_m[i] for i in idx))
        segs.append(
            {
                "s0": float(s0),
                "s1": float(s1),
                "len_m": float(s1 - s0),
                "time_loss_s": seg_time_loss,
                "mean_err_m": seg_mean_err,
                "max_err_m": seg_max_err,
            }
        )

    segs_sorted = sorted(segs, key=lambda x: x["time_loss_s"], reverse=True)

    resp = {
        "track_id": track_id,
        "driver_kind": safe_kind(req.driver_kind),
        "updated": now_iso(),
        "solver": {"converged": bool(sol.get("converged")), "lap_time_s": sol.get("lap_time_s"), "N": int(sol.get("N", N))},
        "series": {
            "s_m": [float(x) for x in s_opt],
            "error_m": [float(x) for x in error_m],
            "time_loss_s": [float(x) for x in time_loss_s],
            "opt_v_mps": [float(x) for x in opt_v_mps],
            "drv_v_mps": [float(x) for x in drv_v_mps],
        },
        "stats": {
            "error_m": _stats(error_m),
            "time_loss_s": _stats([x for x in time_loss_s[1:]]),
            "total_time_loss_s": float(sum(time_loss_s)),
        },
        "segments": {
            "segment_len_m": seg_len,
            "top": segs_sorted[:15],
        },
        "polylines": {
            "outer": [{"lat": float(a), "lon": float(b)} for (a, b) in outer_latlon],
            "inner": [{"lat": float(a), "lon": float(b)} for (a, b) in inner_latlon],
            "driver": [{"lat": float(a), "lon": float(b)} for (a, b) in driver_latlon],
            "optimal": [{"lat": float(lat0 + (y / 111_320.0)), "lon": float(lon0 + (x / (111_320.0 * math.cos(math.radians(lat0)) + 1e-9)))} for (x, y) in opt_xy],
        },
    }
    return resp

async def compare_driver_vs_optimal_internal(track_id: str, driver_kind: str = "driver", n_points: int = 250) -> dict:
    req = CompareRequest(driver_kind=driver_kind, n_points=n_points)
    dummy_token = {"sub": "internal"}  # verify_token не се ползва вътре
    return await compare(track_id, req, token=dummy_token)

def save_compare_json(track_id: str, payload: dict, driver_kind: str = "driver") -> str:
    p = track_path(track_id) / f"compare_{safe_kind(driver_kind)}_vs_optimal.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)

