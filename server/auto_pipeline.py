from __future__ import annotations

import asyncio
import json
import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from server.storage import load_xy
from server.tracks import boundaries_csv_path, resample_polyline, track_path

try:
    from firmware.Optimal_Control.solver_api import OptimizeOptions, optimize_trajectory_from_two_lines
except Exception as e:
    OptimizeOptions = None  # type: ignore
    optimize_trajectory_from_two_lines = None  # type: ignore
    _IMPORT_ERR = str(e)
else:
    _IMPORT_ERR = ""


@dataclass
class PendingTrack:
    inner_lap_id: Optional[str] = None
    outer_lap_id: Optional[str] = None
    running: bool = False
    last_pair: Optional[Tuple[str, str]] = None


_pending: Dict[str, PendingTrack] = {}
_locks: Dict[str, asyncio.Lock] = {}


def _lock(track_id: str) -> asyncio.Lock:
    if track_id not in _locks:
        _locks[track_id] = asyncio.Lock()
    return _locks[track_id]


def _state(track_id: str) -> PendingTrack:
    if track_id not in _pending:
        _pending[track_id] = PendingTrack()
    return _pending[track_id]


def _latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    R = 6371000.0
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)
    x = (lon_r - lon0_r) * math.cos(lat0_r) * R
    y = (lat_r - lat0_r) * R
    return x, y


def _xy_to_latlon_m(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    R = 6371000.0
    lat0_r = math.radians(lat0)
    dlat = y / R
    dlon = x / (R * math.cos(lat0_r))
    lat = math.degrees(math.radians(lat0) + dlat)
    lon = math.degrees(math.radians(lon0) + dlon)
    return lat, lon


def _choose_origin_mean(outer_ll: List[Tuple[float, float]], inner_ll: List[Tuple[float, float]]) -> Tuple[float, float]:
    all_pts = outer_ll + inner_ll
    lat0 = sum(p[0] for p in all_pts) / len(all_pts)
    lon0 = sum(p[1] for p in all_pts) / len(all_pts)
    return lat0, lon0


def build_boundaries_csv(track_id: str, outer_lap_id: str, inner_lap_id: str, n_points: int) -> Path:
    outer_ll = load_xy(outer_lap_id)
    inner_ll = load_xy(inner_lap_id)

    if len(outer_ll) < 10 or len(inner_ll) < 10:
        raise ValueError("not enough points in one of the laps")

    N = int(n_points)
    outer_ll = resample_polyline(outer_ll, N)
    inner_ll = resample_polyline(inner_ll, N)

    out_path = boundaries_csv_path(track_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "outer_lat", "outer_lon", "inner_lat", "inner_lon"])
        for i in range(N):
            w.writerow([i, outer_ll[i][0], outer_ll[i][1], inner_ll[i][0], inner_ll[i][1]])

    return out_path


def optimize_from_boundaries(track_id: str, n_points: int, ipopt_linear_solver: str = "ma57", ipopt_print_level: int = 0) -> dict:
    if optimize_trajectory_from_two_lines is None or OptimizeOptions is None:
        raise RuntimeError(f"Firmware optimizer unavailable: {_IMPORT_ERR}")

    path = boundaries_csv_path(track_id)
    if not path.exists():
        raise FileNotFoundError("boundaries.csv not found for track")

    outer_ll: List[Tuple[float, float]] = []
    inner_ll: List[Tuple[float, float]] = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            outer_ll.append((float(row["outer_lat"]), float(row["outer_lon"])))
            inner_ll.append((float(row["inner_lat"]), float(row["inner_lon"])))

    if len(outer_ll) < 10 or len(inner_ll) < 10:
        raise ValueError("not enough points in boundaries.csv")

    N = int(n_points)
    outer_ll = resample_polyline(outer_ll, N)
    inner_ll = resample_polyline(inner_ll, N)

    lat0, lon0 = _choose_origin_mean(outer_ll, inner_ll)

    left_xy = [{"x": _latlon_to_xy_m(lat, lon, lat0, lon0)[0], "y": _latlon_to_xy_m(lat, lon, lat0, lon0)[1]} for (lat, lon) in outer_ll]
    right_xy = [{"x": _latlon_to_xy_m(lat, lon, lat0, lon0)[0], "y": _latlon_to_xy_m(lat, lon, lat0, lon0)[1]} for (lat, lon) in inner_ll]

    opts = OptimizeOptions(
        n_points=N,
        ipopt_max_iter=2000,
        ipopt_print_level=int(ipopt_print_level),
        ipopt_tol=1e-4,
        ipopt_acceptable_tol=1e-3,
        ipopt_linear_solver=str(ipopt_linear_solver),
    )

    result = optimize_trajectory_from_two_lines(left_xy, right_xy, options=opts)

    if isinstance(result, dict):
        if "optimal" in result and isinstance(result["optimal"], list):
            opt_ll = []
            for p in result["optimal"]:
                lat, lon = _xy_to_latlon_m(float(p["x"]), float(p["y"]), lat0, lon0)
                opt_ll.append({"lat": lat, "lon": lon, "time_s": p.get("time_s"), "speed_kmh": p.get("speed_kmh")})
            result["optimal_latlon"] = opt_ll

        if "centerline" in result and isinstance(result["centerline"], list):
            cen_ll = []
            for p in result["centerline"]:
                lat, lon = _xy_to_latlon_m(float(p["x"]), float(p["y"]), lat0, lon0)
                cen_ll.append({"lat": lat, "lon": lon})
            result["centerline_latlon"] = cen_ll

        result["origin_latlon"] = {"lat0": lat0, "lon0": lon0}
        result["track_id"] = track_id
        result["boundaries_path"] = str(path)

    return result


def save_optimal_artifacts(track_id: str, result: dict) -> None:
    root = track_path(track_id)
    root.mkdir(parents=True, exist_ok=True)

    (root / "optimal.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    opt = result.get("optimal_latlon")
    if not (isinstance(opt, list) and opt):
        return

    R = 6371000.0

    def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1r = math.radians(lat1)
        lat2r = math.radians(lat2)
        dlat = lat2r - lat1r
        dlon = math.radians(lon2 - lon1)
        mlat = (lat1r + lat2r) * 0.5
        x = dlon * math.cos(mlat) * R
        y = dlat * R
        return math.hypot(x, y)

    lat = []
    lon = []
    t = []
    speed_kmh = []

    for p in opt:
        lat.append(float(p["lat"]))
        lon.append(float(p["lon"]))
        t.append(float(p.get("time_s", 0.0)))
        speed_kmh.append(p.get("speed_kmh"))

    N = len(lat)
    if N < 3:
        return

    has_speed = True
    v = []

    for i in range(N):
        s = speed_kmh[i]
        if s is None:
            has_speed = False
            break
        try:
            v.append(float(s) / 3.6)
        except Exception:
            has_speed = False
            break

    if not has_speed:
        R = 6371000.0

        def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            lat1r = math.radians(lat1)
            lat2r = math.radians(lat2)
            dlat = lat2r - lat1r
            dlon = math.radians(lon2 - lon1)
            mlat = (lat1r + lat2r) * 0.5
            x = dlon * math.cos(mlat) * R
            y = dlat * R
            return math.hypot(x, y)

        v = []
        for i in range(N - 1):
            dt = max(t[i + 1] - t[i], 1e-3)
            d = _dist_m(lat[i], lon[i], lat[i + 1], lon[i + 1])
            v.append(d / dt)
        v.append(v[-1])

    # a = dv/dt
    a = []
    for i in range(N - 1):
        dt = max(t[i + 1] - t[i], 1e-3)
        a.append((v[i + 1] - v[i]) / dt)
    a.append(a[-1])

    # soft clamp to keep sane numbers
    A_MAX = 12.0  # m/s^2 ~ 1.2g (tune later)
    a = [max(-A_MAX, min(A_MAX, float(x))) for x in a]

    # optional smoothing (EMA)
    alpha = 0.2
    af = [a[0]]
    for i in range(1, N):
        af.append((1 - alpha) * af[-1] + alpha * a[i])
    a = af


    out_csv = root / "optimal_latlon.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "lat", "lon", "time_s", "speed_kmh", "a_mps2"])
        for i in range(N):
            w.writerow([
                i,
                lat[i],
                lon[i],
                t[i],
                speed_kmh[i] if speed_kmh[i] is not None else "",
                a[i],
            ])




async def register_completed_lap_and_maybe_run(*, track_id: str, lap_type: str, lap_id: str, n_points: int = 900) -> None:
    lk = _lock(track_id)
    async with lk:
        st = _state(track_id)
        if lap_type == "inner":
            st.inner_lap_id = lap_id
        elif lap_type == "outer":
            st.outer_lap_id = lap_id
        else:
            return

        inner_id = st.inner_lap_id
        outer_id = st.outer_lap_id
        if not inner_id or not outer_id:
            return

        pair = (inner_id, outer_id)
        if st.running or st.last_pair == pair:
            return

        st.running = True
        st.last_pair = pair

    try:
        await asyncio.to_thread(build_boundaries_csv, track_id, outer_id, inner_id, int(n_points))
        result = await asyncio.to_thread(
            optimize_from_boundaries,
            track_id,
            int(n_points),
            "ma57",
            0,
        )

        if isinstance(result, dict):
            await asyncio.to_thread(save_optimal_artifacts, track_id, result)

            from server.ws import manager
            await manager.broadcast(
                {
                    "type": "optimal_ready",
                    "track_id": track_id,
                    "n_points": int(n_points),
                    "artifacts": {
                        "optimal_json": f"/api/track/{track_id}/optimal.json",
                        "optimal_latlon_csv": f"/api/track/{track_id}/optimal_latlon.csv",
                        "boundaries_csv": f"/api/track/{track_id}/boundaries",
                    },
                }
            )

    except Exception as e:
        from server.ws import manager
        await manager.broadcast({"type": "optimal_failed", "track_id": track_id, "error": str(e)})
        raise


    finally:
        async with _lock(track_id):
            _state(track_id).running = False

