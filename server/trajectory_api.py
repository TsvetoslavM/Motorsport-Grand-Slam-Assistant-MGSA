from __future__ import annotations

from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from server.auth import verify_token  # reuse existing auth (no circular import)
from server.storage import load_xy  # returns (lat,lon)
from server.tracks import boundaries_csv_path, resample_polyline  # resamples (lat,lon)

import math
import csv
from pathlib import Path


try:
    from firmware.Optimal_Control.solver_api import (
        OptimizeOptions,
        optimize_trajectory_from_two_lines,
    )
except Exception as e:  # pragma: no cover
    # Server can still start; endpoint will return a clear error if called.
    OptimizeOptions = None  # type: ignore
    optimize_trajectory_from_two_lines = None  # type: ignore
    _IMPORT_ERR = str(e)
else:
    _IMPORT_ERR = ""


router = APIRouter(prefix="/api/trajectory", tags=["trajectory"])


class XYPoint(BaseModel):
    x: float
    y: float


class TwoLinesOptimizeRequest(BaseModel):
    left_line: List[XYPoint] = Field(..., description="Left boundary polyline (x,y) points")
    right_line: List[XYPoint] = Field(..., description="Right boundary polyline (x,y) points")

    n_points: int = Field(250, ge=30, le=5000)
    ipopt_max_iter: int = Field(2000, ge=50, le=20000)
    ipopt_print_level: int = Field(0, ge=0, le=12)
    ipopt_tol: float = Field(1e-4, gt=0)
    ipopt_acceptable_tol: float = Field(1e-3, gt=0)
    ipopt_linear_solver: str = Field("mumps")


@router.post("/optimize_from_two_lines")
async def optimize_from_two_lines(req: TwoLinesOptimizeRequest, token: dict = Depends(verify_token)):
    if optimize_trajectory_from_two_lines is None or OptimizeOptions is None:
        raise HTTPException(
            status_code=500,
            detail=f"Firmware optimizer unavailable (casadi/ipopt missing or import error): {_IMPORT_ERR}",
        )

    if len(req.left_line) < 3 or len(req.right_line) < 3:
        raise HTTPException(status_code=400, detail="Each line must contain at least 3 points")

    opts = OptimizeOptions(
        n_points=req.n_points,
        ipopt_max_iter=req.ipopt_max_iter,
        ipopt_print_level=req.ipopt_print_level,
        ipopt_tol=req.ipopt_tol,
        ipopt_acceptable_tol=req.ipopt_acceptable_tol,
        ipopt_linear_solver=req.ipopt_linear_solver,
    )

    try:
        result = optimize_trajectory_from_two_lines(
            [p.model_dump() for p in req.left_line],
            [p.model_dump() for p in req.right_line],
            options=opts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    return result

class TwoLapsOptimizeRequest(BaseModel):
    outer_lap_id: str
    inner_lap_id: str

    n_points: int = Field(250, ge=30, le=5000)

    ipopt_max_iter: int = Field(2000, ge=50, le=20000)
    ipopt_print_level: int = Field(0, ge=0, le=12)
    ipopt_tol: float = Field(1e-4, gt=0)
    ipopt_acceptable_tol: float = Field(1e-3, gt=0)
    ipopt_linear_solver: str = Field("mumps")

    return_latlon: bool = Field(True, description="Add optimal_latlon/centerline_latlon for mapping")
    origin_mode: str = Field("outer0", description="outer0 | mean")


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


def _choose_origin(outer_ll: List[Tuple[float, float]], inner_ll: List[Tuple[float, float]], mode: str) -> Tuple[float, float]:
    if mode == "mean":
        all_pts = outer_ll + inner_ll
        lat0 = sum(p[0] for p in all_pts) / len(all_pts)
        lon0 = sum(p[1] for p in all_pts) / len(all_pts)
        return lat0, lon0
    # default outer0
    return outer_ll[0][0], outer_ll[0][1]


@router.post("/optimize_from_two_laps")
async def optimize_from_two_laps(req: TwoLapsOptimizeRequest, token: dict = Depends(verify_token)):
    if optimize_trajectory_from_two_lines is None or OptimizeOptions is None:
        raise HTTPException(
            status_code=500,
            detail=f"Firmware optimizer unavailable (casadi/ipopt missing or import error): {_IMPORT_ERR}",
        )

    outer_ll = load_xy(req.outer_lap_id)
    inner_ll = load_xy(req.inner_lap_id)

    if len(outer_ll) < 10 or len(inner_ll) < 10:
        raise HTTPException(status_code=400, detail="not enough points in one of the laps")

    N = int(req.n_points)
    outer_ll = resample_polyline(outer_ll, N)
    inner_ll = resample_polyline(inner_ll, N)

    lat0, lon0 = _choose_origin(outer_ll, inner_ll, req.origin_mode)

    left_xy = [{"x": _latlon_to_xy_m(lat, lon, lat0, lon0)[0], "y": _latlon_to_xy_m(lat, lon, lat0, lon0)[1]} for (lat, lon) in outer_ll]
    right_xy = [{"x": _latlon_to_xy_m(lat, lon, lat0, lon0)[0], "y": _latlon_to_xy_m(lat, lon, lat0, lon0)[1]} for (lat, lon) in inner_ll]

    opts = OptimizeOptions(
        n_points=N,
        ipopt_max_iter=req.ipopt_max_iter,
        ipopt_print_level=req.ipopt_print_level,
        ipopt_tol=req.ipopt_tol,
        ipopt_acceptable_tol=req.ipopt_acceptable_tol,
        ipopt_linear_solver=req.ipopt_linear_solver,
    )

    try:
        result = optimize_trajectory_from_two_lines(left_xy, right_xy, options=opts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    # solver returns "optimal" and "centerline" (XY meters)
    if req.return_latlon and isinstance(result, dict):
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

    return result

class OptimizeFromTrackRequest(BaseModel):
    track_id: str
    n_points: int = Field(250, ge=30, le=5000)

    ipopt_max_iter: int = Field(2000, ge=50, le=20000)
    ipopt_print_level: int = Field(0, ge=0, le=12)
    ipopt_tol: float = Field(1e-4, gt=0)
    ipopt_acceptable_tol: float = Field(1e-3, gt=0)
    ipopt_linear_solver: str = Field("mumps")

    return_latlon: bool = Field(True, description="Add optimal_latlon/centerline_latlon for mapping")
    origin_mode: str = Field("mean", description="mean | outer0")


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


def _read_boundaries_latlon_csv(path: Path) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    outer: List[Tuple[float, float]] = []
    inner: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = {"outer_lat", "outer_lon", "inner_lat", "inner_lon"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"boundaries.csv missing columns: {required}")
        for row in r:
            try:
                outer.append((float(row["outer_lat"]), float(row["outer_lon"])))
                inner.append((float(row["inner_lat"]), float(row["inner_lon"])))
            except Exception:
                continue
    return outer, inner


def _choose_origin(outer_ll: List[Tuple[float, float]], inner_ll: List[Tuple[float, float]], mode: str) -> Tuple[float, float]:
    if mode == "outer0":
        return outer_ll[0][0], outer_ll[0][1]
    # default mean
    all_pts = outer_ll + inner_ll
    lat0 = sum(p[0] for p in all_pts) / len(all_pts)
    lon0 = sum(p[1] for p in all_pts) / len(all_pts)
    return lat0, lon0


@router.post("/optimize_from_track_boundaries")
async def optimize_from_track_boundaries(req: OptimizeFromTrackRequest, token: dict = Depends(verify_token)):
    if optimize_trajectory_from_two_lines is None or OptimizeOptions is None:
        raise HTTPException(
            status_code=500,
            detail=f"Firmware optimizer unavailable (casadi/ipopt missing or import error): {_IMPORT_ERR}",
        )

    path = boundaries_csv_path(req.track_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="boundaries.csv not found for track")

    try:
        outer_ll, inner_ll = _read_boundaries_latlon_csv(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read boundaries.csv: {e}")

    if len(outer_ll) < 10 or len(inner_ll) < 10:
        raise HTTPException(status_code=400, detail="not enough points in boundaries.csv")

    N = int(req.n_points)
    outer_ll = resample_polyline(outer_ll, N)
    inner_ll = resample_polyline(inner_ll, N)

    lat0, lon0 = _choose_origin(outer_ll, inner_ll, req.origin_mode)

    left_xy = []
    right_xy = []
    for (lat, lon) in outer_ll:
        x, y = _latlon_to_xy_m(lat, lon, lat0, lon0)
        left_xy.append({"x": x, "y": y})
    for (lat, lon) in inner_ll:
        x, y = _latlon_to_xy_m(lat, lon, lat0, lon0)
        right_xy.append({"x": x, "y": y})

    opts = OptimizeOptions(
        n_points=N,
        ipopt_max_iter=req.ipopt_max_iter,
        ipopt_print_level=req.ipopt_print_level,
        ipopt_tol=req.ipopt_tol,
        ipopt_acceptable_tol=req.ipopt_acceptable_tol,
        ipopt_linear_solver=req.ipopt_linear_solver,
    )

    try:
        result = optimize_trajectory_from_two_lines(left_xy, right_xy, options=opts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    if req.return_latlon and isinstance(result, dict):
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
        result["track_id"] = req.track_id
        result["boundaries_path"] = str(path)

    return result