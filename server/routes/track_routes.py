import csv
import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from ..auth import verify_token
from ..models import BuildBoundariesRequest, BoundariesUpload, RacingLineUpload, BuildRacingLineFromLapRequest
from ..db import db_get_lap
from ..storage import load_xy, load_lap_points
from ..tracks import (
    track_path, boundaries_csv_path, resample_polyline,
    write_boundaries_csv, write_boundaries_meta,
    safe_kind
)
from ..runtime import now_iso
from ..ws import manager
from datetime import datetime


router = APIRouter(prefix="/api/track", tags=["tracks"])

@router.post("/{track_id}/boundaries/build")
async def build_boundaries(track_id: str, req: BuildBoundariesRequest, token: dict = Depends(verify_token)):
    outer = db_get_lap(req.outer_lap_id)
    inner = db_get_lap(req.inner_lap_id)
    if not outer or not inner:
        raise HTTPException(status_code=404, detail="outer/inner lap not found")

    outer_xy = load_xy(req.outer_lap_id)
    inner_xy = load_xy(req.inner_lap_id)
    if len(outer_xy) < 10 or len(inner_xy) < 10:
        raise HTTPException(status_code=400, detail="not enough points")

    n = int(req.n_points)
    if n < 10:
        n = 10
    if n > 5000:
        n = 5000

    outer_r = resample_polyline(outer_xy, n)
    inner_r = resample_polyline(inner_xy, n)

    out_path = write_boundaries_csv(track_id, outer_r, inner_r)
    write_boundaries_meta(
        track_id,
        {
            "outer_lap_id": req.outer_lap_id,
            "inner_lap_id": req.inner_lap_id,
            "n_points": n,
            "path": str(out_path),
        },
    )

    await manager.broadcast({"type": "boundaries_ready", "track_id": track_id, "updated": now_iso()})
    return {"status": "ok", "track_id": track_id, "n_points": n, "path": str(out_path)}

@router.get("/{track_id}/boundaries")
async def download_boundaries(track_id: str, token: dict = Depends(verify_token)):
    out = boundaries_csv_path(track_id)
    if not out.exists():
        raise HTTPException(status_code=404, detail="boundaries.csv not found")
    return FileResponse(str(out), media_type="text/csv", filename="boundaries.csv")

@router.post("/{track_id}/boundaries/upload_json")
async def upload_boundaries_json(track_id: str, req: BoundariesUpload, token: dict = Depends(verify_token)):
    if not req.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    out_path = boundaries_csv_path(track_id)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "outer_lat", "outer_lon", "inner_lat", "inner_lon"])
        for s in req.samples:
            w.writerow([s.time_s, s.outer_lat, s.outer_lon, s.inner_lat, s.inner_lon])

    meta = track_path(track_id) / "boundaries.meta.json"
    meta.write_text(
        json.dumps(
            {"updated": now_iso(), "track_id": track_id, "source": "json_upload", "samples": len(req.samples), "path": str(out_path)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    await manager.broadcast({"type": "boundaries_updated", "track_id": track_id, "updated": now_iso()})
    return {"status": "ok", "track_id": track_id, "samples": len(req.samples), "path": str(out_path)}

@router.post("/{track_id}/ideal/upload")
async def upload_ideal(track_id: str, file: UploadFile = File(...), token: dict = Depends(verify_token)):
    data = await file.read()
    out = track_path(track_id) / "ideal.csv"
    out.write_bytes(data)

    meta = track_path(track_id) / "ideal.meta.json"
    meta.write_text(json.dumps({"updated": now_iso(), "size": len(data)}, ensure_ascii=False), encoding="utf-8")

    await manager.broadcast({"type": "ideal_updated", "track_id": track_id, "updated": now_iso()})
    return {"status": "ok", "track_id": track_id, "bytes": len(data), "path": str(out)}

@router.get("/{track_id}/ideal")
async def download_ideal(track_id: str, token: dict = Depends(verify_token)):
    out = track_path(track_id) / "ideal.csv"
    if not out.exists():
        raise HTTPException(status_code=404, detail="Ideal trajectory not found")
    return FileResponse(str(out), media_type="text/csv", filename="ideal.csv")

@router.get("/{track_id}/ideal/meta")
async def ideal_meta(track_id: str, token: dict = Depends(verify_token)):
    meta = track_path(track_id) / "ideal.meta.json"
    if not meta.exists():
        return {"exists": False}
    return json.loads(meta.read_text(encoding="utf-8"))

@router.post("/{track_id}/racing_line/upload")
async def upload_racing_line(track_id: str, req: RacingLineUpload, token: dict = Depends(verify_token)):
    if not req.points:
        raise HTTPException(status_code=400, detail="No points provided")

    kind = safe_kind(req.kind)
    out = track_path(track_id) / f"racing_{kind}.csv"

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "lat", "lon", "speed_kmh"])
        for p in req.points:
            w.writerow([p.time_s, p.lat, p.lon, p.speed_kmh])

    meta = track_path(track_id) / f"racing_{kind}.meta.json"
    meta.write_text(
        json.dumps(
            {"updated": now_iso(), "track_id": track_id, "kind": kind, "points": len(req.points), "path": str(out)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    await manager.broadcast({"type": "racing_line_updated", "track_id": track_id, "kind": kind, "updated": now_iso()})
    return {"status": "ok", "track_id": track_id, "kind": kind, "points": len(req.points), "path": str(out)}

@router.get("/{track_id}/racing_line/{kind}")
async def download_racing_line(track_id: str, kind: str, token: dict = Depends(verify_token)):
    kind = safe_kind(kind)
    out = track_path(track_id) / f"racing_{kind}.csv"
    if not out.exists():
        raise HTTPException(status_code=404, detail="Racing line not found")
    return FileResponse(str(out), media_type="text/csv", filename=f"racing_{kind}.csv")

@router.get("/{track_id}/racing_line/{kind}/meta")
async def racing_line_meta(track_id: str, kind: str, token: dict = Depends(verify_token)):
    kind = safe_kind(kind)
    meta = track_path(track_id) / f"racing_{kind}.meta.json"
    if not meta.exists():
        return {"exists": False}
    return json.loads(meta.read_text(encoding="utf-8"))

def _parse_iso(ts: str) -> datetime | None:
    try:
        # handles "2026-01-21T10:55:06.529000+00:00"
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None
        
@router.post("/{track_id}/racing_line/build_from_lap")
async def build_racing_line_from_lap(track_id: str, req: BuildRacingLineFromLapRequest, token: dict = Depends(verify_token)):
    lap = db_get_lap(req.lap_id)
    if not lap:
        raise HTTPException(status_code=404, detail="lap not found")

    pts = load_lap_points(req.lap_id)
    if len(pts) < 10:
        raise HTTPException(status_code=400, detail="not enough points in lap")

    kind = safe_kind(req.kind)
    out = track_path(track_id) / f"racing_{kind}.csv"

    t0 = None
    time_s_list = []

    for i, r in enumerate(pts):
        ts = r.get("timestamp")
        # build time_s from timestamp if possible, else fallback to index
        dt = _parse_iso(ts) if isinstance(ts, str) else None
        if dt is None:
            time_s_list.append(float(i))
            continue
        if t0 is None:
            t0 = dt
        time_s_list.append((dt - t0).total_seconds())

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "lat", "lon", "speed_kmh"])
        for i, r in enumerate(pts):
            try:
                lat = float(r["latitude"])
                lon = float(r["longitude"])
                speed_mps = float(r.get("speed", 0.0))  # stored as m/s in lap csv
                speed_kmh = speed_mps * 3.6
                w.writerow([float(time_s_list[i]), lat, lon, speed_kmh])
            except Exception:
                continue

    meta = track_path(track_id) / f"racing_{kind}.meta.json"
    meta.write_text(
        json.dumps(
            {
                "updated": now_iso(),
                "track_id": track_id,
                "kind": kind,
                "source": "lap_build",
                "source_lap_id": req.lap_id,
                "points": len(pts),
                "path": str(out),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    await manager.broadcast({"type": "racing_line_updated", "track_id": track_id, "kind": kind, "updated": now_iso()})
    return {"status": "ok", "track_id": track_id, "kind": kind, "points": len(pts), "path": str(out)}

@router.get("/{track_id}/optimal.json")
async def get_optimal_json(track_id: str, token: dict = Depends(verify_token)):
    p = track_path(track_id) / "optimal.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="optimal.json not found")

    await manager.broadcast({
        "type": "optimal_ready",
        "track_id": track_id,
        "updated": now_iso(),
        "artifacts": {
            "optimal_json": f"/api/track/{track_id}/optimal.json",
            "optimal_latlon_csv": f"/api/track/{track_id}/optimal_latlon.csv",
            "boundaries_csv": f"/api/track/{track_id}/boundaries",
        }
    })
    return FileResponse(str(p), media_type="application/json", filename="optimal.json")


@router.get("/{track_id}/optimal_latlon.csv")
async def get_optimal_latlon_csv(track_id: str, token: dict = Depends(verify_token)):
    p = track_path(track_id) / "optimal_latlon.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="optimal_latlon.csv not found")

    await manager.broadcast({
        "type": "optimal_ready",
        "track_id": track_id,
        "updated": now_iso(),
        "artifacts": {
            "optimal_json": f"/api/track/{track_id}/optimal.json",
            "optimal_latlon_csv": f"/api/track/{track_id}/optimal_latlon.csv",
            "boundaries_csv": f"/api/track/{track_id}/boundaries",
        }
    })
    return FileResponse(str(p), media_type="text/csv", filename="optimal_latlon.csv")

@router.get("/{track_id}/compare_driver_vs_optimal.json")
async def get_compare_driver_vs_optimal(track_id: str, token: dict = Depends(verify_token)):
    p = track_path(track_id) / "compare_driver_vs_optimal.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="compare json not found")
    return FileResponse(str(p), media_type="application/json", filename="compare_driver_vs_optimal.json")

