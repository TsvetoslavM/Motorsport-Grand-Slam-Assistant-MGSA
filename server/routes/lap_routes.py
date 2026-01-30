import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
import asyncio
from pathlib import Path
import json
import time

from ..auth import verify_token
from ..models import StartLapRequest
from ..db import (
    db_insert_lap, db_get_lap, db_update_lap_stop
)
from ..runtime import now_iso, current_lap_id, set_current_lap_id
from ..storage import ensure_csv_header, lap_csv_path, load_lap_points
from ..ws import manager
from ..auto_pipeline import register_completed_lap_and_maybe_run
from ..tracks import track_path
from ..routes.track_routes import build_racing_line_from_lap
from ..models import BuildRacingLineFromLapRequest
from ..analysis_routes import CompareRequest
from ..analysis_routes import compare as compare_endpoint


logger = logging.getLogger("mgsa-server")
router = APIRouter(prefix="/api/lap", tags=["laps"])

@router.post("/start")
async def start_lap(req: StartLapRequest, token: dict = Depends(verify_token)):
    if current_lap_id():
        raise HTTPException(status_code=400, detail="Lap already in progress")

    lap_id = f"lap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = now_iso()

    db_insert_lap(lap_id, req.track_name, start_time)
    set_current_lap_id(lap_id)
    ensure_csv_header(lap_csv_path(lap_id))
    meta_path = track_path(req.track_name) / f"{lap_id}.meta.json"
    meta_path.write_text(
        json.dumps({"lap_id": lap_id, "track_name": req.track_name, "lap_type": req.lap_type}, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Started lap: {lap_id} track={req.track_name}")

    await manager.broadcast({
    "type": "lap_started",
    "lap_id": lap_id,
    "track_name": req.track_name,
    "lap_type": req.lap_type,
    "start_time": start_time
    })
    return {"lap_id": lap_id, "status": "recording", "start_time": start_time}

@router.post("/stop")
async def stop_lap(token: dict = Depends(verify_token)):
    lap_id = current_lap_id()
    if not lap_id:
        raise HTTPException(status_code=400, detail="No lap in progress")

    lap = db_get_lap(lap_id)
    if not lap:
        set_current_lap_id(None)
        raise HTTPException(status_code=500, detail="Lap state corrupted")

    end_time_dt = datetime.now(timezone.utc)
    start_time_dt = datetime.fromisoformat(lap["start_time"])
    lap_time = (end_time_dt - start_time_dt).total_seconds()

    points = load_lap_points(lap_id)
    point_count = len(points)

    n_points = min(2000, max(50, point_count))

    db_update_lap_stop(lap_id, end_time_dt.isoformat(), lap_time, point_count)
    await manager.broadcast({"type": "lap_completed", "lap_id": lap_id, "lap_time": lap_time, "points": point_count})

    logger.info(f"Completed lap {lap_id}: {lap_time:.2f}s points={point_count}")
    track_name = lap.get("track_name")

    lap_type = None
    try:
        meta_path = track_path(track_name) / f"{lap_id}.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            lap_type = meta.get("lap_type")
    except Exception:
        lap_type = None

    set_current_lap_id(None)

    if track_name and lap_type in ("inner", "outer"):
        n_points = min(300, max(50, int(point_count)))
        t = asyncio.create_task(
            register_completed_lap_and_maybe_run(
                track_id=track_name,
                lap_type=lap_type,
                lap_id=lap_id,
                n_points=n_points,
            )
        )

        def _log_task_result(f):
            exc = f.exception()
            if exc:
                logger.exception("auto_pipeline failed", exc_info=exc)

        t.add_done_callback(_log_task_result)


    if track_name and lap_type == "driver":
        await build_racing_line_from_lap(
            track_id=track_name,
            req=BuildRacingLineFromLapRequest(lap_id=lap_id, kind="driver"),
            token=token,
        )

    if track_name and lap_type == "driver":
        asyncio.create_task(_auto_compare_driver(track_name, token, point_count))



    return {"lap_id": lap_id, "lap_time": lap_time, "points": point_count, "status": "saved"}

@router.get("/{lap_id}")
async def get_lap_data(lap_id: str, token: dict = Depends(verify_token)):
    lap = db_get_lap(lap_id)
    if not lap:
        raise HTTPException(status_code=404, detail="Lap not found")
    points = load_lap_points(lap_id)
    return {"info": lap, "points": points}

async def _wait_for_file(path: Path, timeout_s: float = 30.0, poll_s: float = 0.2) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists() and path.stat().st_size > 10:
            return True
        await asyncio.sleep(poll_s)
    return False


async def _auto_compare_driver(track_id: str, token: dict, point_count: int) -> None:
    root = track_path(track_id)
    optimal_json = root / "optimal.json"
    driver_csv = root / "racing_driver.csv"

    ok_opt = await _wait_for_file(optimal_json, timeout_s=60.0)
    ok_drv = await _wait_for_file(driver_csv, timeout_s=10.0)

    if not ok_drv:
        await manager.broadcast({"type": "driver_vs_optimal_failed", "track_id": track_id, "error": "missing racing_driver.csv"})
        return
    if not ok_opt:
        await manager.broadcast({"type": "driver_vs_optimal_failed", "track_id": track_id, "error": "optimal not ready (optimal.json missing)"})
        return

    N = min(900, max(100, int(point_count)))

    payload = await compare_endpoint(
        track_id,
        CompareRequest(driver_kind="driver", n_points=N, segment_len_m=20.0, ipopt_print_level=0),
        token=token,
    )

    out_path = root / "compare_driver_vs_optimal.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    await manager.broadcast({
        "type": "driver_vs_optimal_ready",
        "track_id": track_id,
        "n_points": N,
        "artifacts": {
            "compare_json": f"/api/track/{track_id}/compare_driver_vs_optimal.json",
        },
        "summary": payload.get("stats", {}),
    })
