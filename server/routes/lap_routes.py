import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from ..auth import verify_token
from ..models import StartLapRequest
from ..db import (
    db_insert_lap, db_get_lap, db_update_lap_stop, db_list_laps
)
from ..runtime import now_iso, current_lap_id, set_current_lap_id
from ..storage import ensure_csv_header, lap_csv_path, load_lap_points
from ..ws import manager

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

    logger.info(f"Started lap: {lap_id} track={req.track_name}")

    await manager.broadcast({"type": "lap_started", "lap_id": lap_id, "track_name": req.track_name, "start_time": start_time})
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

    db_update_lap_stop(lap_id, end_time_dt.isoformat(), lap_time, point_count)
    await manager.broadcast({"type": "lap_completed", "lap_id": lap_id, "lap_time": lap_time, "points": point_count})

    logger.info(f"Completed lap {lap_id}: {lap_time:.2f}s points={point_count}")
    set_current_lap_id(None)
    return {"lap_id": lap_id, "lap_time": lap_time, "points": point_count, "status": "saved"}

@router.get("/{lap_id}")
async def get_lap_data(lap_id: str, token: dict = Depends(verify_token)):
    lap = db_get_lap(lap_id)
    if not lap:
        raise HTTPException(status_code=404, detail="Lap not found")
    points = load_lap_points(lap_id)
    return {"info": lap, "points": points}
