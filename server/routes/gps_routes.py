from fastapi import APIRouter, Depends
from ..auth import verify_token
from ..models import GPSPoint
from ..runtime import current_lap_id
from ..storage import append_point_csv
from ..db import db_inc_point_count
from ..ws import manager

router = APIRouter(prefix="/api/gps", tags=["gps"])

gps_buffer_count: int = 0

@router.post("/point")
async def receive_gps(point: GPSPoint, token: dict = Depends(verify_token)):
    global gps_buffer_count
    lap_id = current_lap_id()

    recording = lap_id is not None
    if recording:
        append_point_csv(lap_id, point)
        db_inc_point_count(lap_id, 1)
        gps_buffer_count += 1

    await manager.broadcast({"type": "gps_point", "data": point.model_dump(), "recording": recording, "lap_id": lap_id})
    return {"status": "received", "recording": recording, "lap_id": lap_id}
