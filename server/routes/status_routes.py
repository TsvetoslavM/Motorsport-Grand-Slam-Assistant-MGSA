from fastapi import APIRouter, Depends
from ..auth import verify_token
from ..runtime import current_lap_id
from ..db import db_get_lap, db_list_laps
from ..config import DATA_DIR, DB_PATH, SERVER_VERSION
from ..ws import manager

router = APIRouter(tags=["status"])

@router.get("/")
async def root():
    lap_id = current_lap_id()
    return {
        "status": "online",
        "service": "MGSA Server",
        "version": SERVER_VERSION,
        "active_lap": lap_id,
        "total_laps": len(db_list_laps()),
    }

@router.get("/api/laps")
async def get_laps(token: dict = Depends(verify_token)):
    laps = db_list_laps()
    return {"total": len(laps), "laps": laps}

@router.get("/api/status")
async def get_status(token: dict = Depends(verify_token)):
    lap_id = current_lap_id()
    lap = db_get_lap(lap_id) if lap_id else None
    return {
        "recording": lap_id is not None,
        "current_lap": lap,
        "websocket_clients": len(manager.connections),
        "data_directory": str(DATA_DIR.absolute()),
        "db_path": str(DB_PATH.absolute()),
    }
