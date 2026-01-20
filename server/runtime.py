from typing import Optional
from datetime import datetime, timezone
from .db import db_get_runtime, db_set_runtime, db_clear_runtime

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def current_lap_id() -> Optional[str]:
    return db_get_runtime("current_lap_id")

def set_current_lap_id(lap_id: Optional[str]):
    if lap_id is None:
        db_clear_runtime("current_lap_id")
    else:
        db_set_runtime("current_lap_id", lap_id)
