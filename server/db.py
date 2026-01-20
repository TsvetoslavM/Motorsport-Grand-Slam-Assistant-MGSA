import sqlite3
from typing import Optional, Dict, Any, List
from .config import DB_PATH

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS laps (
            lap_id TEXT PRIMARY KEY,
            track_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            lap_time REAL,
            point_count INTEGER DEFAULT 0
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    conn.commit()
    conn.close()

def db_set_runtime(key: str, value: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runtime(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()

def db_get_runtime(key: str) -> Optional[str]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT value FROM runtime WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None

def db_clear_runtime(key: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM runtime WHERE key=?", (key,))
    conn.commit()
    conn.close()

def db_insert_lap(lap_id: str, track_name: str, start_time: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO laps(lap_id, track_name, start_time, point_count) VALUES(?,?,?,0)",
        (lap_id, track_name, start_time),
    )
    conn.commit()
    conn.close()

def db_update_lap_stop(lap_id: str, end_time: str, lap_time: float, point_count: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE laps
        SET end_time=?, lap_time=?, point_count=?
        WHERE lap_id=?
        """,
        (end_time, lap_time, point_count, lap_id),
    )
    conn.commit()
    conn.close()

def db_inc_point_count(lap_id: str, delta: int = 1):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "UPDATE laps SET point_count = COALESCE(point_count,0) + ? WHERE lap_id=?",
        (delta, lap_id),
    )
    conn.commit()
    conn.close()

def db_get_lap(lap_id: str) -> Optional[Dict[str, Any]]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM laps WHERE lap_id=?", (lap_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def db_list_laps() -> List[Dict[str, Any]]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM laps ORDER BY start_time DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_delete_lap(lap_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM laps WHERE lap_id=?", (lap_id,))
    conn.commit()
    conn.close()
