# server.py - MGSA Laptop Server (Field Test Ready)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import asyncio
import csv
import json
import logging
import os
import sqlite3
from pathlib import Path

# jwt is still used indirectly via server/auth.py; keep dependency installed.

# =========================
# CONFIG
# =========================
SERVER_VERSION = "1.1.0-field"
SECRET_KEY = os.getenv("MGSA_SECRET_KEY", "mgsa-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

DATA_DIR = Path("./mgsa_data")
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "mgsa.db"

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mgsa-server")

# =========================
# APP
# =========================
app = FastAPI(
    title="MGSA Server",
    description="Motorsport GPS Analysis - Laptop Server",
    version=SERVER_VERSION,
)

from server.auth import create_token, users, verify_token

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # за тестове; после стесни
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# EXTRA ROUTERS
# =========================
try:
    from server.trajectory_api import router as trajectory_router

    app.include_router(trajectory_router)
except Exception as e:
    logger.warning(f"Trajectory router not loaded: {e}")

# =========================
# DB HELPERS
# =========================
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


# =========================
# MODELS
# =========================
class GPSPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: float = 0.0
    fix_quality: int = Field(0, ge=0, le=9)  # 0=invalid, 1=GPS, 2=DGPS, 4=RTK fix...
    speed: float = Field(0.0, ge=0)  # m/s или km/h? -> приемаме m/s? (по-долу ще кажа)
    timestamp: str  # ISO 8601

    # OPTIONAL extras (не чупят клиента, ако ги няма)
    hdop: Optional[float] = None
    sats: Optional[int] = None
    source: Optional[str] = None  # "gpsd" / "nmea" / etc.


class LapInfo(BaseModel):
    lap_id: str
    track_name: str
    start_time: str
    end_time: Optional[str] = None
    lap_time: Optional[float] = None
    point_count: int = 0


class LoginRequest(BaseModel):
    username: str
    password: str


class StartLapRequest(BaseModel):
    track_name: str


# =========================
# AUTH
# =========================
# Moved to `server/auth.py` to avoid circular imports with extra routers.


# =========================
# WS MANAGER
# =========================
class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.connections.append(websocket)
        logger.info(f"WebSocket connected. Total={len(self.connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.connections:
                self.connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total={len(self.connections)}")

    async def broadcast(self, message: dict):
        dead: List[WebSocket] = []
        async with self._lock:
            conns = list(self.connections)
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()

# =========================
# RUNTIME STATE
# =========================
# текущата обиколка държим в runtime таблица, за да преживее рестарт
# runtime key: "current_lap_id"
# buffer държим само за status; истинските точки са в CSV
gps_buffer_count: int = 0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_lap_id() -> Optional[str]:
    return db_get_runtime("current_lap_id")


def set_current_lap_id(lap_id: Optional[str]):
    if lap_id is None:
        db_clear_runtime("current_lap_id")
    else:
        db_set_runtime("current_lap_id", lap_id)


def lap_csv_path(lap_id: str) -> Path:
    return DATA_DIR / f"{lap_id}.csv"


def ensure_csv_header(path: Path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["timestamp", "latitude", "longitude", "altitude", "speed", "fix_quality", "hdop", "sats", "source"]
            )


def append_point_csv(lap_id: str, point: GPSPoint):
    path = lap_csv_path(lap_id)
    ensure_csv_header(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                point.timestamp,
                point.latitude,
                point.longitude,
                point.altitude,
                point.speed,
                point.fix_quality,
                point.hdop,
                point.sats,
                point.source,
            ]
        )


def load_lap_points(lap_id: str) -> List[Dict[str, Any]]:
    path = lap_csv_path(lap_id)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(row)
    return out

def load_xy(lap_id: str):
    pts = load_lap_points(lap_id)
    out = []
    for r in pts:
        try:
            out.append((float(r["latitude"]), float(r["longitude"])))
        except Exception:
            pass
    return out

def resample_polyline(points, n: int):
    # points: [(lat, lon), ...] -> връща n точки равномерно по дължина
    if n <= 1 or len(points) < 2:
        return points[:]

    import math

    # кумулативна дължина
    d = [0.0]
    for i in range(1, len(points)):
        lat1, lon1 = points[i - 1]
        lat2, lon2 = points[i]
        dd = math.hypot(lat2 - lat1, lon2 - lon1)
        d.append(d[-1] + dd)

    total = d[-1]
    if total <= 0:
        return [points[0]] * n

    out = []
    step = total / (n - 1)
    j = 1
    for k in range(n):
        target = k * step
        while j < len(d) and d[j] < target:
            j += 1
        if j >= len(d):
            out.append(points[-1])
            continue
        i0 = j - 1
        i1 = j
        t0 = d[i0]
        t1 = d[i1]
        if t1 <= t0:
            out.append(points[i1])
            continue
        a = (target - t0) / (t1 - t0)
        lat0, lon0 = points[i0]
        lat1, lon1 = points[i1]
        out.append((lat0 + a * (lat1 - lat0), lon0 + a * (lon1 - lon0)))
    return out


def boundaries_csv_path(track_id: str) -> Path:
    return track_path(track_id) / "boundaries.csv"



# =========================
# ENDPOINTS
# =========================
@app.get("/")
async def root():
    lap_id = current_lap_id()
    return {
        "status": "online",
        "service": "MGSA Server",
        "version": SERVER_VERSION,
        "active_lap": lap_id,
        "total_laps": len(db_list_laps()),
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = users.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(req.username, user["role"])
    logger.info(f"User logged in: {req.username}")
    return {"access_token": token, "token_type": "bearer", "username": req.username, "role": user["role"]}


@app.post("/api/lap/start")
async def start_lap(req: StartLapRequest, token: dict = Depends(verify_token)):
    if current_lap_id():
        raise HTTPException(status_code=400, detail="Lap already in progress")

    lap_id = f"lap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = now_iso()

    db_insert_lap(lap_id, req.track_name, start_time)
    set_current_lap_id(lap_id)

    # създай CSV веднага
    ensure_csv_header(lap_csv_path(lap_id))

    logger.info(f"Started lap: {lap_id} track={req.track_name}")

    await manager.broadcast({"type": "lap_started", "lap_id": lap_id, "track_name": req.track_name, "start_time": start_time})
    return {"lap_id": lap_id, "status": "recording", "start_time": start_time}


@app.post("/api/lap/stop")
async def stop_lap(token: dict = Depends(verify_token)):
    lap_id = current_lap_id()
    if not lap_id:
        raise HTTPException(status_code=400, detail="No lap in progress")

    lap = db_get_lap(lap_id)
    if not lap:
        # ако DB е повреден, поне спираме runtime
        set_current_lap_id(None)
        raise HTTPException(status_code=500, detail="Lap state corrupted")

    end_time_dt = datetime.now(timezone.utc)
    start_time_dt = datetime.fromisoformat(lap["start_time"])
    lap_time = (end_time_dt - start_time_dt).total_seconds()

    # броим точки по DB (point_count)
    lap2 = db_get_lap(lap_id)
    point_count = int(lap2["point_count"]) if lap2 else 0

    db_update_lap_stop(lap_id, end_time_dt.isoformat(), lap_time, point_count)

    await manager.broadcast({"type": "lap_completed", "lap_id": lap_id, "lap_time": lap_time, "points": point_count})

    logger.info(f"Completed lap {lap_id}: {lap_time:.2f}s points={point_count}")

    set_current_lap_id(None)
    return {"lap_id": lap_id, "lap_time": lap_time, "points": point_count, "status": "saved"}


@app.post("/api/gps/point")
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


@app.get("/api/laps")
async def get_laps(token: dict = Depends(verify_token)):
    laps = db_list_laps()
    return {"total": len(laps), "laps": laps}


@app.get("/api/lap/{lap_id}")
async def get_lap_data(lap_id: str, token: dict = Depends(verify_token)):
    lap = db_get_lap(lap_id)
    if not lap:
        raise HTTPException(status_code=404, detail="Lap not found")

    points = load_lap_points(lap_id)
    return {"info": lap, "points": points}


@app.delete("/api/lap/{lap_id}")
async def delete_lap(lap_id: str, token: dict = Depends(verify_token)):
    lap = db_get_lap(lap_id)
    if not lap:
        raise HTTPException(status_code=404, detail="Lap not found")

    # delete CSV
    p = lap_csv_path(lap_id)
    if p.exists():
        p.unlink()

    # delete DB row
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM laps WHERE lap_id=?", (lap_id,))
    conn.commit()
    conn.close()

    # ако триеш текущата обиколка
    if current_lap_id() == lap_id:
        set_current_lap_id(None)

    return {"status": "deleted", "lap_id": lap_id}


@app.get("/api/status")
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

# class OptimizeRequest(BaseModel):
#     outer_lap_id: str
#     inner_lap_id: str
#     track_id: str


# @app.post("/api/trajectory/optimize")
# async def optimize(req: OptimizeRequest, token: dict = Depends(verify_token)):
#     outer = db_get_lap(req.outer_lap_id)
#     inner = db_get_lap(req.inner_lap_id)
#     if not outer or not inner:
#         raise HTTPException(status_code=404, detail="outer/inner lap not found")

#     outer_xy = load_xy(req.outer_lap_id)
#     inner_xy = load_xy(req.inner_lap_id)
#     if len(outer_xy) < 10 or len(inner_xy) < 10:
#         raise HTTPException(status_code=400, detail="not enough points")

#     ideal = [{"lat": lat, "lon": lon, "i": i} for i, (lat, lon) in enumerate(outer_xy)]

#     payload = {
#         "track_id": req.track_id,
#         "outer_lap_id": req.outer_lap_id,
#         "inner_lap_id": req.inner_lap_id,
#         "created_at": now_iso(),
#         "ideal": ideal,
#         "meta": {
#             "method": "mvp_outer_as_ideal",
#             "outer_points": len(outer_xy),
#             "inner_points": len(inner_xy),
#         },
#     }

#     (DATA_DIR / f"{req.track_id}_ideal.json").write_text(
#         json.dumps(payload), encoding="utf-8"
#     )

#     await manager.broadcast({"type": "ideal_ready", "track_id": req.track_id})
#     return payload

class BuildBoundariesRequest(BaseModel):
    track_id: str
    outer_lap_id: str
    inner_lap_id: str
    n_points: int = 800  # можеш 300..2000, според GPS честота/дължина


@app.post("/api/track/{track_id}/boundaries/build")
async def build_boundaries(track_id: str, req: BuildBoundariesRequest, token: dict = Depends(verify_token)):
    if req.track_id != track_id:
        raise HTTPException(status_code=400, detail="track_id mismatch")

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

    out_path = boundaries_csv_path(track_id)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "outer_lat", "outer_lon", "inner_lat", "inner_lon"])
        for i in range(n):
            o_lat, o_lon = outer_r[i]
            in_lat, in_lon = inner_r[i]
            w.writerow([i, o_lat, o_lon, in_lat, in_lon])

    meta = track_path(track_id) / "boundaries.meta.json"
    meta.write_text(
        json.dumps(
            {
                "updated": now_iso(),
                "track_id": track_id,
                "outer_lap_id": req.outer_lap_id,
                "inner_lap_id": req.inner_lap_id,
                "n_points": n,
                "path": str(out_path),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    await manager.broadcast({"type": "boundaries_ready", "track_id": track_id, "updated": now_iso()})
    return {"status": "ok", "track_id": track_id, "n_points": n, "path": str(out_path)}

@app.get("/api/track/{track_id}/boundaries")
async def download_boundaries(track_id: str, token: dict = Depends(verify_token)):
    out = boundaries_csv_path(track_id)
    if not out.exists():
        raise HTTPException(status_code=404, detail="boundaries.csv not found")
    return FileResponse(str(out), media_type="text/csv", filename="boundaries.csv")


# =========================
# WEBSOCKET
# =========================
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # keep-alive: клиентът може да праща "ping"
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)



# =========================
# LIFECYCLE
# =========================
@app.on_event("startup")
async def on_startup():
    db_init()
    logger.info("=" * 60)
    logger.info("MGSA Server Started")
    logger.info(f"Data dir: {DATA_DIR.absolute()}")
    logger.info(f"DB: {DB_PATH.absolute()}")
    logger.info(f"Active lap: {current_lap_id()}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("MGSA Server shutting down...")
    # close ws
    conns = list(manager.connections)
    for ws in conns:
        try:
            await ws.close()
        except Exception:
            pass


TRACK_DIR = DATA_DIR / "tracks"
TRACK_DIR.mkdir(exist_ok=True)

def track_path(track_id: str) -> Path:
    p = TRACK_DIR / track_id
    p.mkdir(parents=True, exist_ok=True)
    return p

from fastapi import UploadFile, File

@app.post("/api/track/{track_id}/ideal/upload")
async def upload_ideal(track_id: str, file: UploadFile = File(...), token: dict = Depends(verify_token)):
    # приемаме CSV
    data = await file.read()
    out = track_path(track_id) / "ideal.csv"
    out.write_bytes(data)

    meta = track_path(track_id) / "ideal.meta.json"
    meta.write_text(json.dumps({"updated": now_iso(), "size": len(data)}, ensure_ascii=False), encoding="utf-8")

    await manager.broadcast({"type": "ideal_updated", "track_id": track_id, "updated": now_iso()})
    return {"status": "ok", "track_id": track_id, "bytes": len(data), "path": str(out)}

from fastapi.responses import FileResponse

@app.get("/api/track/{track_id}/ideal")
async def download_ideal(track_id: str, token: dict = Depends(verify_token)):
    out = track_path(track_id) / "ideal.csv"
    if not out.exists():
        raise HTTPException(status_code=404, detail="Ideal trajectory not found")
    return FileResponse(str(out), media_type="text/csv", filename="ideal.csv")

@app.get("/api/track/{track_id}/ideal/meta")
async def ideal_meta(track_id: str, token: dict = Depends(verify_token)):
    meta = track_path(track_id) / "ideal.meta.json"
    if not meta.exists():
        return {"exists": False}
    return json.loads(meta.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("MGSA Server - Laptop Backend (Field Test Ready)")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print("Starting server on http://0.0.0.0:8000")
    print("CTRL+C to stop")
    print("=" * 60 + "\n")

    uvicorn.run(
        "server.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # за кола по-добре False
        log_level="info",
    )
