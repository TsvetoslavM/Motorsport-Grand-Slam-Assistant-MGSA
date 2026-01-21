import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


from .config import SERVER_VERSION, DATA_DIR, DB_PATH
from .db import db_init
from .runtime import current_lap_id
from .ws import manager
from .routes import all_routers

from fastapi.staticfiles import StaticFiles
from pathlib import Path


logger = logging.getLogger("mgsa-server")

def create_app() -> FastAPI:
    app = FastAPI(
        title="MGSA Server",
        description="Motorsport GPS Analysis - Laptop Server",
        version=SERVER_VERSION,
    )

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for r in all_routers:
        app.include_router(r)

    try:
        from .trajectory_api import router as trajectory_router
        app.include_router(trajectory_router)
    except Exception as e:
        logger.warning(f"Trajectory router not loaded: {e}")

    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(websocket)

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
        conns = list(manager.connections)
        for ws in conns:
            try:
                await ws.close()
            except Exception:
                pass

    @app.get("/compare")
    async def compare_page():
        p = static_dir / "compare.html"
        if not p.exists():
            raise HTTPException(status_code=404, detail="compare.html not found")
        return FileResponse(str(p), media_type="text/html")
    return app
