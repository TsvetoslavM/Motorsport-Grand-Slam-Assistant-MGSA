from .auth_routes import router as auth_router
from .lap_routes import router as lap_router
from .gps_routes import router as gps_router
from .status_routes import router as status_router
from .track_routes import router as track_router
from server.analysis_routes import router as analysis_router

all_routers = [auth_router, lap_router, gps_router, status_router, track_router, analysis_router]