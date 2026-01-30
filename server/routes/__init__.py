from server.routes.auth_routes import router as auth_router
from server.routes.lap_routes import router as lap_router
from server.routes.gps_routes import router as gps_router
from server.routes.status_routes import router as status_router
from server.routes.track_routes import router as track_router
from server.analysis_routes import router as analysis_router

all_routers = [auth_router, lap_router, gps_router, status_router, track_router, analysis_router]