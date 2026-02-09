from pydantic import BaseModel, Field
from typing import List, Optional


class GPSPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: float = 0.0
    fix_quality: int = Field(0, ge=0, le=9)
    speed: float = Field(0.0, ge=0)
    timestamp: str

    hdop: Optional[float] = None
    sats: Optional[int] = None
    source: Optional[str] = None


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


class BuildBoundariesRequest(BaseModel):
    outer_lap_id: str
    inner_lap_id: str
    n_points: int = 800


class BoundarySample(BaseModel):
    time_s: float
    outer_lat: float
    outer_lon: float
    inner_lat: float
    inner_lon: float


class BoundariesUpload(BaseModel):
    samples: List[BoundarySample]

class StartLapRequest(BaseModel):
    track_name: str
    lap_type: str = "racing"  # "inner" | "outer" | "racing" | "driver"
    session_id: Optional[str] = None


class RacingPoint(BaseModel):
    time_s: float
    lat: float
    lon: float
    speed_kmh: float

class RacingLineUpload(BaseModel):
    kind: str = Field("optimal", description="Line type: 'optimal', 'driver', etc.")
    points: List[RacingPoint]

class BuildRacingLineFromLapRequest(BaseModel):
    lap_id: str
    kind: str = Field("driver", description="racing line kind, e.g. driver/racing/optimal")
