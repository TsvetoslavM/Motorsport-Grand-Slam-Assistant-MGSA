import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class Mode(str, Enum):
    IDLE = "idle"
    RECORD_OUTER = "record_outer"
    RECORD_INNER = "record_inner"
    RACE = "race"

@dataclass
class StartFinishZone:
    lat: float
    lon: float
    radius_m: float
    ts: str

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class MgsaStateMachine:
    def __init__(self, cfg: dict, io):
        self.cfg = cfg
        self.io = io
        self.mode = Mode.IDLE
        self.zone = None

        self.paths = cfg["paths"]
        self.start_finish_cfg = cfg["start_finish"]
        self.zone_path = Path(self.paths["current_start_finish_file"])
        self.zone_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_zone_if_exists()

        self.lap_cfg = cfg.get("lap_crossing", {})
        self._inside_zone = False
        self._lap_active = False
        self._lap_start_t = 0.0
        self._lap_count = 0
        self._last_cross_t = 0.0

    def _load_zone_if_exists(self):
        if not self.zone_path.exists():
            return
        try:
            d = json.loads(self.zone_path.read_text(encoding="utf-8"))
            self.zone = StartFinishZone(
                lat=float(d["lat"]),
                lon=float(d["lon"]),
                radius_m=float(d["radius_m"]),
                ts=str(d.get("ts", "")),
            )
        except Exception:
            self.zone = None

    def _save_zone(self):
        if not self.zone:
            return
        payload = {
            "lat": self.zone.lat,
            "lon": self.zone.lon,
            "radius_m": self.zone.radius_m,
            "ts": self.zone.ts,
        }
        tmp = self.zone_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.zone_path)

    def set_mode(self, mode: Mode):
        if self.mode == mode:
            return
        prev = self.mode
        if prev in (Mode.RECORD_OUTER, Mode.RECORD_INNER) and mode == Mode.IDLE:
            try:
                self.io.finalize_recording()
            except Exception:
                pass

        self.mode = mode
        if mode != self.mode:
            self._inside_zone = False
            self._lap_active = False
            self._lap_start_t = 0.0
            self._lap_count = 0
            self._last_cross_t = 0.0

        self.io.on_mode_changed(self.mode)

    def handle_command(self, cmd: dict):
        t = cmd.get("type")
        if t == "mode":
            m = str(cmd.get("mode", "")).strip().lower()
            if m == "record_outer":
                self.set_mode(Mode.RECORD_OUTER)
            elif m == "record_inner":
                self.set_mode(Mode.RECORD_INNER)
            elif m == "race":
                self.set_mode(Mode.RACE)
            elif m == "idle":
                self.set_mode(Mode.IDLE)
            return

        if t == "start_finish":
            action = str(cmd.get("action", "")).strip().lower()
            if action == "set":
                self._handle_set_start_finish()
            return

        if cmd.get("type") == "trajectory":
            action = cmd.get("action")
            if action == "load":
                force = bool(cmd.get("force", False))
                self.io.trajectory_load(force=force)
            return

    def _handle_set_start_finish(self):
        max_age = float(self.start_finish_cfg.get("max_gps_age_s", 1.0))

        if self.start_finish_cfg.get("require_rtk_fix", False):
            if not self.io.gps_has_good_fix():
                self.io.start_finish_error()
                return
        else:
            if not self.io.gps_has_good_fix():
                self.io.start_finish_error()
                return

        try:
            lat, lon = self.io.gps_get_latlon_wait(timeout_s=2.0)
        except Exception:
            self.io.start_finish_error()
            return

        radius = float(self.start_finish_cfg.get("zone_radius_m", 1.0))
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        self.zone = StartFinishZone(lat=lat, lon=lon, radius_m=radius, ts=ts)
        self._save_zone()
        self.io.start_finish_confirm()

    def tick(self):
        if self.mode == Mode.RECORD_OUTER:
            self._lap_crossing_tick("outer")
        elif self.mode == Mode.RECORD_INNER:
            self._lap_crossing_tick("inner")
        elif self.mode == Mode.RACE:
            self.io.race_tick(self.zone)
        else:
            self.io.idle_tick()

    def _lap_crossing_tick(self, lap_type: str):
        if not self.lap_cfg.get("enabled", True):
            self.io.record_tick(lap_type)
            return

        if not self.zone:
            self.io.record_tick(lap_type)
            return

        try:
            lat, lon = self.io.gps_get_latlon()
        except Exception:
            self.io.record_tick(lap_type)
            return

        speed = float(self.io.gps_get_speed_kmh())
        enter_m = float(self.lap_cfg.get("enter_zone_m", 1.0))
        min_speed = float(self.lap_cfg.get("min_speed_kmh", 5.0))
        min_between = float(self.lap_cfg.get("min_time_between_crossings_s", 10.0))
        min_lap_time = float(self.lap_cfg.get("min_lap_time_s", 20.0))
        stop_after = int(self.lap_cfg.get("stop_after_laps", 1))

        d = haversine_m(lat, lon, self.zone.lat, self.zone.lon)
        in_zone = d <= enter_m

        now = time.time()

        crossed_in = (in_zone and not self._inside_zone)
        self._inside_zone = in_zone

        # always record points (both before and during the lap)
        self.io.record_tick(lap_type)

        if not crossed_in:
            return

        # protection against noise around the line
        if (now - self._last_cross_t) < min_between:
            return
        self._last_cross_t = now

        # is moving
        if speed < min_speed:
            return

        if not self._lap_active:
            # START lap
            self._lap_active = True
            self._lap_start_t = now
            self.io.record_event("lap_start", {"lap_index": self._lap_count + 1})
            return

        # if exist a lap - END lap
        lap_dt = now - self._lap_start_t
        if lap_dt < min_lap_time:
            return

        self._lap_count += 1
        self.io.record_event("lap_end", {"lap_index": self._lap_count, "lap_time_s": lap_dt})

        if self._lap_count >= stop_after:
            self._lap_active = False
            self.set_mode(Mode.IDLE)
            return

        # many laps
        self._lap_active = True
        self._lap_start_t = now
        self.io.record_event("lap_start", {"lap_index": self._lap_count + 1})
