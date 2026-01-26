import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class OptimalSample:
    """Single sample from optimal trajectory."""

    s_m: float
    lat: float
    lon: float
    x_m: float
    y_m: float
    v_opt_mps: float
    a_lat_opt_mps2: float


@dataclass
class RaceFeedback:
    """Result of one race-mode comparison step."""

    ok: bool
    reason: Optional[str]
    has_gps: bool
    gps_fix_ok: bool

    s_idx: int
    s_m: float
    d_lat_m: float

    v_meas_mps: float
    v_opt_mps: float
    delta_v_mps: float
    delta_v_kmh: float

    a_lat_meas_mps2: float
    a_lat_opt_mps2: float
    delta_a_lat_mps2: float

    status: str  # "good" | "too_slow" | "too_fast" | "gps_bad" | "no_trajectory"

    # Simple LED/HUD intents (interpreted by outer code)
    led_mode: str  # "off" | "solid" | "blink"
    led_color: str  # "green" | "blue" | "red" | "yellow"
    led_on_ms: int
    led_off_ms: int

    hud_text: str
    hud_arrow: Optional[str]  # "left" | "right" | None
    hud_warning: Optional[str]


def _latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """
    Convert WGS84 lat/lon to local tangent-plane coordinates (meters)
    using simple equirectangular approximation around (lat0, lon0).
    Good enough for small tracks (few km).
    """
    R = 6371000.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = R * dlon * math.cos(math.radians(lat0))
    y = R * dlat
    return x, y


class RaceModeController:
    """
    Pure comparison & feedback logic for race mode.

    Responsibilities:
      - Load optimal trajectory (lat, lon, v_opt, optional a_lat_opt)
      - Map-match current GPS point to nearest point on trajectory
      - Compute delta_v, lateral deviation, optional delta_a_lat
      - Decide LED/HUD intents
    """

    def __init__(self, cfg: dict, paths: dict):
        self.cfg = cfg
        self.paths = paths

        self.samples: List[OptimalSample] = []
        self._track_id: Optional[str] = None
        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None

        # Smoothing state
        self._dv_ema: Optional[float] = None
        self._dlat_ema: Optional[float] = None
        self._da_lat_ema: Optional[float] = None

        # Config
        rm = cfg.get("race_mode", {})
        sc = rm.get("speed_compare", {})
        self.tol_kmh: float = float(sc.get("tolerance_kmh", 10.0))
        self.map_match_max_dist_m: float = float(
            sc.get("map_match", {}).get("max_distance_m", 5.0)
        )

        led_logic = rm.get("led_logic", {})
        self.led_ok = led_logic.get("ok", {"color": "green", "pattern": "solid"})
        self.led_over = led_logic.get("over_speed", {"color": "red", "pattern": "blink"})
        self.led_under = led_logic.get("under_speed", {"color": "blue", "pattern": "blink"})

        # Defaults if not present
        for d in (self.led_ok, self.led_over, self.led_under):
            d.setdefault("on_ms", 120)
            d.setdefault("off_ms", 120)

        # How aggressively to smooth signals (0..1)
        self.alpha_dv = 0.3
        self.alpha_dlat = 0.3
        self.alpha_da_lat = 0.3

        # Map-matching state
        self._last_idx: Optional[int] = None
        self._last_best_d2: float = float("inf")

    # ------------------------------------------------------------------
    # Trajectory loading
    # ------------------------------------------------------------------
    def ensure_loaded(self, track_id: Optional[str]) -> bool:
        """
        Ensure that optimal trajectory for given track_id is loaded.
        Returns True if loaded and usable.
        """
        if not track_id:
            return False
        if self._track_id == track_id and self.samples:
            return True

        tracks_dir = Path(self.paths["tracks_dir"])
        ideal_path = tracks_dir / track_id / "ideal.csv"
        if not ideal_path.exists():
            self.samples = []
            self._track_id = None
            return False

        try:
            self.samples = self._load_ideal_csv(ideal_path)
            self._track_id = track_id
            # Reset smoothing when trajectory changes
            self._dv_ema = None
            self._dlat_ema = None
            self._da_lat_ema = None
            return bool(self.samples)
        except Exception:
            self.samples = []
            self._track_id = None
            return False

    def _load_ideal_csv(self, path: Path) -> List[OptimalSample]:
        """
        Supported columns (header, order does not matter):
          - lat, lon (required)
          - speed_kmh OR v_opt_kmh OR v_opt_mps (optional)
          - a_lat_mps2 OR a_lat (optional)
          - time_s (ignored here, used only for ordering)
        """
        samples: List[OptimalSample] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            header = [h.strip().lower() for h in header]
            idx = {name: i for i, name in enumerate(header)}

            # Strict schema (server racing line upload):
            #   time_s, lat, lon, speed_kmh
            if {"time_s", "lat", "lon", "speed_kmh"}.issubset(idx.keys()):
                i_speed = idx["speed_kmh"]
            else:
                # Backwards-compatible: more permissive names
                if "lat" not in idx or "lon" not in idx:
                    # Not an optimal racing line file we know how to use
                    return []
                i_speed = (
                    idx.get("speed_kmh")
                    or idx.get("v_opt_kmh")
                    or idx.get("v_opt_mps")
                )

            i_a_lat = idx.get("a_lat_mps2") or idx.get("a_lat")

            lat0 = None
            lon0 = None
            s_acc = 0.0
            prev_xy = None

            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    lat = float(row[idx["lat"]])
                    lon = float(row[idx["lon"]])
                except Exception:
                    continue

                if lat0 is None:
                    lat0, lon0 = lat, lon
                    self._lat0, self._lon0 = lat0, lon0

                x, y = _latlon_to_xy_m(lat, lon, lat0, lon0)

                if prev_xy is not None:
                    dx = x - prev_xy[0]
                    dy = y - prev_xy[1]
                    ds = math.hypot(dx, dy)
                    s_acc += ds
                prev_xy = (x, y)

                v_opt_mps = 0.0
                if i_speed is not None and i_speed < len(row):
                    try:
                        v_val = float(row[i_speed])
                        # Heuristic: if looks like km/h, convert, else keep m/s.
                        # Most optimizers will provide km/h.
                        if v_val > 40.0:
                            v_opt_mps = v_val / 3.6
                        else:
                            v_opt_mps = v_val
                    except Exception:
                        v_opt_mps = 0.0

                a_lat_opt = 0.0
                if i_a_lat is not None and i_a_lat < len(row):
                    try:
                        a_lat_opt = float(row[i_a_lat])
                    except Exception:
                        a_lat_opt = 0.0

                samples.append(
                    OptimalSample(
                        s_m=s_acc,
                        lat=lat,
                        lon=lon,
                        x_m=x,
                        y_m=y,
                        v_opt_mps=v_opt_mps,
                        a_lat_opt_mps2=a_lat_opt,
                    )
                )

        return samples

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(
        self,
        lat: Optional[float],
        lon: Optional[float],
        v_meas_mps: float,
        a_lat_meas_mps2: float,
        a_lon_meas_mps2: float,  # currently unused, kept for future
        gps_fix_ok: bool,
    ) -> RaceFeedback:
        """
        Compute comparison & feedback for one tick.

        IMU values may be zero / dummy if IMU pipeline is not yet wired.
        """
        if not self.samples:
            return self._feedback_no_trajectory()

        if lat is None or lon is None:
            return self._feedback_gps_bad(
                v_meas_mps=v_meas_mps,
                a_lat_meas_mps2=a_lat_meas_mps2,
                gps_fix_ok=gps_fix_ok,
            )

        # Map-match: nearest point in XY
        if self._lat0 is None or self._lon0 is None:
            self._lat0, self._lon0 = self.samples[0].lat, self.samples[0].lon

        x, y = _latlon_to_xy_m(lat, lon, self._lat0, self._lon0)

        # Map-match with windowed search around previous index
        n = len(self.samples)
        window = 200  # +/- points
        if self._last_idx is not None and 0 <= self._last_idx < n:
            s_idx, s_m, d_lat_m, best_d2 = self._nearest_point_windowed(
                x, y, self._last_idx, window
            )
            # If quality degraded, fall back to full search and re-lock
            if best_d2 > self.map_match_max_dist_m ** 2:
                s_idx, s_m, d_lat_m, best_d2 = self._nearest_point_full(x, y)
        else:
            s_idx, s_m, d_lat_m, best_d2 = self._nearest_point_full(x, y)

        # Enforce monotonic progress along trajectory (except simple wrap at start/finish)
        if self._last_idx is not None:
            last = self._last_idx
            if s_idx < last:
                # Allow wrap-around only if we jumped from "end of lap" to "start of lap"
                if not (last > 0.8 * n and s_idx < 0.2 * n):
                    s_idx = last
                    s_m = self.samples[s_idx].s_m
                    d_lat_m = self._lateral_deviation_for_index(s_idx, x, y)

        self._last_idx = s_idx
        self._last_best_d2 = best_d2

        sample = self.samples[s_idx]
        v_opt = sample.v_opt_mps
        a_lat_opt = sample.a_lat_opt_mps2

        delta_v_mps = v_meas_mps - v_opt
        delta_v_kmh = delta_v_mps * 3.6
        delta_a_lat = a_lat_meas_mps2 - a_lat_opt

        # Smoothing
        delta_v_kmh_s = self._smooth("dv", delta_v_kmh, self.alpha_dv)
        d_lat_m_s = self._smooth("dlat", d_lat_m, self.alpha_dlat)
        delta_a_lat_s = self._smooth("da_lat", delta_a_lat, self.alpha_da_lat)

        # Decision logic
        status, led_cfg, hud_arrow, hud_warning = self._decide(
            delta_v_kmh_s, d_lat_m_s, gps_fix_ok
        )

        hud_text = f"{delta_v_kmh_s:+.1f}"

        return RaceFeedback(
            ok=True,
            reason=None,
            has_gps=True,
            gps_fix_ok=gps_fix_ok,
            s_idx=s_idx,
            s_m=s_m,
            d_lat_m=d_lat_m_s,
            v_meas_mps=v_meas_mps,
            v_opt_mps=v_opt,
            delta_v_mps=delta_v_mps,
            delta_v_kmh=delta_v_kmh_s,
            a_lat_meas_mps2=a_lat_meas_mps2,
            a_lat_opt_mps2=a_lat_opt,
            delta_a_lat_mps2=delta_a_lat_s,
            status=status,
            led_mode=led_cfg.get("pattern", "solid"),
            led_color=led_cfg.get("color", "green"),
            led_on_ms=int(led_cfg.get("on_ms", 120)),
            led_off_ms=int(led_cfg.get("off_ms", 120)),
            hud_text=hud_text,
            hud_arrow=hud_arrow,
            hud_warning=hud_warning,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _nearest_point_full(self, x: float, y: float) -> Tuple[int, float, float, float]:
        """
        Full nearest-neighbor search over the trajectory.
        Returns (idx, s_m, d_lat_m, dist2).
        """
        best_idx = 0
        best_d2 = float("inf")

        for i, s in enumerate(self.samples):
            dx = x - s.x_m
            dy = y - s.y_m
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        d_lat_m = self._lateral_deviation_for_index(best_idx, x, y)
        return best_idx, self.samples[best_idx].s_m, d_lat_m, best_d2

    def _nearest_point_windowed(
        self, x: float, y: float, center_idx: int, window: int
    ) -> Tuple[int, float, float, float]:
        """
        Windowed nearest search around last index for performance and stability.
        Returns (idx, s_m, d_lat_m, dist2).
        """
        n = len(self.samples)
        start = max(0, center_idx - window)
        end = min(n - 1, center_idx + window)

        best_idx = center_idx
        best_d2 = float("inf")

        for i in range(start, end + 1):
            s = self.samples[i]
            dx = x - s.x_m
            dy = y - s.y_m
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        d_lat_m = self._lateral_deviation_for_index(best_idx, x, y)
        return best_idx, self.samples[best_idx].s_m, d_lat_m, best_d2

    def _lateral_deviation_for_index(self, idx: int, x: float, y: float) -> float:
        """
        Compute signed lateral deviation from trajectory sample idx to point (x, y).
        """
        s = self.samples[idx]
        n = len(self.samples)
        if idx < n - 1:
            s_next = self.samples[idx + 1]
        else:
            s_next = self.samples[idx - 1]

        tx = s_next.x_m - s.x_m
        ty = s_next.y_m - s.y_m
        t_len = math.hypot(tx, ty) or 1.0
        tx /= t_len
        ty /= t_len

        vx = x - s.x_m
        vy = y - s.y_m

        # Signed lateral deviation via 2D cross product
        cross = tx * vy - ty * vx
        return cross  # left/right sign

    def _smooth(self, kind: str, value: float, alpha: float) -> float:
        if kind == "dv":
            prev = self._dv_ema
        elif kind == "dlat":
            prev = self._dlat_ema
        else:
            prev = self._da_lat_ema

        if prev is None:
            new = value
        else:
            new = alpha * value + (1.0 - alpha) * prev

        if kind == "dv":
            self._dv_ema = new
        elif kind == "dlat":
            self._dlat_ema = new
        else:
            self._da_lat_ema = new

        return new

    def _decide(
        self, delta_v_kmh: float, d_lat_m: float, gps_fix_ok: bool
    ) -> Tuple[str, dict, Optional[str], Optional[str]]:
        """
        Decide overall status and basic LED/HUD intents.
        """
        if not gps_fix_ok:
            led_cfg = {"color": "yellow", "pattern": "blink", "on_ms": 150, "off_ms": 150}
            return "gps_bad", led_cfg, None, "GPS"

        tol = self.tol_kmh

        if abs(delta_v_kmh) <= tol:
            status = "good"
            led_cfg = self.led_ok
        elif delta_v_kmh < -tol:
            status = "too_slow"
            led_cfg = self.led_under
        else:
            status = "too_fast"
            led_cfg = self.led_over

        # Lateral deviation arrow (sign only; magnitude used only for HUD)
        arrow = None
        if abs(d_lat_m) > 1.0:  # 1 m dead-band
            arrow = "left" if d_lat_m > 0.0 else "right"

        return status, led_cfg, arrow, None

    def _feedback_no_trajectory(self) -> RaceFeedback:
        return RaceFeedback(
            ok=False,
            reason="no_trajectory",
            has_gps=False,
            gps_fix_ok=False,
            s_idx=-1,
            s_m=0.0,
            d_lat_m=0.0,
            v_meas_mps=0.0,
            v_opt_mps=0.0,
            delta_v_mps=0.0,
            delta_v_kmh=0.0,
            a_lat_meas_mps2=0.0,
            a_lat_opt_mps2=0.0,
            delta_a_lat_mps2=0.0,
            status="no_trajectory",
            led_mode="blink",
            led_color="yellow",
            led_on_ms=200,
            led_off_ms=200,
            hud_text="--",
            hud_arrow=None,
            hud_warning="NO TRAJ",
        )

    def _feedback_gps_bad(
        self, v_meas_mps: float, a_lat_meas_mps2: float, gps_fix_ok: bool
    ) -> RaceFeedback:
        return RaceFeedback(
            ok=False,
            reason="gps_bad",
            has_gps=False,
            gps_fix_ok=gps_fix_ok,
            s_idx=-1,
            s_m=0.0,
            d_lat_m=0.0,
            v_meas_mps=v_meas_mps,
            v_opt_mps=0.0,
            delta_v_mps=0.0,
            delta_v_kmh=0.0,
            a_lat_meas_mps2=a_lat_meas_mps2,
            a_lat_opt_mps2=0.0,
            delta_a_lat_mps2=0.0,
            status="gps_bad",
            led_mode="blink",
            led_color="yellow",
            led_on_ms=150,
            led_off_ms=150,
            hud_text="--",
            hud_arrow=None,
            hud_warning="GPS",
        )

