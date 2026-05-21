import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImuSample:
    ts: float
    a_lat_mps2: float
    a_lon_mps2: float
    yaw_rate_rad_s: float


class ImuStateReader:
    def __init__(
        self,
        state_dir: Path,
        max_age_s: float = 0.3,
        alpha: float = 0.3,
        calibration_duration_s: float = 3.0,
    ):
        self._path = Path(state_dir) / "imu_state.json"
        self._max_age_s = float(max_age_s)
        self._alpha = float(alpha)

        self._last_mtime: float = 0.0
        self._raw: Optional[ImuSample] = None
        self._ema: Optional[ImuSample] = None

        self._calibrated = False
        self._calibration_start_ts: Optional[float] = None
        self._calibration_duration_s = float(calibration_duration_s)
        self._calibration_samples: list[tuple[float, float, float]] = []

        self._bias_a_lat = 0.0
        self._bias_a_lon = 0.0
        self._bias_yaw = 0.0

    def _is_plausible(self, a_lat: float, a_lon: float, yaw: float) -> bool:
        if abs(a_lat) > 30.0:
            return False
        if abs(a_lon) > 30.0:
            return False
        if abs(yaw) > 6.0:
            return False
        return True

    def _update_calibration(self, ts: float, a_lat: float, a_lon: float, yaw: float) -> None:
        if self._calibration_start_ts is None:
            self._calibration_start_ts = ts

        if self._is_plausible(a_lat, a_lon, yaw):
            self._calibration_samples.append((a_lat, a_lon, yaw))

        if (ts - self._calibration_start_ts) >= self._calibration_duration_s:
            if not self._calibration_samples:
                self._calibration_start_ts = None
                return

            n = len(self._calibration_samples)

            self._bias_a_lat = sum(x[0] for x in self._calibration_samples) / n
            self._bias_a_lon = sum(x[1] for x in self._calibration_samples) / n
            self._bias_yaw = sum(x[2] for x in self._calibration_samples) / n

            self._calibrated = True
            self._calibration_samples.clear()
            self._ema = None

    def reset_calibration(self) -> None:
        self._calibrated = False
        self._calibration_start_ts = None
        self._calibration_samples.clear()

        self._bias_a_lat = 0.0
        self._bias_a_lon = 0.0
        self._bias_yaw = 0.0

        self._raw = None
        self._ema = None

    def is_calibrated(self) -> bool:
        return self._calibrated

    def get_bias(self) -> tuple[float, float, float]:
        return self._bias_a_lat, self._bias_a_lon, self._bias_yaw

    def _read_file_if_updated(self) -> None:
        try:
            st = self._path.stat()
        except FileNotFoundError:
            return

        if st.st_mtime <= self._last_mtime:
            return

        try:
            raw = self._path.read_text(encoding="utf-8")
            d = json.loads(raw)
        except Exception:
            return

        try:
            ts = float(d.get("ts", time.time()))
            a_lat = float(d.get("a_lat_mps2", 0.0))
            a_lon = float(d.get("a_lon_mps2", 0.0))
            yaw = float(d.get("yaw_rate_rad_s", 0.0))
        except Exception:
            return

        self._last_mtime = st.st_mtime

        if not self._is_plausible(a_lat, a_lon, yaw):
            return

        if not self._calibrated:
            self._update_calibration(ts, a_lat, a_lon, yaw)
            return

        a_lat -= self._bias_a_lat
        a_lon -= self._bias_a_lon
        yaw -= self._bias_yaw

        self._raw = ImuSample(
            ts=ts,
            a_lat_mps2=a_lat,
            a_lon_mps2=a_lon,
            yaw_rate_rad_s=yaw,
        )

    def get_latest(self, now: Optional[float] = None) -> Optional[ImuSample]:
        if now is None:
            now = time.time()

        self._read_file_if_updated()

        if not self._calibrated:
            return None

        if not self._raw:
            return None

        if (now - self._raw.ts) > self._max_age_s:
            self._ema = None
            return None

        if self._ema is None:
            self._ema = self._raw
        else:
            a = self._alpha
            r = self._raw
            e = self._ema

            self._ema = ImuSample(
                ts=r.ts,
                a_lat_mps2=a * r.a_lat_mps2 + (1.0 - a) * e.a_lat_mps2,
                a_lon_mps2=a * r.a_lon_mps2 + (1.0 - a) * e.a_lon_mps2,
                yaw_rate_rad_s=a * r.yaw_rate_rad_s + (1.0 - a) * e.yaw_rate_rad_s,
            )

        return self._ema