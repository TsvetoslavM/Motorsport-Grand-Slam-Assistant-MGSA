import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImuSample:
    """
    Single IMU sample in vehicle coordinates.

    Expected JSON schema in state_dir/imu_state.json (produced by your IMU daemon):
      {
        "ts": <unix_time_seconds>,
        "a_lat_mps2": <float>,   # lateral acceleration [m/s^2] (left +)
        "a_lon_mps2": <float>,   # longitudinal acceleration [m/s^2] (forward +)
        "yaw_rate_rad_s": <float>
      }
    """

    ts: float
    a_lat_mps2: float
    a_lon_mps2: float
    yaw_rate_rad_s: float


class ImuStateReader:
    """
    Lightweight bridge to an external IMU daemon.

    The daemon is responsible for reading the SEN0142 over I2C and periodically
    writing the latest fused sample into state_dir/imu_state.json using the schema
    documented in ImuSample above.

    This reader:
      - Reads the JSON file when it changes (mtime-based).
      - Rejects samples that are too old (max_age_s).
      - Applies an EMA to smooth a_lat/a_lon/yaw_rate to the race tick rate.
    """

    def __init__(self, state_dir: Path, max_age_s: float = 0.3, alpha: float = 0.3):
        self._path = Path(state_dir) / "imu_state.json"
        self._max_age_s = float(max_age_s)
        self._alpha = float(alpha)

        self._last_mtime: float = 0.0
        self._raw: Optional[ImuSample] = None

        # Smoothed values
        self._ema: Optional[ImuSample] = None

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
        self._raw = ImuSample(ts=ts, a_lat_mps2=a_lat, a_lon_mps2=a_lon, yaw_rate_rad_s=yaw)

    def get_latest(self, now: Optional[float] = None) -> Optional[ImuSample]:
        """
        Return latest *smoothed* IMU sample, or None if data is missing / too old.
        """
        if now is None:
            now = time.time()

        self._read_file_if_updated()
        if not self._raw:
            return None

        if (now - self._raw.ts) > self._max_age_s:
            # Too old compared to race tick, treat as unavailable
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

