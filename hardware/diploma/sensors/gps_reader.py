import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import serial
import pynmea2


@dataclass
class GpsFix:
    ts: float
    lat: Optional[float]
    lon: Optional[float]
    speed_kmh: Optional[float]
    fix_ok: bool
    fix_quality: int          # from GGA: 0 invalid, 1 GPS, 2 DGPS, 4 RTK fixed, 5 RTK float...
    sats: int
    hdop: Optional[float]


class GpsReader:
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = int(baud)

        self._ser = None
        self._th = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._last: Optional[GpsFix] = None

        self._last_lat = None
        self._last_lon = None
        self._last_speed_kmh = None
        self._last_fix_quality = 0
        self._last_sats = 0
        self._last_hdop = None

    def start(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=1)
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    def get_latest(self) -> Optional[GpsFix]:
        with self._lock:
            return self._last

    def get_latlon(self) -> Tuple[float, float]:
        g = self.get_latest()
        if not g or g.lat is None or g.lon is None:
            raise RuntimeError("GPS: no position yet")
        return float(g.lat), float(g.lon)

    def get_speed_kmh(self) -> float:
        g = self.get_latest()
        if not g or g.speed_kmh is None:
            return 0.0
        return float(g.speed_kmh)

    def has_fix(self, max_age_s: float = 1.0) -> bool:
        g = self.get_latest()
        if not g:
            return False
        if (time.time() - g.ts) > max_age_s:
            return False
        return bool(g.fix_ok)

    def is_rtk_fix(self, max_age_s: float = 1.0) -> bool:
        g = self.get_latest()
        if not g:
            return False
        if (time.time() - g.ts) > max_age_s:
            return False
        return g.fix_quality == 4

    def _publish(self):
        ts = time.time()
        fix_ok = (self._last_fix_quality != 0) and (self._last_lat is not None) and (self._last_lon is not None)
        g = GpsFix(
            ts=ts,
            lat=self._last_lat,
            lon=self._last_lon,
            speed_kmh=self._last_speed_kmh,
            fix_ok=fix_ok,
            fix_quality=int(self._last_fix_quality),
            sats=int(self._last_sats),
            hdop=self._last_hdop,
        )
        with self._lock:
            self._last = g

    def _run(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(4096)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if not line.startswith(b"$") and not line.startswith(b"!"):
                        continue
                    try:
                        msg = pynmea2.parse(line.decode("ascii", errors="ignore"))
                    except Exception:
                        continue

                    tname = msg.__class__.__name__

                    # Position + fix quality from GGA
                    if tname == "GGA":
                        try:
                            lat = msg.latitude if msg.latitude != 0 else None
                            lon = msg.longitude if msg.longitude != 0 else None
                            fq = int(getattr(msg, "gps_qual", 0) or 0)
                            sats = int(getattr(msg, "num_sats", 0) or 0)
                            hdop = getattr(msg, "horizontal_dil", None)
                            hdop = float(hdop) if hdop not in (None, "") else None

                            if lat is not None and lon is not None:
                                self._last_lat = float(lat)
                                self._last_lon = float(lon)
                            self._last_fix_quality = fq
                            self._last_sats = sats
                            self._last_hdop = hdop
                            self._publish()
                        except Exception:
                            pass

                    # Speed from RMC (knots)
                    elif tname == "RMC":
                        try:
                            sp_knots = getattr(msg, "spd_over_grnd", None)
                            if sp_knots not in (None, ""):
                                sp_knots = float(sp_knots)
                                self._last_speed_kmh = sp_knots * 1.852
                                self._publish()
                        except Exception:
                            pass

                    # Speed from VTG (km/h)
                    elif tname == "VTG":
                        try:
                            sp_kmh = getattr(msg, "spd_over_grnd_kmph", None)
                            if sp_kmh not in (None, ""):
                                self._last_speed_kmh = float(sp_kmh)
                                self._publish()
                        except Exception:
                            pass

            except Exception:
                time.sleep(0.2)
                continue
