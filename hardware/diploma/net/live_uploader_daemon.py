import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from sensors.gps_reader import GpsReader


def now_iso():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class NetCfg:
    base_url: str
    timeout_s: float
    username: str
    password: str
    attempts: int
    backoff_s: float


class LiveUploader:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = cfg["paths"]

        self.cmd_path = Path(self.paths["command_file"])
        self.track_id_path = Path(self.paths["current_track_id_file"])
        self.state_dir = Path(self.paths["state_dir"])
        self.pid_path = self.state_dir / "live_uploader.pid"

        net = cfg.get("net", {})
        server = net.get("server", {})
        auth = net.get("auth", {})
        retry = server.get("retry", {})

        self.net = NetCfg(
            base_url=server["base_url"].rstrip("/"),
            timeout_s=float(server.get("timeout_s", 10)),
            username=auth["username"],
            password=auth["password"],
            attempts=int(retry.get("attempts", 5)),
            backoff_s=float(retry.get("backoff_s", 1.5)),
        )

        gps_serial = cfg.get("gps", {}).get("serial", {})
        self.gps = GpsReader(
            port=gps_serial.get("port", "/dev/ttyUSB0"),
            baud=int(gps_serial.get("baud", 460800)),
        )

        self.s = requests.Session()
        self.token = None

        self._last_cmd_mtime = 0.0
        self._recording = False
        self._lap_type = None
        self._lap_id = None
        self._last_sent_ts = 0.0

        self.send_hz = 10.0
        self.max_age_s = 1.0
        
        self.state_file = self.state_dir / "runtime_state.json"
        self._last_led_key = None


    def write_pid(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(__import__("os").getpid()), encoding="utf-8")

    def login(self):
        url = f"{self.net.base_url}/api/auth/login"
        r = self.s.post(
            url,
            json={"username": self.net.username, "password": self.net.password},
            timeout=self.net.timeout_s,
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]
        self.s.headers.update({"Authorization": f"Bearer {self.token}"})

    def post_retry(self, path: str, payload: dict):
        url = f"{self.net.base_url}{path}"
        last = None
        for i in range(self.net.attempts):
            try:
                r = self.s.post(url, json=payload, timeout=self.net.timeout_s)
                if r.status_code == 401:
                    self.login()
                    r = self.s.post(url, json=payload, timeout=self.net.timeout_s)
                r.raise_for_status()
                return r.json() if r.content else {}
            except Exception as e:
                last = e
                self._write_led("blink", "yellow", 250, 250)

                time.sleep(self.net.backoff_s * (i + 1))

        self._write_led("blink", "red", 120, 120)

        raise last

    def read_track_name(self):
        if not self.track_id_path.exists():
            return "mytrack"
        t = self.track_id_path.read_text(encoding="utf-8").strip()
        return t or "mytrack"

    def start_lap(self, lap_type: str):
        track_name = self.read_track_name()
        res = self.post_retry("/api/lap/start", {"track_name": track_name, "lap_type": lap_type})
        self._lap_id = res.get("lap_id")
        self._lap_type = lap_type
        self._recording = True

        if lap_type == "outer":
            self._write_led("solid", "blue")
        elif lap_type == "inner":
            self._write_led("solid", "magenta")
        else:
            self._write_led("solid", "white")


    def stop_lap(self):
        self.post_retry("/api/lap/stop", {})
        self._recording = False
        self._lap_type = None
        self._lap_id = None

        self._write_led("off", "black")

    def read_command(self):
        try:
            st = self.cmd_path.stat()
        except FileNotFoundError:
            return None
        if st.st_mtime <= self._last_cmd_mtime:
            return None
        self._last_cmd_mtime = st.st_mtime
        try:
            return json.loads(self.cmd_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def handle_mode(self, mode: str):
        if mode == "record_outer":
            if self._recording and self._lap_type == "outer":
                return
            if self._recording:
                self.stop_lap()
            self.start_lap("outer")
            return

        if mode == "record_inner":
            if self._recording and self._lap_type == "inner":
                return
            if self._recording:
                self.stop_lap()
            self.start_lap("inner")
            return

        if mode == "idle":
            if self._recording:
                self.stop_lap()
            return

    def send_point(self):
        if not self._recording:
            return

        now = time.time()
        if now - self._last_sent_ts < (1.0 / self.send_hz):
            return

        if not self.gps.has_fix(max_age_s=self.max_age_s):
            return

        g = self.gps.get_latest()
        if not g or g.lat is None or g.lon is None:
            return

        speed_kmh = g.speed_kmh if g.speed_kmh is not None else 0.0
        speed_mps = float(speed_kmh) / 3.6

        payload = {
            "latitude": float(g.lat),
            "longitude": float(g.lon),
            "altitude": 0.0,
            "fix_quality": int(g.fix_quality),
            "speed": float(speed_mps),
            "timestamp": now_iso(),
            "hdop": float(g.hdop) if g.hdop is not None else None,
            "sats": int(g.sats) if g.sats is not None else None,
            "source": "pi_gps",
        }
        if payload["hdop"] is None:
            payload.pop("hdop")
        if payload["sats"] is None:
            payload.pop("sats")

        self.post_retry("/api/gps/point", payload)

        self._last_ok_led = getattr(self, "_last_ok_led", 0.0)
        if now - self._last_ok_led > 1.0:
            self._write_led("blink", "green", 60, 120)
            self._last_ok_led = now

        if self._lap_type == "outer":
            self._write_led("solid", "blue")
        elif self._lap_type == "inner":
            self._write_led("solid", "magenta")

        self._last_sent_ts = now

    def run(self):
        self.write_pid()
        self.gps.start()
        self.login()
        while True:
            cmd = self.read_command()
            if cmd and cmd.get("type") == "mode":
                self.handle_mode(cmd.get("mode", ""))
            self.send_point()
            time.sleep(0.01)

    def _write_led(self, mode: str, color: str, on_ms: int = 120, off_ms: int = 120):
        led_cfg = {"mode": mode, "color": color, "on_ms": int(on_ms), "off_ms": int(off_ms)}
        key = json.dumps(led_cfg, sort_keys=True)
        if key == self._last_led_key:
            return
        self._last_led_key = key

        d = {}
        try:
            if self.state_file.exists():
                raw = self.state_file.read_text(encoding="utf-8").strip()
                if raw:
                    d = json.loads(raw)
        except Exception:
            d = {}

        d["led"] = led_cfg

        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_file)



def load_cfg(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    LiveUploader(load_cfg(args.config)).run()
