import argparse
import json
import time
from pathlib import Path

import requests
import yaml

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - uploader - %(levelname)s - %(message)s")
log = logging.getLogger("uploader")

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def now_iso_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

def read_track_id(cfg):
    p = Path(cfg["paths"]["current_track_id_file"])
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None

def find_latest_done_session(sessions_dir: Path, suffix: str):
    cands = []
    for p in sessions_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.endswith(suffix):
            continue
        if not (p / "DONE.json").exists():
            continue
        cands.append(p)
    cands.sort(key=lambda x: x.name)
    return cands[-1] if cands else None

class ServerClient:
    def __init__(self, base_url: str, username: str, password: str, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout_s = timeout_s
        self.token = None

    def login(self):
        r = requests.post(
            f"{self.base_url}/api/auth/login",
            json={"username": self.username, "password": self.password},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]

        log.info("Logged in to server")

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def status(self) -> dict:
        r = requests.get(
            f"{self.base_url}/api/status",
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def start_lap(self, track_name: str) -> str:
        r = requests.post(
            f"{self.base_url}/api/lap/start",
            json={"track_name": track_name},
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        if r.status_code == 400:
            try:
                st = self.status()
                if st.get("recording"):
                    log.warning("Server has active lap. Auto-stopping it...")
                    rs = requests.post(
                        f"{self.base_url}/api/lap/stop",
                        headers=self._headers(),
                        timeout=self.timeout_s,
                    )
                    rs.raise_for_status()
                    time.sleep(0.2)
                    r = requests.post(
                        f"{self.base_url}/api/lap/start",
                        json={"track_name": track_name},
                        headers=self._headers(),
                        timeout=self.timeout_s,
                    )
            except Exception:
                pass

        r.raise_for_status()
        return r.json()["lap_id"]

    def send_point(self, point: dict):
        r = requests.post(
            f"{self.base_url}/api/gps/point",
            json=point,
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()

    def stop_lap(self):
        r = requests.post(
            f"{self.base_url}/api/lap/stop",
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()

def iter_points_from_events(events_path: Path):
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        gps = d.get("gps") or {}
        lat = gps.get("lat")
        lon = gps.get("lon")
        if lat is None or lon is None:
            continue
        t = d.get("t")
        ts = now_iso_utc() if t is None else time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(float(t)))
        yield {
            "latitude": float(lat),
            "longitude": float(lon),
            "altitude": 0.0,
            "fix_quality": 1,
            "speed": 0.0,     # developing
            "timestamp": ts,
            "hdop": None,
            "sats": None,
            "source": "mgsa_pi",
        }

def upload_session(sc: ServerClient, track_name: str, session_dir: Path):
    events = session_dir / "events.jsonl"
    if not events.exists():
        log.warning(f"Missing events.jsonl in {session_dir}")
        return False

    log.info(f"Uploading {track_name} from {session_dir.name}")
    lap_id = sc.start_lap(track_name)
    log.info(f"Server lap started: lap_id={lap_id} track_name={track_name}")

    sent = 0
    for p in iter_points_from_events(events):
        sc.send_point(p)
        sent += 1
        if sent % 200 == 0:
            log.info(f"Sent {sent} points for {track_name}")
            time.sleep(0.05)

    sc.stop_lap()
    log.info(f"Uploaded OK: {track_name} points={sent}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if not cfg.get("net", {}).get("enabled", True):
        return

    base_url = cfg["net"]["server"]["base_url"]
    timeout_s = float(cfg["net"]["server"].get("timeout_s", 10))
    username = cfg["net"]["auth"]["username"]
    password = cfg["net"]["auth"]["password"]

    sessions_dir = Path(cfg["paths"]["sessions_dir"])
    state_dir = Path(cfg["paths"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    sent_flag = state_dir / "last_uploaded_sessions.json"

    sc = ServerClient(base_url, username, password, timeout_s)
    sc.login()

    while True:
        try:
            track_id = read_track_id(cfg)
            if not track_id:
                time.sleep(2.0)
                continue

            outer = find_latest_done_session(sessions_dir, "_record_outer")
            inner = find_latest_done_session(sessions_dir, "_record_inner")
            if not outer or not inner:
                time.sleep(2.0)
                continue

            log.info(f"Found DONE sessions: outer={outer.name} inner={inner.name} track_id={track_id}")

            key = {"track_id": track_id, "outer": outer.name, "inner": inner.name}
            if sent_flag.exists():
                try:
                    if json.loads(sent_flag.read_text(encoding="utf-8")) == key:
                        time.sleep(3.0)
                        continue
                except Exception:
                    pass

            ok1 = upload_session(sc, f"{track_id}__outer", outer)
            ok2 = upload_session(sc, f"{track_id}__inner", inner)

            if ok1 and ok2:
                sent_flag.write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")

        except Exception as e:
            log.exception("Uploader loop error")
            try:
                sc.login()
                log.info("Re-logged in to server")
            except Exception:
                log.exception("Re-login failed")

        time.sleep(3.0)

if __name__ == "__main__":
    main()
