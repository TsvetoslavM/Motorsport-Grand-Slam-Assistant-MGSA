import argparse
import json
import logging
import time
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - track_dl - %(levelname)s - %(message)s")
log = logging.getLogger("track_dl")

def load_cfg(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def read_track_id(cfg):
    p = Path(cfg["paths"]["current_track_id_file"])
    if not p.exists():
        return None
    s = p.read_text(encoding="utf-8").strip()
    return s or None

class ServerClient:
    def __init__(self, base_url, username, password, timeout_s):
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
        log.info("Logged in")

    def headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def get_meta(self, track_id: str):
        r = requests.get(
            f"{self.base_url}/api/track/{track_id}/ideal/meta",
            headers=self.headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def download_ideal(self, track_id: str) -> bytes:
        r = requests.get(
            f"{self.base_url}/api/track/{track_id}/ideal",
            headers=self.headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    base_url = cfg["net"]["server"]["base_url"]
    timeout_s = float(cfg["net"]["server"].get("timeout_s", 10))
    username = cfg["net"]["auth"]["username"]
    password = cfg["net"]["auth"]["password"]

    out_base = Path(cfg["paths"].get("tracks_dir", "/home/cveto-msga/mgsa/diploma/data/tracks"))
    out_base.mkdir(parents=True, exist_ok=True)

    state_dir = Path(cfg["paths"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    local_meta_path = state_dir / "ideal_local_meta.json"

    local_meta = {}
    if local_meta_path.exists():
        try:
            local_meta = json.loads(local_meta_path.read_text(encoding="utf-8"))
        except Exception:
            local_meta = {}

    sc = ServerClient(base_url, username, password, timeout_s)

    while True:
        try:
            if sc.token is None:
                sc.login()
            track_id = read_track_id(cfg)
            if not track_id:
                time.sleep(2.0)
                continue

            meta = sc.get_meta(track_id)
            if not meta.get("exists", True) and "updated" not in meta:
                time.sleep(3.0)
                continue

            remote_updated = meta.get("updated")
            if local_meta.get("track_id") == track_id and local_meta.get("updated") == remote_updated:
                time.sleep(3.0)
                continue

            log.info(f"New ideal available for {track_id}. Downloading...")
            data = sc.download_ideal(track_id)

            tdir = out_base / track_id
            tdir.mkdir(parents=True, exist_ok=True)
            out_csv = tdir / "ideal.csv"
            out_csv.write_bytes(data)

            local_meta = {"track_id": track_id, "updated": remote_updated, "bytes": len(data)}
            local_meta_path.write_text(json.dumps(local_meta, ensure_ascii=False), encoding="utf-8")

            log.info(f"Saved {out_csv} bytes={len(data)} updated={remote_updated}")

        except Exception:
            log.exception("Downloader error")
            sc.token = None
            time.sleep(2.0)

        time.sleep(3.0)

if __name__ == "__main__":
    main()
