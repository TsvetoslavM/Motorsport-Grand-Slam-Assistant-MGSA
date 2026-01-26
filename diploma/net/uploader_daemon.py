import argparse
import json
import time
from pathlib import Path

import requests
import yaml

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_track_id(cfg):
    p = Path(cfg["paths"]["current_track_id_file"])
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None

def find_latest_done_session(sessions_dir: Path, kind: str):
    cands = []
    for p in sessions_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.endswith(f"_{kind}"):
            continue
        if not (p / "DONE.json").exists():
            continue
        cands.append(p)
    cands.sort(key=lambda x: x.name)
    return cands[-1] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    base_url = cfg["net"]["server"]["base_url"].rstrip("/")
    sessions_dir = Path(cfg["paths"]["sessions_dir"])
    cache_dir = Path(cfg["paths"]["track_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_dir = Path(cfg["paths"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    sent_flag = state_dir / "last_upload.json"

    while True:
        try:
            track_id = read_track_id(cfg)
            if not track_id:
                time.sleep(1.0)
                continue

            outer = find_latest_done_session(sessions_dir, "record_outer")
            inner = find_latest_done_session(sessions_dir, "record_inner")

            if not outer or not inner:
                time.sleep(1.0)
                continue

            key = {"track_id": track_id, "outer": outer.name, "inner": inner.name}

            if sent_flag.exists():
                try:
                    if json.loads(sent_flag.read_text(encoding="utf-8")) == key:
                        time.sleep(2.0)
                        continue
                except Exception:
                    pass

            payload = {
                "track_id": track_id,
                "outer_session": outer.name,
                "inner_session": inner.name,
                "outer_events": (outer / "events.jsonl").read_text(encoding="utf-8"),
                "inner_events": (inner / "events.jsonl").read_text(encoding="utf-8"),
                "start_finish": json.loads(Path(cfg["paths"]["current_start_finish_file"]).read_text(encoding="utf-8")) if Path(cfg["paths"]["current_start_finish_file"]).exists() else None,
            }

            url = f"{base_url}/api/trajectory/optimize"
            r = requests.post(url, json=payload, timeout=20)
            r.raise_for_status()
            data = r.json()

            (cache_dir / "latest_optimal.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            sent_flag.write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")

        except Exception:
            pass

        time.sleep(2.0)

if __name__ == "__main__":
    main()
