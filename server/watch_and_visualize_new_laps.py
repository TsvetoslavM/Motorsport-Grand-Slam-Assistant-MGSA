from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def _http_json(method: str, url: str, payload: dict | None = None, headers: dict | None = None) -> dict:
    data = None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data, headers=hdrs)
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _login(base_url: str, username: str, password: str) -> str:
    resp = _http_json("POST", f"{base_url}/api/auth/login", {"username": username, "password": password})
    return resp["access_token"]


def _safe_kind(s: str) -> str:
    out = "".join(c for c in s if c.isalnum() or c in ("_", "-"))
    return out or "line"


def _parse_iso(ts: str) -> datetime:
    t = ts.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return datetime.fromisoformat(t)


def _speed_to_kmh(v: float) -> float:
    if v < 0:
        return 0.0
    if v > 90.0:
        return float(v)
    return float(v) * 3.6


def _fetch_lap_points(base_url: str, lap_id: str, token: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}"}
    resp = _http_json("GET", f"{base_url}/api/lap/{lap_id}", headers=headers)
    points = resp.get("points") or []
    return points


def _points_to_racing(points: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not points:
        return []

    t0 = None
    out: list[dict[str, float]] = []
    for i, p in enumerate(points):
        try:
            ts = str(p.get("timestamp") or "")
            dt = _parse_iso(ts)
            if t0 is None:
                t0 = dt
            time_s = (dt - t0).total_seconds()
            lat = float(p["latitude"])
            lon = float(p["longitude"])
            speed_raw = float(p.get("speed", 0.0))
            speed_kmh = _speed_to_kmh(speed_raw)
            out.append({"time_s": float(time_s), "lat": lat, "lon": lon, "speed_kmh": speed_kmh})
        except Exception:
            continue

    if len(out) >= 2:
        for i in range(1, len(out)):
            if out[i]["time_s"] <= out[i - 1]["time_s"]:
                out[i]["time_s"] = out[i - 1]["time_s"] + 0.02

    return out


def _upload_racing_points(base_url: str, track_id: str, kind: str, token: str, points: list[dict[str, float]]) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    _http_json(
        "POST",
        f"{base_url}/api/track/{track_id}/racing_line/upload",
        {"kind": kind, "points": points},
        headers=headers,
    )


def _run_visualizer(base_url: str, track_id: str, username: str, password: str, out_html: Path, kinds: list[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "tests.try_visualize_track_compare",
        "--base-url",
        base_url,
        "--track-id",
        track_id,
        "--username",
        username,
        "--password",
        password,
        "--out",
        str(out_html),
    ]
    if kinds:
        cmd.extend(["--kinds"] + kinds)
    return subprocess.run(cmd).returncode


async def _ws_loop(base_url: str, username: str, password: str, track_id: str, out_dir: Path, kinds_extra: list[str]) -> int:
    try:
        import websockets
    except Exception:
        print("[ERROR] Missing dependency: websockets")
        print("Install: pip install websockets")
        return 1

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.rstrip("/") + "/ws/live"

    out_dir.mkdir(parents=True, exist_ok=True)

    token = _login(base_url, username, password)
    print(f"[OK] Logged in as {username}")
    print(f"[*] Listening: {ws_url}")
    print(f"[*] Track: {track_id}")
    print(f"[*] Out dir: {out_dir}")

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                try:
                    msg = msg.decode("utf-8")
                except Exception:
                    continue
            try:
                data = json.loads(msg)
            except Exception:
                continue

            if data.get("type") != "lap_completed":
                continue

            lap_id = str(data.get("lap_id") or "").strip()
            if not lap_id:
                continue

            print(f"\n[EVENT] lap_completed: {lap_id}")

            try:
                pts = _fetch_lap_points(base_url, lap_id, token)
                racing = _points_to_racing(pts)
                if len(racing) < 5:
                    print("[WARN] Not enough points to upload/visualize")
                    continue

                kind = _safe_kind(f"lap_{lap_id}")
                _upload_racing_points(base_url, track_id, kind, token, racing)
                print(f"[OK] Uploaded racing line kind='{kind}' points={len(racing)}")

                out_html = out_dir / f"{track_id}_{kind}.html"
                kinds = [kind] + [k for k in kinds_extra if k and k != kind]
                rc = _run_visualizer(base_url, track_id, username, password, out_html, kinds)

                if rc == 0:
                    print(f"[OK] Visualization saved: {out_html}")
                else:
                    print(f"[ERROR] Visualizer returned code {rc}")

            except Exception as e:
                print(f"[ERROR] Failed to process lap {lap_id}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch WS for lap_completed and auto-generate HTML visualization.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--track-id", required=True)
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--out-dir", default="lap_visualizations")
    ap.add_argument("--also-kinds", nargs="*", default=[], help="Extra racing-line kinds to include (e.g. optimal driver).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    return asyncio.run(_ws_loop(args.base_url, args.username, args.password, args.track_id, out_dir, list(args.also_kinds)))


if __name__ == "__main__":
    raise SystemExit(main())
