from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta


def _http_json(method: str, url: str, payload: dict | None = None, headers: dict | None = None) -> dict:
    data = None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, method=method, data=data, headers=hdrs)

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"\n[HTTP {e.code}] {method} {url}\n{err_body}\n", file=sys.stderr)
        raise


def _login(base_url: str, username: str, password: str) -> str:
    resp = _http_json("POST", f"{base_url}/api/auth/login", {"username": username, "password": password})
    return resp["access_token"]


def _get_status(base_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    return _http_json("GET", f"{base_url}/api/status", None, headers=headers)


def _stop_lap_safe(base_url: str, token: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        _http_json("POST", f"{base_url}/api/lap/stop", None, headers=headers)
    except Exception:
        pass


def _start_lap(base_url: str, token: str, track_name: str, lap_type: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    resp = _http_json(
        "POST",
        f"{base_url}/api/lap/start",
        {"track_name": track_name, "lap_type": lap_type},
        headers=headers,
    )
    return str(resp["lap_id"])


def _stop_lap(base_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    return _http_json("POST", f"{base_url}/api/lap/stop", None, headers=headers)


def _send_point(base_url: str, token: str, point: dict) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    _http_json("POST", f"{base_url}/api/gps/point", point, headers=headers)


def _build_racing_line_from_lap(base_url: str, token: str, track_id: str, lap_id: str, kind: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    return _http_json(
        "POST",
        f"{base_url}/api/track/{track_id}/racing_line/build_from_lap",
        {"lap_id": lap_id, "kind": kind},
        headers=headers,
    )


def _meters_to_latlon(dx_east_m: float, dy_north_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0
    lat = lat0 + (dy_north_m / m_per_deg_lat)
    lon = lon0 + (dx_east_m / m_per_deg_lon)
    return lat, lon


def _generate_loop_points(
    *,
    lat0: float,
    lon0: float,
    radius_m: float,
    n_points: int,
    speed_kmh: float,
    noise_m: float,
    fix_quality: int,
    seed: int,
    start_ts: datetime,
    dt_s: float,
) -> list[dict]:
    random.seed(seed)
    pts: list[dict] = []
    speed_mps = max(0.0, float(speed_kmh)) / 3.6

    for i in range(n_points):
        a = 2.0 * math.pi * (i / max(1, n_points))
        x = radius_m * math.cos(a)
        y = radius_m * math.sin(a)

        if noise_m > 0:
            x += (random.random() * 2.0 - 1.0) * noise_m
            y += (random.random() * 2.0 - 1.0) * noise_m

        lat, lon = _meters_to_latlon(x, y, lat0, lon0)
        ts = (start_ts + timedelta(seconds=i * dt_s)).isoformat()

        pts.append(
            {
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude": 550.0,
                "fix_quality": int(fix_quality),
                "speed": float(speed_mps),  # m/s (server stores this; build_from_lap converts to km/h)
                "timestamp": ts,
                "hdop": 0.7,
                "sats": 16,
                "source": "sim",
            }
        )
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description="Sim client: send a single RACING lap and build racing line from lap_id.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--track-id", required=True)

    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")

    ap.add_argument("--lap-type", default="racing", help="lap_type sent to /api/lap/start (default: racing)")
    ap.add_argument("--kind", default="driver", help="racing line kind to save as racing_<kind>.csv (default: driver)")

    ap.add_argument("--center-lat", type=float, default=42.7144)
    ap.add_argument("--center-lon", type=float, default=23.2743)
    ap.add_argument("--radius-m", type=float, default=50.0)

    ap.add_argument("--n-points", type=int, default=500)
    ap.add_argument("--hz", type=float, default=10.0)

    ap.add_argument("--speed-kmh", type=float, default=90.0)
    ap.add_argument("--noise-m", type=float, default=0.3)
    ap.add_argument("--fix-quality", type=int, default=4)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--no-build", action="store_true", help="Do not call build_from_lap; only record lap.")
    args = ap.parse_args()

    dt_s = 1.0 / max(0.1, float(args.hz))
    n = int(args.n_points)
    if n < 50:
        n = 50

    print(f"[*] Logging in to {args.base_url} as {args.username}...")
    token = _login(args.base_url, args.username, args.password)
    print("[OK] Logged in")

    st = _get_status(args.base_url, token)
    if st.get("recording"):
        print("[WARN] Server already recording. Stopping active lap first...")
        _stop_lap_safe(args.base_url, token)

    t0 = datetime.now(timezone.utc)

    pts = _generate_loop_points(
        lat0=float(args.center_lat),
        lon0=float(args.center_lon),
        radius_m=float(args.radius_m),
        n_points=n,
        speed_kmh=float(args.speed_kmh),
        noise_m=float(args.noise_m),
        fix_quality=int(args.fix_quality),
        seed=int(args.seed),
        start_ts=t0,
        dt_s=dt_s,
    )

    track_name = args.track_id
    lap_id = _start_lap(args.base_url, token, track_name=track_name, lap_type=str(args.lap_type))
    print(f"[OK] Started lap: {lap_id} track_name={track_name} lap_type={args.lap_type}")

    for i, p in enumerate(pts, 1):
        _send_point(args.base_url, token, p)
        if i % 50 == 0 or i == len(pts):
            print(f"  sent {i}/{len(pts)}")
        if args.realtime:
            time.sleep(dt_s)


    res = _stop_lap(args.base_url, token)
    print(f"[OK] Stopped lap: {lap_id} lap_time={res.get('lap_time')} points={res.get('points')}")

    if not args.no_build:
        print(f"[*] Building racing line from lap -> track={args.track_id} kind={args.kind} ...")
        out = _build_racing_line_from_lap(args.base_url, token, args.track_id, lap_id, kind=str(args.kind))
        print(f"[OK] build_from_lap: {out}")

    print(f"\n[RESULT] lap_id={lap_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
