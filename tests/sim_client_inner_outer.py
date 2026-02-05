from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"\n[HTTP {e.code}] {method} {url}\n{err_body}\n", file=sys.stderr)
        raise


def _http_bytes(method: str, url: str, headers: dict | None = None) -> tuple[int, bytes]:
    hdrs = {}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()


def _login(base_url: str, username: str, password: str) -> str:
    resp = _http_json("POST", f"{base_url}/api/auth/login", {"username": username, "password": password})
    return resp["access_token"]


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


def _get_status(base_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    return _http_json("GET", f"{base_url}/api/status", None, headers=headers)


def _stop_lap_safe(base_url: str, token: str) -> None:
    try:
        _stop_lap(base_url, token)
    except Exception:
        pass


def _send_point(base_url: str, token: str, point: dict) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    _http_json("POST", f"{base_url}/api/gps/point", point, headers=headers)


def _meters_to_latlon(dx_east_m: float, dy_north_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0
    lat = lat0 + (dy_north_m / m_per_deg_lat)
    lon = lon0 + (dx_east_m / m_per_deg_lon)
    return lat, lon

def _signed_area_xy(pts: list[tuple[float, float]]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a  # >0 CCW, <0 CW


def _ensure_ccw_xy(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return pts if _signed_area_xy(pts) > 0.0 else list(reversed(pts))

def _wavy_roadcourse_center_xy(
    n_points: int,
    base_radius_m: float,
    a2: float,
    a3: float,
    a4: float,
    rotate_deg: float = 0.0,
    squash_y: float = 0.75,
) -> list[tuple[float, float]]:
    n = max(20, int(n_points))
    R0 = max(5.0, float(base_radius_m))

    rot = math.radians(float(rotate_deg))
    cr = math.cos(rot)
    sr = math.sin(rot)

    pts: list[tuple[float, float]] = []
    for i in range(n):
        th = 2.0 * math.pi * (i / n)

        r = R0 * (1.0 + float(a2) * math.cos(2.0 * th) + float(a3) * math.cos(3.0 * th) + float(a4) * math.cos(4.0 * th))

        x = r * math.cos(th)
        y = r * math.sin(th) * float(squash_y)

        xr = x * cr - y * sr
        yr = x * sr + y * cr
        pts.append((xr, yr))

    return _ensure_ccw_xy(pts)



def _normals_closed_xy(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Unit left normals for closed polyline using central differences."""
    n = len(pts)
    out: list[tuple[float, float]] = []
    for i in range(n):
        x_prev, y_prev = pts[(i - 1) % n]
        x_next, y_next = pts[(i + 1) % n]
        dx = x_next - x_prev
        dy = y_next - y_prev
        L = math.hypot(dx, dy)
        if L < 1e-9:
            out.append((0.0, 0.0))
            continue
        tx = dx / L
        ty = dy / L
        # left normal
        nx = -ty
        ny = tx
        out.append((nx, ny))
    return out


def _generate_inner_outer_stadium(
    *,
    lat0: float,
    lon0: float,
    center_xy: list[tuple[float, float]],
    track_width_m: float,
    fix_quality: int,
    seed: int,
    start_ts: datetime,
    dt_s: float,
    kind: str,   # "inner" | "outer"
) -> list[dict]:
    random.seed(seed)

    n = len(center_xy)
    normals = _normals_closed_xy(center_xy)

    half_w = float(track_width_m) * 0.5
    sign = -1.0 if kind == "inner" else +1.0

    pts: list[dict] = []
    for i in range(n):
        cx, cy = center_xy[i]
        nx, ny = normals[i]

        x = cx + sign * half_w * nx
        y = cy + sign * half_w * ny

        lat, lon = _meters_to_latlon(x, y, lat0, lon0)
        ts = (start_ts + timedelta(seconds=i * dt_s)).isoformat()

        pts.append(
            {
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude": 550.0,
                "fix_quality": int(fix_quality),
                "speed": 0.0,
                "timestamp": ts,
                "hdop": 0.7,
                "sats": 16,
                "source": f"sim_{kind}",
            }
        )

    return pts



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
                "speed": float(speed_mps),
                "timestamp": ts,
                "hdop": 0.7,
                "sats": 16,
                "source": "sim",
            }
        )
    return pts


def _send_lap_fast(
    *,
    base_url: str,
    token: str,
    track_name: str,
    lap_type: str,
    points: list[dict],
    workers: int,
) -> str:
    lap_id = _start_lap(base_url, token, track_name, lap_type)
    print(f"[OK] Started lap: {lap_id} track_name={track_name} lap_type={lap_type}")

    if workers <= 1:
        for p in points:
            _send_point(base_url, token, p)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_send_point, base_url, token, p) for p in points]
            done = 0
            for f in as_completed(futs):
                _ = f.result()
                done += 1
                if done % 100 == 0 or done == len(points):
                    print(f"  sent {done}/{len(points)}")

    res = _stop_lap(base_url, token)
    print(f"[OK] Stopped lap: {lap_id} lap_time={res.get('lap_time')} points={res.get('points')}")
    return lap_id


def wait_for_optimal(
    *,
    base_url: str,
    token: str,
    track_id: str,
    timeout_s: float,
    poll_s: float,
) -> None:
    headers = {"Authorization": f"Bearer {token}"}

    url_csv = f"{base_url}/api/track/{track_id}/optimal_latlon.csv"
    url_json = f"{base_url}/api/track/{track_id}/optimal.json"

    t_end = time.time() + float(timeout_s)
    print(f"[*] Waiting for optimal from server (timeout {timeout_s}s)...")

    last_code = None
    while time.time() < t_end:
        code, data = _http_bytes("GET", url_csv, headers=headers)
        if code == 200 and data:
            out_csv = f"optimal_{track_id}_latlon.csv"
            with open(out_csv, "wb") as f:
                f.write(data)
            print(f"[OK] optimal_latlon.csv ready -> saved: {out_csv}")

            code2, data2 = _http_bytes("GET", url_json, headers=headers)
            if code2 == 200 and data2:
                out_json = f"optimal_{track_id}.json"
                with open(out_json, "wb") as f:
                    f.write(data2)
                print(f"[OK] optimal.json ready -> saved: {out_json}")
            return

        last_code = code
        time.sleep(float(poll_s))

    print(f"[FAIL] optimal not ready. last HTTP code for CSV was: {last_code}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sim client: send INNER + OUTER and wait for server auto-optimal.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--track-id", required=True)
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")

    ap.add_argument("--center-lat", type=float, default=42.7144)
    ap.add_argument("--center-lon", type=float, default=23.2743)

    ap.add_argument("--radius-inner-m", type=float, default=45.0)
    ap.add_argument("--track-width-m", type=float, default=10.0)

    ap.add_argument("--n-points", type=int, default=250, help="Use smaller for fast tests (e.g. 150-300).")
    ap.add_argument("--hz", type=float, default=50.0, help="Only affects timestamps; no sleep in fast mode.")

    ap.add_argument("--speed-kmh", type=float, default=80.0)
    ap.add_argument("--noise-m", type=float, default=0.2)
    ap.add_argument("--fix-quality", type=int, default=4)

    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--workers", type=int, default=8, help="Parallel point sends. 1 = sequential.")
    ap.add_argument("--wait-timeout-s", type=float, default=120.0)
    ap.add_argument("--wait-poll-s", type=float, default=1.0)

    ap.add_argument("--straight-m", type=float, default=120.0)
    ap.add_argument("--turn-radius-m", type=float, default=35.0)

    ap.add_argument("--wavy-base-radius-m", type=float, default=90.0)
    ap.add_argument("--wavy-a2", type=float, default=0.28)
    ap.add_argument("--wavy-a3", type=float, default=0.14)
    ap.add_argument("--wavy-a4", type=float, default=0.08)
    ap.add_argument("--wavy-rotate-deg", type=float, default=15.0)
    ap.add_argument("--wavy-squash-y", type=float, default=0.70)



    args = ap.parse_args()

    n = int(args.n_points)
    if n < 50:
        n = 50
    if n > 5000:
        n = 5000

    dt_s = 1.0 / max(0.1, float(args.hz))

    inner_r = float(args.radius_inner_m)
    outer_r = float(args.radius_inner_m) + float(args.track_width_m)

    print(f"[*] Logging in to {args.base_url} as {args.username}...")
    token = _login(args.base_url, args.username, args.password)
    print("[OK] Logged in")

    st = _get_status(args.base_url, token)
    if st.get("recording"):
        print("[WARN] Server already recording. Stopping active lap first...")
        _stop_lap_safe(args.base_url, token)

    t0 = datetime.now(timezone.utc)

    center = _wavy_roadcourse_center_xy(
        n_points=n,
        base_radius_m=float(args.wavy_base_radius_m),
        a2=float(args.wavy_a2),
        a3=float(args.wavy_a3),
        a4=float(args.wavy_a4),
        rotate_deg=float(args.wavy_rotate_deg),
        squash_y=float(args.wavy_squash_y),
    )



    inner_pts = _generate_inner_outer_stadium(
        lat0=float(args.center_lat),
        lon0=float(args.center_lon),
        center_xy=center,
        track_width_m=float(args.track_width_m),
        fix_quality=int(args.fix_quality),
        seed=1,
        start_ts=t0,
        dt_s=dt_s,
        kind="inner",
    )

    outer_pts = _generate_inner_outer_stadium(
        lat0=float(args.center_lat),
        lon0=float(args.center_lon),
        center_xy=center,
        track_width_m=float(args.track_width_m),
        fix_quality=int(args.fix_quality),
        seed=2,
        start_ts=t0,
        dt_s=dt_s,
        kind="outer",
    )



    track_name = args.track_id  # IMPORTANT: server expects clean track_id

    print("\n[*] Sending INNER lap (fast)...")
    inner_lap_id = _send_lap_fast(
        base_url=args.base_url,
        token=token,
        track_name=track_name,
        lap_type="inner",
        points=inner_pts,
        workers=int(args.workers),
    )

    print("\n[*] Sending OUTER lap (fast)...")
    outer_lap_id = _send_lap_fast(
        base_url=args.base_url,
        token=token,
        track_name=track_name,
        lap_type="outer",
        points=outer_pts,
        workers=int(args.workers),
    )

    print(f"\n[RESULT] inner_lap_id={inner_lap_id}")
    print(f"[RESULT] outer_lap_id={outer_lap_id}")

    wait_for_optimal(
        base_url=args.base_url,
        token=token,
        track_id=args.track_id,
        timeout_s=float(args.wait_timeout_s),
        poll_s=float(args.wait_poll_s),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
