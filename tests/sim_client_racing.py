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

def _stadium_center_xy(n_points: int, straight_m: float, radius_m: float) -> list[tuple[float, float]]:
    n = max(40, int(n_points))
    Ls = float(straight_m)
    R = float(radius_m)

    if Ls < 1e-3:
        Ls = 1.0
    if R < 1.0:
        R = 1.0

    # Stadium aligned on X axis:
    # top straight:  (-Ls/2,+R) -> (+Ls/2,+R)
    # right arc:    center (+Ls/2,0) from +90° to -90° (clockwise, right turn)
    # bottom straight: (+Ls/2,-R) -> (-Ls/2,-R)
    # left arc:     center (-Ls/2,0) from -90° to +90° (clockwise, left turn)
    total = 2.0 * Ls + 2.0 * math.pi * R
    ds = total / n

    pts: list[tuple[float, float]] = []
    s = 0.0

    while len(pts) < n:
        if s < Ls:
            # top straight
            x = -Ls / 2.0 + s
            y = +R
            pts.append((x, y))

        elif s < Ls + math.pi * R:
            # right arc: +90 -> -90 (clockwise)
            u = (s - Ls) / R  # 0..pi
            ang = (math.pi / 2.0) - u
            cx, cy = +Ls / 2.0, 0.0
            x = cx + R * math.cos(ang)
            y = cy + R * math.sin(ang)
            pts.append((x, y))

        elif s < 2.0 * Ls + math.pi * R:
            # bottom straight
            t = s - (Ls + math.pi * R)
            x = +Ls / 2.0 - t
            y = -R
            pts.append((x, y))

        else:
            # left arc: -90 -> +90 (clockwise)
            t = s - (2.0 * Ls + math.pi * R)
            u = t / R  # 0..pi
            ang = (-math.pi / 2.0) + u
            cx, cy = -Ls / 2.0, 0.0
            x = cx + R * math.cos(ang)
            y = cy + R * math.sin(ang)
            pts.append((x, y))

        s += ds

    return pts


def _normals_closed_xy(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
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

        # left normal (CCW)
        out.append((-ty, tx))
    return out


def _generate_driver_points_stadium(
    *,
    lat0: float,
    lon0: float,
    center_xy: list[tuple[float, float]],
    track_width_m: float,
    driver_bias: float,   # 0..1
    v_straight_kmh: float,
    v_corner_kmh: float,
    noise_m: float,
    fix_quality: int,
    seed: int,
    start_ts: datetime,
    dt_s: float,
) -> list[dict]:
    random.seed(seed)
    n = len(center_xy)
    normals = _normals_closed_xy(center_xy)

    # signed corner factor using atan2(cross, dot)
    corner_mag = [0.0] * n
    corner_sign = [1.0] * n

    for i in range(n):
        x0, y0 = center_xy[i - 1]
        x1, y1 = center_xy[i]
        x2, y2 = center_xy[(i + 1) % n]

        v1x, v1y = (x1 - x0), (y1 - y0)
        v2x, v2y = (x2 - x1), (y2 - y1)

        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y

        da_signed = math.atan2(cross, dot)  # -pi..pi
        mag = abs(da_signed)
        corner_mag[i] = min(1.0, mag / (math.pi / 3.0))
        corner_sign[i] = 1.0 if da_signed >= 0.0 else -1.0

    # smooth magnitude and sign (moving average)
    sm_mag = [0.0] * n
    sm_sgn = [0.0] * n
    for i in range(n):
        sm_mag[i] = (
            corner_mag[i - 2] + corner_mag[i - 1] + corner_mag[i] +
            corner_mag[(i + 1) % n] + corner_mag[(i + 2) % n]
        ) / 5.0
        sm_sgn[i] = (
            corner_sign[i - 2] + corner_sign[i - 1] + corner_sign[i] +
            corner_sign[(i + 1) % n] + corner_sign[(i + 2) % n]
        ) / 5.0

    corner_mag = sm_mag
    corner_sign = [1.0 if x >= 0.0 else -1.0 for x in sm_sgn]

    half_w = float(track_width_m) * 0.5
    bias = max(0.0, min(1.0, float(driver_bias)))

    pts: list[dict] = []
    for i in range(n):
        cx, cy = center_xy[i]
        nx, ny = normals[i]

        c = corner_mag[i]          # 0..1
        sgn = corner_sign[i]       # -1 or +1

        # NOTE:
        # For your current normals (left normal), "inside" for a right turn is typically -normal,
        # and for a left turn is +normal. The sgn from atan2(cross,dot) matches that.
        target = (half_w) * c * sgn
        offset = bias * target

        x = cx + offset * nx
        y = cy + offset * ny

        if noise_m > 0.0:
            x += (random.random() * 2.0 - 1.0) * noise_m
            y += (random.random() * 2.0 - 1.0) * noise_m

        v_kmh = (1.0 - c) * float(v_straight_kmh) + c * float(v_corner_kmh)
        v_mps = max(0.0, v_kmh) / 3.6

        lat, lon = _meters_to_latlon(x, y, lat0, lon0)
        ts = (start_ts + timedelta(seconds=i * dt_s)).isoformat()

        pts.append(
            {
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude": 550.0,
                "fix_quality": int(fix_quality),
                "speed": float(v_mps),
                "timestamp": ts,
                "hdop": 0.7,
                "sats": 16,
                "source": "sim_driver",
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

    ap.add_argument("--straight-m", type=float, default=140.0)
    ap.add_argument("--turn-radius-m", type=float, default=35.0)
    ap.add_argument("--track-width-m", type=float, default=10.0)

    ap.add_argument("--driver-bias", type=float, default=0.25, help="0..1 how much driver aims for racing line (0=center).")
    ap.add_argument("--v-straight-kmh", type=float, default=110.0)
    ap.add_argument("--v-corner-kmh", type=float, default=70.0)

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

    center_xy = _stadium_center_xy(n_points=n, straight_m=float(args.straight_m), radius_m=float(args.turn_radius_m))

    pts = _generate_driver_points_stadium(
        lat0=float(args.center_lat),
        lon0=float(args.center_lon),
        center_xy=center_xy,
        track_width_m=float(args.track_width_m),
        driver_bias=float(args.driver_bias),
        v_straight_kmh=float(args.v_straight_kmh),
        v_corner_kmh=float(args.v_corner_kmh),
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
