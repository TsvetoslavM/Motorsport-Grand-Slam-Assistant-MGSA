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
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def _get_status(base_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    return _http_json("GET", f"{base_url}/api/status", None, headers=headers)


def _start_lap(base_url: str, token: str, track_name: str, lap_type: str, session_id: str | None = None) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "track_name": track_name,
        "lap_type": lap_type,
    }
    if session_id is not None:
        payload["session_id"] = session_id

    resp = _http_json("POST", f"{base_url}/api/lap/start", payload, headers=headers)
    return str(resp["lap_id"])


def _stop_lap(base_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    return _http_json("POST", f"{base_url}/api/lap/stop", None, headers=headers)


def _stop_lap_safe(base_url: str, token: str) -> None:
    try:
        _stop_lap(base_url, token)
    except Exception:
        pass


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


def wait_for_compare_json(
    *,
    base_url: str,
    token: str,
    track_id: str,
    timeout_s: float = 60.0,
    poll_s: float = 1.0,
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{base_url}/api/track/{track_id}/compare_driver_vs_optimal.json"

    t_end = time.time() + float(timeout_s)
    last_code = None

    while time.time() < t_end:
        code, data = _http_bytes("GET", url, headers=headers)
        last_code = code

        if code == 200 and data:
            try:
                return json.loads(data.decode("utf-8"))
            except Exception:
                pass

        time.sleep(float(poll_s))

    raise TimeoutError(f"compare_driver_vs_optimal.json not ready, last HTTP code={last_code}")


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
    return 0.5 * a


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
    n = max(40, int(n_points))
    R0 = max(5.0, float(base_radius_m))

    rot = math.radians(float(rotate_deg))
    cr = math.cos(rot)
    sr = math.sin(rot)

    pts: list[tuple[float, float]] = []
    for i in range(n):
        th = 2.0 * math.pi * (i / n)

        r = R0 * (
            1.0
            + float(a2) * math.cos(2.0 * th)
            + float(a3) * math.cos(3.0 * th)
            + float(a4) * math.cos(4.0 * th)
        )

        x = r * math.cos(th)
        y = r * math.sin(th) * float(squash_y)

        xr = x * cr - y * sr
        yr = x * sr + y * cr
        pts.append((xr, yr))

    return _ensure_ccw_xy(pts)


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
        out.append((-ty, tx))
    return out


def _generate_inner_outer_points(
    *,
    lat0: float,
    lon0: float,
    center_xy: list[tuple[float, float]],
    track_width_m: float,
    fix_quality: int,
    seed: int,
    start_ts: datetime,
    dt_s: float,
    kind: str,
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


def _generate_driver_points_from_centerline(
    *,
    lat0: float,
    lon0: float,
    center_xy: list[tuple[float, float]],
    track_width_m: float,
    driver_bias: float,
    v_straight_kmh: float,
    v_corner_kmh: float,
    noise_m: float,
    fix_quality: int,
    seed: int,
    start_ts: datetime,
    dt_s: float,
    source: str = "sim_driver",
    global_offset_m: float = 0.0,
    speed_noise_kmh: float = 0.0,
) -> list[dict]:
    random.seed(seed)
    n = len(center_xy)
    normals = _normals_closed_xy(center_xy)

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

        da_signed = math.atan2(cross, dot)
        mag = abs(da_signed)
        corner_mag[i] = min(1.0, mag / (math.pi / 3.0))
        corner_sign[i] = 1.0 if da_signed >= 0.0 else -1.0

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

        c = corner_mag[i]
        sgn = corner_sign[i]

        target = half_w * c * sgn
        offset = bias * target + float(global_offset_m)

        x = cx + offset * nx
        y = cy + offset * ny

        if noise_m > 0.0:
            x += random.gauss(0.0, noise_m)
            y += random.gauss(0.0, noise_m)

        v_kmh = (1.0 - c) * float(v_straight_kmh) + c * float(v_corner_kmh)
        if speed_noise_kmh > 0.0:
            v_kmh += random.gauss(0.0, speed_noise_kmh)

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
                "source": source,
            }
        )
    return pts


def _send_lap_points(
    *,
    base_url: str,
    token: str,
    track_name: str,
    lap_type: str,
    points: list[dict],
    workers: int,
    realtime: bool,
    dt_s: float,
    session_id: str | None = None,
) -> dict:
    lap_id = _start_lap(base_url, token, track_name, lap_type, session_id=session_id)
    print(f"[OK] Started lap: {lap_id} track_name={track_name} lap_type={lap_type}")

    if realtime and workers > 1:
        print("[WARN] realtime mode -> forcing workers=1")
        workers = 1

    if workers <= 1:
        for i, p in enumerate(points, 1):
            _send_point(base_url, token, p)
            if i % 50 == 0 or i == len(points):
                print(f"  sent {i}/{len(points)}")
            if realtime:
                time.sleep(dt_s)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_send_point, base_url, token, p) for p in points]
            done = 0
            total = len(futs)
            for f in as_completed(futs):
                _ = f.result()
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"  sent {done}/{total}")

    res = _stop_lap(base_url, token)
    print(f"[OK] Stopped lap: {lap_id} lap_time={res.get('lap_time')} points={res.get('points')}")
    return {
        "lap_id": lap_id,
        "lap_time": res.get("lap_time"),
        "points": res.get("points"),
        "status": res.get("status"),
    }


def _profile_params(profile: str) -> tuple[float, float, float]:
    profile = profile.lower()
    if profile == "fast":
        return 0.90, 130.0, 85.0
    if profile == "medium":
        return 0.55, 105.0, 72.0
    if profile == "slow":
        return 0.20, 85.0, 60.0
    raise ValueError(f"Unknown profile: {profile}")


def _save_results_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_monte_carlo_tests(
    *,
    base_url: str,
    token: str,
    track_id: str,
    center_xy: list[tuple[float, float]],
    lat0: float,
    lon0: float,
    track_width_m: float,
    fix_quality: int,
    dt_s: float,
    workers: int,
    realtime: bool,
    base_bias: float,
    base_v_straight_kmh: float,
    base_v_corner_kmh: float,
    runs: int,
    compare_timeout_s: float = 60.0,
) -> list[dict]:
    results = []

    for run_idx in range(runs):
        seed = 1000 + run_idx

        noise_m = random.uniform(0.02, 0.10)
        bias_run = max(0.0, min(1.0, base_bias + random.uniform(-0.10, 0.10)))
        v_straight_run = base_v_straight_kmh + random.uniform(-5.0, 5.0)
        v_corner_run = base_v_corner_kmh + random.uniform(-3.0, 3.0)
        global_offset_m = random.uniform(-0.10, 0.10)
        speed_noise_kmh = random.uniform(0.0, 2.0)

        t_run = datetime.now(timezone.utc)

        driver_pts = _generate_driver_points_from_centerline(
            lat0=lat0,
            lon0=lon0,
            center_xy=center_xy,
            track_width_m=track_width_m,
            driver_bias=bias_run,
            v_straight_kmh=v_straight_run,
            v_corner_kmh=v_corner_run,
            noise_m=noise_m,
            fix_quality=fix_quality,
            seed=seed,
            start_ts=t_run,
            dt_s=dt_s,
            source=f"mc_driver_{run_idx}",
            global_offset_m=global_offset_m,
            speed_noise_kmh=speed_noise_kmh,
        )

        print(
            f"\n[*] MC run {run_idx + 1}/{runs} "
            f"noise={noise_m:.3f}m "
            f"bias={bias_run:.3f} "
            f"offset={global_offset_m:.3f}m "
            f"v_straight={v_straight_run:.1f} "
            f"v_corner={v_corner_run:.1f} "
            f"speed_noise={speed_noise_kmh:.2f}"
        )

        lap_info = _send_lap_points(
            base_url=base_url,
            token=token,
            track_name=track_id,
            lap_type="driver",
            session_id=None,
            points=driver_pts,
            workers=workers,
            realtime=realtime,
            dt_s=dt_s,
        )

        compare_payload = wait_for_compare_json(
            base_url=base_url,
            token=token,
            track_id=track_id,
            timeout_s=compare_timeout_s,
            poll_s=1.0,
        )

        stats = compare_payload.get("stats", {})

        row = {
            "run_idx": run_idx + 1,
            "lap_id": lap_info.get("lap_id"),
            "lap_time": lap_info.get("lap_time"),
            "points": lap_info.get("points"),
            "noise_m": round(noise_m, 4),
            "bias": round(bias_run, 4),
            "global_offset_m": round(global_offset_m, 4),
            "v_straight_kmh": round(v_straight_run, 2),
            "v_corner_kmh": round(v_corner_run, 2),
            "speed_noise_kmh": round(speed_noise_kmh, 2),
        }

        for k, v in stats.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f"stats_{k}"] = v

        results.append(row)

    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Send INNER + OUTER, wait 10s, then send one DRIVER lap or run Monte Carlo driver tests."
    )

    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--track-id", required=True)

    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")

    ap.add_argument("--profile", choices=["fast", "medium", "slow"], default="medium")
    ap.add_argument("--wait-after-boundaries", type=float, default=10.0)

    ap.add_argument("--kind", default="driver", help="racing line kind for build_from_lap")

    ap.add_argument("--center-lat", type=float, default=42.7144)
    ap.add_argument("--center-lon", type=float, default=23.2743)

    ap.add_argument("--n-points", type=int, default=300)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--fix-quality", type=int, default=4)
    ap.add_argument("--noise-m", type=float, default=0.25)
    ap.add_argument("--track-width-m", type=float, default=10.0)

    ap.add_argument("--wavy-base-radius-m", type=float, default=110.0)
    ap.add_argument("--wavy-a2", type=float, default=0.18)
    ap.add_argument("--wavy-a3", type=float, default=0.08)
    ap.add_argument("--wavy-a4", type=float, default=0.04)
    ap.add_argument("--wavy-rotate-deg", type=float, default=15.0)
    ap.add_argument("--wavy-squash-y", type=float, default=0.72)

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--realtime", action="store_true")

    ap.add_argument("--wait-timeout-s", type=float, default=120.0)
    ap.add_argument("--wait-poll-s", type=float, default=1.0)

    ap.add_argument("--monte-carlo-runs", type=int, default=0)
    ap.add_argument("--compare-timeout-s", type=float, default=60.0)

    args = ap.parse_args()

    n = max(50, min(5000, int(args.n_points)))
    dt_s = 1.0 / max(0.1, float(args.hz))

    print(f"[*] Logging in to {args.base_url} as {args.username}...")
    token = _login(args.base_url, args.username, args.password)
    print("[OK] Logged in")

    st = _get_status(args.base_url, token)
    if st.get("recording"):
        print("[WARN] Server already recording. Stopping active lap first...")
        _stop_lap_safe(args.base_url, token)

    center_xy = _wavy_roadcourse_center_xy(
        n_points=n,
        base_radius_m=float(args.wavy_base_radius_m),
        a2=float(args.wavy_a2),
        a3=float(args.wavy_a3),
        a4=float(args.wavy_a4),
        rotate_deg=float(args.wavy_rotate_deg),
        squash_y=float(args.wavy_squash_y),
    )

    boundary_session_id = f"bset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    print(f"[*] boundary_session_id={boundary_session_id}")

    t0 = datetime.now(timezone.utc)

    inner_pts = _generate_inner_outer_points(
        lat0=float(args.center_lat),
        lon0=float(args.center_lon),
        center_xy=center_xy,
        track_width_m=float(args.track_width_m),
        fix_quality=int(args.fix_quality),
        seed=1,
        start_ts=t0,
        dt_s=dt_s,
        kind="inner",
    )

    outer_pts = _generate_inner_outer_points(
        lat0=float(args.center_lat),
        lon0=float(args.center_lon),
        center_xy=center_xy,
        track_width_m=float(args.track_width_m),
        fix_quality=int(args.fix_quality),
        seed=2,
        start_ts=t0,
        dt_s=dt_s,
        kind="outer",
    )

    print("\n[*] Sending INNER lap...")
    inner_lap_info = _send_lap_points(
        base_url=args.base_url,
        token=token,
        track_name=args.track_id,
        lap_type="inner",
        session_id=boundary_session_id,
        points=inner_pts,
        workers=int(args.workers),
        realtime=args.realtime,
        dt_s=dt_s,
    )

    print("\n[*] Sending OUTER lap...")
    outer_lap_info = _send_lap_points(
        base_url=args.base_url,
        token=token,
        track_name=args.track_id,
        lap_type="outer",
        session_id=boundary_session_id,
        points=outer_pts,
        workers=int(args.workers),
        realtime=args.realtime,
        dt_s=dt_s,
    )

    print(f"\n[RESULT] inner_lap_id={inner_lap_info['lap_id']}")
    print(f"[RESULT] outer_lap_id={outer_lap_info['lap_id']}")

    wait_for_optimal(
        base_url=args.base_url,
        token=token,
        track_id=args.track_id,
        timeout_s=float(args.wait_timeout_s),
        poll_s=float(args.wait_poll_s),
    )

    print(f"\n[*] Waiting {args.wait_after_boundaries} seconds before driver lap(s)...")
    time.sleep(float(args.wait_after_boundaries))

    driver_bias, v_straight_kmh, v_corner_kmh = _profile_params(args.profile)
    print(
        f"[*] Racing profile={args.profile} "
        f"driver_bias={driver_bias} "
        f"v_straight_kmh={v_straight_kmh} "
        f"v_corner_kmh={v_corner_kmh}"
    )

    if int(args.monte_carlo_runs) > 0:
        results = run_monte_carlo_tests(
            base_url=args.base_url,
            token=token,
            track_id=args.track_id,
            center_xy=center_xy,
            lat0=float(args.center_lat),
            lon0=float(args.center_lon),
            track_width_m=float(args.track_width_m),
            fix_quality=int(args.fix_quality),
            dt_s=dt_s,
            workers=int(args.workers),
            realtime=args.realtime,
            base_bias=driver_bias,
            base_v_straight_kmh=v_straight_kmh,
            base_v_corner_kmh=v_corner_kmh,
            runs=int(args.monte_carlo_runs),
            compare_timeout_s=float(args.compare_timeout_s),
        )

        out_csv = f"mc_results_{args.track_id}.csv"
        _save_results_csv(out_csv, results)
        print(f"\n[OK] Monte Carlo results saved to {out_csv}")
        print("[DONE] Monte Carlo scenario completed successfully.")
        return 0

    t1 = datetime.now(timezone.utc)

    driver_pts = _generate_driver_points_from_centerline(
        lat0=float(args.center_lat),
        lon0=float(args.center_lon),
        center_xy=center_xy,
        track_width_m=float(args.track_width_m),
        driver_bias=driver_bias,
        v_straight_kmh=v_straight_kmh,
        v_corner_kmh=v_corner_kmh,
        noise_m=float(args.noise_m),
        fix_quality=int(args.fix_quality),
        seed=7,
        start_ts=t1,
        dt_s=dt_s,
        source=f"sim_driver_{args.profile}",
        global_offset_m=0.0,
        speed_noise_kmh=0.0,
    )

    print("\n[*] Sending DRIVER lap...")
    lap_info = _send_lap_points(
        base_url=args.base_url,
        token=token,
        track_name=args.track_id,
        lap_type="driver",
        session_id=None,
        points=driver_pts,
        workers=int(args.workers),
        realtime=args.realtime,
        dt_s=dt_s,
    )

    compare_payload = wait_for_compare_json(
        base_url=args.base_url,
        token=token,
        track_id=args.track_id,
        timeout_s=float(args.compare_timeout_s),
        poll_s=1.0,
    )

    print(f"\n[RESULT] lap_info={lap_info}")
    print(f"[RESULT] compare stats={compare_payload.get('stats', {})}")
    print("[DONE] Full scenario completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())