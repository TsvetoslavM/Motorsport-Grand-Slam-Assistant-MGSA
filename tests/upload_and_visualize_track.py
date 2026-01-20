"""
Upload boundaries + racing lines to server, then generate comparison visualization.

This script:
  1. Uploads boundaries.csv (from local file or generates sample)
  2. Optionally uploads racing lines
  3. Generates comparison HTML

Usage:
  python -m tests.upload_and_visualize_track --base-url http://127.0.0.1:8001 --track-id mytrack --boundaries-csv boundaries.csv --compute-optimal --simulate-line driver"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

import subprocess


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
    """Login and return Bearer token."""
    resp = _http_json("POST", f"{base_url}/api/auth/login", {"username": username, "password": password})
    return resp["access_token"]


def _upload_boundaries_json(base_url: str, track_id: str, token: str, csv_path: Path) -> None:
    """Upload boundaries.csv as JSON samples."""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            samples.append({
                "time_s": float(row["time_s"]),
                "outer_lat": float(row["outer_lat"]),
                "outer_lon": float(row["outer_lon"]),
                "inner_lat": float(row["inner_lat"]),
                "inner_lon": float(row["inner_lon"]),
            })
    
    headers = {"Authorization": f"Bearer {token}"}
    resp = _http_json("POST", f"{base_url}/api/track/{track_id}/boundaries/upload_json", {"samples": samples}, headers=headers)
    print(f"[OK] Uploaded boundaries: {len(samples)} points")


def _upload_racing_line(base_url: str, track_id: str, kind: str, token: str, csv_path: Path) -> None:
    """Upload racing line CSV (time_s,lat,lon,speed_kmh)."""
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            points.append({
                "time_s": float(row["time_s"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "speed_kmh": float(row.get("speed_kmh", 0.0)),
            })
    
    headers = {"Authorization": f"Bearer {token}"}
    resp = _http_json("POST", f"{base_url}/api/track/{track_id}/racing_line/upload", {"kind": kind, "points": points}, headers=headers)
    print(f"[OK] Uploaded racing line '{kind}': {len(points)} points")


def _latlon_to_xy_m(lat: float, lon: float, *, lat0: float, lon0: float) -> tuple[float, float]:
    """Local tangent-plane approximation: x east (m), y north (m)."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat
    return x, y


def _estimate_curvature(points_xy: list[tuple[float, float]]) -> list[float]:
    """Very rough discrete curvature proxy per point using turning angle / segment length."""
    n = len(points_xy)
    if n < 3:
        return [0.0] * n
    out = [0.0] * n
    for i in range(n):
        x0, y0 = points_xy[(i - 1) % n]
        x1, y1 = points_xy[i]
        x2, y2 = points_xy[(i + 1) % n]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        a1 = math.atan2(v1y, v1x)
        a2 = math.atan2(v2y, v2x)
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        ds = max(1e-3, math.hypot(v1x, v1y) + math.hypot(v2x, v2y))
        out[i] = abs(da) / ds
    return out


def _generate_simulated_racing_line_from_boundaries(
    boundaries_csv: Path,
    *,
    seed: int = 1,
    base_speed_kmh: float = 110.0,
    speed_noise_kmh: float = 6.0,
    deviation_frac_amp: float = 0.22,
    deviation_noise_amp: float = 0.05,
    wave_cycles: float = 2.5,
) -> list[dict]:
    """
    Create an imperfect 'driver' line by blending between inner/outer with smooth oscillation + noise.
    Output points: {time_s, lat, lon, speed_kmh}
    """
    random.seed(seed)
    rows = []
    with open(boundaries_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise ValueError("boundaries.csv is empty")

    # Reference for local meters
    lat0 = sum(float(r["outer_lat"]) for r in rows) / len(rows)
    lon0 = sum(float(r["outer_lon"]) for r in rows) / len(rows)

    # Build simulated lat/lon points
    pts_latlon: list[tuple[float, float]] = []
    for i, r in enumerate(rows):
        o_lat = float(r["outer_lat"]); o_lon = float(r["outer_lon"])
        in_lat = float(r["inner_lat"]); in_lon = float(r["inner_lon"])

        # f in [0,1]: 0=inner, 1=outer. Oscillate + noise, then clamp.
        phase = 2.0 * math.pi * wave_cycles * (i / max(1, len(rows) - 1))
        f = 0.5 + deviation_frac_amp * math.sin(phase) + deviation_noise_amp * (random.random() * 2.0 - 1.0)
        f = max(0.05, min(0.95, f))

        lat = in_lat + f * (o_lat - in_lat)
        lon = in_lon + f * (o_lon - in_lon)
        pts_latlon.append((lat, lon))

    # Estimate curvature in meters to modulate speed (slower in corners)
    pts_xy = [_latlon_to_xy_m(lat, lon, lat0=lat0, lon0=lon0) for lat, lon in pts_latlon]
    kappa = _estimate_curvature(pts_xy)

    points = []
    for i, r in enumerate(rows):
        t = float(r.get("time_s", i))
        lat, lon = pts_latlon[i]

        # Speed model: base - gain*curvature + noise
        v = base_speed_kmh - 650.0 * kappa[i] + (random.random() * 2.0 - 1.0) * speed_noise_kmh
        v = max(10.0, v)

        points.append({"time_s": t, "lat": lat, "lon": lon, "speed_kmh": v})

    return points


def _upload_racing_points(base_url: str, track_id: str, kind: str, token: str, points: list[dict]) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    _http_json("POST", f"{base_url}/api/track/{track_id}/racing_line/upload", {"kind": kind, "points": points}, headers=headers)
    print(f"[OK] Uploaded simulated racing line '{kind}': {len(points)} points")


def _generate_optimal_line_from_boundaries_via_solver(
    boundaries_csv: Path,
    *,
    n_points: int = 250,
    ipopt_max_iter: int = 2000,
) -> list[dict]:
    """
    Use firmware.Optimal_Control.solver_api via server.simulate_and_view_two_lines.optimize_from_boundaries_csv
    to compute an \"optimal\" line from the inner/outer boundaries.
    Produces points with time_s, lat, lon, speed_kmh.
    """
    from server.simulate_and_view_two_lines import optimize_from_boundaries_csv

    import pandas as pd

    df_opt = optimize_from_boundaries_csv(boundaries_csv, n_points=n_points, ipopt_max_iter=ipopt_max_iter)
    if df_opt.empty:
        raise ValueError("Optimal DF is empty")

    lat0 = float(df_opt["opt_lat"].mean())
    lon0 = float(df_opt["opt_lon"].mean())

    # Convert to local meters to estimate speed from distance / dt
    xy: list[tuple[float, float]] = []
    for lat, lon in zip(df_opt["opt_lat"], df_opt["opt_lon"]):
        xy.append(_latlon_to_xy_m(float(lat), float(lon), lat0=lat0, lon0=lon0))

    time_s = [float(t) for t in df_opt["time_s"].tolist()]
    n = len(time_s)
    eps = 1e-3
    speeds_mps: list[float] = [0.0] * n
    for i in range(1, n):
        x0, y0 = xy[i - 1]
        x1, y1 = xy[i]
        ds = math.hypot(x1 - x0, y1 - y0)
        dt = max(eps, time_s[i] - time_s[i - 1])
        speeds_mps[i] = ds / dt
    # simple smoothing
    for i in range(1, n - 1):
        speeds_mps[i] = 0.25 * speeds_mps[i - 1] + 0.5 * speeds_mps[i] + 0.25 * speeds_mps[i + 1]

    points: list[dict] = []
    for t, lat, lon, v_mps in zip(time_s, df_opt["opt_lat"], df_opt["opt_lon"], speeds_mps):
        points.append(
            {
                "time_s": float(t),
                "lat": float(lat),
                "lon": float(lon),
                "speed_kmh": float(max(5.0, v_mps * 3.6)),
            }
        )
    return points


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload track data and visualize")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="Server base URL")
    ap.add_argument("--track-id", required=True, help="Track ID")
    ap.add_argument("--username", default="admin", help="Login username")
    ap.add_argument("--password", default="admin123", help="Login password")
    ap.add_argument("--boundaries-csv", type=Path, help="Local boundaries.csv path (time_s,outer_lat,outer_lon,inner_lat,inner_lon)")
    ap.add_argument("--racing-line", nargs=2, metavar=("KIND", "CSV"), action="append", help="Upload racing line: --racing-line optimal raceline.csv (can repeat)")
    ap.add_argument("--compute-optimal", action="store_true", help="Compute optimal line from boundaries.csv via CasADi/IPOPT and upload as 'optimal'")
    ap.add_argument("--simulate-line", metavar="KIND", help="Generate and upload an imperfect simulated racing line from boundaries.csv as KIND (e.g. driver)")
    ap.add_argument("--simulate-seed", type=int, default=1)
    ap.add_argument("--simulate-base-speed", type=float, default=110.0)
    ap.add_argument("--simulate-speed-noise", type=float, default=6.0)
    ap.add_argument("--simulate-deviation-amp", type=float, default=0.22, help="Lateral deviation amplitude as fraction between inner/outer (0..0.45)")
    ap.add_argument("--simulate-deviation-noise", type=float, default=0.05, help="Extra random deviation fraction")
    ap.add_argument("--kinds", nargs="+", help="Racing line kinds to visualize (default: all uploaded)")
    ap.add_argument("--out", default="track_compare.html", help="Output HTML path")
    ap.add_argument("--skip-upload", action="store_true", help="Skip upload, only visualize")
    args = ap.parse_args()

    if not args.skip_upload:
        print(f"[*] Logging in as {args.username}...")
        token = _login(args.base_url, args.username, args.password)
        print(f"[OK] Logged in")

        if args.boundaries_csv:
            if not args.boundaries_csv.exists():
                print(f"[ERROR] Boundaries CSV not found: {args.boundaries_csv}", file=sys.stderr)
                return 1
            print(f"[*] Uploading boundaries from {args.boundaries_csv}...")
            _upload_boundaries_json(args.base_url, args.track_id, token, args.boundaries_csv)
        else:
            print("[WARN] No --boundaries-csv provided, skipping boundaries upload")

        if args.racing_line:
            for kind, csv_path in args.racing_line:
                csv_p = Path(csv_path)
                if not csv_p.exists():
                    print(f"[ERROR] Racing line CSV not found: {csv_path}", file=sys.stderr)
                    return 1
                print(f"[*] Uploading racing line '{kind}' from {csv_path}...")
                _upload_racing_line(args.base_url, args.track_id, kind, token, csv_p)
        else:
            print("[WARN] No --racing-line provided, skipping racing line upload")

        # Optional: generate simulated line from boundaries.csv
        if args.simulate_line:
            if not args.boundaries_csv or not args.boundaries_csv.exists():
                print("[ERROR] --simulate-line requires --boundaries-csv to exist locally", file=sys.stderr)
                return 1
            dev_amp = max(0.0, min(0.45, float(args.simulate_deviation_amp)))
            dev_noise = max(0.0, min(0.20, float(args.simulate_deviation_noise)))
            pts = _generate_simulated_racing_line_from_boundaries(
                args.boundaries_csv,
                seed=int(args.simulate_seed),
                base_speed_kmh=float(args.simulate_base_speed),
                speed_noise_kmh=float(args.simulate_speed_noise),
                deviation_frac_amp=dev_amp,
                deviation_noise_amp=dev_noise,
            )
            _upload_racing_points(args.base_url, args.track_id, args.simulate_line, token, pts)

        # Optional: compute optimal from boundaries via solver
        if args.compute_optimal:
            if not args.boundaries_csv or not args.boundaries_csv.exists():
                print("[ERROR] --compute-optimal requires --boundaries-csv to exist locally", file=sys.stderr)
                return 1
            print("[*] Computing optimal line from boundaries via CasADi/IPOPT...")
            pts_opt = _generate_optimal_line_from_boundaries_via_solver(args.boundaries_csv)
            _upload_racing_points(args.base_url, args.track_id, "optimal", token, pts_opt)

    # Now visualize
    kinds = args.kinds
    auto_kinds: list[str] = []
    if args.racing_line:
        auto_kinds.extend(k for k, _ in args.racing_line)
    if args.simulate_line:
        auto_kinds.append(args.simulate_line)
    if args.compute_optimal:
        auto_kinds.append("optimal")

    if not kinds and auto_kinds:
        # default to all uploaded kinds if user didn't specify
        kinds = []
        for k in auto_kinds:
            if k not in kinds:
                kinds.append(k)
    elif kinds:
        # ensure we include any generated kinds (optimal / simulate_line) if user already listed others
        for k in auto_kinds:
            if k not in kinds:
                kinds.append(k)
    
    print(f"\n[*] Generating visualization...")
    cmd = [
        sys.executable, "-m", "tests.try_visualize_track_compare",
        "--base-url", args.base_url,
        "--track-id", args.track_id,
        "--username", args.username,
        "--password", args.password,
        "--out", args.out,
    ]
    if kinds:
        cmd.extend(["--kinds"] + kinds)
    # If no kinds specified, visualization will show only boundaries (which is fine)
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n[OK] Visualization saved: {args.out}")
        print(f"     Open in browser: file:///{Path(args.out).absolute()}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
