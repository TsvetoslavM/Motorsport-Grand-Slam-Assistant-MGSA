"""
Quick sanity test for the new two-lines trajectory optimizer.

Usage (server mode):
  1) Start server:  python server/server.py
  2) Run:           python tests/try_trajectory_optimize.py --server

Usage (local mode, no server):
  python tests/try_trajectory_optimize.py

Notes:
  - Server mode uses default credentials from server/server.py: admin/admin123
  - If CasADi/IPOPT is not installed, local mode (and the server endpoint) will fail.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from typing import Dict, List


def _make_demo_lines(n: int = 120) -> Dict[str, List[Dict[str, float]]]:
    """
    Build a simple closed "oval" track as left/right boundaries.
    Two polylines share the same parameterization so widths are stable.
    """
    # Centerline: slightly squashed circle
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x = 60.0 * math.cos(t)
        y = 35.0 * math.sin(t)
        pts.append((x, y))

    # Tangent + normal (closed)
    left_line = []
    right_line = []
    track_half_width = 6.0
    for i in range(n):
        x_prev, y_prev = pts[(i - 1) % n]
        x_next, y_next = pts[(i + 1) % n]
        tx = x_next - x_prev
        ty = y_next - y_prev
        norm = math.hypot(tx, ty) or 1.0
        tx /= norm
        ty /= norm
        # left normal
        nx = -ty
        ny = tx
        x, y = pts[i]
        left_line.append({"x": x + nx * track_half_width, "y": y + ny * track_half_width})
        right_line.append({"x": x - nx * track_half_width, "y": y - ny * track_half_width})

    return {"left_line": left_line, "right_line": right_line}


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


def run_local(n_points: int, ipopt_max_iter: int) -> int:
    from firmware.Optimal_Control.solver_api import OptimizeOptions, optimize_trajectory_from_two_lines

    demo = _make_demo_lines()
    opts = OptimizeOptions(n_points=n_points, ipopt_max_iter=ipopt_max_iter, ipopt_print_level=0)
    result = optimize_trajectory_from_two_lines(demo["left_line"], demo["right_line"], options=opts)

    print("=== LOCAL OPTIMIZATION RESULT ===")
    print(f"converged: {result.get('converged')}")
    print(f"N: {result.get('N')}")
    print(f"track_length_m: {result.get('track_length_m'):.2f}")
    print(f"lap_time_s: {result.get('lap_time_s'):.3f}")
    print("first 3 optimal points:")
    for p in result["optimal"][:3]:
        print(f"  x={p['x']:.3f}, y={p['y']:.3f}, v={p['speed_kmh']:.1f} km/h, t={p['time_s']:.3f}s")
    return 0


def run_server(base_url: str, username: str, password: str, n_points: int, ipopt_max_iter: int) -> int:
    demo = _make_demo_lines()

    # login
    login = _http_json(
        "POST",
        f"{base_url}/api/auth/login",
        {"username": username, "password": password},
    )
    token = login["access_token"]

    payload = {
        **demo,
        "n_points": n_points,
        "ipopt_max_iter": ipopt_max_iter,
        "ipopt_print_level": 0,
    }

    result = _http_json(
        "POST",
        f"{base_url}/api/trajectory/optimize_from_two_lines",
        payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    print("=== SERVER OPTIMIZATION RESULT ===")
    print(f"converged: {result.get('converged')}")
    print(f"N: {result.get('N')}")
    print(f"track_length_m: {result.get('track_length_m'):.2f}")
    print(f"lap_time_s: {result.get('lap_time_s'):.3f}")
    print("first 3 optimal points:")
    for p in result["optimal"][:3]:
        print(f"  x={p['x']:.3f}, y={p['y']:.3f}, v={p['speed_kmh']:.1f} km/h, t={p['time_s']:.3f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="store_true", help="Call the FastAPI endpoint instead of local function")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--n-points", type=int, default=250)
    ap.add_argument("--ipopt-max-iter", type=int, default=2000)
    args = ap.parse_args()

    try:
        if args.server:
            return run_server(args.base_url, args.username, args.password, args.n_points, args.ipopt_max_iter)
        return run_local(args.n_points, args.ipopt_max_iter)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        print(f"HTTP error: {e.code} {e.reason}\n{detail}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

