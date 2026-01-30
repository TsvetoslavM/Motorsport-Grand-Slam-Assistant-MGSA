"""
Fetch boundaries + racing lines from MGSA server and generate a comparison HTML page.

What it shows:
  - Outer boundary (yellow)
  - Inner boundary (blue)
  - Any number of racing lines (different colors)
  - Start/finish rectangle (10x5m) centered between boundaries

Usage:
  1) Start server (example): python -m uvicorn server.server:app --host 0.0.0.0 --port 8001
  2) Run:
     python -m tests.try_visualize_track_compare --base-url http://127.0.0.1:8001 --track-id mytrack --kinds optimal driver

Defaults:
  - username/password: admin/admin123 (from server/auth.py)
"""

from __future__ import annotations

import argparse
import csv
import html as html_escape
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import folium


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


def _http_text(url: str, headers: dict | None = None) -> str:
    req = urllib.request.Request(url, method="GET", headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8")


def _rectangle_latlon(lat: float, lon: float, *, lat0: float, length_m: float = 10.0, width_m: float = 5.0) -> list[tuple[float, float]]:
    """Axis-aligned rectangle around (lat,lon) sized length x width meters."""
    dx = length_m / 2.0
    dy = width_m / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0
    dlat = dy / m_per_deg_lat
    dlon = dx / m_per_deg_lon
    return [
        (lat - dlat, lon - dlon),
        (lat - dlat, lon + dlon),
        (lat + dlat, lon + dlon),
        (lat + dlat, lon - dlon),
        (lat - dlat, lon - dlon),
    ]


def _parse_boundaries_csv(text: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    r = csv.DictReader(text.splitlines())
    outer: List[Tuple[float, float]] = []
    inner: List[Tuple[float, float]] = []
    for row in r:
        outer.append((float(row["outer_lat"]), float(row["outer_lon"])))
        inner.append((float(row["inner_lat"]), float(row["inner_lon"])))
    if not outer or not inner:
        raise ValueError("boundaries.csv has no data rows")
    return outer, inner


def _parse_racing_csv(text: str) -> List[Tuple[float, float, float, float]]:
    """Return list of (time_s, lat, lon, speed_kmh)."""
    r = csv.DictReader(text.splitlines())
    pts = []
    for row in r:
        pts.append((float(row["time_s"]), float(row["lat"]), float(row["lon"]), float(row["speed_kmh"])))
    if not pts:
        raise ValueError("racing line CSV has no data rows")
    return pts


def _wrap_map_page(*, title: str, subtitle: str, legend_html: str, map_html: str, footer_left: str, footer_right: str) -> str:
    iframe_srcdoc = html_escape.escape(map_html, quote=True)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_escape.escape(title)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    html, body {{ margin:0; padding:0; background:#0b1220; }}
    .checkered-bg {{ background: repeating-linear-gradient(45deg,#000,#000 16px,#fff 16px,#fff 32px); height: 4px; }}
    .header {{ padding:18px 22px; background:linear-gradient(135deg,#1a1a1a 0%,#0d0d0d 100%); color:#fff; border-bottom:3px solid #dc2626; }}
    .container {{ max-width:1200px; margin:0 auto; padding:0 22px; }}
    .title {{ font-family:'Orbitron',sans-serif; font-size:22px; font-weight:900; letter-spacing:2px; text-transform:uppercase; color:#dc2626; }}
    .kicker {{ font-family:'Orbitron',sans-serif; font-size:12px; color:#fbbf24; font-weight:700; text-transform:uppercase; letter-spacing:1px; }}
    .text {{ background:#0f172a; border-bottom:2px solid #1e293b; padding:16px 0; color:#e2e8f0; font-family:'Orbitron',sans-serif; line-height:1.6; }}
    .legend {{ margin-top:10px; font-size:13px; color:#94a3b8; padding:10px; background:#1e293b; border-left:3px solid #dc2626; border-radius:4px; }}
    .map-wrap {{ background:#0b1220; padding:16px 0; }}
    iframe {{ width:100%; height:72vh; border:0; border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,0.35); background:#0b1220; }}
    .footer {{ padding:14px 0; background:linear-gradient(135deg,#0d0d0d 0%,#1a1a1a 100%); color:#94a3b8; border-top:2px solid #1e293b; font-family:'Orbitron',sans-serif; font-size:11px; }}
    .footer-row {{ display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; align-items:center; }}
  </style>
</head>
<body>
  <div class="checkered-bg"></div>
  <div class="header">
    <div class="container" style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
      <div class="title">{html_escape.escape(title)}</div>
      <div class="kicker">{html_escape.escape(subtitle)}</div>
    </div>
  </div>
  <div class="text">
    <div class="container">
      <div class="legend">{legend_html}</div>
    </div>
  </div>
  <div class="map-wrap">
    <div class="container">
      <iframe srcdoc="{iframe_srcdoc}"></iframe>
    </div>
  </div>
  <div class="checkered-bg"></div>
  <div class="footer">
    <div class="container footer-row">
      <div style="font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;">{html_escape.escape(footer_left)}</div>
      <div style="color:#64748b;">{html_escape.escape(footer_right)}</div>
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--track-id", required=True)
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--kinds", nargs="+", help="Racing line kinds to overlay (e.g. optimal driver). If omitted, only boundaries are shown.")
    ap.add_argument("--out", default="track_compare.html")
    args = ap.parse_args()

    # Login
    login = _http_json("POST", f"{args.base_url}/api/auth/login", {"username": args.username, "password": args.password})
    token = login["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Download boundaries
    try:
        boundaries_text = _http_text(f"{args.base_url}/api/track/{args.track_id}/boundaries", headers=headers)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[ERROR] Boundaries not found for track '{args.track_id}'", file=sys.stderr)
            print(f"        Upload boundaries first using:", file=sys.stderr)
            print(f"        python -m tests.upload_and_visualize_track --track-id {args.track_id} --boundaries-csv boundaries.csv", file=sys.stderr)
            return 1
        raise
    outer, inner = _parse_boundaries_csv(boundaries_text)
    center_lat = sum(p[0] for p in outer) / len(outer)
    center_lon = sum(p[1] for p in outer) / len(outer)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, control_scale=True)
    folium.PolyLine(outer, color="#fbbf24", weight=4, tooltip="Outer Boundary").add_to(m)
    folium.PolyLine(inner, color="#3b82f6", weight=4, tooltip="Inner Boundary").add_to(m)

    # Start/finish rectangle centered between boundaries at index 0
    start_lat = 0.5 * (outer[0][0] + inner[0][0])
    start_lon = 0.5 * (outer[0][1] + inner[0][1])
    rect = _rectangle_latlon(start_lat, start_lon, lat0=center_lat, length_m=10.0, width_m=5.0)
    folium.Polygon(rect, color="#10b981", weight=2, fill=True, fill_color="#10b981", fill_opacity=0.4, tooltip="Start/Finish").add_to(m)

    # Racing lines overlay
    palette = ["#dc2626", "#a855f7", "#22c55e", "#06b6d4", "#f97316", "#eab308"]
    legend_parts = [
        '<strong style="color:#fbbf24;">Legend:</strong> '
        '<span style="color:#fbbf24;">Yellow</span>=Outer • '
        '<span style="color:#3b82f6;">Blue</span>=Inner • '
        '<span style="color:#10b981;">Green</span>=Start/Finish'
    ]

    if args.kinds:
        for idx, kind in enumerate(args.kinds):
            try:
                racing_text = _http_text(f"{args.base_url}/api/track/{args.track_id}/racing_line/{kind}", headers=headers)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"[WARN] Racing line '{kind}' not found, skipping", file=sys.stderr)
                    continue
                raise RuntimeError(f"Error fetching racing line '{kind}': {e.code} {e.reason}")

            pts = _parse_racing_csv(racing_text)
            coords = [(lat, lon) for (_t, lat, lon, _v) in pts]
            color = palette[idx % len(palette)]
            folium.PolyLine(coords, color=color, weight=5, tooltip=f"Racing Line: {kind}").add_to(m)
            legend_parts.append(f' • <span style="color:{color};">{html_escape.escape(kind)}</span>')

    map_html = m.get_root().render()
    page = _wrap_map_page(
        title="MGSA • Track Compare",
        subtitle=f"track_id={args.track_id}",
        legend_html="".join(legend_parts),
        map_html=map_html,
        footer_left="Generated by MGSA compare tool",
        footer_right="Tip: toggle overlays by clicking lines • zoom to inspect",
    )

    out_path = Path(args.out)
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            pass
        print(f"HTTP error: {e.code} {e.reason}\n{detail}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise

