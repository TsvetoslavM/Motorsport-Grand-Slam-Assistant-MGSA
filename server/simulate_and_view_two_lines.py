"""
Generate simulated outer/inner boundary points and visualize them with Folium.

This produces a `boundaries.csv` compatible with `server/view_two_lines.py`
and also writes an interactive HTML map.

Usage:
  python server/simulate_and_view_two_lines.py
  python server/simulate_and_view_two_lines.py --out boundaries.csv --html track_boundaries.html

Tip:
  Open the generated HTML file in your browser.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import folium
import pandas as pd


def generate_boundaries_latlon(
    *,
    n: int = 400,
    center_lat: float = 42.6977,
    center_lon: float = 23.3219,
    radius_m_x: float = 120.0,
    radius_m_y: float = 70.0,
    track_width_m: float = 12.0,
) -> pd.DataFrame:
    """
    Build a closed track in lat/lon (non-circular) so the optimal line is visible.

    We generate a centerline in local meters, compute normals, then offset
    by +/- track_width/2 to get outer/inner lines. We also output a synthetic
    time_s column (cumulative time at constant speed) for each sample.
    """
    n = max(20, int(n))
    half_w = float(track_width_m) / 2.0

    # meters per degree conversions (local approximation)
    lat0 = float(center_lat)
    lon0 = float(center_lon)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0

    # Non-circular centerline in meters: harmonically deformed ellipse
    # (gives distinct corners/straights but remains smooth and closed).
    cx_m: list[float] = []
    cy_m: list[float] = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        # base ellipse
        x = radius_m_x * math.cos(t)
        y = radius_m_y * math.sin(t)
        # add 2nd + 3rd harmonic deformations (creates "corners")
        x += 0.18 * radius_m_x * math.cos(2.0 * t + 0.4)
        y += 0.22 * radius_m_y * math.sin(3.0 * t - 0.7)
        cx_m.append(x)
        cy_m.append(y)

    # compute tangent + left normal on the closed curve
    outer_lat: list[float] = []
    outer_lon: list[float] = []
    inner_lat: list[float] = []
    inner_lon: list[float] = []

    for i in range(n):
        x_prev, y_prev = cx_m[(i - 1) % n], cy_m[(i - 1) % n]
        x_next, y_next = cx_m[(i + 1) % n], cy_m[(i + 1) % n]
        tx = x_next - x_prev
        ty = y_next - y_prev
        norm = math.hypot(tx, ty) or 1.0
        tx /= norm
        ty /= norm
        # left unit normal
        nx = -ty
        ny = tx

        x_c, y_c = cx_m[i], cy_m[i]
        # Outer = left side, Inner = right side (same convention as boundaries build in server)
        x_outer = x_c + nx * half_w
        y_outer = y_c + ny * half_w
        x_inner = x_c - nx * half_w
        y_inner = y_c - ny * half_w

        outer_lat.append(lat0 + (y_outer / m_per_deg_lat))
        outer_lon.append(lon0 + (x_outer / m_per_deg_lon))
        inner_lat.append(lat0 + (y_inner / m_per_deg_lat))
        inner_lon.append(lon0 + (x_inner / m_per_deg_lon))

    # Synthetic time axis (constant speed along centerline)
    # NOTE: Only for CSV convenience/plotting; the optimizer does not use this.
    v_mps = 20.0
    ds = []
    for i in range(n):
        x0, y0 = cx_m[i], cy_m[i]
        x1, y1 = cx_m[(i + 1) % n], cy_m[(i + 1) % n]
        ds.append(math.hypot(x1 - x0, y1 - y0))
    time_s = [0.0]
    for i in range(1, n):
        time_s.append(time_s[-1] + ds[i - 1] / v_mps)

    return pd.DataFrame(
        {
            "time_s": time_s,
            "outer_lat": outer_lat,
            "outer_lon": outer_lon,
            "inner_lat": inner_lat,
            "inner_lon": inner_lon,
        }
    )


def _latlon_to_local_xy_m(lat: float, lon: float, *, lat0: float, lon0: float) -> tuple[float, float]:
    """Approximate conversion from lat/lon to local meters (x east, y north)."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0
    x_m = (lon - lon0) * m_per_deg_lon
    y_m = (lat - lat0) * m_per_deg_lat
    return x_m, y_m


def _local_xy_m_to_latlon(x_m: float, y_m: float, *, lat0: float, lon0: float) -> tuple[float, float]:
    """Approximate conversion from local meters to lat/lon."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    if abs(m_per_deg_lon) < 1e-9:
        m_per_deg_lon = 1.0
    lat = lat0 + (y_m / m_per_deg_lat)
    lon = lon0 + (x_m / m_per_deg_lon)
    return lat, lon


def _rectangle_latlon(
    lat: float,
    lon: float,
    *,
    lat0: float,
    lon0: float,
    length_m: float = 10.0,
    width_m: float = 5.0,
) -> list[tuple[float, float]]:
    """
    Build an axis-aligned rectangle (length x width, meters) around a center point.
    Returned as list of (lat, lon) forming a closed ring.
    """
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


def optimize_from_boundaries_csv(
    csv_path: Path,
    *,
    n_points: int = 250,
    ipopt_max_iter: int = 2000,
) -> pd.DataFrame:
    """
    Load boundaries.csv (lat/lon), run optimizer in local meters, and return DF
    with optimal points back in lat/lon.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty")

    required = {"outer_lat", "outer_lon", "inner_lat", "inner_lon"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    lat0 = float(df["outer_lat"].mean())
    lon0 = float(df["outer_lon"].mean())

    outer_xy = []
    inner_xy = []
    for lat, lon in zip(df["outer_lat"], df["outer_lon"]):
        x_m, y_m = _latlon_to_local_xy_m(float(lat), float(lon), lat0=lat0, lon0=lon0)
        outer_xy.append({"x": x_m, "y": y_m})
    for lat, lon in zip(df["inner_lat"], df["inner_lon"]):
        x_m, y_m = _latlon_to_local_xy_m(float(lat), float(lon), lat0=lat0, lon0=lon0)
        inner_xy.append({"x": x_m, "y": y_m})

    from firmware.Optimal_Control.solver_api import OptimizeOptions, optimize_trajectory_from_two_lines

    opts = OptimizeOptions(n_points=int(n_points), ipopt_max_iter=int(ipopt_max_iter), ipopt_print_level=0)
    result = optimize_trajectory_from_two_lines(outer_xy, inner_xy, options=opts)

    opt_lat = []
    opt_lon = []
    for p in result["optimal"]:
        lat, lon = _local_xy_m_to_latlon(float(p["x"]), float(p["y"]), lat0=lat0, lon0=lon0)
        opt_lat.append(lat)
        opt_lon.append(lon)

    # time_s: align to the CSV time base if present, else generate 0..N
    if "time_s" in df.columns and len(df["time_s"]) >= len(opt_lat):
        time_s = [float(t) for t in df["time_s"].iloc[: len(opt_lat)].tolist()]
    else:
        time_s = [float(i) for i in range(len(opt_lat))]

    return pd.DataFrame({"time_s": time_s, "opt_lat": opt_lat, "opt_lon": opt_lon})


def visualize_boundaries_csv(csv_path: Path, html_path: Path, *, zoom_start: int = 17) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty")

    center_lat = float(df["outer_lat"].mean())
    center_lon = float(df["outer_lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=int(zoom_start), control_scale=True)

    outer_coords = list(zip(df["outer_lat"], df["outer_lon"]))
    inner_coords = list(zip(df["inner_lat"], df["inner_lon"]))

    folium.PolyLine(outer_coords, color="#fbbf24", weight=4, tooltip="Outer Boundary").add_to(m)
    folium.PolyLine(inner_coords, color="#3b82f6", weight=4, tooltip="Inner Boundary").add_to(m)

    # Start/finish rectangle
    start_lat = 0.5 * (outer_coords[0][0] + inner_coords[0][0])
    start_lon = 0.5 * (outer_coords[0][1] + inner_coords[0][1])
    rect = _rectangle_latlon(
        start_lat,
        start_lon,
        lat0=center_lat,
        lon0=center_lon,
        length_m=10.0,
        width_m=5.0,
    )
    folium.Polygon(
        locations=rect,
        color="#10b981",
        weight=2,
        fill=True,
        fill_color="#10b981",
        fill_opacity=0.4,
        tooltip="Start/Finish Line",
    ).add_to(m)

    # Embed the Folium HTML via iframe to avoid layout whitespace issues.
    map_html = m.get_root().render()
    iframe_srcdoc = html.escape(map_html, quote=True)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MGSA • Track Boundaries</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Racing+Sans+One&family=Orbitron:wght@400;700;900&display=swap');
    html, body {{ margin:0; padding:0; background:#0b1220; }}
    .checkered-bg {{ background: repeating-linear-gradient(45deg,#000,#000 20px,#fff 20px,#fff 40px); height: 4px; }}
    .header {{ padding:20px 24px; background:linear-gradient(135deg,#1a1a1a 0%,#0d0d0d 100%); color:#fff; border-bottom:3px solid #dc2626; box-shadow:0 2px 8px rgba(0,0,0,0.3); }}
    .container {{ max-width:1200px; margin:0 auto; padding:0 24px; }}
    .title {{ font-family:'Orbitron',sans-serif; font-size:24px; font-weight:900; letter-spacing:2px; text-transform:uppercase; color:#dc2626; text-shadow:0 0 10px rgba(220,38,38,0.5); }}
    .kicker {{ font-family:'Orbitron',sans-serif; font-size:12px; color:#fbbf24; font-weight:700; text-transform:uppercase; letter-spacing:1px; }}
    .text {{ background:#0f172a; border-bottom:2px solid #1e293b; padding:18px 0; color:#e2e8f0; font-family:'Orbitron',sans-serif; line-height:1.7; }}
    .legend {{ margin-top:10px; font-size:13px; color:#94a3b8; padding:10px; background:#1e293b; border-left:3px solid #dc2626; border-radius:4px; }}
    .map-wrap {{ background:#0b1220; padding:16px 0; }}
    iframe {{ width:100%; height:70vh; border:0; border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,0.35); background:#0b1220; }}
    .footer {{ padding:16px 0; background:linear-gradient(135deg,#0d0d0d 0%,#1a1a1a 100%); color:#94a3b8; border-top:2px solid #1e293b; font-family:'Orbitron',sans-serif; font-size:11px; }}
    .footer-row {{ display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; align-items:center; }}
  </style>
</head>
<body>
  <div class="checkered-bg"></div>
  <div class="header">
    <div class="container" style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
      <div class="title">MGSA • Track Boundaries</div>
      <div class="kicker">Track Visualization</div>
    </div>
  </div>
  <div class="text">
    <div class="container">
      <div style="font-size:16px;font-weight:700;color:#fbbf24;text-transform:uppercase;letter-spacing:1px;">Track Boundary Visualization</div>
      <div style="font-size:14px;color:#cbd5e1;margin-top:8px;">
        This map displays the <span style="color:#fbbf24;font-weight:700;">outer boundary</span> (yellow) and
        <span style="color:#3b82f6;font-weight:700;">inner boundary</span> (blue) of the racing track.
      </div>
      <div class="legend">
        <strong style="color:#fbbf24;">Legend:</strong>
        <span style="color:#fbbf24;">Yellow</span> = Outer •
        <span style="color:#3b82f6;">Blue</span> = Inner •
        <span style="color:#10b981;">Green</span> = Start/Finish (10×5m, centered)
      </div>
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
      <div style="font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Generated by MGSA Simulator</div>
      <div style="color:#64748b;">Tip: Click lines for tooltips • Zoom to inspect boundaries</div>
    </div>
  </div>
</body>
</html>
"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page_html, encoding="utf-8")


def visualize_boundaries_and_optimal(
    csv_path: Path,
    html_path: Path,
    *,
    zoom_start: int = 17,
    n_points: int = 250,
    ipopt_max_iter: int = 2000,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty")

    center_lat = float(df["outer_lat"].mean())
    center_lon = float(df["outer_lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=int(zoom_start), control_scale=True)

    outer_coords = list(zip(df["outer_lat"], df["outer_lon"]))
    inner_coords = list(zip(df["inner_lat"], df["inner_lon"]))
    folium.PolyLine(outer_coords, color="#fbbf24", weight=4, tooltip="Outer Boundary").add_to(m)
    folium.PolyLine(inner_coords, color="#3b82f6", weight=4, tooltip="Inner Boundary").add_to(m)

    opt_df = optimize_from_boundaries_csv(csv_path, n_points=n_points, ipopt_max_iter=ipopt_max_iter)
    opt_coords = list(zip(opt_df["opt_lat"], opt_df["opt_lon"]))
    folium.PolyLine(opt_coords, color="#dc2626", weight=5, tooltip="Optimal Racing Line").add_to(m)

    start_lat = 0.5 * (outer_coords[0][0] + inner_coords[0][0])
    start_lon = 0.5 * (outer_coords[0][1] + inner_coords[0][1])
    rect = _rectangle_latlon(
        start_lat,
        start_lon,
        lat0=center_lat,
        lon0=center_lon,
        length_m=10.0,
        width_m=5.0,
    )
    folium.Polygon(
        locations=rect,
        color="#10b981",
        weight=2,
        fill=True,
        fill_color="#10b981",
        fill_opacity=0.4,
        tooltip="Start/Finish Line",
    ).add_to(m)

    map_html = m.get_root().render()
    iframe_srcdoc = html.escape(map_html, quote=True)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MGSA • Optimal Racing Line</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Racing+Sans+One&family=Orbitron:wght@400;700;900&display=swap');
    html, body {{ margin:0; padding:0; background:#0b1220; }}
    .checkered-bg {{ background: repeating-linear-gradient(45deg,#000,#000 20px,#fff 20px,#fff 40px); height: 4px; }}
    .header {{ padding:20px 24px; background:linear-gradient(135deg,#1a1a1a 0%,#0d0d0d 100%); color:#fff; border-bottom:3px solid #dc2626; box-shadow:0 2px 8px rgba(0,0,0,0.3); }}
    .container {{ max-width:1200px; margin:0 auto; padding:0 24px; }}
    .title {{ font-family:'Orbitron',sans-serif; font-size:24px; font-weight:900; letter-spacing:2px; text-transform:uppercase; color:#dc2626; text-shadow:0 0 10px rgba(220,38,38,0.5); }}
    .kicker {{ font-family:'Orbitron',sans-serif; font-size:12px; color:#fbbf24; font-weight:700; text-transform:uppercase; letter-spacing:1px; }}
    .text {{ background:#0f172a; border-bottom:2px solid #1e293b; padding:18px 0; color:#e2e8f0; font-family:'Orbitron',sans-serif; line-height:1.7; }}
    .legend {{ margin-top:10px; font-size:13px; color:#94a3b8; padding:10px; background:#1e293b; border-left:3px solid #dc2626; border-radius:4px; }}
    .map-wrap {{ background:#0b1220; padding:16px 0; }}
    iframe {{ width:100%; height:70vh; border:0; border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,0.35); background:#0b1220; }}
    .footer {{ padding:16px 0; background:linear-gradient(135deg,#0d0d0d 0%,#1a1a1a 100%); color:#94a3b8; border-top:2px solid #1e293b; font-family:'Orbitron',sans-serif; font-size:11px; }}
    .footer-row {{ display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; align-items:center; }}
  </style>
</head>
<body>
  <div class="checkered-bg"></div>
  <div class="header">
    <div class="container" style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
      <div class="title">MGSA • Optimal Racing Line</div>
      <div class="kicker">Track Analysis</div>
    </div>
  </div>
  <div class="text">
    <div class="container">
      <div style="font-size:16px;font-weight:700;color:#fbbf24;text-transform:uppercase;letter-spacing:1px;">Track Boundaries & Optimal Trajectory</div>
      <div style="font-size:14px;color:#cbd5e1;margin-top:8px;">
        This visualization shows the <span style="color:#fbbf24;font-weight:700;">outer boundary</span> (yellow),
        <span style="color:#3b82f6;font-weight:700;">inner boundary</span> (blue), and the
        <span style="color:#dc2626;font-weight:700;">optimal racing line</span> (red) computed using CasADi/IPOPT optimization.
      </div>
      <div class="legend">
        <strong style="color:#fbbf24;">Legend:</strong>
        <span style="color:#fbbf24;">Yellow</span> = Outer •
        <span style="color:#3b82f6;">Blue</span> = Inner •
        <span style="color:#dc2626;">Red</span> = Optimal •
        <span style="color:#10b981;">Green</span> = Start/Finish (10×5m, centered)
      </div>
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
      <div style="font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Generated by MGSA Simulator + CasADi/IPOPT</div>
      <div style="color:#64748b;">Tip: Click lines for tooltips • Zoom to inspect trajectory</div>
    </div>
  </div>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page_html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="boundaries.csv", help="Output CSV path")
    ap.add_argument("--html", default="track_boundaries.html", help="Output HTML path")
    ap.add_argument("--n", type=int, default=400, help="Number of points")
    ap.add_argument("--center-lat", type=float, default=42.6977)
    ap.add_argument("--center-lon", type=float, default=23.3219)
    ap.add_argument("--rx", type=float, default=120.0, help="Ellipse radius X in meters")
    ap.add_argument("--ry", type=float, default=70.0, help="Ellipse radius Y in meters")
    ap.add_argument("--width", type=float, default=12.0, help="Track width in meters")
    ap.add_argument("--zoom", type=int, default=17, help="Initial map zoom")
    ap.add_argument("--optimize", action="store_true", help="Also compute and draw optimal line (CasADi/IPOPT)")
    ap.add_argument("--opt-n-points", type=int, default=250, help="Points used for optimization")
    ap.add_argument("--opt-ipopt-max-iter", type=int, default=2000, help="IPOPT max iterations")
    args = ap.parse_args()

    csv_path = Path(args.out)
    html_path = Path(args.html)

    df = generate_boundaries_latlon(
        n=args.n,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        radius_m_x=args.rx,
        radius_m_y=args.ry,
        track_width_m=args.width,
    )
    df.to_csv(csv_path, index=False)

    if args.optimize:
        visualize_boundaries_and_optimal(
            csv_path,
            html_path,
            zoom_start=args.zoom,
            n_points=args.opt_n_points,
            ipopt_max_iter=args.opt_ipopt_max_iter,
        )
    else:
        visualize_boundaries_csv(csv_path, html_path, zoom_start=args.zoom)

    print(f"Wrote: {csv_path.resolve()}")
    print(f"Wrote: {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

