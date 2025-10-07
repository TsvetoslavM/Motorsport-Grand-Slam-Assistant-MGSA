import os
import glob
import csv
from typing import List, Tuple, Optional

import numpy as np

from ..vmax_raceline.vmax import VehicleParams, speed_profile, compute_lap_time_seconds


Point = Tuple[float, float]


def load_points_csv(path: str) -> List[Point]:
    pts: List[Point] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            pts.append((float(row[0]), float(row[1])))
    return pts


def find_latest_turn_dir(base_turns_dir: Optional[str] = None) -> str:
    firmware_dir = os.path.dirname(os.path.dirname(__file__))
    turns_dir = base_turns_dir or os.path.join(firmware_dir, "Turns")
    pattern = os.path.join(turns_dir, "segments_*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        raise FileNotFoundError(f"No segments_* folder found under: {turns_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def compare_paths_html_from_segments(segments_dir: str, out_html: Optional[str] = None) -> str:
    """Visualize only the Turn_#.csv and Straight_#.csv corridors (x_l,y_l,x_r,y_r) and the optimized trajectory.

    The optimized trajectory is expected as Turn_1_best.csv in the same directory.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise RuntimeError("plotly is required. Install with: pip install plotly")

    def read_edges_csv(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            cols = [c.strip().lower() for c in header]
            required = {"x_l", "y_l", "x_r", "y_r"}
            if not required.issubset(set(cols)):
                return None
            ix_l = cols.index("x_l"); iy_l = cols.index("y_l")
            ix_r = cols.index("x_r"); iy_r = cols.index("y_r")
            left: List[Tuple[float, float]] = []
            right: List[Tuple[float, float]] = []
            for row in reader:
                if not row:
                    continue
                left.append((float(row[ix_l]), float(row[iy_l])))
                right.append((float(row[ix_r]), float(row[iy_r])))
        return np.asarray(left, dtype=float), np.asarray(right, dtype=float)

    # Gather segments
    turn_files = sorted(glob.glob(os.path.join(segments_dir, "Turn_*.csv")))
    straight_files = sorted(glob.glob(os.path.join(segments_dir, "Straight_*.csv")))

    # Plot setup
    fig = go.Figure()

    # Add straights (black)
    for path in straight_files:
        edges = read_edges_csv(path)
        if edges is None:
            continue
        left, right = edges
        fig.add_trace(go.Scatter(x=left[:, 0], y=left[:, 1], mode="lines", name=os.path.basename(path)+"_L",
                                 line=dict(color="#111111", width=2)))
        fig.add_trace(go.Scatter(x=right[:, 0], y=right[:, 1], mode="lines", name=os.path.basename(path)+"_R",
                                 line=dict(color="#111111", width=2)))

    # Add turns (blue)
    for path in turn_files:
        edges = read_edges_csv(path)
        if edges is None:
            continue
        left, right = edges
        fig.add_trace(go.Scatter(x=left[:, 0], y=left[:, 1], mode="lines", name=os.path.basename(path)+"_L",
                                 line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(x=right[:, 0], y=right[:, 1], mode="lines", name=os.path.basename(path)+"_R",
                                 line=dict(color="#2563eb", width=2)))

    # Add optimized trajectory, if present
    best_csv = os.path.join(segments_dir, "Turn_1_best.csv")
    if os.path.isfile(best_csv):
        best_pts = load_points_csv(best_csv)
        best = np.asarray(best_pts, dtype=float)
        fig.add_trace(go.Scatter(x=best[:, 0], y=best[:, 1], mode="lines",
                                 name="Optimized Trajectory", line=dict(color="#22c55e", width=3)))

    # Figure layout
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="x [m]")
    fig.update_yaxes(title_text="y [m]")
    fig.update_layout(
        title="Segments and Optimized Trajectory",
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99),
        height=800,
        margin=dict(t=60, l=60, r=40, b=60),
    )

    if out_html is None:
        out_html = os.path.join(segments_dir, "segments_and_optimized.html")
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True, auto_open=False)
    return out_html


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Straight_#/Turn_# corridors and optimized trajectory")
    parser.add_argument("--turns-dir", type=str, default=None, help="Base Turns directory (defaults to firmware/Turns)")
    parser.add_argument("--segments-dir", type=str, default=None, help="Explicit segments_* directory to visualize")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    if args.segments_dir and os.path.isdir(args.segments_dir):
        seg_dir = args.segments_dir
    else:
        seg_dir = find_latest_turn_dir(args.turns_dir)

    out_html = compare_paths_html_from_segments(seg_dir, out_html=args.out)
    print(f"Wrote segments visualization HTML to: {out_html}")


if __name__ == "__main__":
    main()


