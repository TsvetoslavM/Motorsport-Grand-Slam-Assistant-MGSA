from .curvature import curvature, curvature_vectorized
from .smoothing import smooth_curvatures_vectorized, smooth_curvatures_multi_scale
from .segmentation import segment_track_multi_scale, segment_by_median_mad
from .visualization import (
    plot_curvature_heatmap,
    plotly_curvature_heatmap_html,
    plotly_track_outline_from_widths_html,
)
from .examples import visualize_curvature_concept, test_segmentation_debug, compare_performance
from .io import load_points
from .track_coloring import plot_turns_and_straights
from .vmax_raceline import VehicleParams, speed_profile, export_csv, plot_speed_vs_s
import os
import argparse
import json
from datetime import datetime


def parse_args() -> argparse.Namespace:
    """Create and parse CLI arguments for curvature demos and utilities."""
    parser = argparse.ArgumentParser(description="Curvature demos and segmentation utilities")
    parser.add_argument("--points", type=str, default=None, help="Path to CSV/JSON points file")
    parser.add_argument("--factor", type=float, default=3.0, help="MAD threshold factor (for --mad)")
    parser.add_argument("--mad", action="store_true", help="Run median+MAD segmentation on points file")
    parser.add_argument("--heatmap", action="store_true", help="Render curvature heatmap (requires matplotlib)")
    parser.add_argument("--web", type=str, default=None, help="Export interactive 2D heatmap to HTML at given path (requires plotly)")
    parser.add_argument("--raceline", type=str, default=None, help="Optional CSV with x,y to overlay as racing line on 2D web export")
    parser.add_argument("--outline-csv", type=str, default=None, help="CSV with columns x,y,left,right to render track outline")
    parser.add_argument("--outline-web", type=str, default=None, help="Output HTML path for outline render")
    parser.add_argument("--print-json", action="store_true", help="Print MAD segments as JSON output")
    parser.add_argument("--turns-web", type=str, default=None, help="Output HTML path for turns/straights coloring (blue turns, black straights)")
    parser.add_argument("--turns-factor", type=float, default=3.0, help="MAD factor for turns/straights segmentation")
    parser.add_argument("--v_max", action="store_true", help="Compute vmax speed profile using data/raceline.csv only")
    return parser.parse_args()


def resolve_points_file(args: argparse.Namespace) -> str | None:
    """Resolve the points file path from args or environment or default CSV."""
    points_file = args.points or os.environ.get("TRACK_POINTS_FILE")
    if not points_file:
        default_csv = os.path.join(os.path.dirname(__file__), "data", "monza.csv")
        points_file = default_csv if os.path.isfile(default_csv) else None
    return points_file


def print_segments(segments: list, header: str) -> None:
    """Pretty print segmentation results to stdout."""
    print(header)
    for i, s in enumerate(segments, 1):
        if s.get("type") == "turn":
            print(f"  {i}. turn  entry={s['entry_point']} apex={s['apex_point']} exit={s['exit_point']}")
        else:
            print(f"  {i}. straight  start={s['start_point']} end={s['end_point']}")


def run_mad_segmentation(points: list[tuple[float, float]], factor: float, print_json_flag: bool) -> None:
    """Run median+MAD segmentation on given points and print results."""
    curvs = curvature_vectorized(points)
    segments = segment_by_median_mad(curvs, factor=factor, points=points)
    if print_json_flag:
        print(json.dumps(segments, ensure_ascii=False, indent=2))
    else:
        print_segments(segments, header=f"MAD segmentation on {len(points)} points (factor={factor}): {len(segments)} segments")


def handle_outline_render(args: argparse.Namespace) -> None:
    """Render track outline HTML, optionally overlaying a raceline and running MAD on raceline."""
    if not (args.outline_csv and args.outline_web):
        return
    try:
        from .io import load_centerline_with_widths
        from .vmax_raceline import VehicleParams, speed_profile
        cws = load_centerline_with_widths(args.outline_csv)
        raceline_pts = None
        raceline_vmax = None
        raceline_vopt = None
        if args.raceline:
            try:
                from .io import load_points as _load_points
                raceline_pts = _load_points(args.raceline)
                # Compute v_max and optimal speed along raceline for hover (use shared defaults)
                params = VehicleParams()
                s, kappa, v_lat, v = speed_profile(raceline_pts, params)
                raceline_vmax = v_lat
                raceline_vopt = v
            except Exception as e:
                print(f"Failed to load raceline '{args.raceline}': {e}")
        ok = plotly_track_outline_from_widths_html(
            cws,
            args.outline_web,
            title=f"Spa, Belgium",
            raceline_points=raceline_pts,
            raceline_vmax=raceline_vmax,
            raceline_vopt=raceline_vopt,
            css_file= "templates/outline.css",
        )
        if ok:
            print(f"Wrote outline HTML to {args.outline_web}")

        # Also export per-segment CSVs based on available geometry (prefer centerline with widths)
        try:
            # Prefer using centerline-with-widths so we can compute left/right edges for inner/outer
            export_points = cws
            if export_points and len(export_points) >= 3:
                curvs_for_export = curvature_vectorized(export_points)
                mad_value = getattr(args, "turns_factor", None)
                if mad_value is None:
                    mad_value = getattr(args, "factor", 3.0)
                export_segments_csv(export_points, curvs_for_export, base_output_dir=None, mad_factor=mad_value)
        except Exception as e:
            print(f"Failed to export segment CSVs (outline flow): {e}")

        if args.mad:
            if raceline_pts is None:
                print("MAD requested but no raceline provided. Use --raceline <path>.")
            else:
                run_mad_segmentation(cws, factor=args.factor, print_json_flag=args.print_json)

    except Exception as e:
        print(f"Failed outline render: {e}")


def render_static_heatmaps(points: list[tuple[float, float]], curvs, points_file: str, heatmap_2d: bool) -> None:
    """Render 2D matplotlib heatmap if requested."""
    if heatmap_2d:
        res = plot_curvature_heatmap(points, curvs, title=f"Curvature Heatmap ({os.path.basename(points_file)})")
        if res is not None:
            plt, fig, ax = res
            plt.show()


def export_web_heatmaps(points: list[tuple[float, float]], curvs, args: argparse.Namespace, points_file: str) -> None:
    """Export Plotly-based heatmaps to HTML, optionally overlay raceline for 2D."""
    if args.web:
        raceline_pts = None
        if args.raceline:
            try:
                from .io import load_points as _load_points
                raceline_pts = _load_points(args.raceline)
            except Exception as e:
                print(f"Failed to load raceline '{args.raceline}': {e}")
        ok = plotly_curvature_heatmap_html(
            points,
            curvs,
            args.web,
            title=f"Curvature Heatmap ({os.path.basename(points_file)})",
            raceline_points=raceline_pts,
        )
        if ok:
            print(f"Wrote HTML to {args.web}")


def export_turns_straights(points: list[tuple[float, float]], curvs, args: argparse.Namespace, points_file: str) -> None:
    """Export turns/straights colored HTML using MAD-based segmentation."""
    if not args.turns_web:
        return
    ok = plot_turns_and_straights(
        points,
        curvs,
        output_html=args.turns_web,
        title=f"Turns/Straights ({os.path.basename(points_file)})",
        mad_factor=args.turns_factor,
    )
    if ok:
        print(f"Wrote HTML to {args.turns_web}")


def export_segments_csv(points: list[tuple[float, float]] | list[tuple[float, float, float, float]], curvs, base_output_dir: str | None = None, mad_factor: float = 3.0) -> str | None:
    """Create a new folder and write CSV files with edge coordinates for each turn and straight.

    - Output columns: x_l, y_l, x_r, y_r (left and right edges)
    - Folder name: data/segments_YYYYMMDD_HHMMSS under repository root (or provided base_output_dir)
    - Files:
        Turn_1.csv, Turn_2.csv, ... for turns
        Straight_1.csv, Straight_2.csv, ... for straights

    If input points include widths as (x, y, left, right), edges are computed using those widths.
    Otherwise, edges fall back to the centerline duplicated on both sides.

    Returns the created directory path, or None on failure.
    """
    try:
        from .segmentation import segment_by_median_mad as _seg_fn
        import csv as _csv
        import numpy as _np
    except Exception as e:
        print(f"Cannot export segments CSV: missing dependencies: {e}")
        return None

    # Harmonize curvature length to points length if needed
    pts = list(points)
    curvatures = _np.asarray(curvs, dtype=float)
    if len(curvatures) != len(pts):
        if len(curvatures) == max(len(pts) - 2, 0):
            curvatures = _np.pad(curvatures, (1, 1), mode="edge")
        else:
            curvatures = _np.interp(_np.arange(len(pts)),
                                    _np.linspace(0, len(pts) - 1, max(len(curvatures), 2)),
                                    curvatures if len(curvatures) > 0 else _np.zeros(2))

    segments = _seg_fn(curvatures, factor=mad_factor, points=pts)

    # Resolve output directory
    if base_output_dir is None:
        firmware_dir = os.path.normpath(os.path.dirname(__file__))
        turns_base_dir = os.path.join(firmware_dir, "Turns")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(turns_base_dir, f"segments_{timestamp}")
    else:
        out_dir = base_output_dir
    os.makedirs(out_dir, exist_ok=True)

    turn_idx = 0
    straight_idx = 0

    def _write_csv(path: str, rows: list[tuple[float, float, float, float]]):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["x_l", "y_l", "x_r", "y_r"])
            for x_l, y_l, x_r, y_r in rows:
                writer.writerow([float(x_l), float(y_l), float(x_r), float(y_r)])

    def _compute_edges_for_slice(start_idx: int, end_idx: int) -> list[tuple[float, float, float, float]]:
        # inclusive slice indices
        idxs = list(range(start_idx, end_idx + 1))
        rows: list[tuple[float, float, float, float]] = []
        # Determine whether points include widths per point
        has_widths = len(pts) > 0 and isinstance(pts[0], (tuple, list)) and len(pts[0]) >= 4

        for i in idxs:
            p_i = pts[i]
            if has_widths:
                x, y, left_w, right_w = float(p_i[0]), float(p_i[1]), float(p_i[2]), float(p_i[3])
            else:
                x, y = float(p_i[0]), float(p_i[1])
                left_w, right_w = 0.0, 0.0  # fallback when widths are unavailable

            # Estimate tangent using neighbors for a stable normal
            if i == 0:
                x_prev, y_prev = float(pts[i][0]), float(pts[i][1])
            else:
                x_prev, y_prev = float(pts[i - 1][0]), float(pts[i - 1][1])
            if i == len(pts) - 1:
                x_next, y_next = float(pts[i][0]), float(pts[i][1])
            else:
                x_next, y_next = float(pts[i + 1][0]), float(pts[i + 1][1])

            tx, ty = (x_next - x_prev), (y_next - y_prev)
            norm = (_np.hypot(tx, ty) if (tx != 0.0 or ty != 0.0) else 1.0)
            tx, ty = (tx / norm, ty / norm)
            # Left normal is (-ty, tx)
            nx_l, ny_l = -ty, tx
            nx_r, ny_r = ty, -tx

            x_l = x + nx_l * left_w
            y_l = y + ny_l * left_w
            x_r = x + nx_r * right_w
            y_r = y + ny_r * right_w
            rows.append((x_l, y_l, x_r, y_r))

        return rows

    for seg in segments:
        start = max(0, int(seg["start_idx"]))
        end = max(start, int(seg["end_idx"]))
        # inclusive slice, compute edges (x_l,y_l,x_r,y_r)
        edge_rows = _compute_edges_for_slice(start, end)
        if seg["type"] == "turn":
            turn_idx += 1
            filename = f"Turn_{turn_idx}.csv"
            _write_csv(os.path.join(out_dir, filename), edge_rows)
        else:
            straight_idx += 1
            filename = f"Straight_{straight_idx}.csv"
            _write_csv(os.path.join(out_dir, filename), edge_rows)

    print(f"Wrote {turn_idx} turns and {straight_idx} straights to {out_dir}")
    return out_dir


def run_default_demo(points_file: str | None) -> None:
    """Execute the default demo flow when no explicit flags are provided."""
    import math
    visualize_curvature_concept()
    test_segmentation_debug(points_file)
    # compare_performance()
    track_points = [(i, 3 * math.sin(i / 10)) for i in range(50)]
    _ = segment_track_multi_scale(track_points, window_sizes=[3, 5, 9], threshold_factors=[0.5, 0.5, 0.5])


def main() -> None:
    args = parse_args()
    points_file = resolve_points_file(args)

    # If --v_max is provided, run the vmax pipeline strictly on data/raceline.csv
    if args.v_max:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required. Install with: pip install matplotlib")
            return

        raceline_csv = os.path.join(os.path.dirname(__file__), "..", "data", "monza_raceline.csv")
        raceline_csv = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "monza_raceline.csv")))
        if not os.path.isfile(raceline_csv):
            # Fallback to repository root data/raceline.csv
            raceline_csv = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "monza_raceline.csv"))
        if not os.path.isfile(raceline_csv):
            print("Cannot find data/monza_raceline.csv. Please place it under the repository data/ folder.")
            return

        pts = load_points(raceline_csv)
        params = VehicleParams()
        s, kappa, v_lat, v = speed_profile(pts, params)
        out_csv = os.path.join(os.path.dirname(raceline_csv), "raceline_vmax.csv")
        export_csv(out_csv, pts, s, kappa, v)
        print(f"Wrote vmax CSV: {out_csv}")
        from .vmax_raceline import compute_lap_time_seconds
        lap_time = compute_lap_time_seconds(s, v)
        print(f"Estimated lap time: {lap_time:.3f} s")
        plot_speed_vs_s(s, v, title=f"Vmax profile (lap: {lap_time:.2f}s)")
        plt.show()
        return

    # Outline rendering can be independent of points file
    handle_outline_render(args)

    # Operations requiring a points file
    if (args.mad or args.heatmap or args.web or args.turns_web or True) and points_file:
        pts = load_points(points_file)
        curvs = curvature_vectorized(pts)

        if args.mad:
            run_mad_segmentation(pts, factor=args.factor, print_json_flag=args.print_json)

        render_static_heatmaps(pts, curvs, points_file, heatmap_2d=args.heatmap)
        export_web_heatmaps(pts, curvs, args, points_file)
        export_turns_straights(pts, curvs, args, points_file)

        # Always export per-segment CSVs to a new folder on run
        export_segments_csv(pts, curvs, base_output_dir=None, mad_factor=args.turns_factor)


    # If no explicit action flags were provided, run the default demo
    if not (args.mad or args.heatmap or args.web or args.turns_web or (args.outline_csv and args.outline_web)):
        run_default_demo(points_file)


if __name__ == "__main__":
    main()