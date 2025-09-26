from .curvature import curvature, curvature_vectorized
from .smoothing import smooth_curvatures_vectorized, smooth_curvatures_multi_scale
from .segmentation import segment_track_multi_scale, segment_by_median_mad
from .visualization import (
    ascii_track_visualization_scaled,
    plot_curvature_heatmap,
    plot_curvature_heatmap_3d,
    plotly_curvature_heatmap_html,
    plotly_curvature_heatmap_3d_html,
    plotly_track_outline_from_widths_html,
)
from .examples import visualize_curvature_concept, test_segmentation_debug, compare_performance
from .io import load_points
from .track_coloring import plot_turns_and_straights
import os
import argparse
import json


def parse_args() -> argparse.Namespace:
    """Create and parse CLI arguments for curvature demos and utilities."""
    parser = argparse.ArgumentParser(description="Curvature demos and segmentation utilities")
    parser.add_argument("--points", type=str, default=None, help="Path to CSV/JSON points file")
    parser.add_argument("--factor", type=float, default=3.0, help="MAD threshold factor (for --mad)")
    parser.add_argument("--mad", action="store_true", help="Run median+MAD segmentation on points file")
    parser.add_argument("--heatmap", action="store_true", help="Render curvature heatmap (requires matplotlib)")
    parser.add_argument("--heatmap3d", action="store_true", help="Render 3D curvature heatmap (requires matplotlib)")
    parser.add_argument("--web", type=str, default=None, help="Export interactive 2D heatmap to HTML at given path (requires plotly)")
    parser.add_argument("--web3d", type=str, default=None, help="Export interactive 3D heatmap to HTML at given path (requires plotly)")
    parser.add_argument("--raceline", type=str, default=None, help="Optional CSV with x,y to overlay as racing line on 2D web export")
    parser.add_argument("--outline-csv", type=str, default=None, help="CSV with columns x,y,left,right to render track outline")
    parser.add_argument("--outline-web", type=str, default=None, help="Output HTML path for outline render")
    parser.add_argument("--print-json", action="store_true", help="Print MAD segments as JSON output")
    parser.add_argument("--turns-web", type=str, default=None, help="Output HTML path for turns/straights coloring (blue turns, black straights)")
    parser.add_argument("--turns-factor", type=float, default=3.0, help="MAD factor for turns/straights segmentation")
    return parser.parse_args()


def resolve_points_file(args: argparse.Namespace) -> str | None:
    """Resolve the points file path from args or environment or default CSV."""
    points_file = args.points or os.environ.get("TRACK_POINTS_FILE")
    if not points_file:
        default_csv = os.path.join(os.path.dirname(__file__), "data", "simple_track.csv")
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
        cws = load_centerline_with_widths(args.outline_csv)
        raceline_pts = None
        if args.raceline:
            try:
                from .io import load_points as _load_points
                raceline_pts = _load_points(args.raceline)
            except Exception as e:
                print(f"Failed to load raceline '{args.raceline}': {e}")
        ok = plotly_track_outline_from_widths_html(
            cws,
            args.outline_web,
            title=f"Spa, Belgium",
            raceline_points=raceline_pts,
            css_file= "templates/outline.css",
        )
        if ok:
            print(f"Wrote outline HTML to {args.outline_web}")

        if args.mad:
            if raceline_pts is None:
                print("MAD requested but no raceline provided. Use --raceline <path>.")
            else:
                run_mad_segmentation(cws, factor=args.factor, print_json_flag=args.print_json)

    except Exception as e:
        print(f"Failed outline render: {e}")


def render_static_heatmaps(points: list[tuple[float, float]], curvs, points_file: str, heatmap_2d: bool, heatmap_3d: bool) -> None:
    """Render 2D/3D matplotlib heatmaps if requested."""
    if heatmap_2d:
        res = plot_curvature_heatmap(points, curvs, title=f"Curvature Heatmap ({os.path.basename(points_file)})")
        if res is not None:
            plt, fig, ax = res
            plt.show()
    if heatmap_3d:
        res3 = plot_curvature_heatmap_3d(points, curvs, title=f"Curvature Heatmap 3D ({os.path.basename(points_file)})")
        if res3 is not None:
            plt3, fig3, ax3 = res3
            plt3.show()


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
    if args.web3d:
        ok3 = plotly_curvature_heatmap_3d_html(
            points,
            curvs,
            args.web3d,
            title=f"Curvature Heatmap 3D ({os.path.basename(points_file)})",
        )
        if ok3:
            print(f"Wrote HTML to {args.web3d}")


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


def run_default_demo(points_file: str | None) -> None:
    """Execute the default demo flow when no explicit flags are provided."""
    import math
    visualize_curvature_concept()
    test_segmentation_debug(points_file)
    # compare_performance()
    track_points = [(i, 3 * math.sin(i / 10)) for i in range(50)]
    segments = segment_track_multi_scale(track_points, window_sizes=[3, 5, 9], threshold_factors=[0.5, 0.5, 0.5])
    ascii_track_visualization_scaled(segments, width=60, height=15)


def main() -> None:
    args = parse_args()
    points_file = resolve_points_file(args)

    # Outline rendering can be independent of points file
    handle_outline_render(args)

    # Operations requiring a points file
    if (args.mad or args.heatmap or args.heatmap3d or args.web or args.web3d or args.turns_web) and points_file:
        pts = load_points(points_file)
        curvs = curvature_vectorized(pts)

        if args.mad:
            run_mad_segmentation(pts, factor=args.factor, print_json_flag=args.print_json)

        render_static_heatmaps(pts, curvs, points_file, heatmap_2d=args.heatmap, heatmap_3d=args.heatmap3d)
        export_web_heatmaps(pts, curvs, args, points_file)
        export_turns_straights(pts, curvs, args, points_file)

    # If no explicit action flags were provided, run the default demo
    if not (args.mad or args.heatmap or args.heatmap3d or args.web or args.web3d or args.turns_web or (args.outline_csv and args.outline_web)):
        run_default_demo(points_file)


if __name__ == "__main__":
    main()