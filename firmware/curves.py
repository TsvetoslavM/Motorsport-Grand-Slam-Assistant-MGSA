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
import os
import argparse
import json


if __name__=="__main__":
    import math

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
    args = parser.parse_args()

    # Resolve points file
    points_file = args.points or os.environ.get("TRACK_POINTS_FILE")
    if not points_file:
        default_csv = os.path.join(os.path.dirname(__file__), "data", "sample_track.csv")
        points_file = default_csv if os.path.isfile(default_csv) else None

    if (args.mad or args.heatmap or args.heatmap3d or args.web or args.web3d) and points_file:
        pts = load_points(points_file)
        curvs = curvature_vectorized(pts)
        if args.mad:
            segments = segment_by_median_mad(curvs, factor=args.factor, points=pts)
            if args.print_json:
                print(json.dumps(segments, ensure_ascii=False, indent=2))
            else:
                print(f"MAD segmentation on {len(pts)} points (factor={args.factor}): {len(segments)} segments")
                for i, s in enumerate(segments, 1):
                    if s["type"] == "turn":
                        print(f"  {i}. turn  entry={s['entry_point']} apex={s['apex_point']} exit={s['exit_point']}")
        else:
                        print(f"  {i}. straight  start={s['start_point']} end={s['end_point']}")
        if args.heatmap:
            res = plot_curvature_heatmap(pts, curvs, title=f"Curvature Heatmap ({os.path.basename(points_file)})")
            if res is not None:
                plt, fig, ax = res
                plt.show()
        if args.heatmap3d:
            res3 = plot_curvature_heatmap_3d(pts, curvs, title=f"Curvature Heatmap 3D ({os.path.basename(points_file)})")
            if res3 is not None:
                plt3, fig3, ax3 = res3
                plt3.show()
        if args.web:
            raceline_pts = None
            if args.raceline:
                try:
                    from .io import load_points as _load_points
                    raceline_pts = _load_points(args.raceline)
                except Exception as e:
                    print(f"Failed to load raceline '{args.raceline}': {e}")
            ok = plotly_curvature_heatmap_html(pts, curvs, args.web, title=f"Curvature Heatmap ({os.path.basename(points_file)})", raceline_points=raceline_pts)
            if ok:
                print(f"Wrote HTML to {args.web}")
        if args.outline_csv and args.outline_web:
            try:
                from .io import load_centerline_with_widths
                cws = load_centerline_with_widths(args.outline_csv)
                ok = plotly_track_outline_from_widths_html(cws, args.outline_web, title=f"Track Outline ({os.path.basename(args.outline_csv)})")
                print("Hello")
                if ok:
                    print(f"Wrote outline HTML to {args.outline_web}")
            except Exception as e:
                print(f"Failed outline render: {e}")
        if args.web3d:
            ok3 = plotly_curvature_heatmap_3d_html(pts, curvs, args.web3d, title=f"Curvature Heatmap 3D ({os.path.basename(points_file)})")
            if ok3:
                print(f"Wrote HTML to {args.web3d}")
            else:
    visualize_curvature_concept()
        # test_segmentation_debug(points_file)
        # compare_performance()
        # track_points = [(i, 3*math.sin(i/10)) for i in range(50)]
        # segments = segment_track_multi_scale(track_points, window_sizes=[3,5,9], threshold_factors=[0.5,0.5,0.5])
        # ascii_track_visualization_scaled(segments, width=60, height=15)
    ascii_track_visualization_scaled(segments, width=60, height=15)