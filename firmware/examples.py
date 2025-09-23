import math
import numpy as np

from .curvature import curvature, curvature_vectorized
from .segmentation import segment_track_multi_scale
from .io import load_points


def visualize_curvature_concept():
    print("=" * 70)
    print("UNDERSTANDING CURVATURE AT CORNERS")
    print("=" * 70)

    # Example: What happens at a 90-degree corner
    corner_points = [(0,0), (1,0), (1,1)]  # Sharp 90° right turn
    kappa = curvature(corner_points[0], corner_points[1], corner_points[2])
    print(f"90° corner {corner_points}: κ = {kappa:.4f}")

    # Example: What happens on a straight line
    straight_points = [(0,0), (1,1), (2,2)]  # Perfect diagonal
    kappa = curvature(straight_points[0], straight_points[1], straight_points[2])
    print(f"Straight line {straight_points}: κ = {kappa:.4f}")

    # Example: What happens in a gentle curve
    curve_points = [(0,0), (1,0.1), (2,0.4)]  # Gentle curve
    kappa = curvature(curve_points[0], curve_points[1], curve_points[2])
    print(f"Gentle curve {curve_points}: κ = {kappa:.4f}")

    print("\nKEY INSIGHT:")
    print("- Sharp corners = HIGH curvature (κ >> 0)")
    print("- Straight lines = ZERO curvature (κ ≈ 0)") 
    print("- Gentle curves = LOW curvature (0 < κ < sharp_corner)")
    print()
    print("Your track has SHARP CORNERS, not smooth racing curves!")
    print("That's why you see high κ values only at corner points.")

    print("\n" + "=" * 70)
    print("CREATING A REALISTIC SMOOTH CURVE")
    print("=" * 70)

    # Generate a smooth curve using trigonometry
    smooth_curve = []
    for i in range(11):
        t = i / 10.0  # Parameter from 0 to 1
        x = 10 * t
        y = 3 * math.sin(math.pi * t)  # Sine wave creates smooth curve
        smooth_curve.append((round(x, 1), round(y, 1)))

    print("Smooth sine curve:", smooth_curve)

    segments = segment_track_multi_scale(smooth_curve)

    print(f"\nSegments found:")
    for i, seg in enumerate(segments):
        seg_type, seg_points, entry, apex, exit, apex_kappa = seg
        print(f"\nSegment {i+1} - {seg_type}: {len(seg_points)} points")
        if seg_type == "завой":
            print(f"  Entry: {entry}, Apex: {apex} (κ={apex_kappa:.4f}), Exit: {exit}")
        else:
            print(f"  From {seg_points[0]} to {seg_points[-1]}")


def test_segmentation_debug(points_file: str | None = None):
    print("=" * 70)
    print("SIMPLE TEST CASE")
    print("=" * 70)

    # Very clear test case
    simple_points = [
        (0, 0), (1, 0), (2, 0),           # Straight: κ ≈ 0
        (3, 1), (4, 3), (5, 5),           # Curved: κ > 0
        (6, 5), (7, 5), (8, 5)            # Straight: κ ≈ 0
    ]

    print("Points:", simple_points)
    print()

    segments = segment_track_multi_scale(simple_points)

    print(f"\nSegments found:")
    for i, seg in enumerate(segments):
        seg_type, seg_points, entry, apex, exit, apex_kappa = seg
        print(f"\nSegment {i+1} - {seg_type}:")
        print(f"  Points: {seg_points}")
        if seg_type == "завой":
            print(f"  Entry: {entry}, Apex: {apex} (κ={apex_kappa:.4f}), Exit: {exit}")

    print("\n" + "=" * 70)
    print("YOUR ORIGINAL DATA - FIXED")
    print("=" * 70)

    if points_file:
        try:
            original_points = load_points(points_file)
        except Exception as e:
            print(f"Failed to load points from '{points_file}': {e}")
            print("Falling back to built-in sample points.")
            original_points = [
                (0,0), (5,0), (10,0),
                (10,2), (10,5), (10,8), (10,10),
                (15,10), (20,10)
            ]
    else:
        original_points = [
            (0,0), (5,0), (10,0),
            (10,2), (10,5), (10,8), (10,10),
            (15,10), (20,10)
        ]

    print("Points:", original_points)

    # Test with different thresholds
    for thresh in [0.1, 0.2, 0.3]:
        print(f"\n--- Threshold: {thresh} ---")
        segments = segment_track_multi_scale(original_points, threshold_factors=[thresh]*3)
        print(f"Segments found:")
        for i, seg in enumerate(segments):
            seg_type, seg_points, entry, apex, exit, apex_kappa = seg
            print(f"  Segment {i+1} - {seg_type}: {len(seg_points)} points")
            if seg_type == "завой":
                print(f"    Entry: {entry}, Apex: {apex} (κ={apex_kappa:.4f}), Exit: {exit}")
            else:
                print(f"    From {seg_points[0]} to {seg_points[-1]}")


def compare_performance():
    """Compare performance between original and vectorized implementations"""
    import time

    # Create a large track for testing
    large_track = []
    for i in range(1000):
        x = i
        y = 10 * math.sin(i/50)  # Create a wavy track
        large_track.append((x, y))

    print("Performance comparison for large track (1000 points):")

    # Test original implementation
    start_time = time.time()
    curvatures_orig = []
    for i in range(1, len(large_track)-1):
        kappa = curvature(large_track[i-1], large_track[i], large_track[i+1])
        curvatures_orig.append(kappa)
    orig_time = time.time() - start_time
    print(f"Original implementation: {orig_time:.4f} seconds")

    # Test vectorized implementation
    start_time = time.time()
    curvatures_vec = curvature_vectorized(large_track)
    vec_time = time.time() - start_time
    print(f"Vectorized implementation: {vec_time:.4f} seconds")

    print(f"Speedup: {orig_time/vec_time:.2f}x")

    # Verify results are similar
    max_diff = np.max(np.abs(np.array(curvatures_orig) - curvatures_vec[1:-1]))
    print(f"Maximum difference between implementations: {max_diff:.6f}")


