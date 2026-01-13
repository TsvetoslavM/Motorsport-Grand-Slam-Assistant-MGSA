# Expose common APIs at package level (optional re-exports)
from .curvature import curvature, curvature_vectorized
from .smoothing import smooth_curvatures_vectorized, smooth_curvatures_multi_scale
from .segmentation import segment_track_multi_scale, segment_by_median_mad
from .track_coloring import plot_turns_and_straights

__all__ = [
    "curvature",
    "curvature_vectorized",
    "smooth_curvatures_vectorized",
    "smooth_curvatures_multi_scale",
    "segment_track_multi_scale",
    "segment_by_median_mad",
    "plot_turns_and_straights",
]


