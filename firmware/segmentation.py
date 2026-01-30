import numpy as np
from .smoothing import smooth_curvatures_multi_scale


def segment_track_multi_scale(points, window_sizes=[3,5,7], threshold_factors=[0.5,0.5,0.5], min_segment_length=2):
    """Segment track using multi-scale curvature analysis"""
    if len(points)<3:
        return []

    multi_curvs = smooth_curvatures_multi_scale(points, window_sizes)

    # Determine per-scale thresholds
    thresholds = {}
    for i,w in enumerate(window_sizes):
        curvs = multi_curvs[w]
        thresholds[w] = np.mean(curvs) + threshold_factors[i]*np.std(curvs)

    # Combine multi-scale curvatures
    combined = np.zeros(len(points)-2, dtype=bool)
    for w in window_sizes:
        # multi_curvs[w] already corresponds to middle points (len(points) - 2)
        curvs = multi_curvs[w]
        combined[:len(curvs)] |= (curvs >= thresholds[w])

    states = np.where(combined, "turn", "straight")

    # Build segments
    segments = []
    i = 0
    while i < len(states):
        current_state = states[i]
        start_idx = i
        while i<len(states) and states[i]==current_state:
            i+=1
        end_idx = i-1

        point_start = start_idx+1 if start_idx>0 else 0
        point_end = end_idx+2 if end_idx<len(states)-1 else len(points)
        seg_points = points[point_start:point_end]

        if len(seg_points)>=min_segment_length:
            if current_state=="turn":
                mid_window = window_sizes[len(window_sizes)//2]
                curvs_mid = multi_curvs[mid_window][point_start:point_end]
                if len(curvs_mid) == 0:
                    # Fallback: choose geometric midpoint if no curvature samples available
                    apex_idx = (point_start + point_end - 1) // 2
                    apex_point = points[apex_idx]
                    apex_kappa = 0.0
                else:
                    if len(curvs_mid)>2:
                        apex_rel = np.argmax(curvs_mid[1:-1])+1
                    else:
                        apex_rel = np.argmax(curvs_mid)
                    apex_idx = point_start + apex_rel
                    apex_point = points[apex_idx]
                    apex_kappa = curvs_mid[apex_rel]
                entry, exit = seg_points[0], seg_points[-1]
                segments.append((current_state, seg_points, entry, apex_point, exit, apex_kappa))
            else:
                segments.append((current_state, seg_points, None, None, None, None))
    return segments


def segment_by_median_mad(curvatures, factor: float = 3.0, points=None, min_segment_length: int = 1):
    """Segment turns and straights from curvature values using a robust threshold.

    Parameters:
        curvatures: Sequence[float] - curvature per point (same length as points if provided)
        factor: float - threshold factor applied to MAD (default 3.0)
        points: Optional[Sequence[Tuple[float,float]]] - if provided, segment endpoints/apex are returned as coordinates as well
        min_segment_length: int - minimum number of points required to keep a segment

    Returns: List[dict]
        Each dict has keys:
            - type: "turn" | "straight"
            - start_idx, end_idx (inclusive indices)
            - For turns: entry_idx, apex_idx, exit_idx
            - If points is provided: entry_point, apex_point, exit_point (for turns) and start_point, end_point (for straights)
    """
    import numpy as _np

    curvatures = _np.asarray(curvatures, dtype=float)
    if curvatures.size == 0:
        return []

    med = _np.median(curvatures)
    mad = _np.median(_np.abs(curvatures - med))
    threshold = med + factor * mad

    is_turn = curvatures >= threshold

    segments = []
    i = 0
    n = len(curvatures)
    while i < n:
        current_state = is_turn[i]
        seg_start = i
        while i < n and is_turn[i] == current_state:
            i += 1
        seg_end = i - 1

        if (seg_end - seg_start + 1) < min_segment_length:
            continue

        if current_state:
            # turn segment
            seg_slice = slice(seg_start, seg_end + 1)
            apex_rel = int(_np.argmax(curvatures[seg_slice]))
            apex_idx = seg_start + apex_rel

            seg_info = {
                "type": "turn",
                "start_idx": seg_start,
                "end_idx": seg_end,
                "entry_idx": seg_start,
                "apex_idx": apex_idx,
                "exit_idx": seg_end,
                "threshold": float(threshold),
                "median": float(med),
                "mad": float(mad),
            }
            if points is not None:
                entry_point = points[seg_start]
                apex_point = points[apex_idx]
                exit_point = points[seg_end]
                seg_info.update({
                    "entry_point": entry_point,
                    "apex_point": apex_point,
                    "exit_point": exit_point,
                })
        else:
            # straight segment
            seg_info = {
                "type": "straight",
                "start_idx": seg_start,
                "end_idx": seg_end,
                "threshold": float(threshold),
                "median": float(med),
                "mad": float(mad),
            }
            if points is not None:
                seg_info.update({
                    "start_point": points[seg_start],
                    "end_point": points[seg_end],
                })

        segments.append(seg_info)

    return segments


