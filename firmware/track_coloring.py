import numpy as np

from .segmentation import segment_by_median_mad


def plot_turns_and_straights(points,
                             curvatures,
                             output_html: str | None = None,
                             title: str = "Track: Straights (black) & Turns (blue)",
                             mad_factor: float = 3.0):
    """Render a track where straights are black and turns are blue.

    Parameters:
        points: Sequence of (x, y) pairs describing the polyline order
        curvatures: Sequence of curvature values aligned to points (len == len(points)
                    or len(points)-2; shorter will be padded)
        output_html: If provided, writes an interactive Plotly HTML to this path and
                     returns True; otherwise returns the Plotly figure object
        title: Plot title
        mad_factor: Multiplier for MAD threshold when segmenting turns

    Returns:
        - If output_html is provided: bool success
        - Else: plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise RuntimeError("plotly is required. Install with: pip install plotly")

    if points is None or len(points) < 2:
        raise ValueError("Need at least 2 points to render track.")

    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    # Harmonize curvature length to points length if needed
    curvs = np.asarray(curvatures, dtype=float)
    if curvs.size != len(pts):
        if curvs.size == max(len(pts) - 2, 0):
            # Pad ends if curvature corresponds to middle points only
            curvs = np.pad(curvs, (1, 1), mode="edge")
        else:
            # Interpolate to match length
            curvs = np.interp(np.arange(len(pts)),
                              np.linspace(0, len(pts) - 1, max(len(curvs), 2)),
                              curvs if curvs.size > 0 else np.zeros(2))

    segments = segment_by_median_mad(curvs, factor=mad_factor, points=pts)

    fig = go.Figure()

    # Helper to add a polyline chunk as a trace
    def _add_chunk(chunk_pts: np.ndarray, color: str, name: str):
        if chunk_pts.shape[0] < 2:
            return
        fig.add_trace(
            go.Scatter(
                x=chunk_pts[:, 0],
                y=chunk_pts[:, 1],
                mode="lines",
                line=dict(color=color, width=3),
                name=name,
                hoverinfo="skip",
            )
        )

    # Build contiguous chunks per type
    for seg in segments:
        if seg["type"] == "turn":
            start = seg["start_idx"]
            end = seg["end_idx"]
            # indices are inclusive; ensure within bounds
            start = max(0, int(start))
            end = min(len(pts) - 1, int(end))
            chunk = pts[start:end + 1]
            _add_chunk(chunk, color="blue", name="Turn")
        else:
            start = seg["start_idx"]
            end = seg["end_idx"]
            start = max(0, int(start))
            end = min(len(pts) - 1, int(end))
            chunk = pts[start:end + 1]
            _add_chunk(chunk, color="black", name="Straight")

    fig.update_layout(
        title=title,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        plot_bgcolor="white",
        showlegend=True,
    )

    if output_html:
        fig.write_html(output_html, include_plotlyjs="cdn", full_html=True, auto_open=False)
        return True

    return fig


