def ascii_track_visualization_scaled(segments, width=60, height=15):
    """Visualize segments with curvature intensity"""
    all_y = [y for seg in segments for _,y in seg[1]]
    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max-y_min if y_max!=y_min else 1.0

    for i, seg in enumerate(segments):
        seg_type, pts, entry, apex, exit, apex_kappa = seg
        print(f"\nSegment {i+1} - {seg_type}: {len(pts)} points")
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        for idx, (x,y) in enumerate(pts):
            if len(pts) == 1:
                col = 0
            else:
                col = int((idx/(len(pts)-1))*(width-1))
            row = int((y_max - y)/y_range*(height-1))
            if seg_type=="права":
                symbol = "─"
            else:
                rel = min(abs(y - entry[1]) / (abs(apex[1]-entry[1])+1e-6), 1.0) if entry else 0.5
                if rel<0.33: symbol="░"
                elif rel<0.66: symbol="▒"
                else: symbol="▓"
            canvas[row][col]=symbol
        for row in canvas:
            print("".join(row))


def plot_curvature_heatmap(points, curvatures, cmap="viridis", linewidth=3.0, s=15, show_points=True, title="Curvature Heatmap", colorbar_label="Curvature κ"):
    """Plot a heatmap-like visualization by coloring the track by curvature.

    - Colors each segment between consecutive points by its curvature value (at the second point)
    - Optionally overlays points as a scatter with same colormap
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np
    except ImportError:
        print("matplotlib is required for heatmap visualization. Install with: pip install matplotlib")
        return

    if len(points) == 0:
        print("No points to visualize.")
        return

    pts = np.asarray(points, dtype=float)
    x = pts[:,0]
    y = pts[:,1]

    # Prepare line segments between consecutive points
    if len(pts) >= 2:
        segments = np.stack([np.stack([pts[:-1], pts[1:]], axis=1)], axis=0)[0]
        # Use curvature values aligned to the segment end (index 1..n-1)
        curvs_for_segments = np.asarray(curvatures, dtype=float)
        if len(curvs_for_segments) != len(pts):
            # If curvatures are for middle points only, pad to match length
            if len(curvs_for_segments) == len(pts) - 2:
                curvs_for_segments = np.pad(curvs_for_segments, (1,1), mode='edge')
            else:
                # Fallback: interpolate or broadcast
                curvs_for_segments = np.interp(np.arange(len(pts)), np.linspace(0, len(pts)-1, len(curvatures)), curvatures)
        values = curvs_for_segments[1:]  # one per segment

        norm = plt.Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)) if float(np.max(values))>0 else 1.0)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth)
        lc.set_array(values)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label(colorbar_label)

        if show_points:
            sc = ax.scatter(x, y, c=curvs_for_segments, cmap=cmap, norm=norm, s=s, edgecolor='none')

        return plt, fig, ax
    else:
        # Single point: just scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(x, y, c=[curvatures[0] if len(curvatures)>0 else 0.0], cmap=cmap, s=s)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(colorbar_label)
        return plt, fig, ax


def plot_curvature_heatmap_3d(points, curvatures, cmap="viridis", linewidth=2.0, s=10, title="Curvature Heatmap 3D", colorbar_label="Curvature κ"):
    """3D visualization: x-y track colored by curvature with z as curvature height.

    - Plots the polyline in 3D where z-axis equals curvature magnitude
    - Colors line and points by curvature using the given colormap
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import numpy as np
    except ImportError:
        print("matplotlib is required for 3D heatmap visualization. Install with: pip install matplotlib")
        return

    if len(points) == 0:
        print("No points to visualize.")
        return

    pts = np.asarray(points, dtype=float)
    x = pts[:,0]
    y = pts[:,1]

    curvs = np.asarray(curvatures, dtype=float)
    if len(curvs) != len(pts):
        if len(curvs) == len(pts) - 2:
            curvs = np.pad(curvs, (1,1), mode='edge')
        else:
            curvs = np.interp(np.arange(len(pts)), np.linspace(0, len(pts)-1, len(curvatures)), curvatures)
    z = curvs

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('κ')

    # Draw 3D line colored by curvature by plotting many short segments
    cmap_obj = plt.get_cmap(cmap)
    z_min, z_max = float(z.min()), float(z.max()) if float(z.max())>0 else 1.0
    rng = max(z_max - z_min, 1e-9)
    for i in range(len(pts)-1):
        xs = [x[i], x[i+1]]
        ys = [y[i], y[i+1]]
        zs = [z[i], z[i+1]]
        color = cmap_obj((z[i] - z_min) / rng)
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth)

    # Scatter points with same colormap
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    sc = ax.scatter(x, y, z, c=z, cmap=cmap, norm=norm, s=s)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(colorbar_label)

    # Aspect
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() or 1.0
    mid_x = (x.max()+x.min())/2
    mid_y = (y.max()+y.min())/2
    mid_z = (z.max()+z.min())/2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(max(0, mid_z - max_range/2), mid_z + max_range/2)

    return plt, fig, ax


def plotly_curvature_heatmap_html(points, curvatures, output_html, title="MGSA Headmap", raceline_points=None, smooth_iterations: int = 3):
    """Export an interactive 2D curvature heatmap to a self-contained HTML file using Plotly.

    points: sequence of (x,y)
    curvatures: list/array, length == len(points) or len(points)-2 (will be padded)
    output_html: path to write HTML file
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        print("plotly is required for web export. Install with: pip install plotly")
        return False

    if len(points)==0:
        print("No points to export.")
        return False

    pts = np.asarray(points, dtype=float)
    x = pts[:,0]
    y = pts[:,1]

    # Smooth the polyline for nicer outlines (Chaikin's corner cutting)
    def _chaikin_open(poly: np.ndarray, iterations: int = 3) -> np.ndarray:
        if len(poly) < 3:
            return poly
        out = poly.copy()
        for _ in range(iterations):
            new_pts = [out[0]]
            for i in range(len(out) - 1):
                p = out[i]
                q = out[i + 1]
                Q = 0.75 * p + 0.25 * q
                R = 0.25 * p + 0.75 * q
                new_pts.extend([Q, R])
            new_pts.append(out[-1])
            out = np.asarray(new_pts)
        return out

    smoothed = _chaikin_open(pts, iterations=smooth_iterations)
    sx, sy = smoothed[:,0], smoothed[:,1]

    curvs = np.asarray(curvatures, dtype=float)
    if len(curvs) != len(pts):
        if len(curvs) == len(pts)-2:
            curvs = np.pad(curvs, (1,1), mode='edge')
        else:
            curvs = np.interp(np.arange(len(pts)), np.linspace(0, len(pts)-1, len(curvatures)), curvatures)

    fig = go.Figure()
    # Outline layer
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                             line=dict(color="black", width=15),
                             hoverinfo="skip", showlegend=False))
    # Inner white layer
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                             line=dict(color="white", width=8),
                             hoverinfo="skip", showlegend=False))
    # Points with black border
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(color="white", size=1, line=dict(color="white", width=1)),
                             showlegend=False))
    

    # Optional racing line overlay
    if raceline_points is not None:
        rp = np.asarray(raceline_points, dtype=float)
        if len(rp) >= 2:
            rp_sm = _chaikin_open(rp, iterations=smooth_iterations)
            rx, ry = rp_sm[:,0], rp_sm[:,1]
            fig.add_trace(go.Scatter(x=rx, y=ry, name="Racing line", mode="lines",
                                     line=dict(color="blue", width=3),
                                     hoverinfo="skip", showlegend=True))
    fig.add_trace(go.Scatter(x=x, y=y, name="Optimal treactory", mode="lines",
                             line=dict(color="red", width=3),
                             hoverinfo="skip", showlegend=True))
                             
    fig.update_layout(title=title, yaxis_scaleratio=1, plot_bgcolor="white")
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True, auto_open=False)
    return True


def plotly_curvature_heatmap_3d_html(points, curvatures, output_html, title="Curvature Heatmap 3D (web)"):
    """Export an interactive 3D curvature heatmap to self-contained HTML using Plotly."""
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        print("plotly is required for web export. Install with: pip install plotly")
        return False

    if len(points)==0:
        print("No points to export.")
        return False

    pts = np.asarray(points, dtype=float)
    x = pts[:,0]
    y = pts[:,1]

    curvs = np.asarray(curvatures, dtype=float)
    if len(curvs) != len(pts):
        if len(curvs) == len(pts)-2:
            curvs = np.pad(curvs, (1,1), mode='edge')
        else:
            curvs = np.interp(np.arange(len(pts)), np.linspace(0, len(pts)-1, len(curvatures)), curvatures)
    z = curvs

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers",
                               marker=dict(size=3, color=z, colorscale="Viridis", colorbar=dict(title="κ")),
                               line=dict(width=4, color=z, colorscale="Viridis")))
    fig.update_layout(title=title, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="κ"))
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True, auto_open=False)
    return True


def _write_html_with_css(fig,
                         output_html: str,
                         css_text: str | None = None,
                         css_file: str | None = None) -> bool:
    try:
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        css = css_text
        if css is None and css_file is not None:
            try:
                with open(css_file, "r", encoding="utf-8") as f:
                    css = f.read()
            except Exception as e:
                print(f"Failed to read CSS file '{css_file}': {e}")
        if css:
            inject = f"\n<style>\n{css}\n</style>\n"
            head_end = html.find("</head>")
            if head_end != -1:
                html = html[:head_end] + inject + html[head_end:]
            else:
                # Fallback: prepend
                html = inject + html
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)
        return True
    except Exception as e:
        print(f"Failed to write HTML with CSS: {e}")
        return False


def plotly_track_outline_from_widths_html(centerline_with_widths,
                                          output_html,
                                          title="SPA",
                                          smooth_iterations: int = 2,
                                          raceline_points=None,
                                          curvatures=None,
                                          mad_factor: float = 4.0,
                                          color_edges_by_turns: bool = True,
                                          css_text: str | None = None,
                                          css_file: str | None = None,
                                          raceline_vmax=None,
                                          raceline_vopt=None):
    """Draw left/right edges from centerline and per-point left/right widths.

    - If color_edges_by_turns is True, colors the left/right edges similar to
      plot_turns_and_straights: straights black, turns blue. Segmentation is
      computed from centerline curvature (provided via `curvatures` or
      computed internally).

    Parameters:
    centerline_with_widths: list of (x, y, left_width, right_width)
        output_html: destination HTML path
        title: chart title
        smooth_iterations: smoothing iterations for the faint background outline
        raceline_points: optional (x,y) sequence to overlay
        curvatures: optional array aligned to centerline points; if None will
                    be computed
        mad_factor: MAD segmentation factor
        color_edges_by_turns: whether to add colored left/right edges
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        print("plotly is required for web export. Install with: pip install plotly")
        return False

    if len(centerline_with_widths) < 2:
        print("Need at least 2 points for outline.")
        return False

    arr = np.asarray(centerline_with_widths, dtype=float)
    x = arr[:,0]; y = arr[:,1]; lw = arr[:,2]; rw = arr[:,3]
    pts = np.stack([x, y], axis=1)

    # Compute tangents and normals
    def _tangents(p: np.ndarray) -> np.ndarray:
        d = np.zeros_like(p)
        d[1:-1] = (p[2:] - p[:-2]) / 2.0
        d[0] = p[1] - p[0]
        d[-1] = p[-1] - p[-2]
        return d

    t = _tangents(pts)
    # Normal: rotate tangent by +90 degrees: (dx, dy) -> (-dy, dx)
    n = np.stack([-t[:,1], t[:,0]], axis=1)
    # Normalize normals, avoid zero-length
    norms = np.linalg.norm(n, axis=1)
    norms[norms == 0] = 1.0
    n = n / norms[:,None]

    left_edge_raw = pts + n * lw[:,None]
    right_edge_raw = pts - n * rw[:,None]

    # Optional smoothing (Chaikin) for nicer outlines
    def _chaikin_open(poly: np.ndarray, iterations: int = 2) -> np.ndarray:
        if len(poly) < 3:
            return poly
        out = poly.copy()
        for _ in range(iterations):
            new_pts = [out[0]]
            for i in range(len(out) - 1):
                p = out[i]; q = out[i + 1]
                Q = 0.75 * p + 0.25 * q
                R = 0.25 * p + 0.75 * q
                new_pts.extend([Q, R])
            new_pts.append(out[-1])
            out = np.asarray(new_pts)
        return out

    # Smoothed copies for background polygon only (to keep indices consistent for coloring)
    left_edge_bg = left_edge_raw
    right_edge_bg = right_edge_raw
    if smooth_iterations > 0:
        left_edge_bg = _chaikin_open(left_edge_bg, iterations=smooth_iterations)
        right_edge_bg = _chaikin_open(right_edge_bg, iterations=smooth_iterations)

    fig = go.Figure()
        # Close the outline by connecting right edge back to left edge (background polygon)
    poly_x = list(left_edge_bg[:,0]) + list(right_edge_bg[::-1,0]) + [left_edge_bg[0,0]]
    poly_y = list(left_edge_bg[:,1]) + list(right_edge_bg[::-1,1]) + [left_edge_bg[0,1]]

    # Optionally add colored edges (unsmoothed, to align with segmentation indices)
    if color_edges_by_turns:
        try:
            from .segmentation import segment_by_median_mad
            if curvatures is None:
                from .curvature import curvature_vectorized
                curvatures = curvature_vectorized(pts)
            # Harmonize curvature length to points length if needed
            curvs = np.asarray(curvatures, dtype=float)
            if curvs.size != len(pts):
                if curvs.size == max(len(pts) - 2, 0):
                    curvs = np.pad(curvs, (1, 1), mode="edge")
                else:
                    curvs = np.interp(np.arange(len(pts)),
                                      np.linspace(0, len(pts) - 1, max(len(curvs), 2)),
                                      curvs if curvs.size > 0 else np.zeros(2))
            segments = segment_by_median_mad(curvs, factor=mad_factor, points=pts)

            edge_trace_indices: list[int] = []
            turn_edge_trace_indices: list[int] = []
            straight_edge_trace_indices: list[int] = []

            def _add_edge_chunk(edge_pts: np.ndarray, color: str, name: str, show_in_legend: bool) -> int:
                if edge_pts.shape[0] < 2:
                    return -1
                before = len(fig.data)
                fig.add_trace(
                    go.Scatter(
                        x=edge_pts[:, 0],
                        y=edge_pts[:, 1],
                        mode="lines",
                        line=dict(color=color, width=1),
                        name=name,
                        hoverinfo="skip",
                        showlegend=show_in_legend,
                    )
                )
                idx = before
                edge_trace_indices.append(idx)
                return idx

            turn_index = 0
            # For sizing annotations, estimate a reasonable default radius
            x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
            y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
            scale_ref = max(x_max - x_min, y_max - y_min, 1.0)

            # Collect positions/sizes for label markers (to avoid overlaps)
            label_x: list[float] = []
            label_y: list[float] = []
            label_text: list[str] = []
            label_sizes: list[float] = []
            for seg in segments:
                start = max(0, int(seg["start_idx"]))
                end = min(len(pts) - 1, int(seg["end_idx"]))
                if end <= start:
                    continue
                if seg["type"] == "turn":
                    turn_index += 1
                    turn_name = f"Turn {turn_index}"
                    # Show legend only once per turn (left edge)
                    idx_left = _add_edge_chunk(left_edge_raw[start:end + 1], color="blue", name=turn_name, show_in_legend=True)
                    _ = _add_edge_chunk(right_edge_raw[start:end + 1], color="blue", name=turn_name, show_in_legend=False)
                    if idx_left >= 0:
                        turn_edge_trace_indices.append(idx_left)
                    # Add numeric label marker (circle+text) offset off the track
                    mid = (start + end) // 2
                    # Choose offset direction by available width
                    left_w = float(lw[mid]) if mid < len(lw) else 0.0
                    right_w = float(rw[mid]) if mid < len(rw) else 0.0
                    # Always place on the left side of travel (normal direction)
                    side = 15.0
                    offset_dist = max(left_w, 1e-6) * 1.5
                    ax = float(pts[mid, 0] + n[mid, 0] * side * offset_dist)
                    ay = float(pts[mid, 1] + n[mid, 1] * side * offset_dist)
                    # Circle radius, relative to track size but not too big
                    r = max(0.012 * scale_ref, 0.2 * max(left_w, right_w, 1e-6))
                    # Prevent overlaps by pushing further along the left normal until no intersection
                    def _overlaps_any(px: float, py: float, pr: float) -> bool:
                        for (qx, qy, qr) in zip(label_x, label_y, label_sizes):
                            dx = px - qx
                            dy = py - qy
                            if (dx*dx + dy*dy) < (pr + qr) * (pr + qr):
                                return True
                        return False
                    while _overlaps_any(ax, ay, r):
                        offset_dist += r * 0.6
                        ax = float(pts[mid, 0] + n[mid, 0] * side * offset_dist)
                        ay = float(pts[mid, 1] + n[mid, 1] * side * offset_dist)
                    label_x.append(ax)
                    label_y.append(ay)
                    label_text.append(str(turn_index))
                    # Plotly marker size is in px; map world radius to a reasonable px size
                    # Use a fixed size for consistency but keep list for future tuning
                    label_sizes.append(r)
                else:
                    # Straights: draw but do not show in legend
                    idx_s_left = _add_edge_chunk(left_edge_raw[start:end + 1], color="black", name="Straight", show_in_legend=False)
                    _ = _add_edge_chunk(right_edge_raw[start:end + 1], color="black", name="Straight", show_in_legend=False)
                    if idx_s_left >= 0:
                        straight_edge_trace_indices.append(idx_s_left)
        except Exception as e:
            print(f"Failed to color edges by turns: {e}")
    
    # Add a labeled markers trace for turn numbers so it can be toggled from legend
    labels_trace_index = None
    if color_edges_by_turns:
        try:
            if len(label_x) > 0:
                # Use a single scatter for labels with text inside circular-looking markers
                # Marker size in px; choose a readable fixed size
                marker_size = 18
                before = len(fig.data)
                fig.add_trace(go.Scatter(
                    x=label_x,
                    y=label_y,
                    mode="markers+text",
                    name="Turn numbers",
                    text=label_text,
                    textposition="middle center",
                    textfont=dict(color="blue", size=10),
                    marker=dict(size=marker_size, color="white", line=dict(color="blue", width=2)),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                labels_trace_index = before
        except Exception as e:
            print(f"Failed to add turn number labels: {e}")
    raceline_trace_index = None
    if raceline_points is not None:
        rp = np.asarray(raceline_points, dtype=float)
        if len(rp) >= 2:
            # Use original raceline points for hover consistency
            rx, ry = rp[:,0], rp[:,1]
            # Build text per-point including v_max and optimal speed if provided
            text = []
            vmax_arr = np.asarray(raceline_vmax, dtype=float).reshape(-1) if raceline_vmax is not None else None
            vopt_arr = np.asarray(raceline_vopt, dtype=float).reshape(-1) if raceline_vopt is not None else None
            have_vmax = vmax_arr is not None and vmax_arr.size == rp.shape[0]
            have_vopt = vopt_arr is not None and vopt_arr.size == rp.shape[0]
            # Compute cumulative time along the raceline using trapezoidal integration
            cum_time = None
            if have_vopt:
                try:
                    # Local import to avoid heavy dependency at module import time
                    from .vmax_raceline.vmax import compute_arc_length
                except Exception:
                    compute_arc_length = None
                if compute_arc_length is not None:
                    s, ds = compute_arc_length(rp)
                    # Trapezoidal average speed per segment
                    v0 = np.maximum(vopt_arr[:-1], 1e-3)
                    v1 = np.maximum(vopt_arr[1:], 1e-3)
                    v_avg = 0.5 * (v0 + v1)
                    dt = np.zeros_like(s)
                    if ds.size == v_avg.size and ds.size > 0:
                        dt[1:] = ds / v_avg
                    cum_time = np.cumsum(dt)

            def _format_time_mm_ss_cs(seconds: float) -> str:
                if not np.isfinite(seconds) or seconds < 0:
                    return "-"
                minutes = int(seconds // 60)
                sec = int(seconds % 60)
                centi = int(round((seconds - minutes * 60 - sec) * 100))
                # Handle rounding overflow (e.g., 59.995 -> 60.00)
                if centi >= 100:
                    centi -= 100
                    sec += 1
                    if sec >= 60:
                        sec -= 60
                        minutes += 1
                return f"{minutes}.{sec:02d}.{centi:02d}"
            for i in range(rp.shape[0]):
                parts = [f"x: {rx[i]:.3f}", f"y: {ry[i]:.3f}"]
                if have_vmax:
                    vdisp = vmax_arr[i]
                    if np.isfinite(vdisp):
                        parts.append(f"v_max: {float(vdisp):.2f} m/s")
                    else:
                        parts.append("v_max: -")
                if have_vopt:
                    kmh = float(vopt_arr[i]) * 3.6
                    parts.append(f"optimal: {kmh:.1f} km/h")
                if cum_time is not None and i < cum_time.size:
                    parts.append(f"time: {_format_time_mm_ss_cs(float(cum_time[i]))}")
                text.append("<br>".join(parts))

            before = len(fig.data)
            fig.add_trace(go.Scatter(
                x=rx,
                y=ry,
                name="Racing line",
                mode="lines+markers",
                line=dict(color="gray", width=2),
                marker=dict(color="blue", size=1),
                showlegend=True,
                text=text,
                hovertemplate="%{text}<extra></extra>",
            ))
            raceline_trace_index = before

    # Add buttons to toggle labels and show racing line only
    try:
        total_traces = len(fig.data)
        # Default: all visible
        vis_all = [True] * total_traces
        # Hide labels only
        vis_no_labels = vis_all.copy()
        if labels_trace_index is not None:
            vis_no_labels[labels_trace_index] = False
        # Racing line only: hide everything except raceline (if present)
        vis_race_only = [False] * total_traces
        if raceline_trace_index is not None:
            vis_race_only[raceline_trace_index] = True

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.12,
                xanchor="center",
                yanchor="bottom",
                buttons=[
                    dict(label="All", method="update", args=[{"visible": vis_all}]),
                    dict(label="Hide turn numbers", method="update", args=[{"visible": vis_no_labels}]),
                    dict(label="Racing line only", method="update", args=[{"visible": vis_race_only}]),
                ],
                showactive=True,
            )]
        )
    except Exception as e:
        print(f"Failed to add toggle buttons: {e}")

    fig.update_layout(title=title, plot_bgcolor="white")
    return _write_html_with_css(fig, output_html, css_text=css_text, css_file=css_file)


    return True

