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


def plotly_track_outline_from_widths_html(centerline_with_widths, output_html, title="Track Outline (web)", smooth_iterations: int = 2):
    """Draw left/right edges from centerline and per-point left/right widths.

    centerline_with_widths: list of (x, y, left_width, right_width)
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

    left_edge = pts + n * lw[:,None]
    right_edge = pts - n * rw[:,None]

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

    if smooth_iterations > 0:
        left_edge = _chaikin_open(left_edge, iterations=smooth_iterations)
        right_edge = _chaikin_open(right_edge, iterations=smooth_iterations)

    fig = go.Figure()
    # Draw left and right edges as black outlines with white inner stroke
    for edge, name in [(left_edge, "Left Edge"), (right_edge, "Right Edge")]:
        ex, ey = edge[:,0], edge[:,1]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name=name,
                                 line=dict(color="black", width=12), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(color="white", width=8), hoverinfo="skip", showlegend=False))

    # Close the outline by connecting right edge back to left edge
    poly_x = list(left_edge[:,0]) + list(right_edge[::-1,0]) + [left_edge[0,0]]
    poly_y = list(left_edge[:,1]) + list(right_edge[::-1,1]) + [left_edge[0,1]]
    fig.add_trace(go.Scatter(x=poly_x, y=poly_y, mode="lines", name="Outline",
                             line=dict(color="black", width=2), hoverinfo="skip", showlegend=False, opacity=0.2))

    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y", yaxis_scaleanchor="x", yaxis_scaleratio=1, plot_bgcolor="white")
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True, auto_open=False)
    return True
