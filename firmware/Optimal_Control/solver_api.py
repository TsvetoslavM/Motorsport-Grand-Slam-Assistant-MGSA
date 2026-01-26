from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from firmware.vehicle import VehicleParams

from .constraints import (
    add_constraints,
    create_objective_with_racing_line,
    initialize_with_proper_racing_line,
)
from .CasADi_IPOPT import (
    adaptive_path_discretization,
    compute_curvature_closed,
    get_advanced_ipopt_options,
    smooth_curvature_closed,
)


PointLike = Union[Tuple[float, float], List[float], Dict[str, float]]


def _to_xy(p: PointLike) -> Tuple[float, float]:
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """
    Resample polyline to n points uniformly by arc-length.
    points: (M,2)
    """
    if n <= 1:
        return points[:1].copy()
    if points.shape[0] < 2:
        return np.repeat(points[:1], n, axis=0)

    seg = points[1:] - points[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(points[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=float)
    j = 1
    for i, t in enumerate(targets):
        while j < len(cum) and cum[j] < t:
            j += 1
        if j >= len(cum):
            out[i] = points[-1]
            continue
        i0 = j - 1
        i1 = j
        t0 = cum[i0]
        t1 = cum[i1]
        if t1 <= t0 + 1e-12:
            out[i] = points[i1]
            continue
        a = (t - t0) / (t1 - t0)
        out[i] = points[i0] * (1.0 - a) + points[i1] * a
    return out


def _compute_normals_closed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return unit normals (N,2) for a closed loop centerline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # central difference with wrap
    dx = np.roll(x, -1) - np.roll(x, 1)
    dy = np.roll(y, -1) - np.roll(y, 1)
    nrm = np.hypot(dx, dy)
    nrm[nrm == 0.0] = 1.0
    tx = dx / nrm
    ty = dy / nrm
    # left normal
    nx = -ty
    ny = tx
    normals = np.stack([nx, ny], axis=1)
    return normals


def _ds_closed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    ds = np.hypot(dx, dy)
    ds[ds == 0.0] = 1e-6
    return ds


@dataclass
class OptimizeOptions:
    n_points: int = 250
    ipopt_max_iter: int = 2000
    ipopt_print_level: int = 0
    ipopt_tol: float = 1e-4
    ipopt_acceptable_tol: float = 1e-3
    ipopt_linear_solver: str = "mumps"
    vehicle: Optional[VehicleParams] = None
    # Adaptive discretization options
    use_adaptive_discretization: bool = True
    curvature_threshold: float = 0.01
    ds_straight: float = 5.0
    ds_corner: float = 1.5
    curvature_smooth_sigma: float = 2.0


def optimize_trajectory_from_two_lines(
    left_line: Iterable[PointLike],
    right_line: Iterable[PointLike],
    *,
    options: Optional[OptimizeOptions] = None,
) -> Dict[str, Any]:
    """
    Compute an optimal trajectory using CasADi+IPOPT from two boundary lines.

    Input:
      - left_line/right_line: iterables of points (x,y) or {"x":..,"y":..}
    Output:
      - dict with x_opt, y_opt, speed_mps, time_s, lap_time_s, plus some debug info
    """
    if options is None:
        options = OptimizeOptions()
    vehicle = options.vehicle or VehicleParams()

    left = np.array([_to_xy(p) for p in left_line], dtype=float)
    right = np.array([_to_xy(p) for p in right_line], dtype=float)
    if left.shape[0] < 3 or right.shape[0] < 3:
        raise ValueError("left_line and right_line must each contain at least 3 points")

    # Initial uniform resampling to get consistent centerline
    N_initial = int(max(30, min(5000, options.n_points)))
    left_r = _resample_polyline(left, N_initial)
    right_r = _resample_polyline(right, N_initial)

    center = 0.5 * (left_r + right_r)
    x_center = center[:, 0]
    y_center = center[:, 1]

    normals = _compute_normals_closed(x_center, y_center)  # left normal

    # Project boundaries onto normals to get signed widths
    vec_left = left_r - center
    vec_right = right_r - center
    w_left = np.maximum(0.5, np.sum(vec_left * normals, axis=1))
    w_right = np.maximum(0.5, -np.sum(vec_right * normals, axis=1))

    # If user swapped lines, widths may be mostly negative -> fix by swapping
    if float(np.median(w_left)) < 0.0 or float(np.median(w_right)) < 0.0:
        left_r, right_r = right_r, left_r
        vec_left = left_r - center
        vec_right = right_r - center
        w_left = np.maximum(0.5, np.sum(vec_left * normals, axis=1))
        w_right = np.maximum(0.5, -np.sum(vec_right * normals, axis=1))

    # Apply adaptive discretization if enabled
    if options.use_adaptive_discretization:
        x_center, y_center, w_left, w_right, ds_array = adaptive_path_discretization(
            x_center, y_center, w_left, w_right,
            curvature_threshold=options.curvature_threshold,
            ds_straight=options.ds_straight,
            ds_corner=options.ds_corner,
            smooth_sigma=options.curvature_smooth_sigma,
        )
        # Recompute normals after adaptive resampling
        normals = _compute_normals_closed(x_center, y_center)
    else:
        ds_array = _ds_closed(x_center, y_center)

    N = len(x_center)

    # Compute and smooth curvature using shareable functions
    curvature = compute_curvature_closed(x_center, y_center)
    curvature = smooth_curvature_closed(curvature, sigma=options.curvature_smooth_sigma)

    # --- CasADi/IPOPT optimization (same structure as CasADi_IPOPT.py, but callable) ---
    import casadi as ca  # local import so server can start even if casadi missing

    opti = ca.Opti()
    n = opti.variable(N)  # lateral offset (normal direction)
    v = opti.variable(N)  # speed
    a_lon = opti.variable(N)
    slack_power = opti.variable(N)

    opti.set_initial(slack_power, 0)
    opti.subject_to(slack_power >= 0)
    opti.subject_to(slack_power <= 50000)

    a_lat, corner_types, corner_phases = add_constraints(
        opti=opti,
        vehicle=vehicle,
        n=n,
        v=v,
        a_lon=a_lon,
        slack_power=slack_power,
        w_left=w_left,
        w_right=w_right,
        ds_array=ds_array,
        curvature=curvature,
    )

    create_objective_with_racing_line(
        opti,
        v,
        a_lon,
        slack_power,
        n,
        curvature,
        w_left,
        w_right,
        corner_phases,
        ds_array,
        N,
        vehicle,
    )

    # Use advanced IPOPT configuration with user overrides
    opts = get_advanced_ipopt_options(
        max_iter=int(options.ipopt_max_iter),
        tol=float(options.ipopt_tol),
        acceptable_tol=float(options.ipopt_acceptable_tol),
        print_level=int(options.ipopt_print_level),
        linear_solver=str(options.ipopt_linear_solver),
    )
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        converged = True
    except Exception:
        sol = opti.debug
        converged = False

    n_opt = np.array(sol.value(n)).reshape(-1)
    v_opt = np.array(sol.value(v)).reshape(-1)
    a_lon_opt = np.array(sol.value(a_lon)).reshape(-1)
    a_lat_opt = np.array(sol.value(a_lat)).reshape(-1)

    x_opt = x_center + n_opt * normals[:, 0]
    y_opt = y_center + n_opt * normals[:, 1]

    eps = 1e-6
    time_s = np.cumsum(np.concatenate(([0.0], ds_array[:-1] / np.maximum(v_opt[:-1], eps))))

    lap_time_s = float(np.sum(ds_array / np.maximum(v_opt, eps)))
    track_length_m = float(np.sum(ds_array))

    return {
        "converged": converged,
        "N": int(N),
        "track_length_m": track_length_m,
        "lap_time_s": lap_time_s,
        "centerline": [{"x": float(x_center[i]), "y": float(y_center[i])} for i in range(N)],
        "optimal": [
            {
                "x": float(x_opt[i]),
                "y": float(y_opt[i]),
                "speed_mps": float(v_opt[i]),
                "speed_kmh": float(v_opt[i] * 3.6),
                "time_s": float(time_s[i]),
                "a_lon": float(a_lon_opt[i]),
                "a_lat": float(a_lat_opt[i]),
            }
            for i in range(N)
        ],
        "widths": {"w_left": w_left.tolist(), "w_right": w_right.tolist()},
        "debug": {
            "corner_types": {k: (len(v) if isinstance(v, list) else v) for k, v in (corner_types).items()},
        },
    }

