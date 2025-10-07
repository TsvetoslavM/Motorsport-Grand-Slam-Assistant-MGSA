import numpy as np
from typing import List, Tuple
from scipy.interpolate import splprep, splev


def resample_to_match(reference: List[Tuple[float, float]], target: List[Tuple[float, float]]) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if len(ref) < 2 or len(tgt) < 2:
        return tgt
    d = np.linalg.norm(np.diff(tgt, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= max(s[-1], 1e-9)
    t_q = np.linspace(0.0, 1.0, len(ref))
    x = np.interp(t_q, s, tgt[:, 0])
    y = np.interp(t_q, s, tgt[:, 1])
    return np.stack([x, y], axis=1)


def resample_polyline_arclength(points: List[Tuple[float, float]], target_n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2 or target_n <= 2:
        return pts
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= max(s[-1], 1e-9)
    t_q = np.linspace(0.0, 1.0, target_n)
    x = np.interp(t_q, s, pts[:, 0])
    y = np.interp(t_q, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def chaikin_smooth(path_pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    curve = path_pts.copy()
    for _ in range(max(0, iterations)):
        q = 0.25 * curve[:-1] + 0.75 * curve[1:]
        r = 0.75 * curve[:-1] + 0.25 * curve[1:]
        curve = np.vstack([curve[0:1], np.column_stack([q, r]).reshape(-1, 2), curve[-1:]])
    return curve


def estimate_curvature_series(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 3:
        return np.zeros((pts.shape[0],), dtype=float)
    d1 = np.diff(pts, axis=0)
    d2 = np.diff(d1, axis=0)
    t = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-9)
    dt = d2 / (np.linalg.norm(d1[1:], axis=1, keepdims=True) + 1e-9)
    kappa = np.zeros((pts.shape[0],), dtype=float)
    kappa[1:-1] = np.linalg.norm(dt, axis=1)
    return kappa


def compute_tangents_and_normals(points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points")
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1) + 1e-9
    tangents = tangents / norms[:, None]
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return tangents, normals


def resample_polyline_by_curvature(points: List[Tuple[float, float]], target_n: int, beta: float = 2.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3 or target_n <= 2:
        return resample_polyline_arclength(points, target_n)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = max(s[-1], 1e-9)
    t = s / total
    kappa = estimate_curvature_series(pts)
    kappa = np.abs(kappa)
    if np.max(kappa) > 0:
        k_norm = kappa / np.max(kappa)
    else:
        k_norm = kappa
    density = 1.0 + float(beta) * k_norm
    seg_density = 0.5 * (density[:-1] + density[1:])
    seg_len = np.diff(t)
    weight = seg_density * seg_len
    cumw = np.concatenate([[0.0], np.cumsum(weight)])
    cumw /= max(cumw[-1], 1e-9)
    tq = np.linspace(0.0, 1.0, target_n)
    x = np.interp(tq, cumw, pts[:, 0])
    y = np.interp(tq, cumw, pts[:, 1])
    return np.stack([x, y], axis=1)


def build_path_from_corridor_bspline(inner: np.ndarray, outer: np.ndarray, alphas: np.ndarray, smoothing: float = 0.0, degree: int = 3) -> np.ndarray:
    """Build smooth path via B-spline through control points defined by alphas between inner/outer edges."""
    alphas = np.clip(alphas, 0.0, 1.0)
    control_points = (1.0 - alphas[:, None]) * inner + alphas[:, None] * outer
    n = len(control_points)
    if n < degree + 1:
        return control_points
    try:
        tck, u = splprep([control_points[:, 0], control_points[:, 1]], s=float(smoothing), k=int(min(degree, n - 1)), per=False)
        u_new = np.linspace(0.0, 1.0, n)
        x_new, y_new = splev(u_new, tck)
        return np.stack([x_new, y_new], axis=1)
    except Exception:
        return control_points


