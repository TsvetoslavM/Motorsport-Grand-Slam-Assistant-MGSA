import os
import csv
import glob
import math
import random
import time
from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..curvature import curvature_vectorized
from ..vmax_raceline.vmax import VehicleParams, speed_profile, compute_lap_time_seconds
from .ga_operators import vectorized_tournament_select, uniform_crossover, gaussian_mutation, adaptive_crossover_weighted
from .geometry import (
    resample_to_match as geom_resample_to_match,
    resample_polyline_arclength as geom_resample_arclength,
    resample_polyline_by_curvature as geom_resample_by_curvature,
    estimate_curvature_series as geom_curvature_series,
    compute_tangents_and_normals as geom_tangents_normals,
)
from .fitness import (
    path_roughness_penalty as fit_path_roughness,
    heading_spike_penalty as fit_heading_spike,
    offsets_smoothness_penalty as fit_offsets_smooth,
    curvature_limit_penalty as fit_curvature_limit,
    straightness_reward as fit_straight_reward,
    force_straight_alphas as fit_force_straight,
    count_curvature_violations as fit_count_curv_viol,
)
from .population import compute_adaptive_population_size, adjust_population_size
# Named constants
EPSILON = 1e-9
MIN_POINTS_FOR_CURVATURE = 3
DEFAULT_SMOOTH_REPEATS = 2

from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size: int = 10000) -> None:
        self.cache: "OrderedDict[Tuple[Any, ...], float]" = OrderedDict()
        self.max_size = max_size

    def get(self, key: Tuple[Any, ...]) -> Optional[float]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: Tuple[Any, ...], value: float) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
from .geometry import (
    resample_to_match as geom_resample_to_match,
    resample_polyline_arclength as geom_resample_arclength,
    resample_polyline_by_curvature as geom_resample_by_curvature,
    estimate_curvature_series as geom_curvature_series,
    compute_tangents_and_normals as geom_tangents_normals,
)
from .fitness import (
    path_roughness_penalty as fit_path_roughness,
    heading_spike_penalty as fit_heading_spike,
    offsets_smoothness_penalty as fit_offsets_smooth,
)


Point = Tuple[float, float]


@dataclass
class GAParams:
    population_size: int = 60
    generations: int = 200
    elite_fraction: float = 0.1
    mutation_rate: float = 0.25
    mutation_sigma: float = 0.10
    crossover_rate: float = 0.9
    tournament_k: int = 4
    # Objective prioritizes minimum lap time; penalties are light regularization
    smoothness_lambda: float = 0.0016
    offset_smooth_lambda: float = 0.012
    curvature_lambda: float = 0.0004
    curvature_jerk_lambda: float = 0.0002
    center_deviation_lambda: float = 0.00
    # Longitudinal dynamics realism (accel/brake and jerk)
    long_accel_lambda: float = 0.0002
    long_jerk_lambda: float = 0.0002
    # Single-resolution GA (no coarse/refine stages)
    # Smoothness hardening (limit kinks)
    alpha_delta_max: float = 0.08  # max change between consecutive alphas
    alpha_rate_lambda: float = 1.0
    heading_max_delta_rad: float = 0.35  # ~20 degrees per segment
    heading_spike_lambda: float = 0.01
    # Vehicle/dynamics parameters (forwarded to VehicleParams)
    mass_kg: float = 798.0
    mu_friction: float = 2.0
    gravity: float = 9.81
    rho_air: float = 1.225
    cL_downforce: float = 4.0
    frontal_area_m2: float = 1.6
    engine_power_watts: float = 735000.0
    a_brake_max: float = 54.0
    a_accel_cap: float = 20.0
    cD_drag: float = 1.0
    c_rr: float = 0.004
    safety_speed_margin: float = 1.00
    brake_power_watts: float = 1200000.0
    target_samples: int = 400
    # Adaptation and diversity controls
    use_annealing: bool = True
    mutation_sigma_start: float = 0.25
    mutation_sigma_end: float = 0.05
    immigrants_fraction: float = 0.10
    stagnation_patience: int = 60
    # Advanced GA controls
    enable_parallel: bool = True
    parallel_workers: int = max(1, os.cpu_count() or 1)
    cache_granularity: float = 0.002
    coarse_to_fine: bool = True
    coarse_factor: float = 0.25  # fraction of samples in coarse stage
    curvature_sampling_beta: float = 2.0  # weight curvature in adaptive sampling
    adaptive_penalty_decay: float = 0.95  # stronger decay: penalties -> ~5% at end
    diversity_target: float = 0.05  # target std of alphas for diversity scaling
    # Missing earlier; used by offset/selection helpers
    group_step: int = 1
    use_straights_only: bool = False
    curvature_threshold: float = 0.1
    max_offset: float = 0.5
    # Steering rate penalty weight
    steering_rate_lambda: float = 0.3
    # Curvature constraint tuning
    max_curvature: float = 0.015
    curvature_hardness: float = 30.0
    curvature_hard_penalty: float = 100.0
    curvature_soft_penalty: float = 5.0
    straightness_bonus: float = 0.3
    alpha_post_max_change: float = 0.005
    # Adaptive crossover and B-spline controls
    use_adaptive_crossover: bool = True
    use_bspline: bool = True
    bspline_smoothing: float = 0.01
    bspline_degree: int = 3
    # Dynamic population sizing
    use_dynamic_population: bool = False
    population_min_ratio: float = 0.5
    population_max_ratio: float = 1.5
    # Weighted fitness controls
    time_weight: float = 1000.0
    penalty_weight: float = 0.5
    adaptive_time_weight: bool = True
    # Internal per-generation weights (set in params_for_gen)
    current_time_weight: float = 1000.0
    current_penalty_weight: float = 0.5


# In-memory cache for fitness evaluations (alpha vectors -> fitness), keyed by quantized tuple
# Bounded LRU cache for fitness evaluations (alpha vectors -> fitness)
_fitness_cache = LRUCache(max_size=10000)


def _quantize_array(arr: np.ndarray, step: float) -> Tuple[float, ...]:
    if arr.size == 0:
        return ()
    q = np.round(arr.astype(float) / float(step)) * float(step)
    return tuple(np.clip(q, 0.0, 1.0).tolist())


def load_latest_turn1_csv(base_dir: Optional[str] = None) -> str:
    """Find the most recent firmware/Turns/segments_*/Turn_1.csv."""
    firmware_dir = os.path.dirname(os.path.dirname(__file__))
    turns_dir = os.path.join(firmware_dir, "Turns")
    if base_dir:
        turns_dir = base_dir
    pattern = os.path.join(turns_dir, "segments_*", "Turn_1.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No Turn_1.csv found under {pattern}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def read_points_csv(path: str) -> List[Point]:
    """Read a polyline from CSV.

    Supports two formats:
    1) centerline: columns [x, y]
    2) corridor edges: columns [x_l, y_l, x_r, y_r] -> returns averaged centerline
    """
    pts: List[Point] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_edges = False
        if header:
            cols = [c.strip().lower() for c in header]
            has_edges = {"x_l", "y_l", "x_r", "y_r"}.issubset(set(cols))
        if has_edges:
            # Build centerline as midpoints of left/right edges
            idx_xl = cols.index("x_l"); idx_yl = cols.index("y_l")
            idx_xr = cols.index("x_r"); idx_yr = cols.index("y_r")
            for row in reader:
                if not row:
                    continue
                x_l = float(row[idx_xl]); y_l = float(row[idx_yl])
                x_r = float(row[idx_xr]); y_r = float(row[idx_yr])
                pts.append(((x_l + x_r) * 0.5, (y_l + y_r) * 0.5))
        else:
            for row in reader:
                if not row:
                    continue
                x = float(row[0]); y = float(row[1])
                pts.append((x, y))
    return pts


def read_corridor_from_segment_csv(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read inner/outer edges from a segment CSV with columns [x_l, y_l, x_r, y_r].

    Returns (inner_edge, outer_edge) as np.ndarray with shape (N, 2), or None if not in the expected format.
    The notion of "inner" vs "outer" is arbitrary here; we keep left as inner and right as outer consistently.
    """
    left: List[Point] = []
    right: List[Point] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        cols = [c.strip().lower() for c in header]
        required = {"x_l", "y_l", "x_r", "y_r"}
        if not required.issubset(set(cols)):
            return None
        idx_xl = cols.index("x_l"); idx_yl = cols.index("y_l")
        idx_xr = cols.index("x_r"); idx_yr = cols.index("y_r")
        for row in reader:
            if not row:
                continue
            left.append((float(row[idx_xl]), float(row[idx_yl])))
            right.append((float(row[idx_xr]), float(row[idx_yr])))
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def write_points_csv(path: str, pts: List[Point]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"]) 
        for x, y in pts:
            w.writerow([float(x), float(y)])


def resample_to_match(reference: List[Point], target: List[Point]) -> np.ndarray:
    """Resample target polyline to have same number of points as reference using arc-length parameterization."""
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if len(ref) < 2 or len(tgt) < 2:
        return tgt
    # Arc length for target
    d = np.linalg.norm(np.diff(tgt, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= max(s[-1], 1e-9)
    # Uniform samples matching reference count
    t_q = np.linspace(0.0, 1.0, len(ref))
    x = np.interp(t_q, s, tgt[:, 0])
    y = np.interp(t_q, s, tgt[:, 1])
    return np.stack([x, y], axis=1)


def resample_polyline_arclength(points: List[Point], target_n: int) -> np.ndarray:
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


def resample_polyline_by_curvature(points: List[Point], target_n: int, beta: float = 2.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3 or target_n <= 2:
        return resample_polyline_arclength(points, target_n)
    # Base arclength parameter t in [0,1]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = max(s[-1], 1e-9)
    t = s / total
    # Curvature estimate per vertex
    kappa = estimate_curvature_series(pts)
    kappa = np.abs(kappa)
    # Sampling density proportional to 1 + beta * normalized curvature
    if np.max(kappa) > 0:
        k_norm = kappa / np.max(kappa)
    else:
        k_norm = kappa
    density = 1.0 + float(beta) * k_norm
    # Build cumulative density along t
    # Use mid-segment densities for integration
    seg_density = 0.5 * (density[:-1] + density[1:])
    seg_len = np.diff(t)
    weight = seg_density * seg_len
    cumw = np.concatenate([[0.0], np.cumsum(weight)])
    cumw /= max(cumw[-1], 1e-9)
    tq = np.linspace(0.0, 1.0, target_n)
    # Invert mapping cumw(t) -> t
    x = np.interp(tq, cumw, pts[:, 0])
    y = np.interp(tq, cumw, pts[:, 1])
    return np.stack([x, y], axis=1)


def chaikin_smooth(path_pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    curve = path_pts.copy()
    for _ in range(max(0, iterations)):
        q = 0.25 * curve[:-1] + 0.75 * curve[1:]
        r = 0.75 * curve[:-1] + 0.25 * curve[1:]
        curve = np.vstack([curve[0:1], np.column_stack([q, r]).reshape(-1, 2), curve[-1:]])
    return curve


def estimate_curvature_series(pts: np.ndarray) -> np.ndarray:
    """Оценява кривината, избягва division by zero"""
    if pts.shape[0] < 3:
        return np.zeros((pts.shape[0],), dtype=float)
    
    d1 = np.diff(pts, axis=0)
    
    # SAFEGUARD: Провери за нулеви разстояния
    norms = np.linalg.norm(d1, axis=1)
    if np.any(norms < 1e-6):
        # Има дублирани или почти идентични точки
        print(f"[WARNING] Found {np.sum(norms < 1e-6)} near-duplicate points in curvature calculation")
        # Филтрирай точките преди изчисление
        valid_mask = np.concatenate([[True], norms >= 1e-6, [True]])
        pts_filtered = pts[valid_mask]
        if pts_filtered.shape[0] < 3:
            return np.zeros((pts.shape[0],), dtype=float)
        return estimate_curvature_series(pts_filtered)  # Рекурсивно с филтрирани точки
    
    d2 = np.diff(d1, axis=0)
    
    # Tangent and its rate of change
    t = d1 / (norms[:, None] + 1e-9)
    
    # Safeguard за norms[1:]
    norms2 = norms[1:]
    norms2 = np.maximum(norms2, 1e-9)  # Избягвай division by zero
    
    dt = d2 / (norms2[:, None] + 1e-9)
    
    # Curvature magnitude approximation via |dt/ds|
    kappa = np.zeros((pts.shape[0],), dtype=float)
    kappa[1:-1] = np.linalg.norm(dt, axis=1)
    
    # Clamp extreme values
    kappa = np.clip(kappa, -10.0, 10.0)
    
    return kappa


def select_indices_group_and_straights(midline: np.ndarray, group_step: int, use_straights_only: bool, curvature_threshold: float) -> np.ndarray:
    n = midline.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)
    # Grouping baseline
    idxs = np.arange(0, n, max(1, int(group_step)))
    # Ensure last index included
    if idxs[-1] != n - 1:
        idxs = np.concatenate([idxs, np.array([n - 1], dtype=int)])
    if use_straights_only:
        # Compute curvature and filter
        try:
            kappa = curvature_vectorized(midline.tolist())
        except Exception:
            kappa = estimate_curvature_series(midline)
        kappa = np.asarray(kappa, dtype=float)
        # Harmonize length if needed
        if kappa.shape[0] != n:
            if kappa.shape[0] == max(n - 2, 0):
                kappa = np.pad(kappa, (1, 1), mode="edge")
            else:
                # Fallback interpolate
                kappa = np.interp(np.arange(n), np.linspace(0, n - 1, max(kappa.shape[0], 2)), kappa if kappa.shape[0] > 0 else np.zeros(2))
        straight_mask = np.abs(kappa) <= float(curvature_threshold)
        # Always keep endpoints
        straight_mask[0] = True
        straight_mask[-1] = True
        idxs = idxs[straight_mask[idxs]] if idxs.size > 0 else np.where(straight_mask)[0]
        if idxs.size < 2:
            idxs = np.array([0, n - 1], dtype=int)
    return idxs


def build_path_from_corridor(inner: np.ndarray, outer: np.ndarray, alphas: np.ndarray, params: Optional[GAParams] = None) -> np.ndarray:
    """Interpolate between inner and outer edges; optionally use B-spline smoothing if enabled in params."""
    alphas = np.clip(alphas, 0.0, 1.0)
    if params is not None and getattr(params, "use_bspline", False):
        from .geometry import build_path_from_corridor_bspline
        return build_path_from_corridor_bspline(inner, outer, alphas, smoothing=float(getattr(params, "bspline_smoothing", 0.0)), degree=int(getattr(params, "bspline_degree", 3)))
    return (1.0 - alphas)[:, None] * inner + alphas[:, None] * outer


def compute_tangents_and_normals(points: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points")
    # Central differences for internal points, forward/backward for ends
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    # Normalize
    norms = np.linalg.norm(tangents, axis=1) + 1e-9
    tangents = tangents / norms[:, None]
    # 2D normal is (-ty, tx)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return tangents, normals


# Offset mode removed per user request; only corridor mode remains


def estimate_time_seconds(path_pts: np.ndarray, params: GAParams) -> float:
    vp = VehicleParams(
        mass_kg=params.mass_kg,
        mu_friction=params.mu_friction,
        gravity=params.gravity,
        rho_air=params.rho_air,
        cL_downforce=params.cL_downforce,
        frontal_area_m2=params.frontal_area_m2,
        engine_power_watts=params.engine_power_watts,
        a_brake_max=params.a_brake_max,
        a_accel_cap=params.a_accel_cap,
        cD_drag=params.cD_drag,
        c_rr=params.c_rr,
        safety_speed_margin=params.safety_speed_margin,
        brake_power_watts=params.brake_power_watts,
    )
    s, _kappa, _v_lat, v = speed_profile(path_pts.tolist(), vp)
    return compute_lap_time_seconds(s, v)


def compute_time_and_dynamics(path_pts: np.ndarray, params: GAParams, is_closed: bool = False) -> tuple:
    """
    Args:
        is_closed: True ако е затворена писта, False ако е сегмент
    """
    vp = VehicleParams(
        mass_kg=params.mass_kg,
        mu_friction=params.mu_friction,
        gravity=params.gravity,
        rho_air=params.rho_air,
        cL_downforce=params.cL_downforce,
        frontal_area_m2=params.frontal_area_m2,
        engine_power_watts=params.engine_power_watts,
        a_brake_max=params.a_brake_max,
        a_accel_cap=params.a_accel_cap,
        cD_drag=params.cD_drag,
        c_rr=params.c_rr,
        safety_speed_margin=params.safety_speed_margin,
        brake_power_watts=params.brake_power_watts,
    )
    
    if not is_closed:
        # За отворен сегмент: добави dummy точки за затваряне
        # (само за speed profile computation)
        closing_vec = path_pts[0] - path_pts[-1]
        closing_dist = np.linalg.norm(closing_vec)
        
        if closing_dist > 10.0:  # Ако е отворен
            # Добави проста връзка за затваряне
            n_close = max(5, int(closing_dist / 5.0))
            closing_pts = np.linspace(path_pts[-1], path_pts[0], n_close)
            path_closed = np.vstack([path_pts, closing_pts])
        else:
            path_closed = path_pts
    else:
        path_closed = path_pts
    
    s, _kappa, _v_lat, v = speed_profile(path_closed.tolist(), vp)
    
    # Ако не е затворен, игнорирай dummy частта
    if not is_closed and len(path_pts) < len(path_closed):
        s = s[:len(path_pts)]
        v = v[:len(path_pts)]
    
    lap_t = compute_lap_time_seconds(s, v)
    s_arr = np.asarray(s, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if s_arr.size >= 2:
        dvds = np.gradient(v_arr, s_arr, edge_order=2)
        a_long = v_arr * dvds
    else:
        a_long = np.zeros_like(v_arr)
    return lap_t, s_arr, v_arr, a_long

def close_corridor_segment(inner: np.ndarray, outer: np.ndarray, extension_factor: float = 0.20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Затваря коридор сегмент с удължения преди/след.
    Избягва дублирани точки които причиняват division by zero.
    """
    n = inner.shape[0]
    ext_len = max(10, int(n * extension_factor))
    
    # === ВХОД: добави права преди началото ===
    entry_dir_inner = (inner[1] - inner[0])
    entry_dir_inner = entry_dir_inner / (np.linalg.norm(entry_dir_inner) + 1e-9)
    entry_dir_outer = (outer[1] - outer[0])
    entry_dir_outer = entry_dir_outer / (np.linalg.norm(entry_dir_outer) + 1e-9)
    
    # ВАЖНО: Еднакво разстояние между точките (избягвай дублиране)
    step_size = 5.0  # 5 метра между точките
    
    entry_inner = []
    entry_outer = []
    for i in range(ext_len, 0, -1):
        dist = i * step_size
        p_inner = inner[0] - entry_dir_inner * dist
        p_outer = outer[0] - entry_dir_outer * dist
        entry_inner.append(p_inner)
        entry_outer.append(p_outer)
    
    entry_inner = np.array(entry_inner)
    entry_outer = np.array(entry_outer)
    
    # === ИЗХОД: добави права след края ===
    exit_dir_inner = (inner[-1] - inner[-2])
    exit_dir_inner = exit_dir_inner / (np.linalg.norm(exit_dir_inner) + 1e-9)
    exit_dir_outer = (outer[-1] - outer[-2])
    exit_dir_outer = exit_dir_outer / (np.linalg.norm(exit_dir_outer) + 1e-9)
    
    exit_inner = []
    exit_outer = []
    for i in range(1, ext_len + 1):
        dist = i * step_size
        p_inner = inner[-1] + exit_dir_inner * dist
        p_outer = outer[-1] + exit_dir_outer * dist
        exit_inner.append(p_inner)
        exit_outer.append(p_outer)
    
    exit_inner = np.array(exit_inner)
    exit_outer = np.array(exit_outer)
    
    # === КОМБИНИРАЙ ===
    extended_inner = np.vstack([entry_inner, inner, exit_inner])
    extended_outer = np.vstack([entry_outer, outer, exit_outer])
    
    # === DE-DUPLICATE: Премахни твърде близки точки ===
    def remove_duplicates(pts: np.ndarray, min_dist: float = 0.5) -> np.ndarray:
        """Премахва точки които са по-близо от min_dist"""
        if pts.shape[0] < 2:
            return pts
        
        keep = [0]  # Винаги запази първата точка
        for i in range(1, pts.shape[0]):
            dist = np.linalg.norm(pts[i] - pts[keep[-1]])
            if dist >= min_dist:
                keep.append(i)
        
        return pts[keep]
    
    extended_inner = remove_duplicates(extended_inner, min_dist=1.0)
    extended_outer = remove_duplicates(extended_outer, min_dist=1.0)
    
    print(f"[close_corridor] After de-duplication: inner={extended_inner.shape[0]}, outer={extended_outer.shape[0]} points")
    
    return extended_inner, extended_outer

def offsets_smoothness_penalty(offsets: np.ndarray) -> float:
    # Penalize large offset changes to keep feasible smooth line
    if offsets.size < 3:
        return 0.0
    d1 = np.diff(offsets)
    d2 = np.diff(offsets, n=2)
    return float(np.mean(d1 * d1) + np.mean(d2 * d2))


def path_roughness_penalty(path_pts: np.ndarray) -> float:
    if path_pts.shape[0] < 3:
        return 0.0
    diffs = np.diff(path_pts, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    dheading = np.diff(headings)
    # wrap to [-pi, pi]
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(dheading * dheading))


def heading_spike_penalty(path_pts: np.ndarray, max_delta_rad: float) -> float:
    if path_pts.shape[0] < 3:
        return 0.0
    diffs = np.diff(path_pts, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    dheading = np.diff(headings)
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
    excess = np.abs(dheading) - float(max_delta_rad)
    excess = np.clip(excess, 0.0, None)
    return float(np.mean(excess * excess))


def curvature_penalty(path_pts: np.ndarray) -> float:
    kappa = estimate_curvature_series(path_pts)
    if kappa.size == 0:
        return 0.0
    return float(np.mean(kappa * kappa))


def curvature_jerk_penalty(path_pts: np.ndarray) -> float:
    kappa = estimate_curvature_series(path_pts)
    if kappa.size < 3:
        return 0.0
    dk = np.diff(kappa)
    d2k = np.diff(kappa, n=2) if kappa.size > 3 else np.array([0.0])
    return float(np.mean(dk * dk) + (np.mean(d2k * d2k) if d2k.size > 0 else 0.0))


# Offset evaluation removed (corridor-only GA). Provide stubs to keep interface intact.
def evaluate_individual_offset(base_pts: List[Point], base_normals: np.ndarray, offsets: np.ndarray, params: GAParams) -> float:
    raise NotImplementedError("Offset mode is disabled in this build. Use corridor mode with inner/outer edges.")


def build_offset_path(pts_list: List[Point], normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_list, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if normals.shape[0] != pts.shape[0] or offsets.size != pts.shape[0]:
        raise ValueError("Mismatched shapes for offset path construction")
    return pts + normals * offsets[:, None]


def evaluate_individual_corridor(inner: np.ndarray, outer: np.ndarray, alphas: np.ndarray, params: GAParams) -> float:
    alphas = np.clip(alphas, 0.0, 1.0)
    path_pts = build_path_from_corridor(inner, outer, alphas, params)
    t, s_arr, v_arr, a_long = compute_time_and_dynamics(path_pts, params)
    
    # === БЕЗ DEATH PENALTY - само градуално наказание ===
    # Проверка на кривина (без instant failure)
    kappa_abs = np.abs(estimate_curvature_series(path_pts))
    if kappa_abs.size > 0:
        max_kappa = float(np.max(kappa_abs))
        # Градуално квадратично наказание
        if max_kappa > float(params.max_curvature):
            curvature_excess = (max_kappa / float(params.max_curvature) - 1.0)
            # Мек penalty пропорционален на превишаването
            t = t * (1.0 + 50.0 * curvature_excess ** 2)
    # Smooth alpha variation to avoid zig-zag
    d1 = np.diff(alphas)
    d2 = np.diff(alphas, n=2) if len(alphas) > 2 else np.array([0.0])
    # Minimal time primary objective; small regularization for feasibility/smoothness
    penalty = params.offset_smooth_lambda * (np.mean(d1 * d1) + (np.mean(d2 * d2) if d2.size > 0 else 0.0))
    # Hard-limit-like penalty for large alpha steps (discourage kinks)
    if d1.size > 0:
        excess = np.abs(d1) - float(params.alpha_delta_max)
        excess = np.clip(excess, 0.0, None)
        penalty += params.alpha_rate_lambda * float(np.mean(excess * excess))
    penalty += params.smoothness_lambda * path_roughness_penalty(path_pts)
    penalty += params.curvature_lambda * curvature_penalty(path_pts)
    penalty += params.heading_spike_lambda * heading_spike_penalty(path_pts, params.heading_max_delta_rad)
    penalty += params.center_deviation_lambda * float(np.mean((alphas - 0.5) * (alphas - 0.5)))
    # Longitudinal accel/brake over-limit penalties
    if a_long.size > 0:
        accel_over = np.clip(a_long - params.a_accel_cap, 0.0, None)
        brake_over = np.clip(-(a_long) - params.a_brake_max, 0.0, None)
        denom = max(params.a_accel_cap, params.a_brake_max, 1e-6)
        penalty += params.long_accel_lambda * float(np.mean((accel_over / denom) ** 2 + (brake_over / denom) ** 2))
        if s_arr.size >= 2:
            dads = np.gradient(a_long, s_arr, edge_order=2)
            jerk = dads * v_arr
            jerk_norm = jerk / max(denom * 2.0, 1e-6)
            penalty += params.long_jerk_lambda * float(np.mean(jerk_norm * jerk_norm))

    # Steering rate penalty (driver control rate feasibility)
    try:
        diffs = np.diff(path_pts, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        dheading = np.diff(headings)
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
        seg_lengths = np.linalg.norm(diffs[:-1], axis=1)
        if v_arr.size >= 3 and seg_lengths.size == dheading.size:
            avg_v = 0.5 * (v_arr[:-2] + v_arr[1:-1])
            dt = seg_lengths / np.maximum(avg_v, 1.0)
            steering_rate = np.abs(dheading) / np.maximum(dt, 1e-3)
            max_rate_rad = np.deg2rad(180.0)
            excess = np.clip(steering_rate - max_rate_rad, 0.0, None)
            penalty += params.steering_rate_lambda * float(np.mean(excess * excess))
    except Exception:
        pass

    # Removed optional penalties: lateral jerk, adaptive safety margin, load transfer

    # Curvature constraints and straightness incentives
    try:
        kappa_chk = estimate_curvature_series(path_pts)
        kappa_abs2 = np.abs(kappa_chk)
        hard_violations = float(np.sum(np.maximum(kappa_abs2 - float(params.max_curvature), 0.0)))
        if hard_violations > 1e-3:
            penalty += float(params.curvature_hard_penalty) * hard_violations
        penalty += float(params.curvature_soft_penalty) * fit_curvature_limit(
            path_pts, max_kappa=float(params.max_curvature), hardness=float(params.curvature_hardness)
        )
        # Straightness reward (negative penalty)
        # Scale by configured straightness_bonus via multiplier of default 0.3 baseline
        base_bonus = fit_straight_reward(path_pts, target_kappa=float(params.max_curvature))
        scale = float(params.straightness_bonus) / 0.3 if 0.3 != 0 else 1.0
        penalty += scale * base_bonus
        # Additional alpha smoothness reinforcement
        penalty += 1.0 * params.offset_smooth_lambda * np.mean(d1 ** 2)
    except Exception:
        pass
    # Weighted fitness
    time_w = float(getattr(params, "current_time_weight", params.time_weight))
    pen_w = float(getattr(params, "current_penalty_weight", params.penalty_weight))
    return time_w * float(t) + pen_w * float(penalty)


def _evaluate_with_cache(inner: np.ndarray, outer: np.ndarray, alphas: np.ndarray, params_eval: GAParams, cache_key_extra: Tuple[Any, ...], cache_step: float) -> float:
    key = (_quantize_array(alphas, cache_step),) + cache_key_extra
    got = _fitness_cache.get(key)
    if got is not None:
        return got
    fit = evaluate_individual_corridor(inner, outer, alphas, params_eval)
    _fitness_cache.set(key, fit)
    return fit


def smooth_series_moving_average(values: np.ndarray, repeats: int = 2) -> np.ndarray:
    """Light smoothing with a 5-tap binomial kernel; clamps to [0,1] for alphas."""
    if values.size < 5 or repeats <= 0:
        return values
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()
    out = values.astype(float)
    for _ in range(repeats):
        out = np.convolve(out, kernel, mode="same")
    return out


def limit_series_rate(values: np.ndarray, max_delta: float) -> np.ndarray:
    if values.size <= 1:
        return values
    out = values.astype(float).copy()
    for i in range(1, out.size):
        delta = out[i] - out[i - 1]
        if delta > max_delta:
            out[i] = out[i - 1] + max_delta
        elif delta < -max_delta:
            out[i] = out[i - 1] - max_delta
    return out


def tournament_select(pop: List[np.ndarray], fitness: List[float], k: int) -> int:
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fitness[i])
    return best


def crossover(a: np.ndarray, b: np.ndarray, rate: float) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > rate or a.size < 2:
        return a.copy(), b.copy()
    point = random.randint(1, a.size - 1)
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def mutate(ind: np.ndarray, rate: float, sigma: float, max_abs: float) -> np.ndarray:
    mask = np.random.rand(ind.size) < rate
    noise = np.random.normal(0.0, sigma, size=ind.size)
    ind = ind + mask * noise
    return np.clip(ind, -max_abs, max_abs)


# def run_ga_offset(points: List[Point], params: GAParams) -> Tuple[np.ndarray, np.ndarray, float]:
#     # Resample for stable optimization resolution
#     t_start = time.perf_counter()
#     pts_arr = resample_polyline_arclength(points, max(len(points), params.target_samples))
#     print(f"[GA-offset] Resampled to {pts_arr.shape[0]} points (target_samples={params.target_samples})")
#     pts_list = pts_arr.tolist()
#     _, normals = compute_tangents_and_normals(pts_list)
#     # Optional grouping and straights-only selection
#     sel_idxs = select_indices_group_and_straights(np.asarray(pts_list, dtype=float), params.group_step, params.use_straights_only, params.curvature_threshold)
#     base_pts = [pts_list[i] for i in sel_idxs.tolist()]
#     base_normals = np.asarray(normals, dtype=float)[sel_idxs]
#     n = len(base_pts)
#     print(f"[GA-offset] Using {n}/{len(pts_list)} points (group_step={params.group_step}, straights_only={params.use_straights_only}, curv_thr={params.curvature_threshold})")
#     print(f"[GA-offset] GA config: pop={params.population_size}, gen={params.generations}")
#     pop = [np.random.uniform(-0.2, 0.2, size=n).astype(float) for _ in range(params.population_size)]
#     fitness = [evaluate_individual_offset(base_pts, base_normals, ind, params) for ind in pop]

#     num_elite = max(1, int(params.elite_fraction * params.population_size))

#     best_offsets = pop[int(np.argmin(fitness))].copy()
#     best_time = float(min(fitness))

#     last_report = -1
#     no_improve = 0
#     for _gen in range(params.generations):
#         # Elitism
#         elite_idx = np.argsort(fitness)[:num_elite]
#         new_pop: List[np.ndarray] = [pop[i].copy() for i in elite_idx]

#         # Create rest via tournament selection, crossover, mutation
#         while len(new_pop) < params.population_size:
#             i1 = tournament_select(pop, fitness, params.tournament_k)
#             i2 = tournament_select(pop, fitness, params.tournament_k)
#             c1, c2 = crossover(pop[i1], pop[i2], params.crossover_rate)
#             # Adaptive mutation sigma (annealing + stagnation boost)
#             sigma_prog = (1.0 - (_gen / max(params.generations - 1, 1))) if params.use_annealing else 1.0
#             sigma = params.mutation_sigma_end + (params.mutation_sigma_start - params.mutation_sigma_end) * sigma_prog
#             if no_improve >= params.stagnation_patience // 2:
#                 sigma *= 1.5
#             c1 = mutate(c1, params.mutation_rate, sigma, params.max_offset)
#             if len(new_pop) < params.population_size:
#                 new_pop.append(c1)
#             if len(new_pop) < params.population_size:
#                 c2 = mutate(c2, params.mutation_rate, sigma, params.max_offset)
#                 new_pop.append(c2)

#         pop = new_pop
#         fitness = [evaluate_individual_offset(base_pts, base_normals, ind, params) for ind in pop]

#         gen_best_idx = int(np.argmin(fitness))
#         gen_best = pop[gen_best_idx]
#         gen_best_fit = float(fitness[gen_best_idx])
#         if gen_best_fit < best_time - 1e-6:
#             best_time = gen_best_fit
#             best_offsets = gen_best.copy()
#             no_improve = 0
#         else:
#             no_improve += 1

#         # Random immigrants to maintain diversity
#         if params.immigrants_fraction > 0.0:
#             num_imm = int(params.immigrants_fraction * params.population_size)
#             if num_imm > 0:
#                 worst_idx = np.argsort(fitness)[-num_imm:]
#                 for wi in worst_idx:
#                     pop[wi] = np.random.uniform(-0.2, 0.2, size=n).astype(float)
#                     fitness[wi] = evaluate_individual_offset(base_pts, base_normals, pop[wi], params)
#         if _gen % 10 == 0 and _gen != last_report:
#             last_report = _gen
#             mean_fit = float(np.mean(fitness))
#             std_fit = float(np.std(fitness))
#             print(f"[GA-offset] gen={_gen:4d} best_time={best_time:.4f}  mean={mean_fit:.4f}  std={std_fit:.4f}  no_improve={no_improve}")

#         # Restart non-elite on prolonged stagnation
#         if no_improve >= params.stagnation_patience:
#             base_pop_size = params.population_size - num_elite
#             new_pop = [pop[i].copy() for i in elite_idx]
#             new_pop.extend([np.random.uniform(-0.2, 0.2, size=n).astype(float) for _ in range(max(0, base_pop_size))])
#             pop = new_pop[:params.population_size]
#             fitness = [evaluate_individual_offset(base_pts, base_normals, ind, params) for ind in pop]
#             no_improve = 0
#             print("[GA-offset] Restart triggered (stagnation)")

#     # Interpolate offsets back to full resolution along the selected index parameterization
#     sel_s = np.linspace(0.0, 1.0, n)
#     full_s = np.linspace(0.0, 1.0, len(pts_list))
#     best_offsets_full = np.interp(full_s, sel_s, best_offsets)
#     # Smooth offsets for realistic steering continuity and rebuild path
#     best_offsets_full = smooth_series_moving_average(best_offsets_full, repeats=1)
#     best_path = build_offset_path(pts_list, normals, best_offsets_full)
#     dt = time.perf_counter() - t_start
#     print(f"[GA-offset] Done. best_time={best_time:.4f}  duration={dt:.2f}s")
#     return best_offsets, best_path, best_time

def create_racing_line_initialization(inner: np.ndarray, outer: np.ndarray, apex_fraction: float = 0.55) -> np.ndarray:
    """
    КОНСЕРВАТИВНА racing line за стабилна инициализация
    GA ще я оптимизира към агресивна линия след това
    """
    n = inner.shape[0]
    
    # Намери апекса по кривина
    midline = 0.5 * (inner + outer)
    kappa = np.abs(estimate_curvature_series(midline))
    if kappa.size > 0 and np.max(kappa) > 1e-6:
        apex_idx = int(np.argmax(kappa))
    else:
        apex_idx = n // 2
    
    # ПЛАВНА инициализация (не агресивна!)
    alphas = np.zeros(n)
    
    # ВХОД: синусоидален преход от външно към апекс
    if apex_idx > 0:
        t = np.linspace(0, np.pi/2, apex_idx)
        alphas[:apex_idx] = 0.70 - 0.40 * np.sin(t)  # 0.70 -> 0.30
    
    # АПЕКС: умерено вътрешно
    alphas[apex_idx] = 0.30
    
    # ИЗХОД: синусоидален преход от апекс към външно
    exit_length = n - apex_idx - 1
    if exit_length > 0:
        t = np.linspace(0, np.pi/2, exit_length)
        alphas[apex_idx+1:] = 0.30 + 0.40 * np.sin(t)  # 0.30 -> 0.70
    
    # СИЛНО изглаждане за стабилност
    alphas = smooth_series_moving_average(alphas, repeats=3)
    
    return np.clip(alphas, 0.0, 1.0)

def run_ga_corridor(inner: np.ndarray, outer: np.ndarray, params: GAParams) -> Tuple[np.ndarray, np.ndarray, float]:
    # Timing
    total_start = time.perf_counter()

    def prepare_sampling(inner_arr: np.ndarray, outer_arr: np.ndarray, target_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Adaptive sampling based on curvature on midline
        mid = 0.5 * (inner_arr + outer_arr)
        if target_n <= 0:
            target_n = mid.shape[0]
        if params.curvature_sampling_beta > 0.0:
            inner_rs = resample_polyline_by_curvature(inner_arr.tolist(), max(target_n, 10), beta=params.curvature_sampling_beta)
            outer_rs = resample_polyline_by_curvature(outer_arr.tolist(), max(target_n, 10), beta=params.curvature_sampling_beta)
        else:
            inner_rs = resample_polyline_arclength(inner_arr.tolist(), max(target_n, 10))
            outer_rs = resample_polyline_arclength(outer_arr.tolist(), max(target_n, 10))
        mid_rs = 0.5 * (inner_rs + outer_rs)
        return inner_rs, outer_rs, mid_rs

    # Coarse-to-fine setup
    fitness_history: List[float] = []
    best_alphas_full_res: Optional[np.ndarray] = None
    best_time_overall: float = float("inf")

    stages: List[Tuple[int, float, float]] = []  # (samples, sigma_start, sigma_end)
    full_target = max(int(params.target_samples), max(inner.shape[0], outer.shape[0]))
    if params.coarse_to_fine and full_target >= 40:
        coarse_n = max(10, int(full_target * params.coarse_factor))
        stages.append((coarse_n, params.mutation_sigma_start * 1.2, params.mutation_sigma_end * 1.2))
        stages.append((full_target, params.mutation_sigma_start, params.mutation_sigma_end))
    else:
        stages.append((full_target, params.mutation_sigma_start, params.mutation_sigma_end))

    last_report = -1
    for stage_idx, (target_n, sigma_start, sigma_end) in enumerate(stages):
        t_stage = time.perf_counter()
        inner_rs, outer_rs, midline = prepare_sampling(inner, outer, target_n)
        n = inner_rs.shape[0]
        print(f"[GA-corridor] Stage {stage_idx+1}/{len(stages)}: samples={n}")

        # Initialization with racing line
        rng = np.random.default_rng()
        if best_alphas_full_res is not None:
            # Project previous best to current resolution
            sel_s_prev = np.linspace(0.0, 1.0, best_alphas_full_res.size)
            sel_s_curr = np.linspace(0.0, 1.0, n)
            init_center = np.clip(np.interp(sel_s_curr, sel_s_prev, best_alphas_full_res), 0.0, 1.0)
        else:
            # Създай racing line като начална точка
            init_center = create_racing_line_initialization(inner_rs, outer_rs, apex_fraction=0.5)
            print(f"[GA-corridor] Initialized with racing line (apex detection)")

        racing_line_count = params.population_size // 2
        pop = []

        # Racing line вариации (малки отклонения)
        for _ in range(racing_line_count):
            pop.append(np.clip(init_center + rng.normal(0.0, 0.03, size=n), 0.0, 1.0))

        # Diverse exploration (по-големи отклонения)
        for _ in range(params.population_size - racing_line_count):
            pop.append(np.clip(init_center + rng.normal(0.0, 0.15, size=n), 0.0, 1.0))

        print(f"[GA-corridor] Population: {racing_line_count} near racing line, {params.population_size - racing_line_count} diverse")

        # Diversity helpers
        def population_diversity(pop_arr: List[np.ndarray]) -> float:
            if len(pop_arr) == 0:
                return 0.0
            mat = np.stack(pop_arr, axis=0)
            return float(np.mean(np.std(mat, axis=0)))

        # Curvature for mutation scaling (lower sigma in turns)
        kappa = np.abs(estimate_curvature_series(midline))
        if np.max(kappa) > 0:
            kappa_norm = kappa / np.max(kappa)
        else:
            kappa_norm = kappa
        curvature_sigma_scale = 0.6 + 0.4 * (1.0 - kappa_norm)  # 0.6 in turns -> 1.0 on straights

        num_elite = max(1, int(params.elite_fraction * params.population_size))

        # Fitness function with adaptive penalties over generations (apply via param copy)
        def params_for_gen(gen_idx: int) -> GAParams:
            if params.adaptive_penalty_decay <= 0.0 or len(stages) == 0:
                return params
            prog = gen_idx / max(params.generations - 1, 1)
            decay = (1.0 - prog) * params.adaptive_penalty_decay + (1.0 - params.adaptive_penalty_decay)
            # Adaptive time/penalty weights
            if params.adaptive_time_weight:
                # Increase time weight over generations, keep penalty weight decaying
                time_w = params.time_weight * (0.5 + 0.5 * prog)
                pen_w = params.penalty_weight * decay
            else:
                time_w = params.time_weight
                pen_w = params.penalty_weight
            return replace(
                params,
                mutation_sigma_start=sigma_start,
                mutation_sigma_end=sigma_end,
                smoothness_lambda=params.smoothness_lambda * decay,
                offset_smooth_lambda=params.offset_smooth_lambda * decay,
                curvature_lambda=params.curvature_lambda * decay,
                curvature_jerk_lambda=params.curvature_jerk_lambda * decay,
                heading_spike_lambda=params.heading_spike_lambda * decay,
                center_deviation_lambda=params.center_deviation_lambda * decay,
                long_accel_lambda=params.long_accel_lambda * decay,
                long_jerk_lambda=params.long_jerk_lambda * decay,
                current_time_weight=time_w,
                current_penalty_weight=pen_w,
            )

        # Initial fitness
        cache_extra_key = (
            params.mass_kg, params.mu_friction, params.cL_downforce, params.cD_drag,
            params.a_brake_max, params.a_accel_cap, params.safety_speed_margin,
        )
        # Smart initialization: heuristic + random
        kappa0 = np.abs(estimate_curvature_series(midline))
        base_alphas = np.where(kappa0 > 0.05, 0.3, 0.5)
        pop = []
        heuristic_n = params.population_size // 3
        for _ in range(heuristic_n):
            pop.append(np.clip(base_alphas + rng.normal(0.0, 0.05, size=n), 0.0, 1.0))
        for _ in range(params.population_size - len(pop)):
            pop.append(rng.uniform(0.2, 0.8, size=n).astype(float))

        if params.enable_parallel and params.parallel_workers > 1:
            with ProcessPoolExecutor(max_workers=params.parallel_workers) as ex:
                futures = [ex.submit(_evaluate_with_cache, inner_rs, outer_rs, ind, params_for_gen(0), cache_extra_key, params.cache_granularity) for ind in pop]
                fitness = [f.result() for f in futures]
        else:
            fitness = [_evaluate_with_cache(inner_rs, outer_rs, ind, params_for_gen(0), cache_extra_key, params.cache_granularity) for ind in pop]
            # ДОБАВИ:
            print(f"[DEBUG] Initial fitness range: min={np.min(fitness):.2f}, max={np.max(fitness):.2f}, mean={np.mean(fitness):.2f}")
            print(f"[DEBUG] Initial best alpha range: min={np.min(best_alphas):.3f}, max={np.max(best_alphas):.3f}, mean={np.mean(best_alphas):.3f}")

            # Провери дали има валидни решения
            valid_count = np.sum(np.array(fitness) < 1e6)
            print(f"[DEBUG] Valid solutions: {valid_count}/{len(fitness)}")

        best_idx = int(np.argmin(fitness))
        best_alphas = pop[best_idx].copy()
        best_time = float(fitness[best_idx])
        no_improve = 0

        gen_start_time = time.perf_counter()
        for _gen in range(params.generations):
            # Elitism
            elite_idx = np.argsort(fitness)[:num_elite]
            new_pop: List[np.ndarray] = [pop[i].copy() for i in elite_idx]

            # Adaptive mutation sigma: annealing + diversity control
            sigma_prog = (1.0 - (_gen / max(params.generations - 1, 1))) if params.use_annealing else 1.0
            base_sigma = sigma_end + (sigma_start - sigma_end) * sigma_prog
            diversity = population_diversity(pop)
            if params.diversity_target > 0.0:
                # Increase sigma if diversity low, decrease if high
                base_sigma *= np.clip(params.diversity_target / max(diversity, 1e-6), 0.5, 2.0)
            if no_improve >= params.stagnation_patience // 2:
                base_sigma *= 1.5

            # Dynamic population sizing
            if getattr(params, "use_dynamic_population", False):
                div_now = diversity
                target_size = compute_adaptive_population_size(_gen, params.generations, params.population_size, div_now, params.diversity_target, params.population_min_ratio, params.population_max_ratio)
                if target_size != len(pop):
                    print(f"[GA] Adjusting population: {len(pop)} -> {target_size} (diversity={div_now:.4f})")
                    pop, fitness = adjust_population_size(pop, fitness, target_size, rng)
                    # Recompute elite after resize
                    elite_idx = np.argsort(fitness)[:max(1, int(params.elite_fraction * len(pop)))]
                    new_pop = [pop[i].copy() for i in elite_idx]

            # Vectorized breeding
            need = len(pop) - len(new_pop)
            if need > 0:
                fitness_arr = np.asarray(fitness, dtype=float)
                # select pairs
                idx_a = vectorized_tournament_select(fitness_arr, params.tournament_k, need, rng)
                idx_b = vectorized_tournament_select(fitness_arr, params.tournament_k, need, rng)
                parents_a = np.stack([pop[i] for i in idx_a], axis=0)
                parents_b = np.stack([pop[i] for i in idx_b], axis=0)
                if params.use_adaptive_crossover:
                    fitness_a = fitness_arr[idx_a]
                    fitness_b = fitness_arr[idx_b]
                    offspring = adaptive_crossover_weighted(parents_a, parents_b, fitness_a, fitness_b, params.crossover_rate, rng)
                else:
                    offspring = uniform_crossover(parents_a, parents_b, params.crossover_rate, rng)
                # Curvature-biased per-gene sigma
                per_gene_sigma = base_sigma * curvature_sigma_scale
                offspring = gaussian_mutation(offspring, per_gene_sigma, rng)
                # Append to population list
                new_pop.extend([offspring[i].astype(float) for i in range(offspring.shape[0])])

            pop = new_pop

            # Evaluate fitness (parallel, cached) with adaptive penalties at this generation
            params_eval = params_for_gen(_gen)
            cache_extra_key_gen = (
                params_eval.smoothness_lambda,
                params_eval.offset_smooth_lambda,
                params_eval.curvature_lambda,
                params_eval.curvature_jerk_lambda,
                params_eval.heading_spike_lambda,
                params_eval.center_deviation_lambda,
                params_eval.long_accel_lambda,
                params_eval.long_jerk_lambda,
            ) + cache_extra_key

            if params.enable_parallel and params.parallel_workers > 1:
                with ProcessPoolExecutor(max_workers=params.parallel_workers) as ex:
                    futures = [ex.submit(_evaluate_with_cache, inner_rs, outer_rs, ind, params_eval, cache_extra_key_gen, params.cache_granularity) for ind in pop]
                    fitness = [f.result() for f in futures]
            else:
                fitness = [_evaluate_with_cache(inner_rs, outer_rs, ind, params_eval, cache_extra_key_gen, params.cache_granularity) for ind in pop]
                # ДОБАВИ:
                print(f"[DEBUG] Initial fitness range: min={np.min(fitness):.2f}, max={np.max(fitness):.2f}, mean={np.mean(fitness):.2f}")
                print(f"[DEBUG] Initial best alpha range: min={np.min(best_alphas):.3f}, max={np.max(best_alphas):.3f}, mean={np.mean(best_alphas):.3f}")

                # Провери дали има валидни решения
                valid_count = np.sum(np.array(fitness) < 1e6)
                print(f"[DEBUG] Valid solutions: {valid_count}/{len(fitness)}")

            # Optional logging of time vs penalty ratio on current best individual
            try:
                best_idx_dbg = int(np.argmin(fitness))
                best_alphas_dbg = pop[best_idx_dbg]
                path_dbg = build_path_from_corridor(inner_rs, outer_rs, best_alphas_dbg, params_eval)
                t_dbg, _, _, _ = compute_time_and_dynamics(path_dbg, params_eval)
                # Recompute penalties only (approx): reuse evaluate but subtract weighted time
                fit_dbg = evaluate_individual_corridor(inner_rs, outer_rs, best_alphas_dbg, params_eval)
                time_w_dbg = float(getattr(params_eval, "current_time_weight", params_eval.time_weight))
                pen_w_dbg = float(getattr(params_eval, "current_penalty_weight", params_eval.penalty_weight))
                penalty_dbg = max(fit_dbg - time_w_dbg * float(t_dbg), 0.0) / max(pen_w_dbg, 1e-6)
                ratio = penalty_dbg / max(float(t_dbg), 1e-6)
                print(f"[GA] gen={_gen} time={t_dbg:.3f}s penalty={penalty_dbg:.3f} ratio={ratio:.2f}")
            except Exception:
                pass

            # Fitness sharing (niching) to encourage diversity
            try:
                sigma_share = 0.1
                penalty_factor = 0.3
                pop_mat = np.stack(pop, axis=0)
                # Pairwise distances (approximate using subset for speed if large)
                if pop_mat.shape[0] <= 256:
                    dists = np.linalg.norm(pop_mat[:, None, :] - pop_mat[None, :, :], axis=2)
                    niche_counts = np.clip(1.0 - dists / sigma_share, 0.0, None)
                    # zero self
                    np.fill_diagonal(niche_counts, 0.0)
                    sharing = 1.0 + penalty_factor * np.sum(niche_counts, axis=1)
                    fitness = (np.asarray(fitness) * sharing).tolist()
            except Exception:
                pass

            gen_best_idx = int(np.argmin(fitness))
            gen_best = pop[gen_best_idx]
            gen_best_fit = float(fitness[gen_best_idx])
            fitness_history.append(gen_best_fit)

            improved = False
            if gen_best_fit < best_time - 1e-9:
                best_time = gen_best_fit
                best_alphas = gen_best.copy()
                no_improve = 0
                improved = True
            else:
                no_improve += 1

            # Random immigrants to maintain diversity
            if params.immigrants_fraction > 0.0:
                num_imm = int(params.immigrants_fraction * params.population_size)
                if num_imm > 0:
                    worst_idx = np.argsort(fitness)[-num_imm:]
                    for wi in worst_idx:
                        pop[wi] = np.clip(rng.uniform(0.2, 0.8, size=n).astype(float), 0.0, 1.0)
                        fitness[wi] = _evaluate_with_cache(inner_rs, outer_rs, pop[wi], params_eval, cache_extra_key_gen, params.cache_granularity)

            # Logging
            if _gen % 10 == 0 and _gen != last_report:
                last_report = _gen
                mean_fit = float(np.mean(fitness))
                std_fit = float(np.std(fitness))
                elapsed_gen = time.perf_counter() - gen_start_time
                gen_start_time = time.perf_counter()
                print(f"[GA-corridor] stage={stage_idx+1} gen={_gen:4d} best={best_time:.4f} mean={mean_fit:.4f} std={std_fit:.4f} div={diversity:.4f} dt={elapsed_gen:.2f}s imp={'Y' if improved else 'N'} no_imp={no_improve}")

            # Early stopping
            if no_improve >= params.stagnation_patience:
                print("[GA-corridor] Early stopping (stagnation)")
                break

        # Stage result -> project to full resolution baseline for next stage
        sel_s = np.linspace(0.0, 1.0, n)
        if best_alphas is None or best_alphas.size == 0:
            best_alphas = np.full((n,), 0.5, dtype=float)
        best_alphas_full_res = np.clip(np.interp(np.linspace(0.0, 1.0, full_target), sel_s, best_alphas), 0.0, 1.0)
        best_time_overall = min(best_time_overall, best_time)
        print(f"[GA-corridor] Stage {stage_idx+1} done in {time.perf_counter()-t_stage:.2f}s best={best_time:.4f}")

    # Final smoothing and path construction at full resolution
    # ИЗТРИЙ ВСИЧКО И СЛОЖИ:
    inner_final, outer_final, mid_final = prepare_sampling(inner, outer, full_target)

    # Директно използвай резултата - БЕЗ post-processing!
    best_alphas_full = best_alphas_full_res.copy()

    # Само минимален rate limit за физична възможност
    best_alphas_full = np.clip(limit_series_rate(best_alphas_full, params.alpha_delta_max), 0.0, 1.0)

    # Построй финалната линия
    best_path = build_path_from_corridor(inner_final, outer_final, best_alphas_full)
    # Curvature validation report
    try:
        num_viol, max_kappa_val = fit_count_curv_viol(best_path, max_kappa=float(params.max_curvature))
        print(f"[GA-corridor] Final curvature check: violations={num_viol}, max_κ={max_kappa_val:.4f}")
        if num_viol > 0:
            print(f"[GA-corridor] WARNING: {num_viol} points still violate κ > {float(params.max_curvature):.4f}!")
        else:
            print(f"[GA-corridor] SUCCESS: All points have κ ≤ {float(params.max_curvature):.4f}")
    except Exception:
        pass
    dt_total = time.perf_counter() - total_start
    print(f"[GA-corridor] Done. best_time={best_time_overall:.4f} duration={dt_total:.2f}s")
    return best_alphas_full_res, best_path, best_time_overall, fitness_history


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GA optimizer for Turn_1.csv")
    parser.add_argument("--turns-dir", type=str, default=None, help="Base 'Turns' directory (defaults to firmware/Turns)")
    # Offset mode removed; no max-offset
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gen", type=int, default=400, help="Generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Single-resolution (no coarse/refine, no straights-only)
    parser.add_argument("--inner", type=str, default=None, help="CSV path for inner edge of corridor (x,y)")
    parser.add_argument("--outer", type=str, default=None, help="CSV path for outer edge of corridor (x,y)")
    parser.add_argument("--car-preset", type=str, default=None, choices=["f1_2024", "indy", "gt3"], help="Preset vehicle parameters")
    # Vehicle parameters for high fidelity
    parser.add_argument("--mass", type=float, default=798.0)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--rho", type=float, default=1.225)
    parser.add_argument("--cL", type=float, default=4.0)
    parser.add_argument("--area", type=float, default=1.6)
    parser.add_argument("--power", type=float, default=735000.0)
    parser.add_argument("--abrake", type=float, default=54.0)
    parser.add_argument("--aaccel", type=float, default=20.0)
    parser.add_argument("--cD", type=float, default=1.0)
    parser.add_argument("--crr", type=float, default=0.004)
    parser.add_argument("--safety", type=float, default=1.00)
    parser.add_argument("--brake_power", type=float, default=1200000.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    turn_csv = load_latest_turn1_csv(base_dir=args.turns_dir)
    points = read_points_csv(turn_csv)

    # Corridor setup (optional)
    inner_edge = None
    outer_edge = None
    use_corridor = False
    # 1) Explicit corridor via --inner/--outer takes precedence
    if args.inner and args.outer:
        inner_pts = read_points_csv(args.inner)
        outer_pts = read_points_csv(args.outer)
        # Resample to match reference (turn points)
        inner_edge = resample_to_match(points, inner_pts)
        outer_edge = resample_to_match(points, outer_pts)
        # Ensure shapes
        if inner_edge.shape[0] == len(points) and outer_edge.shape[0] == len(points):
            use_corridor = True
    else:
        # 2) Try to build corridor directly from the segment CSV (x_l,y_l,x_r,y_r)
        corridor = read_corridor_from_segment_csv(turn_csv)
        if corridor is not None:
            inner_edge, outer_edge = corridor
            # If necessary, resample to match point count used for GA
            if inner_edge.shape[0] != len(points):
                inner_edge = resample_to_match(points, inner_edge.tolist())
            if outer_edge.shape[0] != len(points):
                outer_edge = resample_to_match(points, outer_edge.tolist())
        if inner_edge.shape[0] == len(points) and outer_edge.shape[0] == len(points):
            use_corridor = True


  
    # Apply car presets (can be overridden by explicit flags)
    preset = None
    if args.car_preset == "f1_2024":
        preset = VehicleParams(
            mass_kg=798.0, mu_friction=2.2, gravity=9.81, rho_air=1.225, cL_downforce=4.5,
            frontal_area_m2=1.6, engine_power_watts=750000.0, a_brake_max=54.0, a_accel_cap=22.0,
            cD_drag=1.05, c_rr=0.004, safety_speed_margin=1.00, brake_power_watts=1200000.0,
        )
    elif args.car_preset == "indy":
        preset = VehicleParams(
            mass_kg=730.0, mu_friction=2.0, gravity=9.81, rho_air=1.225, cL_downforce=3.5,
            frontal_area_m2=1.7, engine_power_watts=530000.0, a_brake_max=50.0, a_accel_cap=18.0,
            cD_drag=1.0, c_rr=0.005, safety_speed_margin=1.00, brake_power_watts=1000000.0,
        )
    elif args.car_preset == "gt3":
        preset = VehicleParams(
            mass_kg=1300.0, mu_friction=1.6, gravity=9.81, rho_air=1.225, cL_downforce=1.5,
            frontal_area_m2=2.0, engine_power_watts=405000.0, a_brake_max=30.0, a_accel_cap=12.0,
            cD_drag=0.9, c_rr=0.010, safety_speed_margin=1.00, brake_power_watts=600000.0,
        )

    # Start from preset if provided, then override with CLI values
    if preset is not None:
        base_mass = preset.mass_kg; base_mu = preset.mu_friction; base_g = preset.gravity
        base_rho = preset.rho_air; base_cL = preset.cL_downforce; base_area = preset.frontal_area_m2
        base_power = preset.engine_power_watts; base_abrake = preset.a_brake_max; base_aaccel = preset.a_accel_cap
        base_cD = preset.cD_drag; base_crr = preset.c_rr; base_safety = preset.safety_speed_margin
        base_bpw = preset.brake_power_watts
    else:
        base_mass = args.mass; base_mu = args.mu; base_g = args.g
        base_rho = args.rho; base_cL = args.cL; base_area = args.area
        base_power = args.power; base_abrake = args.abrake; base_aaccel = args.aaccel
        base_cD = args.cD; base_crr = args.crr; base_safety = args.safety
        base_bpw = args.brake_power

    params = GAParams(
    # === ПОПУЛАЦИЯ ===
        population_size=150,  # Намали от 200
        generations=400,      # Намали от 600
        elite_fraction=0.15,
        
        # === RESOLUTION - ЗАПОЧНИ ПО-НИСКО ===
        target_samples=300,  # Намали от 500 (по-стабилно)
        
        # === КРИВИНА - РЕАЛИСТИЧНА ===
        max_curvature=0.10,          # Реалистична за F1
        curvature_hardness=5.0,
        curvature_hard_penalty=50.0,
        curvature_soft_penalty=8.0,
        
        # === ALPHA ПРОМЕНИ - УМЕРЕНИ ===
        alpha_delta_max=0.12,
        alpha_rate_lambda=0.6,
        alpha_post_max_change=0.008,
        
        # === STRAIGHTNESS ===
        straightness_bonus=0.15,
        
        # === БАЛАНС ВРЕМЕ/PENALTIES ===
        time_weight=1200.0,     # Умерено високо
        penalty_weight=0.5,     # По-балансирано
        adaptive_time_weight=True,
        
        # === PENALTIES - БАЛАНСИРАНИ ===
        smoothness_lambda=0.0010,
        offset_smooth_lambda=0.008,
        curvature_lambda=0.0002,
        heading_spike_lambda=0.006,
        heading_max_delta_rad=0.45,
        steering_rate_lambda=0.20,
        
        # === ФИЗИКА ===
        long_accel_lambda=0.00012,
        long_jerk_lambda=0.00012,
        
        # === МУТАЦИЯ - УМЕРЕНА ===
        use_annealing=True,
        mutation_sigma_start=0.15,  # ПО-МАЛКА стартова мутация
        mutation_sigma_end=0.03,
        mutation_rate=0.25,
        
        # === DIVERSITY ===
        immigrants_fraction=0.15,   # УВЕЛИЧИ immigrants
        diversity_target=0.05,
        stagnation_patience=50,     # ПО-КРАТКО търпение
        
        # === CROSSOVER ===
        use_adaptive_crossover=True,
        crossover_rate=0.90,
        
        # === B-SPLINE - ИЗКЛЮЧЕН ===
        use_bspline=False,
        bspline_smoothing=0.0,
        
        # === SAMPLING ===
        curvature_sampling_beta=2.0,  # НАМАЛИ от 3.5
        
        # === COARSE-TO-FINE ===
        coarse_to_fine=True,
        coarse_factor=0.30,  # УВЕЛИЧИ coarse фазата (по-дълга)
        adaptive_penalty_decay=0.93,
        
        # === VEHICLE - РЕАЛИСТИЧЕН F1 ===
        mass_kg=798.0,
        mu_friction=2.2,            # Намали от 2.5
        cL_downforce=4.8,           # Намали от 5.5
        engine_power_watts=760000.0,
        a_brake_max=56.0,           # Намали от 65
        a_accel_cap=23.0,           # Намали от 28
        safety_speed_margin=0.98,   # УВЕЛИЧИ от 0.96
        brake_power_watts=1200000.0,
        
        gravity=9.81,
        rho_air=1.225,
        frontal_area_m2=1.6,
        cD_drag=1.0,
        c_rr=0.004,
        
        # === ADVANCED ===
        enable_parallel=True,
        parallel_workers=max(1, os.cpu_count() or 1),
        cache_granularity=0.002,
        use_dynamic_population=False,
    )

    print(f"Optimizing path for: {turn_csv}")
        # След зареждането на коридора:
    if use_corridor and inner_edge is not None and outer_edge is not None:
        print(f"[main] Original corridor: inner={inner_edge.shape[0]}, outer={outer_edge.shape[0]} points")
        
        # === ЗАТВОРИ КОРИДОРА ===
        inner_closed, outer_closed = close_corridor_segment(inner_edge, outer_edge, extension_factor=0.20)
        print(f"[main] Closed corridor: inner={inner_closed.shape[0]}, outer={outer_closed.shape[0]} points (added entries/exits)")
        
        # Оптимизирай затворения коридор
        _ret = run_ga_corridor(inner_closed, outer_closed, params)
        
        if isinstance(_ret, tuple) and len(_ret) == 4:
            best_alphas_result, best_path_full, best_fit, _fitness_history = _ret
        elif isinstance(_ret, tuple) and len(_ret) == 3:
            best_alphas_result, best_path_full, best_fit = _ret
        else:
            raise ValueError(f"Unexpected return: {len(_ret)} values")
        
        # === ИЗВАДИ САМО ОРИГИНАЛНИЯ СЕГМЕНТ (без удълженията) ===
        # Удължението е 20% от оригиналната дължина на всяка страна
        orig_len = inner_edge.shape[0]
        ext_len = int(orig_len * 0.20)
        
        # Извади централната част (оригиналния завой)
        start_idx = ext_len
        end_idx = len(best_path_full) - ext_len
        best_path = best_path_full[start_idx:end_idx]
        
        print(f"[main] Extracted original turn segment: {best_path.shape[0]} points")

    # Resample best path to the same number of points as the original segment for compact CSV
    try:
        best_compact = resample_to_match(points, best_path.tolist())
    except Exception:
        # Fallback: direct arclength resample
        best_compact = resample_polyline_arclength(best_path.tolist(), len(points))

    out_dir = os.path.dirname(turn_csv)
    out_path = os.path.join(out_dir, "Turn_1_best.csv")
    write_points_csv(out_path, best_compact.tolist())
    # След write_points_csv(out_path, best_compact.tolist())

    # Анализ на резултата
    print("\n=== RACE LINE ANALYSIS ===")
    try:
        # Проверка на кривина
        kappa_final = estimate_curvature_series(best_path)
        print(f"Max curvature: {np.max(np.abs(kappa_final)):.6f}")
        print(f"Mean curvature: {np.mean(np.abs(kappa_final)):.6f}")
        
        # Проверка на скорост
        vp = VehicleParams(
            mass_kg=params.mass_kg, mu_friction=params.mu_friction,
            cL_downforce=params.cL_downforce, engine_power_watts=params.engine_power_watts,
            # ... останалите параметри
        )
        s, _, _, v = speed_profile(best_path.tolist(), vp)
        print(f"Min speed: {np.min(v):.2f} m/s ({np.min(v)*3.6:.1f} km/h)")
        print(f"Max speed: {np.max(v):.2f} m/s ({np.max(v)*3.6:.1f} km/h)")
        print(f"Avg speed: {np.mean(v):.2f} m/s ({np.mean(v)*3.6:.1f} km/h)")
        
        # Намери апекса
        apex_idx = np.argmax(np.abs(kappa_final))
        print(f"Apex at point {apex_idx}/{len(best_path)} (speed: {v[apex_idx]*3.6:.1f} km/h)")
        
    except Exception as e:
        print(f"Analysis error: {e}")
    print(f"Wrote best path: {out_path}")

    # В края на main(), след write_points_csv:

    print("\n=== 🏁 AGGRESSIVE RACING LINE ANALYSIS ===")

    vp = VehicleParams(
        mass_kg=params.mass_kg, 
        mu_friction=params.mu_friction,
        cL_downforce=params.cL_downforce, 
        engine_power_watts=params.engine_power_watts,
        a_brake_max=params.a_brake_max, 
        a_accel_cap=params.a_accel_cap,
        cD_drag=params.cD_drag, 
        c_rr=params.c_rr,
        safety_speed_margin=params.safety_speed_margin,
        brake_power_watts=params.brake_power_watts,
        gravity=params.gravity,
        rho_air=params.rho_air,
        frontal_area_m2=params.frontal_area_m2,
    )

    s_agg, kappa_agg, v_lat, v_agg = speed_profile(best_path.tolist(), vp)
    t_agg = compute_lap_time_seconds(s_agg, v_agg)

    # Използвай best_alphas_result (от run_ga_corridor return)
    best_alphas_analysis = np.interp(
        np.linspace(0, 1, len(best_path)), 
        np.linspace(0, 1, len(best_alphas_result)),
        best_alphas_result
    )
    apex_idx = int(np.argmin(best_alphas_analysis))



    print(f"\n⚡ LAP TIME: {t_agg:.3f}s")
    print(f"\n🚀 SPEED ZONES:")
    print(f"  ENTRY speed: {v_agg[0]*3.6:.1f} km/h ← ЦЕЛИ 100+ km/h!")
    print(f"  APEX speed:  {v_agg[apex_idx]*3.6:.1f} km/h")
    print(f"  EXIT speed:  {v_agg[-1]*3.6:.1f} km/h")
    print(f"  MAX speed:   {np.max(v_agg)*3.6:.1f} km/h")
    print(f"  MIN speed:   {np.min(v_agg)*3.6:.1f} km/h")

    print(f"\n🎯 APEX:")
    print(f"  Position: {apex_idx}/{len(best_path)} ({apex_idx/len(best_path)*100:.1f}%)")
    print(f"  Alpha: {best_alphas_analysis[apex_idx]:.3f} (0=inner, 1=outer)")
    print(f"  Track position: {'INNER' if best_alphas_analysis[apex_idx] < 0.3 else 'MID' if best_alphas_analysis[apex_idx] < 0.7 else 'OUTER'}")

    print(f"\n📐 CURVATURE:")
    kappa_final = estimate_curvature_series(best_path)
    print(f"  Max κ: {np.max(np.abs(kappa_final)):.6f}")
    violations = np.sum(np.abs(kappa_final) > params.max_curvature)
    print(f"  Violations: {violations}/{len(kappa_final)} ({violations/len(kappa_final)*100:.1f}%)")

    # Провери дали входната скорост е близо до 100 km/h
    if v_agg[0]*3.6 < 80:
        print(f"\n⚠️  WARNING: Entry speed too low! ({v_agg[0]*3.6:.1f} km/h)")
        print(f"   Try increasing mu_friction or cL_downforce")


if __name__ == "__main__":
    main()
