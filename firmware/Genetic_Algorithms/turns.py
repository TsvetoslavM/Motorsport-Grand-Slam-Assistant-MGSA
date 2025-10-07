import os
import csv
import glob
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from ..curvature import curvature_vectorized
from ..vmax_raceline.vmax import VehicleParams, speed_profile, compute_lap_time_seconds


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
    smoothness_lambda: float = 0.08
    offset_smooth_lambda: float = 0.6
    curvature_lambda: float = 0.02
    curvature_jerk_lambda: float = 0.01
    center_deviation_lambda: float = 0.00
    # Longitudinal dynamics realism (accel/brake and jerk)
    long_accel_lambda: float = 0.02
    long_jerk_lambda: float = 0.01
    # Single-resolution GA (no coarse/refine stages)
    # Smoothness hardening (limit kinks)
    alpha_delta_max: float = 0.15  # max change between consecutive alphas
    alpha_rate_lambda: float = 1.0
    heading_max_delta_rad: float = 0.35  # ~20 degrees per segment
    heading_spike_lambda: float = 0.5
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
    target_samples: int = 300
    # Adaptation and diversity controls
    use_annealing: bool = True
    mutation_sigma_start: float = 0.25
    mutation_sigma_end: float = 0.05
    immigrants_fraction: float = 0.10
    stagnation_patience: int = 60


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
    # Tangent and its rate of change
    t = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-9)
    dt = d2 / (np.linalg.norm(d1[1:], axis=1, keepdims=True) + 1e-9)
    # Curvature magnitude approximation via |dt/ds|
    kappa = np.zeros((pts.shape[0],), dtype=float)
    kappa[1:-1] = np.linalg.norm(dt, axis=1)
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


def build_path_from_corridor(inner: np.ndarray, outer: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Interpolate between inner and outer edges per point: P = (1-alpha)*inner + alpha*outer."""
    alphas = np.clip(alphas, 0.0, 1.0)
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


def compute_time_and_dynamics(path_pts: np.ndarray, params: GAParams) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (lap_time, s, v, a_long) where a_long is signed longitudinal accel along s.

    This lets us compute physically realistic penalties (over-accel/brake and jerk).
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
    s, _kappa, _v_lat, v = speed_profile(path_pts.tolist(), vp)
    lap_t = compute_lap_time_seconds(s, v)
    s_arr = np.asarray(s, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if s_arr.size >= 2:
        dvds = np.gradient(v_arr, s_arr, edge_order=2)
        a_long = v_arr * dvds
    else:
        a_long = np.zeros_like(v_arr)
    return lap_t, s_arr, v_arr, a_long


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


# Offset evaluation removed (corridor-only GA)


def evaluate_individual_corridor(inner: np.ndarray, outer: np.ndarray, alphas: np.ndarray, params: GAParams) -> float:
    alphas = np.clip(alphas, 0.0, 1.0)
    path_pts = build_path_from_corridor(inner, outer, alphas)
    t, s_arr, v_arr, a_long = compute_time_and_dynamics(path_pts, params)
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
    penalty += params.curvature_jerk_lambda * curvature_jerk_penalty(path_pts)
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
    return t + float(penalty)


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


def run_ga_offset(points: List[Point], params: GAParams) -> Tuple[np.ndarray, np.ndarray, float]:
    # Resample for stable optimization resolution
    t_start = time.perf_counter()
    pts_arr = resample_polyline_arclength(points, max(len(points), params.target_samples))
    print(f"[GA-offset] Resampled to {pts_arr.shape[0]} points (target_samples={params.target_samples})")
    pts_list = pts_arr.tolist()
    _, normals = compute_tangents_and_normals(pts_list)
    # Optional grouping and straights-only selection
    sel_idxs = select_indices_group_and_straights(np.asarray(pts_list, dtype=float), params.group_step, params.use_straights_only, params.curvature_threshold)
    base_pts = [pts_list[i] for i in sel_idxs.tolist()]
    base_normals = np.asarray(normals, dtype=float)[sel_idxs]
    n = len(base_pts)
    print(f"[GA-offset] Using {n}/{len(pts_list)} points (group_step={params.group_step}, straights_only={params.use_straights_only}, curv_thr={params.curvature_threshold})")
    print(f"[GA-offset] GA config: pop={params.population_size}, gen={params.generations}")
    pop = [np.random.uniform(-0.2, 0.2, size=n).astype(float) for _ in range(params.population_size)]
    fitness = [evaluate_individual_offset(base_pts, base_normals, ind, params) for ind in pop]

    num_elite = max(1, int(params.elite_fraction * params.population_size))

    best_offsets = pop[int(np.argmin(fitness))].copy()
    best_time = float(min(fitness))

    last_report = -1
    no_improve = 0
    for _gen in range(params.generations):
        # Elitism
        elite_idx = np.argsort(fitness)[:num_elite]
        new_pop: List[np.ndarray] = [pop[i].copy() for i in elite_idx]

        # Create rest via tournament selection, crossover, mutation
        while len(new_pop) < params.population_size:
            i1 = tournament_select(pop, fitness, params.tournament_k)
            i2 = tournament_select(pop, fitness, params.tournament_k)
            c1, c2 = crossover(pop[i1], pop[i2], params.crossover_rate)
            # Adaptive mutation sigma (annealing + stagnation boost)
            sigma_prog = (1.0 - (_gen / max(params.generations - 1, 1))) if params.use_annealing else 1.0
            sigma = params.mutation_sigma_end + (params.mutation_sigma_start - params.mutation_sigma_end) * sigma_prog
            if no_improve >= params.stagnation_patience // 2:
                sigma *= 1.5
            c1 = mutate(c1, params.mutation_rate, sigma, params.max_offset)
            if len(new_pop) < params.population_size:
                new_pop.append(c1)
            if len(new_pop) < params.population_size:
                c2 = mutate(c2, params.mutation_rate, sigma, params.max_offset)
                new_pop.append(c2)

        pop = new_pop
        fitness = [evaluate_individual_offset(base_pts, base_normals, ind, params) for ind in pop]

        gen_best_idx = int(np.argmin(fitness))
        gen_best = pop[gen_best_idx]
        gen_best_fit = float(fitness[gen_best_idx])
        if gen_best_fit < best_time - 1e-6:
            best_time = gen_best_fit
            best_offsets = gen_best.copy()
            no_improve = 0
        else:
            no_improve += 1

        # Random immigrants to maintain diversity
        if params.immigrants_fraction > 0.0:
            num_imm = int(params.immigrants_fraction * params.population_size)
            if num_imm > 0:
                worst_idx = np.argsort(fitness)[-num_imm:]
                for wi in worst_idx:
                    pop[wi] = np.random.uniform(-0.2, 0.2, size=n).astype(float)
                    fitness[wi] = evaluate_individual_offset(base_pts, base_normals, pop[wi], params)
        if _gen % 10 == 0 and _gen != last_report:
            last_report = _gen
            mean_fit = float(np.mean(fitness))
            std_fit = float(np.std(fitness))
            print(f"[GA-offset] gen={_gen:4d} best_time={best_time:.4f}  mean={mean_fit:.4f}  std={std_fit:.4f}  no_improve={no_improve}")

        # Restart non-elite on prolonged stagnation
        if no_improve >= params.stagnation_patience:
            base_pop_size = params.population_size - num_elite
            new_pop = [pop[i].copy() for i in elite_idx]
            new_pop.extend([np.random.uniform(-0.2, 0.2, size=n).astype(float) for _ in range(max(0, base_pop_size))])
            pop = new_pop[:params.population_size]
            fitness = [evaluate_individual_offset(base_pts, base_normals, ind, params) for ind in pop]
            no_improve = 0
            print("[GA-offset] Restart triggered (stagnation)")

    # Interpolate offsets back to full resolution along the selected index parameterization
    sel_s = np.linspace(0.0, 1.0, n)
    full_s = np.linspace(0.0, 1.0, len(pts_list))
    best_offsets_full = np.interp(full_s, sel_s, best_offsets)
    # Smooth offsets for realistic steering continuity and rebuild path
    best_offsets_full = smooth_series_moving_average(best_offsets_full, repeats=1)
    best_path = build_offset_path(pts_list, normals, best_offsets_full)
    dt = time.perf_counter() - t_start
    print(f"[GA-offset] Done. best_time={best_time:.4f}  duration={dt:.2f}s")
    return best_offsets, best_path, best_time


def run_ga_corridor(inner: np.ndarray, outer: np.ndarray, params: GAParams) -> Tuple[np.ndarray, np.ndarray, float]:
    # Resample corridor to target resolution for robust evaluation
    t_start = time.perf_counter()
    inner = resample_polyline_arclength(inner.tolist(), max(inner.shape[0], params.target_samples))
    outer = resample_polyline_arclength(outer.tolist(), max(outer.shape[0], params.target_samples))
    print(f"[GA-corridor] Resampled inner/outer to {inner.shape[0]} points (target_samples={params.target_samples})")
    # Single-resolution: evaluate on full resolution
    midline = 0.5 * (inner + outer)
    inner_sel = inner
    outer_sel = outer
    n = inner_sel.shape[0]
    print(f"[GA-corridor] Using full resolution: {n} points")
    print(f"[GA-corridor] GA config: pop={params.population_size}, gen={params.generations}")
    pop = [np.clip(0.5 + np.random.normal(0.0, 0.08, size=n).astype(float), 0.0, 1.0) for _ in range(params.population_size)]
    fitness = [evaluate_individual_corridor(inner_sel, outer_sel, ind, params) for ind in pop]

    num_elite = max(1, int(params.elite_fraction * params.population_size))
    best_alphas = pop[int(np.argmin(fitness))].copy()
    best_time = float(min(fitness))

    last_report = -1
    no_improve = 0
    for _gen in range(params.generations):
        elite_idx = np.argsort(fitness)[:num_elite]
        new_pop: List[np.ndarray] = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < params.population_size:
            i1 = tournament_select(pop, fitness, params.tournament_k)
            i2 = tournament_select(pop, fitness, params.tournament_k)
            c1, c2 = crossover(pop[i1], pop[i2], params.crossover_rate)
            sigma_prog = (1.0 - (_gen / max(params.generations - 1, 1))) if params.use_annealing else 1.0
            sigma = params.mutation_sigma_end + (params.mutation_sigma_start - params.mutation_sigma_end) * sigma_prog
            if no_improve >= params.stagnation_patience // 2:
                sigma *= 1.5
            c1 = mutate(c1, params.mutation_rate, sigma, 1.0)
            if len(new_pop) < params.population_size:
                new_pop.append(np.clip(c1, 0.0, 1.0))
            if len(new_pop) < params.population_size:
                c2 = mutate(c2, params.mutation_rate, sigma, 1.0)
                new_pop.append(np.clip(c2, 0.0, 1.0))
        pop = new_pop
        fitness = [evaluate_individual_corridor(inner_sel, outer_sel, ind, params) for ind in pop]
        gen_best_idx = int(np.argmin(fitness))
        gen_best = pop[gen_best_idx]
        gen_best_fit = float(fitness[gen_best_idx])
        if gen_best_fit < best_time - 1e-6:
            best_time = gen_best_fit
            best_alphas = gen_best.copy()
            no_improve = 0
        else:
            no_improve += 1

        # Random immigrants
        if params.immigrants_fraction > 0.0:
            num_imm = int(params.immigrants_fraction * params.population_size)
            if num_imm > 0:
                worst_idx = np.argsort(fitness)[-num_imm:]
                for wi in worst_idx:
                    pop[wi] = np.clip(np.random.uniform(0.2, 0.8, size=n).astype(float), 0.0, 1.0)
                    fitness[wi] = evaluate_individual_corridor(inner_sel, outer_sel, pop[wi], params)
        if _gen % 10 == 0 and _gen != last_report:
            last_report = _gen
            mean_fit = float(np.mean(fitness))
            std_fit = float(np.std(fitness))
            print(f"[GA-corridor] gen={_gen:4d} best_time={best_time:.4f}  mean={mean_fit:.4f}  std={std_fit:.4f}  no_improve={no_improve}")

        if no_improve >= params.stagnation_patience:
            base_pop_size = params.population_size - num_elite
            new_pop = [pop[i].copy() for i in elite_idx]
            new_pop.extend([np.clip(np.random.uniform(0.2, 0.8, size=n).astype(float), 0.0, 1.0) for _ in range(max(0, base_pop_size))])
            pop = new_pop[:params.population_size]
            fitness = [evaluate_individual_corridor(inner_sel, outer_sel, ind, params) for ind in pop]
            no_improve = 0
            print("[GA-corridor] Restart triggered (stagnation)")

    # Interpolate alphas back to full resolution using arc-length along selection
    sel_s = np.linspace(0.0, 1.0, n)
    full_s = np.linspace(0.0, 1.0, midline.shape[0])
    best_alphas_full = np.clip(np.interp(full_s, sel_s, best_alphas), 0.0, 1.0)
    # Smooth alphas (stays within corridor after clamping) for realistic curvature continuity
    best_alphas_full = np.clip(smooth_series_moving_average(best_alphas_full, repeats=1), 0.0, 1.0)
    best_alphas_full = np.clip(limit_series_rate(best_alphas_full, params.alpha_delta_max), 0.0, 1.0)
    best_path = build_path_from_corridor(inner, outer, best_alphas_full)
    dt = time.perf_counter() - t_start
    print(f"[GA-corridor] Done. best_time={best_time:.4f}  duration={dt:.2f}s")
    return best_alphas, best_path, best_time


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GA optimizer for Turn_1.csv")
    parser.add_argument("--turns-dir", type=str, default=None, help="Base 'Turns' directory (defaults to firmware/Turns)")
    # Offset mode removed; no max-offset
    parser.add_argument("--pop", type=int, default=160, help="Population size")
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
        population_size=args.pop,
        generations=args.gen,
        mass_kg=base_mass,
        mu_friction=base_mu,
        gravity=base_g,
        rho_air=base_rho,
        cL_downforce=base_cL,
        frontal_area_m2=base_area,
        engine_power_watts=base_power,
        a_brake_max=base_abrake,
        a_accel_cap=base_aaccel,
        cD_drag=base_cD,
        c_rr=base_crr,
        safety_speed_margin=base_safety,
        brake_power_watts=base_bpw,
        target_samples=max(len(points), 300),
    )

    print(f"Optimizing path for: {turn_csv}")
    if use_corridor:
        print("[main] Mode: corridor (alphas)")
    else:
        print("[main] Mode: offset from centerline")
    if use_corridor and inner_edge is not None and outer_edge is not None:
        _, best_path, best_fit = run_ga_corridor(inner_edge, outer_edge, params)
    else:
        _, best_path, best_fit = run_ga_offset(points, params)

    # Resample best path to the same number of points as the original segment for compact CSV
    try:
        best_compact = resample_to_match(points, best_path.tolist())
    except Exception:
        # Fallback: direct arclength resample
        best_compact = resample_polyline_arclength(best_path.tolist(), len(points))

    out_dir = os.path.dirname(turn_csv)
    out_path = os.path.join(out_dir, "Turn_1_best.csv")
    write_points_csv(out_path, best_compact.tolist())
    print(f"Wrote best path: {out_path}")


if __name__ == "__main__":
    main()
