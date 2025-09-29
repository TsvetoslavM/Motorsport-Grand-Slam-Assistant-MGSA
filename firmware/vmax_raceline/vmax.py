import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ..curvature import curvature_vectorized


Point = Tuple[float, float]


@dataclass
class VehicleParams:
    mass_kg: float = 798.0
    mu_friction: float = 2.0
    gravity: float = 9.81
    rho_air: float = 1.225
    cL_downforce: float = 4.0
    frontal_area_m2: float = 1.6
    engine_power_watts: float = 735000.0
    a_brake_max: float = 54.0  # m/s^2, hard cap for braking decel (system limit)
    a_accel_cap: float = 20.0  # m/s^2, optional hard cap for longitudinal accel
    cD_drag: float = 1.2       # aerodynamic drag coefficient
    c_rr: float = 0.004        # rolling resistance coefficient (F1-like)
    safety_speed_margin: float = 1.00  # multiplier for global power cap (≤ 1.0)
    brake_power_watts: float | None = None  # optional power dissipation limit for braking

    def k_aero(self) -> float:
        # k_aero = (0.5 * rho * C_L * A) / m
        return (0.5 * self.rho_air * self.cL_downforce * self.frontal_area_m2) / max(self.mass_kg, 1e-9)

    def k_drag(self) -> float:
        # k_drag = 0.5 * rho * C_D * A
        return 0.5 * self.rho_air * self.cD_drag * self.frontal_area_m2


def compute_arc_length(points: Sequence[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """Return cumulative arc length s[i] and segment lengths ds[i]=s[i+1]-s[i]."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        s = np.zeros(len(pts))
        return s, np.zeros(0)
    diffs = pts[1:] - pts[:-1]
    ds = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, ds


def compute_curvature(points: Sequence[Point]) -> np.ndarray:
    return curvature_vectorized(points)


def vmax_lateral(curvature: np.ndarray, params: VehicleParams, v_global_cap: float | None = None) -> np.ndarray:
    """Lateral-friction-limited speed using v^2*|kappa| <= mu*(g + k_aero*v^2).

    Analytic form when (|kappa| - mu*k_aero) > 0: v = sqrt(mu*g/(|kappa| - mu*k_aero)).
    Otherwise (straights/very small curvature) -> +inf; optional clip to v_global_cap.
    """
    mu = params.mu_friction
    g = params.gravity
    k_a = params.k_aero()
    kap = np.abs(np.asarray(curvature, dtype=float))
    denom = kap - mu * k_a
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sqrt(np.maximum(mu * g, 0.0) / np.maximum(denom, 0.0))
    v[denom <= 0.0] = np.inf
    if v_global_cap is not None and np.isfinite(v_global_cap):
        v = np.minimum(v, v_global_cap)
    return v


def speed_profile(points: Sequence[Point],
                  params: VehicleParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute physically feasible speed profile along the trajectory.

    Returns (s, kappa, v_lat, v_profile)
    """
    s, ds = compute_arc_length(points)
    kappa = compute_curvature(points)
    kap = np.abs(kappa)

    # Power-based global speed cap (approx P = k_drag v^3)
    k_drag = params.k_drag()
    P = max(params.engine_power_watts, 0.0)
    v_power_limit = (P / max(k_drag, 1e-12)) ** (1.0 / 3.0)
    margin = min(max(params.safety_speed_margin, 0.0), 1.0)
    v_global_cap = max(v_power_limit * margin, 1.0)

    v_lat = vmax_lateral(kappa, params, v_global_cap=v_global_cap)

    n = len(s)
    if n == 0:
        return s, kappa, v_lat, np.array([])

    # Internal copy for constraints: inf (no lateral cap) -> use global cap
    v = np.where(np.isfinite(v_lat), v_lat, v_global_cap).astype(float)

    # Iterative forward/backward until convergence
    m = params.mass_kg
    mu = params.mu_friction
    g = params.gravity
    k_a = params.k_aero()
    c_rr = max(params.c_rr, 0.0)

    max_iter = 15
    tol = 1e-3
    for _ in range(max_iter):
        v_prev = v.copy()

        # Backward pass (speed-dependent braking limit)
        for i in range(n - 2, -1, -1):
            seg = max(ds[i], 1e-9)
            v_next = max(v[i + 1], 0.0)
            v_ref = v_next
            g_eff = g + k_a * v_ref * v_ref
            a_total_brake = mu * g_eff
            a_lat = (v_ref * v_ref) * kap[i]
            a_long_brake_max = math.sqrt(max(a_total_brake * a_total_brake - a_lat * a_lat, 0.0))
            if params.a_brake_max is not None:
                a_long_brake_max = min(a_long_brake_max, max(params.a_brake_max, 0.0))
            if params.brake_power_watts is not None and params.brake_power_watts > 0.0:
                a_brake_power = params.brake_power_watts / (m * max(v_ref, 1e-3))
                a_long_brake_max = min(a_long_brake_max, a_brake_power)
            vmax_prev = math.sqrt(max(v_next * v_next + 2.0 * a_long_brake_max * seg, 0.0))
            v[i] = min(v[i], vmax_prev, v_global_cap)

        # Forward pass (traction circle + power limit)
        for i in range(0, n - 1):
            seg = max(ds[i], 1e-9)
            vi = max(v[i], 0.0)

            g_eff = g + k_a * vi * vi
            a_total_max = mu * g_eff
            a_lat = vi * vi * kap[i]
            a_long_friction = math.sqrt(max(a_total_max * a_total_max - a_lat * a_lat, 0.0))

            P_drag = k_drag * (vi ** 3)
            P_roll = c_rr * m * g * vi
            P_avail = max(P - P_drag - P_roll, 0.0)
            a_power = P_avail / (m * max(vi, 1e-3))

            a_cap = params.a_accel_cap
            a_long = min(a_long_friction, a_power, a_cap)

            v_next_max = math.sqrt(max(vi * vi + 2.0 * a_long * seg, 0.0))
            v[i + 1] = min(v[i + 1], v_next_max, v_global_cap)

        # Cyclic consistency (start similar to end)
        v[0] = min(v[0], v[-1], v_global_cap)

        # Check convergence
        if float(np.max(np.abs(v - v_prev))) < tol:
            break

    return s, kappa, v_lat, v


def export_csv(file_path: str,
               points: Sequence[Point],
               s: np.ndarray,
               kappa: np.ndarray,
               v: np.ndarray) -> None:
    import csv
    pts = np.asarray(points, dtype=float)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["s_m", "x", "y", "kappa_1_per_m", "v_m_per_s"])
        for i in range(len(pts)):
            x = float(pts[i, 0])
            y = float(pts[i, 1])
            si = float(s[i]) if i < len(s) else float("nan")
            ki = float(kappa[i]) if i < len(kappa) else float("nan")
            vi = float(v[i]) if i < len(v) else float("nan")
            writer.writerow([si, x, y, ki, vi])


def plot_speed_vs_s(s: np.ndarray, v: np.ndarray, title: str = "Speed vs Distance"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s, v, label="v(s)")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("v [m/s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return plt, fig, ax


def compute_lap_time_seconds(s: np.ndarray, v: np.ndarray, v_epsilon: float = 0.1) -> float:
    """Compute lap time by integrating dt = ds / v using trapezoidal speeds.

    - s: cumulative arc length [m]
    - v: feasible speed profile [m/s]
    - v_epsilon: lower bound to avoid division by zero
    """
    if s.size <= 1 or v.size <= 1:
        return 0.0
    ds = np.diff(s)
    v0 = np.maximum(v[:-1], v_epsilon)
    v1 = np.maximum(v[1:], v_epsilon)
    v_avg = 0.5 * (v0 + v1)
    return float(np.sum(ds / np.maximum(v_avg, v_epsilon)))


