import numpy as np
from typing import List, Tuple
from .geometry import estimate_curvature_series


def offsets_smoothness_penalty(offsets: np.ndarray) -> float:
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


def curvature_limit_penalty(path_pts: np.ndarray, max_kappa: float = 0.02, hardness: float = 30.0) -> float:
    """Exponential penalty for curvature above a threshold with overflow protection.

    Clips violations so that hardness*violation <= 20 to prevent exp overflow.
    """
    kappa = estimate_curvature_series(path_pts)
    kappa_abs = np.abs(kappa)
    violations = np.maximum(kappa_abs - max_kappa, 0.0)
    # Clip to keep exp argument bounded: exp(20) ~ 4.85e8
    h = float(max(hardness, 1e-6))
    max_violation = 20.0 / h
    violations = np.clip(violations, 0.0, max_violation)
    penalty_per_point = np.exp(h * violations) - 1.0
    return float(np.sum(penalty_per_point))


def straightness_reward(path_pts: np.ndarray, target_kappa: float = 0.02) -> float:
    """Reward (negative penalty) proportional to share of near-straight points."""
    kappa = estimate_curvature_series(path_pts)
    kappa_abs = np.abs(kappa)
    straight_mask = kappa_abs <= target_kappa
    straight_ratio = float(np.mean(straight_mask))
    bonus = straight_ratio * 0.3
    return -bonus


def force_straight_alphas(alphas: np.ndarray, max_change: float = 0.005) -> np.ndarray:
    """Post-process alphas to limit per-step change (favor straight segments)."""
    corrected = alphas.copy()
    for i in range(1, len(alphas)):
        delta = corrected[i] - corrected[i - 1]
        if abs(delta) > max_change:
            corrected[i] = corrected[i - 1] + np.sign(delta) * max_change
    for i in range(len(alphas) - 2, -1, -1):
        delta = corrected[i] - corrected[i + 1]
        if abs(delta) > max_change:
            corrected[i] = corrected[i + 1] + np.sign(delta) * max_change
    return np.clip(corrected, 0.0, 1.0)


def count_curvature_violations(path_pts: np.ndarray, max_kappa: float = 0.02) -> Tuple[int, float]:
    kappa = estimate_curvature_series(path_pts)
    kappa_abs = np.abs(kappa)
    violations = int(np.sum(kappa_abs > max_kappa))
    max_kappa_val = float(np.max(kappa_abs)) if kappa_abs.size > 0 else 0.0
    return violations, max_kappa_val


