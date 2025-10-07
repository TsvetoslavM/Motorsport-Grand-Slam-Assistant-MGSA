import math
import numpy as np
import pytest

from firmware.Genetic_Algorithms.fitness import curvature_limit_penalty
from firmware.Genetic_Algorithms.ga_operators import adaptive_crossover_weighted
from firmware.Genetic_Algorithms.geometry import build_path_from_corridor_bspline
from firmware.Genetic_Algorithms.turns import GAParams, evaluate_individual_corridor


def test_curvature_limit_penalty_no_overflow():
    # Construct a highly curved polyline (zigzag) to trigger large violations
    pts = []
    for i in range(50):
        x = i * 0.1
        y = (1 if i % 2 == 0 else -1) * 0.5
        pts.append((x, y))
    pts_arr = np.asarray(pts, dtype=float)
    pen = curvature_limit_penalty(pts_arr, max_kappa=0.02, hardness=30.0)
    assert math.isfinite(pen)
    assert pen > 0.0


def test_adaptive_crossover_weighted_bias():
    rng = np.random.default_rng(0)
    N, L = 16, 20
    parents_a = rng.uniform(0.2, 0.4, size=(N, L))
    parents_b = rng.uniform(0.6, 0.8, size=(N, L))
    # Make A much better (lower fitness)
    fitness_a = np.full((N,), 10.0)
    fitness_b = np.full((N,), 1000.0)
    offspring = adaptive_crossover_weighted(parents_a, parents_b, fitness_a, fitness_b, crossover_rate=1.0, rng=rng)
    # Offspring should be closer to A on average
    dist_a = float(np.mean(np.abs(offspring - parents_a)))
    dist_b = float(np.mean(np.abs(offspring - parents_b)))
    assert dist_a < dist_b


@pytest.mark.skipif(pytest.importorskip("scipy", reason="scipy required for bspline").__class__ is None, reason="scipy missing")
def test_build_path_from_corridor_bspline_shapes():
    N = 60
    s = np.linspace(0.0, 1.0, N)
    inner = np.stack([s, np.zeros_like(s)], axis=1)
    outer = np.stack([s, np.ones_like(s)], axis=1)
    alphas = np.linspace(0.2, 0.8, N)
    path = build_path_from_corridor_bspline(inner, outer, alphas, smoothing=0.0, degree=3)
    assert path.shape == (N, 2)
    assert np.isfinite(path).all()


def test_weighted_fitness_changes_with_weights():
    # Simple straight corridor where penalties are small and comparable across runs
    N = 40
    s = np.linspace(0.0, 1.0, N)
    inner = np.stack([s, np.zeros_like(s)], axis=1)
    outer = np.stack([s, np.ones_like(s)], axis=1)
    alphas = np.full((N,), 0.5)
    base = GAParams(use_bspline=False)
    # Evaluate with low time weight
    p1 = base
    p1 = p1.__class__(**{**p1.__dict__, "current_time_weight": 1.0, "current_penalty_weight": 1.0})
    f1 = evaluate_individual_corridor(inner, outer, alphas, p1)
    # Evaluate with high time weight
    p2 = base
    p2 = p2.__class__(**{**p2.__dict__, "current_time_weight": 100.0, "current_penalty_weight": 1.0})
    f2 = evaluate_individual_corridor(inner, outer, alphas, p2)
    # Since time dominates more in f2, it should be >= f1 (same path)
    assert f2 >= f1


