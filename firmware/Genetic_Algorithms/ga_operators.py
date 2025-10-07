import numpy as np
from typing import Tuple


def vectorized_tournament_select(fitness: np.ndarray, k: int, num: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices of selected parents using k-tournament, vectorized for speed.

    Args:
        fitness: shape (P,)
        k: tournament size
        num: number of selections to draw
        rng: numpy Generator
    Returns:
        indices: shape (num,), dtype=int
    """
    P = fitness.shape[0]
    candidates = rng.integers(0, P, size=(num, k), dtype=np.int64)
    cand_fit = fitness[candidates]
    winners = candidates[np.arange(num), np.argmin(cand_fit, axis=1)]
    return winners.astype(int)


def uniform_crossover(parents_a: np.ndarray, parents_b: np.ndarray, crossover_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover per gene with probability 0.5, applied with crossover_rate per offspring.

    parents_a, parents_b: shape (N, L)
    Returns offspring: shape (N, L)
    """
    N, L = parents_a.shape
    do_cross = rng.random(N) < crossover_rate
    mask = rng.random((N, L)) < 0.5
    offspring = parents_a.copy()
    # Only apply mask for those rows that crossover
    if np.any(do_cross):
        idx = np.where(do_cross)[0]
        m = mask[idx]
        a = parents_a[idx]
        b = parents_b[idx]
        offspring[idx] = np.where(m, a, b)
    return offspring


def gaussian_mutation(offspring: np.ndarray, per_gene_sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise with per-gene sigma, then clip to [0,1].

    offspring: shape (N, L)
    per_gene_sigma: shape (L,)
    """
    noise = rng.normal(0.0, per_gene_sigma, size=offspring.shape)
    mutated = offspring + noise
    return np.clip(mutated, 0.0, 1.0)


def adaptive_crossover_weighted(parents_a: np.ndarray, parents_b: np.ndarray,
                                fitness_a: np.ndarray, fitness_b: np.ndarray,
                                crossover_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Fitness-weighted crossover: better parent contributes more genes (lower fitness = better).

    Args:
        parents_a, parents_b: shape (N, L)
        fitness_a, fitness_b: shape (N,)
        crossover_rate: probability to perform crossover per offspring
        rng: numpy Generator
    Returns:
        offspring: shape (N, L)
    """
    N, L = parents_a.shape
    offspring = parents_a.copy()
    do_cross = rng.random(N) < crossover_rate

    # Convert fitness to probabilities: pA = wA/(wA+wB), w = 1/(f+eps)
    eps = 1e-9
    w_a = 1.0 / (fitness_a + eps)
    w_b = 1.0 / (fitness_b + eps)
    prob_a = w_a / (w_a + w_b)

    if np.any(do_cross):
        idx = np.where(do_cross)[0]
        pa = prob_a[idx][:, None]  # (M,1)
        mask = rng.random((idx.size, L)) < pa
        a = parents_a[idx]
        b = parents_b[idx]
        offspring[idx] = np.where(mask, a, b)

    # For pairs without crossover, copy better parent
    if np.any(~do_cross):
        idxn = np.where(~do_cross)[0]
        choose_a = fitness_a[idxn] <= fitness_b[idxn]
        offspring[idxn] = np.where(choose_a[:, None], parents_a[idxn], parents_b[idxn])

    return offspring


