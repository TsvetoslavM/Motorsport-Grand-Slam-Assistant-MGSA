import numpy as np
from typing import List, Tuple


def compute_adaptive_population_size(gen: int, max_gen: int, base_size: int, diversity: float, diversity_target: float, min_ratio: float = 0.5, max_ratio: float = 1.5) -> int:
    """Compute dynamic population size based on progress and diversity.

    Args:
        gen: current generation (0-indexed)
        max_gen: total generations
        base_size: initial population size
        diversity: current diversity (std of genes averaged)
        diversity_target: desired diversity target
        min_ratio: min size ratio vs base
        max_ratio: max size ratio vs base
    """
    min_size = max(int(base_size * min_ratio), 30)
    max_size = max(int(base_size * max_ratio), min_size)
    progress = gen / max(max_gen - 1, 1)
    current_size = int(base_size - (base_size - min_size) * progress)
    if diversity_target > 0.0 and diversity < diversity_target:
        diversity_ratio = diversity / max(diversity_target, 1e-6)
        size_multiplier = 1.0 + (1.0 - diversity_ratio) * 0.5
        current_size = int(current_size * size_multiplier)
    current_size = max(min_size, min(current_size, max_size))
    return current_size


def adjust_population_size(pop: List[np.ndarray], fitness: List[float], target_size: int, rng: np.random.Generator) -> Tuple[List[np.ndarray], List[float]]:
    """Resize population by trimming worst or adding random immigrants."""
    current_size = len(pop)
    if current_size == target_size:
        return pop, fitness
    if current_size > target_size:
        idx = np.argsort(fitness)[:target_size]
        new_pop = [pop[i] for i in idx]
        new_fit = [fitness[i] for i in idx]
        return new_pop, new_fit
    # Grow
    new_pop = pop.copy()
    new_fit = fitness.copy()
    n_genes = len(pop[0]) if pop else 0
    for _ in range(target_size - current_size):
        ind = rng.uniform(0.2, 0.8, size=n_genes).astype(float)
        new_pop.append(ind)
        new_fit.append(float('inf'))
    return new_pop, new_fit


