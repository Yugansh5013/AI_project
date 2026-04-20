"""
Advanced Pufferfish Optimization Algorithm (A-POA) & Base POA
=============================================================
OPTIMIZED: Tidal Pressure distance matrix computed once per iteration.
All algorithms share the interface:
    run_X(obj_func, dim, max_fes, lb, ub, pop_size) -> (best_pos, best_fit, history)
"""
import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, List, Callable

# ─── Helper Functions (A-POA improvements) ────────────────────

def _get_cppi_target(positions: np.ndarray, fitness: np.ndarray, K: int = 3) -> np.ndarray:
    K = min(K, len(fitness))
    sorted_indices = np.argsort(fitness)
    top_k = sorted_indices[:K]
    elite_pos = positions[top_k]
    elite_fit = fitness[top_k]
    weights = 1.0 / (elite_fit + 1e-8)
    return np.average(elite_pos, axis=0, weights=weights)


def _risu_update(current_pos: np.ndarray, target_pos: np.ndarray, dim: int) -> np.ndarray:
    I = np.random.choice([1, 2])
    direction = target_pos - I * current_pos
    r = np.random.uniform(0.0, 1.0)
    noise = np.random.normal(0.0, 0.01, size=dim)
    return current_pos + r * direction + noise


def _compute_density_factors(positions: np.ndarray) -> np.ndarray:
    """Compute density factors for ALL agents at once (vectorized).
    Returns array of shape (pop_size,) with values in [0, 1].
    1 = crowded, 0 = isolated.
    """
    pop_size = positions.shape[0]
    dist_matrix = cdist(positions, positions, metric="euclidean")
    np.fill_diagonal(dist_matrix, np.nan)
    mean_dists = np.nanmean(dist_matrix, axis=1)
    d_min, d_max = np.min(mean_dists), np.max(mean_dists)
    denom = d_max - d_min + 1e-8
    df = 1.0 - (mean_dists - d_min) / denom
    return np.clip(df, 0.0, 1.0)


def _spine_perturbation(
    position: np.ndarray, fit: float,
    obj_func: Callable, lb: float, ub: float, eps: float = 1e-4
) -> Tuple[np.ndarray, float, int]:
    """Returns (new_pos, new_fit, fes_used)."""
    best_pos, best_fit = position.copy(), fit
    fes = 0
    for j in range(len(position)):
        trial = best_pos.copy()
        trial[j] += eps
        trial = np.clip(trial, lb, ub)
        trial_fit = obj_func(trial)
        fes += 1
        if trial_fit < best_fit:
            return trial, trial_fit, fes
        trial = best_pos.copy()
        trial[j] -= eps
        trial = np.clip(trial, lb, ub)
        trial_fit = obj_func(trial)
        fes += 1
        if trial_fit < best_fit:
            return trial, trial_fit, fes
    return best_pos, best_fit, fes


def _chaotic_drift(dim: int, lb: float, ub: float) -> np.ndarray:
    z = np.random.uniform(0.01, 0.99)
    vec = np.zeros(dim)
    for j in range(dim):
        z = 4.0 * z * (1.0 - z)
        vec[j] = z
    return lb + vec * (ub - lb)


# ─── A-POA (with all 5 improvements) ────────────────────────

def run_apoa(
    obj_func: Callable[[np.ndarray], float],
    dim: int,
    max_fes: int = 60000,
    lb: float = -100.0,
    ub: float = 100.0,
    pop_size: int = 30,
    K: int = 3,
) -> Tuple[np.ndarray, float, List[float]]:
    """Run A-POA with FE-based budget."""
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    buoyancy = np.zeros(pop_size)
    fes = 0
    history: List[float] = []

    # Initial evaluation
    for i in range(pop_size):
        fitness[i] = obj_func(positions[i])
        fes += 1
    g_best_idx = np.argmin(fitness)
    g_best_pos = positions[g_best_idx].copy()
    g_best_fit = fitness[g_best_idx]
    history.append(g_best_fit)

    while fes < max_fes:
        cppi_target = _get_cppi_target(positions, fitness, K)
        t_ratio = fes / max_fes
        base_step = (1.0 - t_ratio) * (ub - lb) * 0.1

        # Compute density factors ONCE per iteration (optimization)
        density_factors = _compute_density_factors(positions)

        for i in range(pop_size):
            if fes >= max_fes:
                break
            old_fit = fitness[i]

            # Chaotic drift for deeply stuck agents
            if buoyancy[i] < -3:
                positions[i] = _chaotic_drift(dim, lb, ub)
                fitness[i] = obj_func(positions[i])
                fes += 1
                buoyancy[i] = 0.0
                if fitness[i] < g_best_fit:
                    g_best_fit = fitness[i]
                    g_best_pos = positions[i].copy()
                continue

            # Phase 1 — Exploration via RISU
            if buoyancy[i] <= 0:
                new_pos = _risu_update(positions[i], cppi_target, dim)
                new_pos = np.clip(new_pos, lb, ub)
                new_fit = obj_func(new_pos)
                fes += 1
                if new_fit < fitness[i]:
                    positions[i] = new_pos
                    fitness[i] = new_fit

            if fes >= max_fes:
                break

            # Phase 2 — Exploitation with Tidal Pressure (uses cached density)
            adaptive_step = base_step * (1.0 + 0.5 * density_factors[i])
            perturbation = np.random.uniform(-1, 1, dim) * adaptive_step
            trial_pos = np.clip(positions[i] + perturbation, lb, ub)
            trial_fit = obj_func(trial_pos)
            fes += 1
            if trial_fit < fitness[i]:
                positions[i] = trial_pos
                fitness[i] = trial_fit

            # Buoyancy update + Spine Perturbation
            if fitness[i] < old_fit:
                buoyancy[i] += 1
                if fes < max_fes:
                    sp_pos, sp_fit, sp_fes = _spine_perturbation(
                        positions[i], fitness[i], obj_func, lb, ub
                    )
                    fes += sp_fes
                    if sp_fit < fitness[i]:
                        positions[i] = sp_pos
                        fitness[i] = sp_fit
            else:
                buoyancy[i] -= 1

            # Update global best
            if fitness[i] < g_best_fit:
                g_best_fit = fitness[i]
                g_best_pos = positions[i].copy()

        history.append(g_best_fit)

    return g_best_pos, g_best_fit, history


# ─── Base POA (NO improvements — vanilla) ────────────────────

def run_base_poa(
    obj_func: Callable[[np.ndarray], float],
    dim: int,
    max_fes: int = 60000,
    lb: float = -100.0,
    ub: float = 100.0,
    pop_size: int = 30,
) -> Tuple[np.ndarray, float, List[float]]:
    """Run vanilla POA (no CPPI, no RISU, no Tidal Pressure, no Spine, no Drift).
    Uses single-elite target, per-dimension random updates, fixed step schedule.
    """
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    fes = 0
    history: List[float] = []

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i])
        fes += 1
    g_best_idx = np.argmin(fitness)
    g_best_pos = positions[g_best_idx].copy()
    g_best_fit = fitness[g_best_idx]
    history.append(g_best_fit)

    while fes < max_fes:
        t_ratio = fes / max_fes
        base_step = (1.0 - t_ratio) * (ub - lb) * 0.1

        for i in range(pop_size):
            if fes >= max_fes:
                break

            # Phase 1 — Exploration toward single best (axis-aligned)
            if np.random.rand() < 0.5:
                r = np.random.uniform(0, 1, dim)
                I = np.random.choice([1, 2])
                new_pos = positions[i] + r * (g_best_pos - I * positions[i])
                new_pos = np.clip(new_pos, lb, ub)
                new_fit = obj_func(new_pos)
                fes += 1
                if new_fit < fitness[i]:
                    positions[i] = new_pos
                    fitness[i] = new_fit

            if fes >= max_fes:
                break

            # Phase 2 — Fixed-step exploitation
            perturbation = np.random.uniform(-1, 1, dim) * base_step
            trial_pos = np.clip(positions[i] + perturbation, lb, ub)
            trial_fit = obj_func(trial_pos)
            fes += 1
            if trial_fit < fitness[i]:
                positions[i] = trial_pos
                fitness[i] = trial_fit

            if fitness[i] < g_best_fit:
                g_best_fit = fitness[i]
                g_best_pos = positions[i].copy()

        history.append(g_best_fit)

    return g_best_pos, g_best_fit, history
