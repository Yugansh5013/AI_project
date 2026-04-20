"""
Comparison Algorithms for CEC Benchmark
========================================
7 metaheuristic optimizers, all sharing the same interface:
    run_X(obj_func, dim, max_fes, lb, ub, pop_size) -> (best_pos, best_fit, history)
"""
import numpy as np
from typing import Tuple, List, Callable


# ═══════════════════════════════════════════════════════════════
# 1. PSO — Particle Swarm Optimization (Kennedy & Eberhart 1995)
# ═══════════════════════════════════════════════════════════════

def run_pso(
    obj_func: Callable, dim: int, max_fes: int = 60000,
    lb: float = -100.0, ub: float = 100.0, pop_size: int = 30,
    w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
) -> Tuple[np.ndarray, float, List[float]]:
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    velocities = np.random.uniform(-(ub-lb)*0.1, (ub-lb)*0.1, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    pbest_pos = positions.copy()
    pbest_fit = np.full(pop_size, np.inf)
    fes = 0
    history = []

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i]); fes += 1
        pbest_fit[i] = fitness[i]; pbest_pos[i] = positions[i].copy()
    g_idx = np.argmin(pbest_fit)
    g_best_pos, g_best_fit = pbest_pos[g_idx].copy(), pbest_fit[g_idx]
    history.append(g_best_fit)

    while fes < max_fes:
        for i in range(pop_size):
            if fes >= max_fes: break
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (pbest_pos[i] - positions[i])
                             + c2 * r2 * (g_best_pos - positions[i]))
            positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
            fitness[i] = obj_func(positions[i]); fes += 1
            if fitness[i] < pbest_fit[i]:
                pbest_fit[i] = fitness[i]; pbest_pos[i] = positions[i].copy()
                if fitness[i] < g_best_fit:
                    g_best_fit = fitness[i]; g_best_pos = positions[i].copy()
        history.append(g_best_fit)
    return g_best_pos, g_best_fit, history


# ═══════════════════════════════════════════════════════════════
# 2. GWO — Grey Wolf Optimizer (Mirjalili et al. 2014)
# ═══════════════════════════════════════════════════════════════

def run_gwo(
    obj_func: Callable, dim: int, max_fes: int = 60000,
    lb: float = -100.0, ub: float = 100.0, pop_size: int = 30,
) -> Tuple[np.ndarray, float, List[float]]:
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    fes = 0; history = []

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i]); fes += 1
    sorted_idx = np.argsort(fitness)
    alpha_pos, alpha_fit = positions[sorted_idx[0]].copy(), fitness[sorted_idx[0]]
    beta_pos, beta_fit = positions[sorted_idx[1]].copy(), fitness[sorted_idx[1]]
    delta_pos, delta_fit = positions[sorted_idx[2]].copy(), fitness[sorted_idx[2]]
    history.append(alpha_fit)

    iter_count = 0
    max_iter_est = max_fes // pop_size
    while fes < max_fes:
        a = 2.0 - 2.0 * (iter_count / max(max_iter_est, 1))
        for i in range(pop_size):
            if fes >= max_fes: break
            r1a, r2a = np.random.rand(dim), np.random.rand(dim)
            r1b, r2b = np.random.rand(dim), np.random.rand(dim)
            r1c, r2c = np.random.rand(dim), np.random.rand(dim)
            A1, A2, A3 = 2*a*r1a-a, 2*a*r1b-a, 2*a*r1c-a
            C1, C2, C3 = 2*r2a, 2*r2b, 2*r2c
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])
            positions[i] = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
            fitness[i] = obj_func(positions[i]); fes += 1
            # Update alpha, beta, delta hierarchy directly
            if fitness[i] < alpha_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos, beta_fit = alpha_pos.copy(), alpha_fit
                alpha_pos, alpha_fit = positions[i].copy(), fitness[i]
            elif fitness[i] < beta_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos, beta_fit = positions[i].copy(), fitness[i]
            elif fitness[i] < delta_fit:
                delta_pos, delta_fit = positions[i].copy(), fitness[i]
        history.append(alpha_fit)
        iter_count += 1
    return alpha_pos, alpha_fit, history


# ═══════════════════════════════════════════════════════════════
# 3. WOA — Whale Optimization Algorithm (Mirjalili & Lewis 2016)
# ═══════════════════════════════════════════════════════════════

def run_woa(
    obj_func: Callable, dim: int, max_fes: int = 60000,
    lb: float = -100.0, ub: float = 100.0, pop_size: int = 30,
) -> Tuple[np.ndarray, float, List[float]]:
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    fes = 0; history = []

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i]); fes += 1
    best_idx = np.argmin(fitness)
    g_best_pos, g_best_fit = positions[best_idx].copy(), fitness[best_idx]
    history.append(g_best_fit)

    iter_count = 0
    max_iter_est = max_fes // pop_size
    while fes < max_fes:
        a = 2.0 - 2.0 * (iter_count / max(max_iter_est, 1))
        a2 = -1.0 - (iter_count / max(max_iter_est, 1))
        for i in range(pop_size):
            if fes >= max_fes: break
            r = np.random.rand()
            A = 2 * a * np.random.rand(dim) - a
            C = 2 * np.random.rand(dim)
            p = np.random.rand()
            b = 1.0
            l = np.random.uniform(a2, 1.0)

            if p < 0.5:
                if np.abs(A).mean() < 1:
                    D = np.abs(C * g_best_pos - positions[i])
                    positions[i] = g_best_pos - A * D
                else:
                    rand_idx = np.random.randint(pop_size)
                    D = np.abs(C * positions[rand_idx] - positions[i])
                    positions[i] = positions[rand_idx] - A * D
            else:
                D = np.abs(g_best_pos - positions[i])
                positions[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + g_best_pos

            positions[i] = np.clip(positions[i], lb, ub)
            fitness[i] = obj_func(positions[i]); fes += 1
            if fitness[i] < g_best_fit:
                g_best_fit = fitness[i]; g_best_pos = positions[i].copy()
        history.append(g_best_fit)
        iter_count += 1
    return g_best_pos, g_best_fit, history


# ═══════════════════════════════════════════════════════════════
# 4. HHO — Harris Hawks Optimization (Heidari et al. 2019)
# ═══════════════════════════════════════════════════════════════

def run_hho(
    obj_func: Callable, dim: int, max_fes: int = 60000,
    lb: float = -100.0, ub: float = 100.0, pop_size: int = 30,
) -> Tuple[np.ndarray, float, List[float]]:
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    fes = 0; history = []

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i]); fes += 1
    best_idx = np.argmin(fitness)
    rabbit_pos, rabbit_fit = positions[best_idx].copy(), fitness[best_idx]
    history.append(rabbit_fit)

    iter_count = 0
    max_iter_est = max_fes // pop_size
    while fes < max_fes:
        E0 = 2 * np.random.rand() - 1
        E = 2 * E0 * (1 - iter_count / max(max_iter_est, 1))

        for i in range(pop_size):
            if fes >= max_fes: break
            q = np.random.rand()
            r = np.random.rand()

            if abs(E) >= 1:
                # Exploration
                if q >= 0.5:
                    rand_idx = np.random.randint(pop_size)
                    X_rand = positions[rand_idx]
                    positions[i] = X_rand - np.random.rand(dim) * np.abs(
                        X_rand - 2 * np.random.rand(dim) * positions[i])
                else:
                    positions[i] = (rabbit_pos - positions.mean(axis=0)
                                    - np.random.rand(dim) * (lb + np.random.rand(dim) * (ub - lb)))
            else:
                # Exploitation
                if r >= 0.5 and abs(E) >= 0.5:
                    positions[i] = rabbit_pos - E * np.abs(rabbit_pos - positions[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    positions[i] = rabbit_pos - E * np.abs(rabbit_pos - positions[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    jump = rabbit_pos - E * np.abs(E * rabbit_pos - positions[i])
                    positions[i] = jump + np.random.rand(dim) * (ub - lb) * 0.01
                else:
                    jump = rabbit_pos - E * np.abs(E * rabbit_pos - positions.mean(axis=0))
                    positions[i] = jump + np.random.rand(dim) * (ub - lb) * 0.01

            positions[i] = np.clip(positions[i], lb, ub)
            fitness[i] = obj_func(positions[i]); fes += 1
            if fitness[i] < rabbit_fit:
                rabbit_fit = fitness[i]; rabbit_pos = positions[i].copy()
        history.append(rabbit_fit)
        iter_count += 1
    return rabbit_pos, rabbit_fit, history


# ═══════════════════════════════════════════════════════════════
# 5. AOA — Arithmetic Optimization Algorithm (Abualigah et al. 2021)
# ═══════════════════════════════════════════════════════════════

def run_aoa(
    obj_func: Callable, dim: int, max_fes: int = 60000,
    lb: float = -100.0, ub: float = 100.0, pop_size: int = 30,
) -> Tuple[np.ndarray, float, List[float]]:
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    fes = 0; history = []
    alpha = 5; mu = 0.499

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i]); fes += 1
    best_idx = np.argmin(fitness)
    g_best_pos, g_best_fit = positions[best_idx].copy(), fitness[best_idx]
    history.append(g_best_fit)

    iter_count = 0
    max_iter_est = max_fes // pop_size
    while fes < max_fes:
        moa = 0.2 + (iter_count / max(max_iter_est, 1)) * 0.8  # Math Optimizer Accelerated
        mop = 1.0 - (iter_count / max(max_iter_est, 1)) ** (1.0 / alpha)  # Probability

        for i in range(pop_size):
            if fes >= max_fes: break
            for j in range(dim):
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                if r1 > moa:  # Exploration (Mul or Div)
                    if r2 > 0.5:
                        positions[i, j] = g_best_pos[j] / (mop + 1e-10) * ((ub - lb) * mu + lb)
                    else:
                        positions[i, j] = g_best_pos[j] * mop * ((ub - lb) * mu + lb)
                else:  # Exploitation (Add or Sub)
                    if r3 > 0.5:
                        positions[i, j] = g_best_pos[j] - mop * ((ub - lb) * mu + lb)
                    else:
                        positions[i, j] = g_best_pos[j] + mop * ((ub - lb) * mu + lb)
            positions[i] = np.clip(positions[i], lb, ub)
            fitness[i] = obj_func(positions[i]); fes += 1
            if fitness[i] < g_best_fit:
                g_best_fit = fitness[i]; g_best_pos = positions[i].copy()
        history.append(g_best_fit)
        iter_count += 1
    return g_best_pos, g_best_fit, history


# ═══════════════════════════════════════════════════════════════
# 6. SCA — Sine Cosine Algorithm (Mirjalili 2016)
# ═══════════════════════════════════════════════════════════════

def run_sca(
    obj_func: Callable, dim: int, max_fes: int = 60000,
    lb: float = -100.0, ub: float = 100.0, pop_size: int = 30,
) -> Tuple[np.ndarray, float, List[float]]:
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    fes = 0; history = []

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i]); fes += 1
    best_idx = np.argmin(fitness)
    g_best_pos, g_best_fit = positions[best_idx].copy(), fitness[best_idx]
    history.append(g_best_fit)

    iter_count = 0
    max_iter_est = max_fes // pop_size
    while fes < max_fes:
        a = 2.0
        r1 = a - (iter_count / max(max_iter_est, 1)) * a
        for i in range(pop_size):
            if fes >= max_fes: break
            for j in range(dim):
                r2 = 2 * np.pi * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                if r4 < 0.5:
                    positions[i, j] += r1 * np.sin(r2) * np.abs(r3 * g_best_pos[j] - positions[i, j])
                else:
                    positions[i, j] += r1 * np.cos(r2) * np.abs(r3 * g_best_pos[j] - positions[i, j])
            positions[i] = np.clip(positions[i], lb, ub)
            fitness[i] = obj_func(positions[i]); fes += 1
            if fitness[i] < g_best_fit:
                g_best_fit = fitness[i]; g_best_pos = positions[i].copy()
        history.append(g_best_fit)
        iter_count += 1
    return g_best_pos, g_best_fit, history


# ═══════════════════════════════════════════════════════════════
# 7. SSA — Salp Swarm Algorithm (Mirjalili et al. 2017)
# ═══════════════════════════════════════════════════════════════

def run_ssa(
    obj_func: Callable, dim: int, max_fes: int = 60000,
    lb: float = -100.0, ub: float = 100.0, pop_size: int = 30,
) -> Tuple[np.ndarray, float, List[float]]:
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    fes = 0; history = []

    for i in range(pop_size):
        fitness[i] = obj_func(positions[i]); fes += 1
    best_idx = np.argmin(fitness)
    food_pos, food_fit = positions[best_idx].copy(), fitness[best_idx]
    history.append(food_fit)

    iter_count = 0
    max_iter_est = max_fes // pop_size
    while fes < max_fes:
        c1 = 2 * np.exp(-((4 * iter_count / max(max_iter_est, 1)) ** 2))
        for i in range(pop_size):
            if fes >= max_fes: break
            if i == 0:  # Leader salp
                for j in range(dim):
                    c2, c3 = np.random.rand(), np.random.rand()
                    if c3 < 0.5:
                        positions[i, j] = food_pos[j] + c1 * ((ub - lb) * c2 + lb)
                    else:
                        positions[i, j] = food_pos[j] - c1 * ((ub - lb) * c2 + lb)
            else:  # Follower salps
                positions[i] = 0.5 * (positions[i] + positions[i - 1])
            positions[i] = np.clip(positions[i], lb, ub)
            fitness[i] = obj_func(positions[i]); fes += 1
            if fitness[i] < food_fit:
                food_fit = fitness[i]; food_pos = positions[i].copy()
        history.append(food_fit)
        iter_count += 1
    return food_pos, food_fit, history


# ═══════════════════════════════════════════════════════════════
# Registry — all algorithms accessible by name
# ═══════════════════════════════════════════════════════════════

ALGORITHMS = {
    "PSO": run_pso,
    "GWO": run_gwo,
    "WOA": run_woa,
    "HHO": run_hho,
    "AOA": run_aoa,
    "SCA": run_sca,
    "SSA": run_ssa,
}
