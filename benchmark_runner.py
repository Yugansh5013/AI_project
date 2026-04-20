"""
Unified Benchmark Runner
=========================
Runs all algorithms on CEC 2014 + CEC 2017 + Engineering Problems.
Saves raw results incrementally to CSV (crash-safe).
"""
import numpy as np
import pandas as pd
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from apoa import run_apoa, run_base_poa
from comparison_algorithms import ALGORITHMS
from engineering_problems import ENGINEERING_PROBLEMS

# ─── Configuration ───────────────────────────────────────────
MAX_FES = 60000
NUM_RUNS = 50
POP_SIZE = 30
DIM = 10  # CEC standard dimension
NUM_WORKERS = max(1, os.cpu_count() - 1)

# All 9 algorithms
ALL_ALGOS = {
    "A-POA": run_apoa,
    "Base-POA": run_base_poa,
    **ALGORITHMS,
}

# ─── CEC Function Registry ──────────────────────────────────

def _build_cec_registry():
    """Build function registry for CEC 2014 and CEC 2017 from opfunu."""
    from opfunu.cec_based import cec2014, cec2017

    registry = {}

    # CEC 2014: F1–F30
    for i in range(1, 31):
        cls_name = f"F{i}2014"
        cls = getattr(cec2014, cls_name, None)
        if cls is not None:
            registry[f"CEC2014_F{i}"] = cls

    # CEC 2017: F1, F3–F29 (F2 is deprecated, F30 missing in opfunu)
    for i in [1] + list(range(3, 30)):
        cls_name = f"F{i}2017"
        cls = getattr(cec2017, cls_name, None)
        if cls is not None:
            registry[f"CEC2017_F{i}"] = cls

    return registry


# ─── Worker Function ────────────────────────────────────────

def _run_single(args):
    """Worker: run one (algo, function, seed) combo."""
    algo_name, func_key, func_class, dim, max_fes, pop_size, seed, algo_func_code = args
    np.random.seed(seed)

    # Import the algorithm function dynamically to avoid pickling issues
    if algo_name == "A-POA":
        from apoa import run_apoa as algo_func
    elif algo_name == "Base-POA":
        from apoa import run_base_poa as algo_func
    else:
        from comparison_algorithms import ALGORITHMS as A
        algo_func = A[algo_name]

    # Build the objective function
    if func_key.startswith("CEC"):
        from opfunu.cec_based import cec2014, cec2017
        if "2014" in func_key:
            module = cec2014
        else:
            module = cec2017
        f_num = func_key.split("_F")[1]
        cls_name = f"F{f_num}{'2014' if '2014' in func_key else '2017'}"
        cls = getattr(module, cls_name)
        func_instance = cls(ndim=dim)
        obj_func = lambda x, fi=func_instance: fi.evaluate(x)
        lb, ub = -100.0, 100.0
    else:
        # Engineering problem
        from engineering_problems import ENGINEERING_PROBLEMS as EP
        obj_func_raw, (lb_arr, ub_arr), prob_dim = EP[func_key]
        obj_func = obj_func_raw
        dim = prob_dim
        lb, ub = float(lb_arr.min()), float(ub_arr.max())
        # For engineering problems, use actual bounds per dimension
        # We pass lb/ub as arrays by wrapping
        positions_init = np.random.uniform(lb_arr, ub_arr, (pop_size, dim))

    # Run the algorithm
    if func_key.startswith("CEC"):
        best_pos, best_fit, history = algo_func(
            obj_func=obj_func, dim=dim, max_fes=max_fes, lb=lb, ub=ub, pop_size=pop_size,
        )
    else:
        # For engineering: use lb/ub arrays
        best_pos, best_fit, history = algo_func(
            obj_func=obj_func, dim=dim, max_fes=max_fes,
            lb=float(lb_arr.min()), ub=float(ub_arr.max()), pop_size=pop_size,
        )

    return {
        "Algorithm": algo_name,
        "Function": func_key,
        "Run": seed,
        "Best_Fitness": best_fit,
        "Convergence_Length": len(history),
        "Final_History_5": history[-5:] if len(history) >= 5 else history,
    }


# ─── Main Runner ────────────────────────────────────────────

def run_cec_benchmark(output_csv: str = "benchmark_results.csv"):
    """Run the full CEC benchmark suite."""
    print("=" * 70)
    print("  FULL CEC BENCHMARK RUNNER")
    print(f"  Algorithms: {len(ALL_ALGOS)} | FEs: {MAX_FES} | Runs: {NUM_RUNS} | Dim: {DIM}")
    print(f"  Workers: {NUM_WORKERS}")
    print("=" * 70)

    registry = _build_cec_registry()
    print(f"\n  CEC functions loaded: {len(registry)}")

    # Build task list
    tasks = []
    for algo_name in ALL_ALGOS:
        for func_key in registry:
            for run_id in range(NUM_RUNS):
                seed = hash((algo_name, func_key, run_id)) % (2**31)
                tasks.append((algo_name, func_key, None, DIM, MAX_FES, POP_SIZE, seed, algo_name))

    total = len(tasks)
    print(f"  Total tasks: {total}\n")

    results = []
    completed = 0
    t_start = time.time()

    # Sequential execution (safe for all environments)
    for task in tasks:
        try:
            result = _run_single(task)
            results.append(result)
        except Exception as e:
            results.append({
                "Algorithm": task[0], "Function": task[1], "Run": task[6],
                "Best_Fitness": float("inf"), "Convergence_Length": 0,
                "Final_History_5": [], "Error": str(e),
            })
        completed += 1
        elapsed = time.time() - t_start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        if completed % 10 == 0 or completed == total:
            print(f"  [{completed:5d}/{total}] "
                  f"{completed/total*100:5.1f}% | "
                  f"{elapsed/60:.1f}min elapsed | "
                  f"ETA {eta/60:.1f}min | "
                  f"{task[0]:8s} {task[1]}", flush=True)

        # Incremental save every 100 tasks
        if completed % 100 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n  ✅ Benchmark complete! {len(df)} results saved to {output_csv}")
    print(f"  Total time: {(time.time()-t_start)/60:.1f} minutes")
    return df


def run_engineering_benchmark(output_csv: str = "engineering_results.csv"):
    """Run engineering problems benchmark."""
    print("=" * 70)
    print("  ENGINEERING PROBLEMS BENCHMARK")
    print(f"  Algorithms: {len(ALL_ALGOS)} | Problems: {len(ENGINEERING_PROBLEMS)}")
    print("=" * 70)

    results = []
    total = len(ALL_ALGOS) * len(ENGINEERING_PROBLEMS) * NUM_RUNS
    completed = 0
    t_start = time.time()

    for algo_name, algo_func in ALL_ALGOS.items():
        for prob_name, (obj_func, (lb_arr, ub_arr), prob_dim) in ENGINEERING_PROBLEMS.items():
            for run_id in range(NUM_RUNS):
                np.random.seed(hash((algo_name, prob_name, run_id)) % (2**31))
                try:
                    best_pos, best_fit, history = algo_func(
                        obj_func=obj_func, dim=prob_dim, max_fes=MAX_FES,
                        lb=float(lb_arr.min()), ub=float(ub_arr.max()),
                        pop_size=POP_SIZE,
                    )
                    results.append({
                        "Algorithm": algo_name, "Problem": prob_name,
                        "Run": run_id, "Best_Fitness": best_fit,
                    })
                except Exception as e:
                    results.append({
                        "Algorithm": algo_name, "Problem": prob_name,
                        "Run": run_id, "Best_Fitness": float("inf"),
                        "Error": str(e),
                    })
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{completed}/{total}] {elapsed/60:.1f}min | {algo_name} {prob_name}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n  ✅ Engineering benchmark complete! Saved to {output_csv}")
    return df


# ─── Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("cec", "all"):
        run_cec_benchmark()
    if mode in ("eng", "all"):
        run_engineering_benchmark()
