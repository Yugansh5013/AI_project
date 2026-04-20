# Engineering Problems Benchmark Runner
# Run on either machine: python run_engineering.py
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from apoa import run_apoa, run_base_poa
from comparison_algorithms import ALGORITHMS
from engineering_problems import ENGINEERING_PROBLEMS

MAX_FES = 60000
NUM_RUNS = 50
POP_SIZE = 30
OUTPUT = "engineering_results.csv"

ALL_ALGOS = {"A-POA": run_apoa, "Base-POA": run_base_poa}
ALL_ALGOS.update(ALGORITHMS)

# Resume support
done_keys = set()
if os.path.exists(OUTPUT):
    existing = pd.read_csv(OUTPUT)
    for _, row in existing.iterrows():
        done_keys.add((row["Algorithm"], row["Problem"], int(row["Run"])))
    print(f"Resuming: {len(done_keys)} tasks already done")

total = len(ALL_ALGOS) * len(ENGINEERING_PROBLEMS) * NUM_RUNS
print("=" * 60)
print(f"  ENGINEERING PROBLEMS BENCHMARK")
print(f"  Algorithms: {len(ALL_ALGOS)} | Problems: {len(ENGINEERING_PROBLEMS)}")
print(f"  Runs: {NUM_RUNS} | FEs: {MAX_FES}")
print(f"  Total tasks: {total} | Already done: {len(done_keys)}")
print("=" * 60)

results = []
if os.path.exists(OUTPUT):
    results = pd.read_csv(OUTPUT).to_dict("records")

completed = len(done_keys)
t_start = time.time()
new_count = 0

for algo_name, algo_func in ALL_ALGOS.items():
    for prob_name, (obj_func, (lb_arr, ub_arr), prob_dim) in ENGINEERING_PROBLEMS.items():
        for run_id in range(NUM_RUNS):
            if (algo_name, prob_name, run_id) in done_keys:
                continue

            np.random.seed(hash((algo_name, prob_name, run_id)) % (2**31))
            try:
                _, best_fit, _ = algo_func(
                    obj_func, prob_dim, MAX_FES,
                    float(lb_arr.min()), float(ub_arr.max()), POP_SIZE,
                )
            except Exception as e:
                best_fit = float("inf")

            results.append({
                "Algorithm": algo_name, "Problem": prob_name,
                "Run": run_id, "Best_Fitness": best_fit,
            })
            completed += 1
            new_count += 1

            if new_count % 10 == 0:
                elapsed = time.time() - t_start
                rate = new_count / elapsed if elapsed > 0 else 1
                remaining = total - completed
                eta = remaining / rate if rate > 0 else 0
                print(f"  [{completed:5d}/{total}] {completed/total*100:5.1f}% | "
                      f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m | {algo_name} {prob_name}",
                      flush=True)

            if new_count % 50 == 0:
                pd.DataFrame(results).to_csv(OUTPUT, index=False)

pd.DataFrame(results).to_csv(OUTPUT, index=False)
total_time = (time.time() - t_start) / 60
print(f"\nDONE! {new_count} new tasks in {total_time:.1f} min")
print(f"Saved: {OUTPUT} ({len(results)} total results)")
