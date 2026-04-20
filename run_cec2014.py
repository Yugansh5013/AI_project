# CEC 2014 Benchmark Runner
# Run on YOUR machine: python run_cec2014.py
import sys, os, time
sys.path.insert(0, r"c:\use this\AI_project")
os.chdir(r"c:\use this\AI_project")

import numpy as np
import pandas as pd
from apoa import run_apoa, run_base_poa
from comparison_algorithms import ALGORITHMS

MAX_FES = 60000
NUM_RUNS = 50
POP_SIZE = 30
DIM = 10
OUTPUT = "cec2014_results.csv"

ALL_ALGOS = {"A-POA": run_apoa, "Base-POA": run_base_poa}
ALL_ALGOS.update(ALGORITHMS)

# Build CEC 2014 functions (F1-F30)
from opfunu.cec_based import cec2014
cec_funcs = {}
for i in range(1, 31):
    cls = getattr(cec2014, f"F{i}2014", None)
    if cls:
        cec_funcs[f"CEC2014_F{i}"] = cls

# Resume support: load existing results
done_keys = set()
if os.path.exists(OUTPUT):
    existing = pd.read_csv(OUTPUT)
    for _, row in existing.iterrows():
        done_keys.add((row["Algorithm"], row["Function"], int(row["Run"])))
    print(f"Resuming: {len(done_keys)} tasks already done")

total = len(ALL_ALGOS) * len(cec_funcs) * NUM_RUNS
print("=" * 60)
print(f"  CEC 2014 BENCHMARK")
print(f"  Algorithms: {len(ALL_ALGOS)} | Functions: {len(cec_funcs)}")
print(f"  Runs: {NUM_RUNS} | FEs: {MAX_FES} | Dim: {DIM}")
print(f"  Total tasks: {total} | Already done: {len(done_keys)}")
print(f"  Remaining: {total - len(done_keys)}")
print("=" * 60)

results = []
if os.path.exists(OUTPUT):
    results = pd.read_csv(OUTPUT).to_dict("records")

completed = len(done_keys)
t_start = time.time()
new_count = 0

for algo_name, algo_func in ALL_ALGOS.items():
    for func_key, func_cls in cec_funcs.items():
        for run_id in range(NUM_RUNS):
            if (algo_name, func_key, run_id) in done_keys:
                continue

            seed = hash((algo_name, func_key, run_id)) % (2**31)
            np.random.seed(seed)
            try:
                instance = func_cls(ndim=DIM)
                obj = lambda x, fi=instance: fi.evaluate(x)
                _, best_fit, _ = algo_func(obj, DIM, MAX_FES, -100.0, 100.0, POP_SIZE)
            except Exception as e:
                best_fit = float("inf")

            results.append({
                "Algorithm": algo_name, "Function": func_key,
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
                      f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m | {algo_name} {func_key}",
                      flush=True)

            # Save every 50 new tasks
            if new_count % 50 == 0:
                pd.DataFrame(results).to_csv(OUTPUT, index=False)

# Final save
pd.DataFrame(results).to_csv(OUTPUT, index=False)
total_time = (time.time() - t_start) / 60
print(f"\nDONE! {new_count} new tasks in {total_time:.1f} min")
print(f"Saved: {OUTPUT} ({len(results)} total results)")
