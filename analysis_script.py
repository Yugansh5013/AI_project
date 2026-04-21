# Step 1 — Imports & Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, ranksums
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_palette("husl")

# Load CEC results (from split files)
dfs = []
if os.path.exists("cec2014_results.csv"):
    dfs.append(pd.read_csv("cec2014_results.csv"))
if os.path.exists("cec2017_results.csv"):
    dfs.append(pd.read_csv("cec2017_results.csv"))

if not dfs:
    print("Warning: No CEC results found! Please run python run_cec2014.py or run_cec2017.py first.")
    # Create empty df to prevent errors later
    df = pd.DataFrame(columns=['Algorithm', 'Function', 'Run', 'Best_Fitness'])
else:
    df = pd.concat(dfs, ignore_index=True)
    print(f"CEC Results Loaded: {len(df)} rows, {df['Algorithm'].nunique()} algorithms, {df['Function'].nunique()} functions")
    print(f"Algorithms: {sorted(df['Algorithm'].unique())}")


# Step 2a — CEC 2014 Summary Table with Ranks
cec2014 = df[df['Function'].str.contains('CEC2014')].copy()

summary_2014 = cec2014.groupby(['Function', 'Algorithm'])['Best_Fitness'].agg(
    Mean='mean', Std='std'
).reset_index()

# Rank algorithms per function (1 = best)
summary_2014['Rank'] = summary_2014.groupby('Function')['Mean'].rank(method='min')

# Pivot for display
pivot_mean = summary_2014.pivot(index='Function', columns='Algorithm', values='Mean')
pivot_std = summary_2014.pivot(index='Function', columns='Algorithm', values='Std')
pivot_rank = summary_2014.pivot(index='Function', columns='Algorithm', values='Rank')

# Sort by function number
import re
pivot_mean['_sort'] = pivot_mean.index.map(lambda x: int(re.search(r'F(\d+)', x).group(1)))
pivot_mean = pivot_mean.sort_values('_sort').drop('_sort', axis=1)
pivot_rank['_sort'] = pivot_rank.index.map(lambda x: int(re.search(r'F(\d+)', x).group(1)))
pivot_rank = pivot_rank.sort_values('_sort').drop('_sort', axis=1)

# Display table
print("=" * 100)
print("CEC 2014 — Mean Error by Algorithm")
print("=" * 100)
print(pivot_mean.to_string(float_format='{:.2e}'.format))
print()
print("CEC 2014 — Rank by Algorithm (1 = best)")
print("=" * 100)
print(pivot_rank.to_string(float_format='{:.0f}'.format))
print()
print("Average Rank:")
print(pivot_rank.mean().sort_values().to_string(float_format='{:.2f}'))


# Step 3 — CEC 2017 Summary Table with Ranks
cec2017 = df[df['Function'].str.contains('CEC2017')].copy()

summary_2017 = cec2017.groupby(['Function', 'Algorithm'])['Best_Fitness'].agg(
    Mean='mean', Std='std'
).reset_index()

summary_2017['Rank'] = summary_2017.groupby('Function')['Mean'].rank(method='min')

pivot_mean_17 = summary_2017.pivot(index='Function', columns='Algorithm', values='Mean')
pivot_rank_17 = summary_2017.pivot(index='Function', columns='Algorithm', values='Rank')

pivot_mean_17['_sort'] = pivot_mean_17.index.map(lambda x: int(re.search(r'F(\d+)', x).group(1)))
pivot_mean_17 = pivot_mean_17.sort_values('_sort').drop('_sort', axis=1)
pivot_rank_17['_sort'] = pivot_rank_17.index.map(lambda x: int(re.search(r'F(\d+)', x).group(1)))
pivot_rank_17 = pivot_rank_17.sort_values('_sort').drop('_sort', axis=1)

print("=" * 100)
print("CEC 2017 — Mean Error by Algorithm")
print("=" * 100)
print(pivot_mean_17.to_string(float_format='{:.2e}'.format))
print()
print("CEC 2017 — Rank by Algorithm (1 = best)")
print("=" * 100)
print(pivot_rank_17.to_string(float_format='{:.0f}'))
print()
print("Average Rank:")
print(pivot_rank_17.mean().sort_values().to_string(float_format='{:.2f}'))


# Step 4 — Wilcoxon Rank-Sum Test (A-POA vs each competitor)
alpha = 0.05
competitors = [a for a in df['Algorithm'].unique() if a != 'A-POA']
functions = sorted(df['Function'].unique())

wilcoxon_results = []
win_loss = {comp: {'win': 0, 'loss': 0, 'tie': 0} for comp in competitors}

for func in functions:
    apoa_vals = df[(df['Function'] == func) & (df['Algorithm'] == 'A-POA')]['Best_Fitness'].values
    for comp in competitors:
        comp_vals = df[(df['Function'] == func) & (df['Algorithm'] == comp)]['Best_Fitness'].values
        if len(apoa_vals) < 2 or len(comp_vals) < 2:
            continue
        stat, p_val = ranksums(apoa_vals, comp_vals)
        sig = 'Yes' if p_val < alpha else 'No'

        apoa_mean = np.mean(apoa_vals)
        comp_mean = np.mean(comp_vals)
        if p_val < alpha:
            if apoa_mean < comp_mean:
                result = '+'  # A-POA wins
                win_loss[comp]['win'] += 1
            else:
                result = '−'  # A-POA loses
                win_loss[comp]['loss'] += 1
        else:
            result = '≈'
            win_loss[comp]['tie'] += 1

        wilcoxon_results.append({
            'Function': func, 'Competitor': comp,
            'p-value': p_val, 'Significant': sig, 'Result': result,
        })

wdf = pd.DataFrame(wilcoxon_results)
pivot_w = wdf.pivot(index='Function', columns='Competitor', values='Result')
print("Wilcoxon Rank-Sum Test: A-POA vs Competitors")
print("+ = A-POA wins | − = A-POA loses | ≈ = No significant difference")
print("=" * 80)
print(pivot_w.to_string())

print()
print("Win/Loss/Tie Summary:")
print("-" * 60)
for comp in competitors:
    w, l, t = win_loss[comp]['win'], win_loss[comp]['loss'], win_loss[comp]['tie']
    print(f"  A-POA vs {comp:10s}: +{w:2d} / −{l:2d} / ≈{t:2d}")


# Step 5 — Convergence Curves
# Note: Full convergence data requires re-running with history tracking.
# This cell generates convergence-style plots from the available data.

# For now, plot mean fitness by algorithm across all CEC 2017 functions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Mean Error Comparison — CEC 2017 Selected Functions', fontsize=16, color='white')

selected = ['CEC2017_F1', 'CEC2017_F5', 'CEC2017_F10', 'CEC2017_F15', 'CEC2017_F20', 'CEC2017_F25']
algos = sorted(df['Algorithm'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))

for ax, func in zip(axes.flat, selected):
    func_data = df[df['Function'] == func]
    means = func_data.groupby('Algorithm')['Best_Fitness'].mean().reindex(algos)
    bars = ax.bar(range(len(algos)), means.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(func.replace('CEC2017_', ''), fontsize=12, color='white')
    ax.set_yscale('log')
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cec2017_comparison_bars.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()
print("Saved: cec2017_comparison_bars.png")


# Step 6 — Engineering Problems
try:
    eng_df = pd.read_csv("engineering_results.csv")
    print(f"Engineering Results: {len(eng_df)} rows")

    eng_summary = eng_df.groupby(['Problem', 'Algorithm'])['Best_Fitness'].agg(
        Mean='mean', Std='std', Best='min', Worst='max'
    ).reset_index()
    eng_summary['Rank'] = eng_summary.groupby('Problem')['Mean'].rank(method='min')

    for prob in eng_summary['Problem'].unique():
        print(f"\n{'='*80}")
        print(f"  {prob}")
        print(f"{'='*80}")
        prob_data = eng_summary[eng_summary['Problem'] == prob].sort_values('Rank')
        print(prob_data[['Algorithm', 'Mean', 'Std', 'Best', 'Worst', 'Rank']].to_string(index=False))

    print(f"\n{'='*80}")
    print("Overall Average Rank (Engineering):")
    print(eng_summary.groupby('Algorithm')['Rank'].mean().sort_values().to_string(float_format='{:.2f}'))
except FileNotFoundError:
    print("engineering_results.csv not found. Run: python benchmark_runner.py eng")


# Step 7 — Overall Summary
print("=" * 80)
print("  OVERALL PERFORMANCE SUMMARY")
print("=" * 80)

# Average rank across all CEC functions
all_functions = df.groupby(['Function', 'Algorithm'])['Best_Fitness'].mean().reset_index()
all_functions['Rank'] = all_functions.groupby('Function')['Best_Fitness'].rank(method='min')
avg_rank = all_functions.groupby('Algorithm')['Rank'].mean().sort_values()

print("\nAverage Rank (All CEC Functions):")
print("-" * 40)
for algo, rank in avg_rank.items():
    marker = " ← BEST" if rank == avg_rank.min() else ""
    marker = " ← PROPOSED" if algo == "A-POA" else marker
    print(f"  {algo:12s}: {rank:.2f}{marker}")

# A-POA position
apoa_rank = avg_rank.get('A-POA', float('inf'))
apoa_position = list(avg_rank.index).index('A-POA') + 1 if 'A-POA' in avg_rank.index else -1
print(f"\n  A-POA finished at position {apoa_position}/{len(avg_rank)} with average rank {apoa_rank:.2f}")


