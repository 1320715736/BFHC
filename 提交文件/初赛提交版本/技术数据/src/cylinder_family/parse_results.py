import csv, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load trial records written by optuna_optimize.py.
DATA = pathlib.Path(__file__).parent / 'data' / 'trials.csv'
FIG_DIR = pathlib.Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(DATA, encoding='utf-8')))
SOLVED_STATUSES = {'OK', 'PRUNE_LIFETIME'}
ok = [r for r in rows if r['status'] in SOLVED_STATUSES]
if not ok:
    print('ERROR: 无成功 trial')
    sys.exit(1)
fields_float = ['r1_mm', 'r2_mm', 'r3_mm', 'r4_mm', 'Vwork_V', 'initialTmax_K', 'lifetimeH', 'initialP03sphere_W', 'initialPradSphere_W', 'lifeAvgP03sphere_W', 'lifeAvgPradSphere_W', 'selfViewLoss_pct', 'erosionSteps', 'elapsed_sec']

# Convert CSV columns into arrays for ranking and plotting.
N = len(ok)
data = {}
for f in fields_float:
    data[f] = np.array([float(r[f]) for r in ok])
trial_ids = np.array([int(r['trial']) for r in ok])
failure_reached = np.array([r['failureReached'] == 'True' for r in ok])
bl_mask = trial_ids == 0
if bl_mask.any():
    bl_idx = np.where(bl_mask)[0][0]
    bl = {f: data[f][bl_idx] for f in fields_float}
else:
    print('WARNING: Trial 0 (baseline) not found')
    bl = None
avg_p03 = data['lifeAvgP03sphere_W']
lifetime = data['lifetimeH']
LIFE_FLOOR = bl['lifetimeH'] * 0.3 if bl else 15.0
feasible = lifetime >= LIFE_FLOOR

# Console summary and top candidate table.
print('=' * 70)
print(f'  圆柱族贝叶斯优化 — 完整汇总 ({N} trials)')
print('=' * 70)
print(f'\n总 trials: {len(rows)}  成功求解: {N}  异常失败: {len(rows) - N}')
print(f'满足寿命硬约束 (≥{LIFE_FLOOR:.1f}h): {feasible.sum()}/{N}')
print(f"总 COMSOL 计算时间: {data['elapsed_sec'].sum() / 3600:.1f} h")
print(f"平均单次耗时: {data['elapsed_sec'].mean():.0f} s")
order = np.argsort(-avg_p03)
print('\n── Top-20 按 lifeAvgP03sphere 排序 ──')
hdr = f'{'#':>4} {'r1':>5} {'r2':>5} {'r3':>5} {'r4':>5} {'Vwork':>6} {'Tmax':>7} {'Life_h':>7} {'AvgP03':>7} {'SV%':>5} {'OK':>3}'
print(hdr)
print('-' * len(hdr))
for i in order[:20]:
    ok_mark = '✓' if feasible[i] else '✗'
    print(f"{trial_ids[i]:>4} {data['r1_mm'][i]:>5.2f} {data['r2_mm'][i]:>5.2f} {data['r3_mm'][i]:>5.2f} {data['r4_mm'][i]:>5.2f} {data['Vwork_V'][i]:>6.3f} {data['initialTmax_K'][i]:>7.1f} {lifetime[i]:>7.1f} {avg_p03[i]:>7.1f} {data['selfViewLoss_pct'][i]:>5.1f} {ok_mark:>3}")
feas_idx = np.where(feasible)[0]

# Pareto front uses lifecycle P03 and lifetime as the two objectives.
if len(feas_idx) > 0:
    feas_p03 = avg_p03[feas_idx]
    feas_life = lifetime[feas_idx]
    pareto_mask = np.zeros(len(feas_idx), dtype=bool)
    for i in range(len(feas_idx)):
        dominated = False
        for j in range(len(feas_idx)):
            if i == j:
                continue
            if feas_p03[j] >= feas_p03[i] and feas_life[j] >= feas_life[i] and (feas_p03[j] > feas_p03[i] or feas_life[j] > feas_life[i]):
                dominated = True
                break
        if not dominated:
            pareto_mask[i] = True
    pareto_global = feas_idx[pareto_mask]
    print(f'\n── Pareto 前沿 ({pareto_mask.sum()} 个非支配解, 满足寿命约束) ──')
    p_order = np.argsort(-avg_p03[pareto_global])
    for rank, pi in enumerate(p_order, 1):
        gi = pareto_global[pi]
        imp = (avg_p03[gi] / bl['lifeAvgP03sphere_W'] - 1) * 100 if bl else 0
        print(f"  Rank {rank}: Trial #{trial_ids[gi]}  [{data['r1_mm'][gi]:.2f}, {data['r2_mm'][gi]:.2f}, {data['r3_mm'][gi]:.2f}, {data['r4_mm'][gi]:.2f}] mm  AvgP03={avg_p03[gi]:.1f}W ({imp:+.1f}%)  Life={lifetime[gi]:.1f}h  SV={data['selfViewLoss_pct'][gi]:.1f}%")
if bl:
    if len(feas_idx) > 0:
        best_feas_i = feas_idx[np.argmax(avg_p03[feas_idx])]
    else:
        best_feas_i = None
    best_all_i = np.argmax(avg_p03)
    print('\n── Baseline 对比 ──')
    print(f'{'指标':>25} {'Baseline':>12} {'最优(约束)':>12} {'最优(无约束)':>12}')
    print('-' * 65)
    rows_cmp = [('Vwork (V)', 'Vwork_V', '.4f'), ('初始 Tmax (K)', 'initialTmax_K', '.1f'), ('寿命 (h)', 'lifetimeH', '.1f'), ('初始 P03sphere (W)', 'initialP03sphere_W', '.1f'), ('寿命均值 P03sphere (W)', 'lifeAvgP03sphere_W', '.1f'), ('自遮挡损失 (%)', 'selfViewLoss_pct', '.1f'), ('侵蚀步数', 'erosionSteps', '.0f')]
    for label, f, fmt in rows_cmp:
        v_bl = bl[f]
        v_uc = data[f][best_all_i]
        v_fc = data[f][best_feas_i] if best_feas_i is not None else float('nan')
        print(f'{label:>25} {v_bl:>12{fmt}} {v_fc:>12{fmt}} {v_uc:>12{fmt}}')
    if best_feas_i is not None:
        imp = (avg_p03[best_feas_i] / bl['lifeAvgP03sphere_W'] - 1) * 100
        print(f"\n★ 满足约束的最优提升: {imp:+.1f}%  (Trial #{trial_ids[best_feas_i]}  [{data['r1_mm'][best_feas_i]:.3f}, {data['r2_mm'][best_feas_i]:.3f}, {data['r3_mm'][best_feas_i]:.3f}, {data['r4_mm'][best_feas_i]:.3f}] mm)")

# Optimization history.
fig, ax = plt.subplots(figsize=(10, 4))
sort_by_trial = np.argsort(trial_ids)
sorted_ids = trial_ids[sort_by_trial]
sorted_p03 = avg_p03[sort_by_trial]
running_best = np.maximum.accumulate(sorted_p03)
ax.scatter(sorted_ids, sorted_p03, s=12, alpha=0.5, label='Per trial', c='steelblue')
ax.plot(sorted_ids, running_best, color='red', linewidth=2, label='Running best')
if bl:
    ax.axhline(bl['lifeAvgP03sphere_W'], color='gray', linestyle='--', label=f"Baseline ({bl['lifeAvgP03sphere_W']:.0f}W)")
ax.set_xlabel('Trial #')
ax.set_ylabel('lifeAvgP03sphere (W)')
ax.set_title('Bayesian Optimization Convergence — Cylinder Family')
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / 'optimization_history.png', dpi=150)
print(f'\n[saved] figures/optimization_history.png')

# Power-lifetime scatter plot with feasible Pareto front.
fig, ax = plt.subplots(figsize=(8, 6))
infeasible = ~feasible
ax.scatter(lifetime[infeasible], avg_p03[infeasible], s=15, alpha=0.3, c='gray', label=f'Infeasible (Life<{LIFE_FLOOR:.0f}h)')
ax.scatter(lifetime[feasible], avg_p03[feasible], s=20, alpha=0.6, c='steelblue', label='Feasible')
if len(feas_idx) > 0 and pareto_mask.any():
    pg = pareto_global
    p_sort = np.argsort(lifetime[pg])
    ax.plot(lifetime[pg][p_sort], avg_p03[pg][p_sort], 'r-o', markersize=6, label='Pareto front', zorder=5)
if bl:
    ax.scatter([bl['lifetimeH']], [bl['lifeAvgP03sphere_W']], s=100, marker='*', c='gold', edgecolors='black', zorder=10, label='Baseline')
ax.axvline(LIFE_FLOOR, color='red', linestyle=':', alpha=0.5, label=f'Life floor ({LIFE_FLOOR:.0f}h)')
ax.set_xlabel('Lifetime (h)')
ax.set_ylabel('lifeAvgP03sphere (W)')
ax.set_title('Pareto Front: Radiation Power vs Lifetime')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pareto_front.png', dpi=150)
print(f'[saved] figures/pareto_front.png')

# Radius patterns among high-performing trials.
fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12, 5))
top30 = order[:30]
param_names = ['r1_mm', 'r2_mm', 'r3_mm', 'r4_mm']
param_labels = ['r1', 'r2', 'r3', 'r4']
colors = plt.cm.RdYlGn(np.linspace(0, 1, 30))
for idx, gi in enumerate(top30):
    vals = [data[p][gi] for p in param_names]
    for j in range(3):
        axes[j].plot([j, j + 1], [vals[j], vals[j + 1]], color=colors[idx], alpha=0.7)
for j in range(4):
    axes[min(j, 3) if j < 4 else 3].set_xticks([])
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 5))
for idx, gi in enumerate(top30):
    vals = [data[p][gi] for p in param_names]
    ax.plot(range(4), vals, color=colors[idx], alpha=0.7, linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(param_labels)
ax.set_ylabel('Radius (mm)')
ax.set_title('Parallel Coordinates — Top 30 by AvgP03')
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(1, 30))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Rank')
fig.tight_layout()
fig.savefig(FIG_DIR / 'parallel_coordinate.png', dpi=150)
print(f'[saved] figures/parallel_coordinate.png')
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(data['r1_mm'], data['r3_mm'], c=avg_p03, cmap='hot', s=20, alpha=0.7)
fig.colorbar(sc, label='lifeAvgP03sphere (W)')
ax.set_xlabel('r1 (mm)')
ax.set_ylabel('r3 (mm)')
ax.set_title('AvgP03 landscape: r1 vs r3')
fig.tight_layout()
fig.savefig(FIG_DIR / 'scatter_r1_r3.png', dpi=150)
print(f'[saved] figures/scatter_r1_r3.png')
print('\n✅ parse_results.py 完成')
