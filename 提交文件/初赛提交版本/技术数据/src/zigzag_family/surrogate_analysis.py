import csv
import math
import os
import pathlib
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Read solved trials and fit surrogate models on the free zigzag variables.
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get('BFHC_TRIALS_CSV', BASE_DIR / 'data' / 'trials.csv'))
FIG_DIR = pathlib.Path(os.environ.get('BFHC_FIG_DIR', BASE_DIR / 'figures'))
FIG_DIR.mkdir(parents=True, exist_ok=True)
CYLINDER_BASELINE_LIFETIME_H = 115.5037
CYLINDER_BASELINE_AVG_P03_W = 428.5727
CYLINDER_OPT_FEASIBLE_AVG_P03_W = 559.6431
LIFE_FLOOR = 0.3 * CYLINDER_BASELINE_LIFETIME_H
ZIGZAG_JAVA_BASELINE_P03 = 3539.04
ZIGZAG_JAVA_BASELINE_LIFE = 7.1277
ZIGZAG_REF_PARAMS = np.array([8.0, 104.0, 0.8])
SOLVED_STATUSES = {'OK', 'PRUNE_LIFETIME'}
FEATURE_NAMES = ['N_RUNS', 'L_RUN_mm', 'z_first_mm']
DISPLAY_NAMES = ['N_RUNS', 'L_RUN', 'z_first']

def safe_float(value):
    try:
        out = float(value)
    except Exception:
        return float('nan')
    return out if math.isfinite(out) else float('nan')

def is_complete_solved(row):
    if row.get('status') not in SOLVED_STATUSES:
        return False
    fields = FEATURE_NAMES + ['lifeAvgP03sphere_W', 'lifetimeH']
    return all((math.isfinite(safe_float(row.get(f, ''))) for f in fields))

def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'[saved] {path}')

def cv_score(model, X, y):
    folds = min(5, len(y))
    if folds < 3:
        return np.array([float('nan')])
    return cross_val_score(model, X, y, cv=folds, scoring='r2')

# Dataset assembly.
if not DATA.exists():
    print(f'ERROR: data file not found: {DATA}')
    sys.exit(1)
with DATA.open('r', encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
complete = [r for r in rows if is_complete_solved(r)]
if len(complete) < 8:
    print(f'ERROR: not enough complete solved rows for surrogate analysis: {len(complete)}')
    sys.exit(1)
X = np.array([[safe_float(r[f]) for f in FEATURE_NAMES] for r in complete], dtype=float)
y_p03 = np.array([safe_float(r['lifeAvgP03sphere_W']) for r in complete], dtype=float)
y_life = np.array([safe_float(r['lifetimeH']) for r in complete], dtype=float)
trial_ids = np.array([int(r['trial']) for r in complete], dtype=int)
status_counts = {}
for row in rows:
    status_counts[row.get('status', '')] = status_counts.get(row.get('status', ''), 0) + 1
print('=' * 72)
print('Zigzag surrogate + SHAP analysis')
print('=' * 72)
print(f'Input CSV: {DATA}')
print(f'Figure dir: {FIG_DIR}')
print(f'Rows: total={len(rows)}, complete_solved={len(complete)}')
print('Status counts: ' + ', '.join((f'{k}={v}' for k, v in sorted(status_counts.items()))))
print(f'Lifetime floor: {LIFE_FLOOR:.2f} h')
bl_match = next((i for i, r in enumerate(complete) if r.get('trial') == '0'), None)
if bl_match is not None:
    bl_p03 = y_p03[bl_match]
    bl_life = y_life[bl_match]
    bl_x = X[bl_match]
else:
    bl_p03 = ZIGZAG_JAVA_BASELINE_P03
    bl_life = ZIGZAG_JAVA_BASELINE_LIFE
    bl_x = ZIGZAG_REF_PARAMS

# Random forests approximate COMSOL outputs for sensitivity analysis.
rf_p03 = RandomForestRegressor(n_estimators=260, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=1)
rf_life = RandomForestRegressor(n_estimators=260, max_depth=12, min_samples_leaf=2, random_state=43, n_jobs=1)
cv_p03 = cv_score(rf_p03, X, y_p03)
cv_life = cv_score(rf_life, X, y_life)
print(f'RF AvgP03 R2: {np.nanmean(cv_p03):.3f} +/- {np.nanstd(cv_p03):.3f}')
print(f'RF Lifetime R2: {np.nanmean(cv_life):.3f} +/- {np.nanstd(cv_life):.3f}')
rf_p03.fit(X, y_p03)
rf_life.fit(X, y_life)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, rf, title in [(axes[0], rf_p03, 'lifeAvgP03sphere'), (axes[1], rf_life, 'Lifetime')]:
    imp = rf.feature_importances_
    order = np.argsort(imp)
    ax.barh([DISPLAY_NAMES[i] for i in order], imp[order], color='steelblue')
    ax.set_xlabel('Random forest importance')
    ax.set_title(title)
fig.tight_layout()
savefig(fig, 'feature_importance.png')
plt.close(fig)
print('\nFeature importance')
for target, rf in [('AvgP03', rf_p03), ('Lifetime', rf_life)]:
    print(f'  {target}:')
    for i in np.argsort(-rf.feature_importances_):
        print(f'    {DISPLAY_NAMES[i]} = {rf.feature_importances_[i]:.3f}')

# SHAP plots explain each free variable contribution.
explainer_p03 = shap.TreeExplainer(rf_p03)
shap_p03 = explainer_p03.shap_values(X)
explainer_life = shap.TreeExplainer(rf_life)
shap_life = explainer_life.shap_values(X)
fig = plt.figure(figsize=(8, 4.5))
shap.summary_plot(shap_p03, X, feature_names=DISPLAY_NAMES, show=False)
plt.title('SHAP Summary - lifeAvgP03sphere')
plt.tight_layout()
savefig(fig, 'shap_summary_p03.png')
plt.close(fig)
fig = plt.figure(figsize=(8, 4.5))
shap.summary_plot(shap_life, X, feature_names=DISPLAY_NAMES, show=False)
plt.title('SHAP Summary - Lifetime')
plt.tight_layout()
savefig(fig, 'shap_summary_life.png')
plt.close(fig)
top_feat_idx = int(np.argmax(rf_p03.feature_importances_))
fig = plt.figure(figsize=(8, 5))
shap.dependence_plot(top_feat_idx, shap_p03, X, feature_names=DISPLAY_NAMES, show=False)
plt.title(f'SHAP Dependence - {DISPLAY_NAMES[top_feat_idx]} effect on AvgP03')
plt.tight_layout()
savefig(fig, 'shap_dependence.png')
plt.close(fig)

# Response surfaces are shown for the most populated N_RUNS values.
n_runs_values = sorted(set(X[:, 0].astype(int)))
n_counts = [(n, int(np.sum(X[:, 0].astype(int) == n))) for n in n_runs_values]
n_counts.sort(key=lambda pair: -pair[1])
top_n = [n for n, _ in n_counts[:3]]
fig, axes = plt.subplots(1, len(top_n), figsize=(6 * len(top_n), 5), squeeze=False)
for ax, n_val in zip(axes[0], top_n):
    L_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 70)
    z_grid = np.linspace(X[:, 2].min(), X[:, 2].max(), 60)
    Lg, Zg = np.meshgrid(L_grid, z_grid)
    Xgrid = np.column_stack([np.full(Lg.size, n_val), Lg.ravel(), Zg.ravel()])
    Ygrid = rf_p03.predict(Xgrid).reshape(Lg.shape)
    cs = ax.contourf(Lg, Zg, Ygrid, levels=20, cmap='hot')
    fig.colorbar(cs, ax=ax, label='Predicted AvgP03 (W)')
    mask_n = X[:, 0].astype(int) == n_val
    ax.scatter(X[mask_n, 1], X[mask_n, 2], c='cyan', s=12, alpha=0.65, edgecolors='none')
    ax.set_xlabel('L_RUN (mm)')
    ax.set_ylabel('z_first (mm)')
    ax.set_title(f'N_RUNS = {n_val}')
fig.suptitle('Response Surface: AvgP03 RF surrogate', y=1.02)
fig.tight_layout()
savefig(fig, 'response_surface.png')
plt.close(fig)

# Distribution by N_RUNS helps check whether a discrete choice dominates.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_vals, title in [(axes[0], y_p03, 'lifeAvgP03sphere (W)'), (axes[1], y_life, 'Lifetime (h)')]:
    box_data = [y_vals[X[:, 0].astype(int) == n] for n in n_runs_values]
    bp = ax.boxplot(box_data, tick_labels=[str(n) for n in n_runs_values], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.set_xlabel('N_RUNS')
    ax.set_ylabel(title)
    ax.set_title(f'{title} by N_RUNS')
    if title.startswith('lifeAvg'):
        ax.axhline(bl_p03, color='red', linestyle='--', label='Zigzag reference')
        ax.axhline(CYLINDER_OPT_FEASIBLE_AVG_P03_W, color='green', linestyle=':', label='Best cylinder feasible')
        ax.legend(fontsize=8)
    else:
        ax.axhline(LIFE_FLOOR, color='red', linestyle=':', label='Life floor')
        ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, 'boxplot_NRUNS.png')
plt.close(fig)
N_RUNS_CHOICES = [4, 6, 8, 10, 12, 14, 16]
L_bounds = (float(X[:, 1].min()), float(X[:, 1].max()))
z_bounds = (float(X[:, 2].min()), float(X[:, 2].max()))

def objective_for(n_runs):

    def _objective(x):
        point = np.array([[n_runs, x[0], x[1]]], dtype=float)
        pred_p03 = rf_p03.predict(point)[0]
        pred_life = rf_life.predict(point)[0]
        penalty = 1200.0 * max(0.0, LIFE_FLOOR - pred_life)
        return -pred_p03 + penalty
    return _objective

# Search the surrogate for a feasible next candidate.
best = None
for n_runs in N_RUNS_CHOICES:
    result = differential_evolution(objective_for(n_runs), [L_bounds, z_bounds], seed=42 + int(n_runs), maxiter=90, popsize=10, tol=1e-06, polish=True, workers=1)
    point = np.array([[n_runs, result.x[0], result.x[1]]], dtype=float)
    pred_p03 = rf_p03.predict(point)[0]
    pred_life = rf_life.predict(point)[0]
    candidate = (result.fun, n_runs, result.x[0], result.x[1], pred_p03, pred_life)
    if best is None or candidate[0] < best[0]:
        best = candidate
_, best_n, L_opt, z_opt, pred_p03_opt, pred_life_opt = best
feasible = y_life >= LIFE_FLOOR
feas_idx = np.where(feasible)[0]
if len(feas_idx) == 0:
    print('ERROR: no feasible measured zigzag designs under cylinder-based lifetime floor')
    sys.exit(1)
best_feas = int(feas_idx[np.argmax(y_p03[feas_idx])])
best_exp_x = X[best_feas]
best_exp_p03 = y_p03[best_feas]
best_exp_life = y_life[best_feas]
print('\nMeasured best feasible zigzag')
print(f'  Trial #{trial_ids[best_feas]}: N={best_exp_x[0]:.0f}, L_RUN={best_exp_x[1]:.1f} mm, z_first={best_exp_x[2]:.2f} mm, AvgP03={best_exp_p03:.1f} W, life={best_exp_life:.2f} h')
print(f'  Gain vs cylinder feasible best: {(best_exp_p03 / CYLINDER_OPT_FEASIBLE_AVG_P03_W - 1) * 100:+.1f}%')
print('\nSurrogate-advised next candidate')
print(f'  N={best_n}, L_RUN={L_opt:.1f} mm, z_first={z_opt:.2f} mm, pred AvgP03={pred_p03_opt:.1f} W, pred life={pred_life_opt:.2f} h')

# Compare the measured best design with the surrogate suggestion.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(y_life[~feasible], y_p03[~feasible], c='gray', s=16, alpha=0.35, label='Below life floor')
ax.scatter(y_life[feasible], y_p03[feasible], c='steelblue', s=20, alpha=0.7, label='Feasible')
ax.scatter([best_exp_life], [best_exp_p03], c='red', s=130, marker='*', label=f'Measured best #{trial_ids[best_feas]}', zorder=5)
ax.scatter([pred_life_opt], [pred_p03_opt], c='gold', s=100, marker='D', edgecolors='black', label='Surrogate next', zorder=5)
ax.axvline(LIFE_FLOOR, color='red', ls=':', alpha=0.65, label=f'Life floor {LIFE_FLOOR:.2f} h')
ax.set_xlabel('Lifetime (h)')
ax.set_ylabel('lifeAvgP03sphere (W)')
ax.set_title('Measured Best vs Surrogate Candidate')
ax.legend(fontsize=8)
ax = axes[1]
x_pos = np.arange(3)
w = 0.25
sur_vals = [best_n, L_opt, z_opt]
ax.bar(x_pos - w, bl_x, w, label='Zigzag reference', color='gray', alpha=0.6)
ax.bar(x_pos, best_exp_x, w, label=f'Measured best #{trial_ids[best_feas]}', color='steelblue')
ax.bar(x_pos + w, sur_vals, w, label='Surrogate next', color='gold', edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(['N_RUNS', 'L_RUN (mm)', 'z_first (mm)'])
ax.set_ylabel('Value')
ax.set_title('Design Parameter Comparison')
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, 'surrogate_optimal.png')
plt.close(fig)
opt_point = np.array([[best_n, L_opt, z_opt]], dtype=float)
shap_vals_opt = explainer_p03.shap_values(opt_point)
fig_force = plt.figure(figsize=(12, 3))
shap.force_plot(explainer_p03.expected_value, shap_vals_opt[0], opt_point[0], feature_names=DISPLAY_NAMES, matplotlib=True, show=False)
plt.title('SHAP Force Plot - Surrogate Candidate')
plt.tight_layout()
savefig(fig_force, 'shap_force_optimal.png')
plt.close(fig_force)
n_summary = []
for n in n_runs_values:
    mask = X[:, 0].astype(int) == n
    feas_mask = mask & feasible
    if np.any(feas_mask):
        n_summary.append((n, float(np.max(y_p03[feas_mask])), int(np.sum(feas_mask))))
n_summary.sort(key=lambda item: -item[1])
print('\nDesign interpretation summary')
print(f'  Most important feature for AvgP03: {DISPLAY_NAMES[int(np.argmax(rf_p03.feature_importances_))]}')
print(f'  Most important feature for Lifetime: {DISPLAY_NAMES[int(np.argmax(rf_life.feature_importances_))]}')
if n_summary:
    print('  Feasible N_RUNS ranking by best measured AvgP03:')
    for n, best_p03, count in n_summary:
        print(f'    N={n}: best AvgP03={best_p03:.1f} W from {count} feasible rows')
print('  The measured submission candidate remains the best feasible COMSOL row;')
print('  surrogate search is used as interpretability and next-sample guidance.')
print('\nsurrogate_analysis.py completed')
