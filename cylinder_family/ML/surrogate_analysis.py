"""
Cylinder-family surrogate model and SHAP analysis.

Important modeling choice:
  r1, r2, r3 are the independent Optuna variables.
  r4 is derived from volume conservation and is not used as an independent
  feature in RF/SHAP:

      r4 = sqrt(4*r0^2 - r1^2 - r2^2 - r3^2), r0 = 2.5 mm

The script keeps the existing figure filenames so downstream documents remain
stable:
  figures/feature_importance.png
  figures/shap_summary_p03.png
  figures/shap_summary_life.png
  figures/shap_dependence_r3.png
  figures/response_surface.png
  figures/surrogate_optimal.png
  figures/shap_force_optimal.png
"""

import csv
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "trials.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOLVED_STATUSES = {"OK", "PRUNE_LIFETIME"}
FEATURE_NAMES = ["r1_mm", "r2_mm", "r3_mm"]
DISPLAY_NAMES = ["r1", "r2", "r3"]
R0_MM = 2.5
TOTAL_SQ = 4.0 * R0_MM**2
R_MIN = 0.8
R_MAX = 4.5
BASELINE_LIFETIME_H = 115.5037
LIFE_FLOOR = 0.30 * BASELINE_LIFETIME_H


def safe_float(value):
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def derived_r4(r1, r2, r3):
    r4_sq = TOTAL_SQ - r1 * r1 - r2 * r2 - r3 * r3
    if r4_sq < 0.0:
        return float("nan")
    return math.sqrt(r4_sq)


def complete_row(row):
    needed = FEATURE_NAMES + ["r4_mm", "lifeAvgP03sphere_W", "lifetimeH"]
    if row.get("status") not in SOLVED_STATUSES:
        return False
    return all(math.isfinite(safe_float(row.get(k, ""))) for k in needed)


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")


with DATA.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

ok = [r for r in rows if complete_row(r)]
if len(ok) < 8:
    raise RuntimeError(f"Not enough complete solved rows: {len(ok)}")

X = np.array([[safe_float(r[f]) for f in FEATURE_NAMES] for r in ok], dtype=float)
r4_csv = np.array([safe_float(r["r4_mm"]) for r in ok], dtype=float)
y_p03 = np.array([safe_float(r["lifeAvgP03sphere_W"]) for r in ok], dtype=float)
y_life = np.array([safe_float(r["lifetimeH"]) for r in ok], dtype=float)
trial_ids = np.array([int(r["trial"]) for r in ok], dtype=int)

print("=" * 72)
print("Cylinder surrogate + SHAP analysis (independent features only)")
print("=" * 72)
print(f"Rows used: {len(ok)}")
print("Features: r1, r2, r3")
print("Derived variable excluded from SHAP: r4 = sqrt(4*r0^2-r1^2-r2^2-r3^2)")
print(f"Lifetime floor: {LIFE_FLOOR:.2f} h")

rf_p03 = RandomForestRegressor(
    n_estimators=260,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1,
)
rf_life = RandomForestRegressor(
    n_estimators=260,
    max_depth=10,
    min_samples_leaf=2,
    random_state=43,
    n_jobs=1,
)

cv_p03 = cross_val_score(rf_p03, X, y_p03, cv=5, scoring="r2")
cv_life = cross_val_score(rf_life, X, y_life, cv=5, scoring="r2")
print(f"RF AvgP03 R2: {cv_p03.mean():.3f} +/- {cv_p03.std():.3f}")
print(f"RF Lifetime R2: {cv_life.mean():.3f} +/- {cv_life.std():.3f}")

rf_p03.fit(X, y_p03)
rf_life.fit(X, y_life)

# Feature importance.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, rf, title in [
    (axes[0], rf_p03, "lifeAvgP03sphere"),
    (axes[1], rf_life, "Lifetime"),
]:
    imp = rf.feature_importances_
    order = np.argsort(imp)
    ax.barh([DISPLAY_NAMES[i] for i in order], imp[order], color="steelblue")
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(f"Independent-variable importance - {title}")
fig.tight_layout()
savefig(fig, "feature_importance.png")
plt.close(fig)

print("\nFeature importance (AvgP03)")
for i in np.argsort(-rf_p03.feature_importances_):
    print(f"  {DISPLAY_NAMES[i]}: {rf_p03.feature_importances_[i]:.3f}")
print("Feature importance (Lifetime)")
for i in np.argsort(-rf_life.feature_importances_):
    print(f"  {DISPLAY_NAMES[i]}: {rf_life.feature_importances_[i]:.3f}")

# SHAP.
explainer_p03 = shap.TreeExplainer(rf_p03)
shap_p03 = explainer_p03.shap_values(X)
explainer_life = shap.TreeExplainer(rf_life)
shap_life = explainer_life.shap_values(X)

fig = plt.figure(figsize=(8, 4.5))
shap.summary_plot(shap_p03, X, feature_names=DISPLAY_NAMES, show=False)
plt.title("SHAP Summary - lifeAvgP03sphere (r1-r3 only)")
plt.tight_layout()
savefig(fig, "shap_summary_p03.png")
plt.close(fig)

fig = plt.figure(figsize=(8, 4.5))
shap.summary_plot(shap_life, X, feature_names=DISPLAY_NAMES, show=False)
plt.title("SHAP Summary - Lifetime (r1-r3 only)")
plt.tight_layout()
savefig(fig, "shap_summary_life.png")
plt.close(fig)

fig = plt.figure(figsize=(8, 5))
shap.dependence_plot("r3", shap_p03, X, feature_names=DISPLAY_NAMES, show=False)
plt.title("SHAP Dependence - r3 effect on AvgP03")
plt.tight_layout()
savefig(fig, "shap_dependence_r3.png")
plt.close(fig)

# Response surface: r1 vs r3 while r2 is fixed at the sample median.
r2_med = float(np.median(X[:, 1]))
r1_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 80)
r3_grid = np.linspace(X[:, 2].min(), X[:, 2].max(), 80)
R1g, R3g = np.meshgrid(r1_grid, r3_grid)
Xgrid = np.column_stack([R1g.ravel(), np.full(R1g.size, r2_med), R3g.ravel()])
Z = rf_p03.predict(Xgrid)
valid = []
for r1, r2, r3 in Xgrid:
    r4 = derived_r4(r1, r2, r3)
    valid.append(math.isfinite(r4) and R_MIN <= r4 <= R_MAX)
Z = np.where(np.array(valid), Z, np.nan).reshape(R1g.shape)

fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(R1g, R3g, Z, levels=20, cmap="hot")
fig.colorbar(cs, ax=ax, label="Predicted AvgP03 (W)")
ax.scatter(X[:, 0], X[:, 2], c="cyan", s=10, alpha=0.55, edgecolors="none")
ax.set_xlabel("r1 (mm)")
ax.set_ylabel("r3 (mm)")
ax.set_title(f"Response Surface (r2={r2_med:.2f} mm; r4 derived)")
fig.tight_layout()
savefig(fig, "response_surface.png")
plt.close(fig)


def feasible_geometry(r1, r2, r3):
    if not (R_MIN <= r1 <= R_MAX and R_MIN <= r2 <= R_MAX and R_MIN <= r3 <= R_MAX):
        return None
    r4 = derived_r4(r1, r2, r3)
    if not (math.isfinite(r4) and R_MIN <= r4 <= R_MAX):
        return None
    return r4


def constrained_objective(x):
    r1, r2, r3 = x
    r4 = feasible_geometry(r1, r2, r3)
    if r4 is None:
        return 1e6
    point = np.array([[r1, r2, r3]], dtype=float)
    pred_p03 = rf_p03.predict(point)[0]
    pred_life = rf_life.predict(point)[0]
    penalty = 1200.0 * max(0.0, LIFE_FLOOR - pred_life)
    return -pred_p03 + penalty


bounds = [(R_MIN, R_MAX), (R_MIN, R_MAX), (R_MIN, R_MAX)]
best_result = None
for seed in range(3):
    result = differential_evolution(
        constrained_objective,
        bounds,
        seed=seed,
        maxiter=120,
        popsize=10,
        tol=1e-6,
        polish=True,
        workers=1,
    )
    if best_result is None or result.fun < best_result.fun:
        best_result = result

r1_opt, r2_opt, r3_opt = best_result.x
r4_opt = derived_r4(r1_opt, r2_opt, r3_opt)
opt_point = np.array([[r1_opt, r2_opt, r3_opt]], dtype=float)
pred_p03_opt = rf_p03.predict(opt_point)[0]
pred_life_opt = rf_life.predict(opt_point)[0]

feasible = y_life >= LIFE_FLOOR
feas_idx = np.where(feasible)[0]
best_feas = int(feas_idx[np.argmax(y_p03[feas_idx])])
best_exp_x = X[best_feas]
best_exp_r4 = r4_csv[best_feas]
best_exp_p03 = y_p03[best_feas]
best_exp_life = y_life[best_feas]

print("\nMeasured best feasible")
print(
    f"  Trial #{trial_ids[best_feas]}: "
    f"[r1,r2,r3,r4]=[{best_exp_x[0]:.3f},{best_exp_x[1]:.3f},"
    f"{best_exp_x[2]:.3f},{best_exp_r4:.3f}] mm, "
    f"AvgP03={best_exp_p03:.1f} W, Lifetime={best_exp_life:.1f} h"
)
print("Surrogate-advised candidate")
print(
    f"  [r1,r2,r3,r4]=[{r1_opt:.3f},{r2_opt:.3f},{r3_opt:.3f},{r4_opt:.3f}] mm, "
    f"pred AvgP03={pred_p03_opt:.1f} W, pred Lifetime={pred_life_opt:.1f} h"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(y_life, y_p03, c="steelblue", s=16, alpha=0.55, label="Solved trials")
ax.scatter([best_exp_life], [best_exp_p03], c="red", s=130, marker="*", label=f"Measured best #{trial_ids[best_feas]}", zorder=5)
ax.scatter([pred_life_opt], [pred_p03_opt], c="gold", s=100, marker="D", edgecolors="black", label="Surrogate candidate", zorder=5)
ax.axvline(LIFE_FLOOR, color="gray", linestyle="--", alpha=0.7, label=f"Life floor {LIFE_FLOOR:.1f} h")
ax.set_xlabel("Lifetime (h)")
ax.set_ylabel("AvgP03sphere (W)")
ax.set_title("Surrogate Candidate vs Measured Best")
ax.legend(fontsize=8)

ax = axes[1]
labels = ["r1", "r2", "r3", "r4*"]
x = np.arange(len(labels))
w = 0.34
ax.bar(x - w / 2, [best_exp_x[0], best_exp_x[1], best_exp_x[2], best_exp_r4], w, color="steelblue", label=f"Measured best #{trial_ids[best_feas]}")
ax.bar(x + w / 2, [r1_opt, r2_opt, r3_opt, r4_opt], w, color="gold", edgecolor="black", label="Surrogate candidate")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Radius (mm)")
ax.set_title("Design Radii (r4 is derived)")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "surrogate_optimal.png")
plt.close(fig)

shap_vals_opt = explainer_p03.shap_values(opt_point)
fig_force = plt.figure(figsize=(12, 3))
shap.force_plot(
    explainer_p03.expected_value,
    shap_vals_opt[0],
    opt_point[0],
    feature_names=DISPLAY_NAMES,
    matplotlib=True,
    show=False,
)
plt.title("SHAP Force Plot - Surrogate Candidate (r1-r3 only)")
plt.tight_layout()
savefig(fig_force, "shap_force_optimal.png")
plt.close(fig_force)

print("\nsurrogate_analysis.py completed")
