"""
Parse zigzag optimization trials and generate submission-ready summary figures.

This script is intentionally read-only with respect to the optimization data.
Use environment variables to analyze a frozen snapshot while Optuna/COMSOL is
still appending to the live trials.csv:

  BFHC_TRIALS_CSV   input CSV path
  BFHC_FIG_DIR      output figure directory
"""

import csv
import math
import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get("BFHC_TRIALS_CSV", BASE_DIR / "data" / "trials.csv"))
FIG_DIR = pathlib.Path(os.environ.get("BFHC_FIG_DIR", BASE_DIR / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Contest lifetime constraint: optimized design lifetime must be at least
# 30% of the initial cylinder blank lifetime.
CYLINDER_BASELINE_LIFETIME_H = 115.5037
CYLINDER_BASELINE_AVG_P03_W = 428.5727
CYLINDER_OPT_FEASIBLE_AVG_P03_W = 559.6431
LIFE_FLOOR = 0.30 * CYLINDER_BASELINE_LIFETIME_H

# Zigzag Java baseline is a reference design, not the lifetime constraint.
ZIGZAG_JAVA_REF = {
    "Vwork_V": 100.0,
    "initialTmax_K": 3209.3,
    "lifetimeH": 7.1277,
    "initialP03sphere_W": 3711.29,
    "initialPradSphere_W": 3881.58,
    "lifeAvgP03sphere_W": 3539.04,
    "lifeAvgPradSphere_W": 3703.73,
    "selfViewLoss_pct": -3.77,
    "erosionSteps": 11.0,
    "elapsed_sec": 0.0,
}

SOLVED_STATUSES = {"OK", "PRUNE_LIFETIME"}
FIELDS_FLOAT = [
    "N_RUNS",
    "L_RUN_mm",
    "z_first_mm",
    "side_mm",
    "Vwork_V",
    "initialTmax_K",
    "lifetimeH",
    "initialP03sphere_W",
    "initialPradSphere_W",
    "lifeAvgP03sphere_W",
    "lifeAvgPradSphere_W",
    "selfViewLoss_pct",
    "erosionSteps",
    "elapsed_sec",
]


def safe_float(value):
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def is_complete_solved(row):
    if row.get("status") not in SOLVED_STATUSES:
        return False
    return all(math.isfinite(safe_float(row.get(f, ""))) for f in FIELDS_FLOAT)


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")


if not DATA.exists():
    print(f"ERROR: data file not found: {DATA}")
    sys.exit(1)

with DATA.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

complete = [r for r in rows if is_complete_solved(r)]
if not complete:
    print("ERROR: no complete solved zigzag trials found")
    sys.exit(1)

status_counts = {}
for row in rows:
    status_counts[row.get("status", "")] = status_counts.get(row.get("status", ""), 0) + 1

data = {f: np.array([safe_float(r[f]) for r in complete], dtype=float) for f in FIELDS_FLOAT}
trial_ids = np.array([int(r["trial"]) for r in complete], dtype=int)
avg_p03 = data["lifeAvgP03sphere_W"]
lifetime = data["lifetimeH"]
feasible = lifetime >= LIFE_FLOOR

trial0 = next((r for r in complete if r.get("trial") == "0"), None)
if trial0 is not None:
    zigzag_ref = {f: safe_float(trial0.get(f, "")) for f in FIELDS_FLOAT}
else:
    zigzag_ref = ZIGZAG_JAVA_REF

print("=" * 78)
print("Zigzag optimization summary")
print("=" * 78)
print(f"Input CSV: {DATA}")
print(f"Figure dir: {FIG_DIR}")
print(f"Total rows: {len(rows)}")
print("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
print(f"Complete solved rows used in plots: {len(complete)}")
print(f"Lifetime floor: {LIFE_FLOOR:.2f} h = cylinder baseline {CYLINDER_BASELINE_LIFETIME_H:.2f} h x 30%")
print(f"Feasible complete solved rows: {int(feasible.sum())}/{len(complete)}")
print(f"COMSOL time represented by complete rows: {data['elapsed_sec'].sum() / 3600:.1f} h")
print(f"Average complete-row runtime: {data['elapsed_sec'].mean():.0f} s")

order = np.argsort(-avg_p03)
print("\nTop 20 by lifecycle average 0-3 um sphere power")
header = (
    f"{'trial':>5} {'N':>3} {'L_mm':>7} {'zf_mm':>6} {'side':>7} "
    f"{'V':>7} {'Tmax':>8} {'life_h':>8} {'AvgP03':>9} {'feas':>5}"
)
print(header)
print("-" * len(header))
for i in order[:20]:
    print(
        f"{trial_ids[i]:>5} {data['N_RUNS'][i]:>3.0f} "
        f"{data['L_RUN_mm'][i]:>7.1f} {data['z_first_mm'][i]:>6.2f} "
        f"{data['side_mm'][i]:>7.3f} {data['Vwork_V'][i]:>7.2f} "
        f"{data['initialTmax_K'][i]:>8.1f} {lifetime[i]:>8.2f} "
        f"{avg_p03[i]:>9.1f} {('Y' if feasible[i] else 'N'):>5}"
    )

feas_idx = np.where(feasible)[0]
pareto_global = np.array([], dtype=int)
if len(feas_idx) > 0:
    feas_p03 = avg_p03[feas_idx]
    feas_life = lifetime[feas_idx]
    pareto_mask = np.zeros(len(feas_idx), dtype=bool)
    for i in range(len(feas_idx)):
        dominated = False
        for j in range(len(feas_idx)):
            if i == j:
                continue
            if (
                feas_p03[j] >= feas_p03[i]
                and feas_life[j] >= feas_life[i]
                and (feas_p03[j] > feas_p03[i] or feas_life[j] > feas_life[i])
            ):
                dominated = True
                break
        pareto_mask[i] = not dominated
    pareto_global = feas_idx[pareto_mask]

    print(f"\nPareto front among feasible rows: {len(pareto_global)} designs")
    p_order = np.argsort(-avg_p03[pareto_global])
    for rank, local_i in enumerate(p_order, 1):
        gi = pareto_global[local_i]
        print(
            f"  Rank {rank}: Trial #{trial_ids[gi]} | "
            f"N={data['N_RUNS'][gi]:.0f}, L={data['L_RUN_mm'][gi]:.1f} mm, "
            f"z_first={data['z_first_mm'][gi]:.2f} mm, side={data['side_mm'][gi]:.3f} mm | "
            f"AvgP03={avg_p03[gi]:.1f} W, life={lifetime[gi]:.2f} h"
        )

best_all_i = int(np.argmax(avg_p03))
best_feas_i = int(feas_idx[np.argmax(avg_p03[feas_idx])]) if len(feas_idx) else None

print("\nReference comparison")
print(f"Zigzag Java/reference baseline: AvgP03={zigzag_ref['lifeAvgP03sphere_W']:.1f} W, life={zigzag_ref['lifetimeH']:.2f} h")
print(f"Cylinder uniform baseline: AvgP03={CYLINDER_BASELINE_AVG_P03_W:.1f} W, life={CYLINDER_BASELINE_LIFETIME_H:.2f} h")
print(f"Cylinder feasible optimized best: AvgP03={CYLINDER_OPT_FEASIBLE_AVG_P03_W:.1f} W")
print(
    f"Best unconstrained zigzag row: Trial #{trial_ids[best_all_i]}, "
    f"AvgP03={avg_p03[best_all_i]:.1f} W, life={lifetime[best_all_i]:.2f} h"
)
if best_feas_i is not None:
    print(
        f"Best feasible zigzag row: Trial #{trial_ids[best_feas_i]}, "
        f"AvgP03={avg_p03[best_feas_i]:.1f} W, life={lifetime[best_feas_i]:.2f} h, "
        f"gain vs cylinder optimized best={(avg_p03[best_feas_i] / CYLINDER_OPT_FEASIBLE_AVG_P03_W - 1) * 100:+.1f}%"
    )

# 1. Convergence history.
fig, ax = plt.subplots(figsize=(10, 4))
sort_by_trial = np.argsort(trial_ids)
sorted_ids = trial_ids[sort_by_trial]
sorted_p03 = avg_p03[sort_by_trial]
running_best = np.maximum.accumulate(sorted_p03)
ax.scatter(sorted_ids, sorted_p03, s=14, alpha=0.5, c="steelblue", label="Complete solved rows")
ax.plot(sorted_ids, running_best, color="red", linewidth=2, label="Running best")
ax.axhline(zigzag_ref["lifeAvgP03sphere_W"], color="gray", linestyle="--", label="Zigzag reference")
ax.axhline(CYLINDER_OPT_FEASIBLE_AVG_P03_W, color="green", linestyle=":", label="Best cylinder feasible")
ax.set_xlabel("Trial #")
ax.set_ylabel("lifeAvgP03sphere (W)")
ax.set_title("Bayesian Optimization Convergence - Zigzag Family")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "optimization_history.png")
plt.close(fig)

# 2. Pareto front.
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(lifetime[~feasible], avg_p03[~feasible], s=16, alpha=0.3, c="gray", label="Below lifetime floor")
ax.scatter(lifetime[feasible], avg_p03[feasible], s=22, alpha=0.7, c="steelblue", label="Feasible")
if len(pareto_global) > 0:
    p_sort = np.argsort(lifetime[pareto_global])
    ax.plot(
        lifetime[pareto_global][p_sort],
        avg_p03[pareto_global][p_sort],
        "r-o",
        markersize=5,
        label="Feasible Pareto front",
        zorder=5,
    )
ax.scatter([zigzag_ref["lifetimeH"]], [zigzag_ref["lifeAvgP03sphere_W"]], s=120, marker="*", c="gold", edgecolors="black", label="Zigzag reference", zorder=8)
ax.axvline(LIFE_FLOOR, color="red", linestyle=":", alpha=0.6, label=f"Life floor ({LIFE_FLOOR:.2f} h)")
ax.set_xlabel("Lifetime (h)")
ax.set_ylabel("lifeAvgP03sphere (W)")
ax.set_title("Radiation Power vs Lifetime - Zigzag Family")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
savefig(fig, "pareto_front.png")
plt.close(fig)

# 3. Parallel coordinates for top 30.
top30 = order[: min(30, len(complete))]
fig, ax = plt.subplots(figsize=(10, 5))
param_names = ["N_RUNS", "L_RUN_mm", "z_first_mm", "side_mm"]
param_labels = ["N_RUNS", "L_RUN (mm)", "z_first (mm)", "side (mm)"]
colors = plt.cm.RdYlGn(np.linspace(1, 0, len(top30)))
norm_data = {}
for p in param_names:
    vals = data[p][top30]
    vmin, vmax = vals.min(), vals.max()
    norm_data[p] = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(vals, 0.5)
for idx in range(len(top30)):
    vals = [norm_data[p][idx] for p in param_names]
    ax.plot(range(len(param_names)), vals, color=colors[idx], alpha=0.75, linewidth=1.4)
ax.set_xticks(range(len(param_names)))
ax.set_xticklabels(param_labels)
ax.set_ylabel("Normalized value among top rows")
ax.set_title("Top Designs - Parallel Coordinates")
fig.tight_layout()
savefig(fig, "parallel_coordinate.png")
plt.close(fig)

# 4. N_RUNS vs L_RUN landscape.
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(data["N_RUNS"], data["L_RUN_mm"], c=avg_p03, cmap="hot", s=26, alpha=0.75)
fig.colorbar(sc, ax=ax, label="lifeAvgP03sphere (W)")
ax.set_xlabel("N_RUNS")
ax.set_ylabel("L_RUN (mm)")
ax.set_title("AvgP03 Landscape: N_RUNS vs L_RUN")
fig.tight_layout()
savefig(fig, "scatter_N_L.png")
plt.close(fig)

print("\nparse_results.py completed")
