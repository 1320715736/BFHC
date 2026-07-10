"""
Rematch Optuna search for the 8-segment cylinder family.

This script intentionally does not reuse first-round trials or databases.
It writes a new C1 study under ``cylinder_family/ML/data`` and evaluates every
candidate with the current rematch COMSOL runner.
"""

import csv
import math
import os
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState

from comsol_runner import COMSOLRunner


ML_DIR = Path(r"D:\VScode\project\BFHC\cylinder_family\ML")
DATA_DIR = ML_DIR / "data"

DB_PATH = DATA_DIR / "rematch_c1_optuna.db"
CSV_PATH = DATA_DIR / "rematch_c1_trials.csv"
STUDY_NAME = "cylinder_rematch_c1"

R0_MM = 2.5
SEG_COUNT = 8
R_MIN_MM = 0.8
R_MAX_MM = 4.5
TOTAL_SQ = 4.0 * R0_MM**2

# Current rematch official cylinder baseline.
BASELINE_LIFETIME_H = 242.07911958397654
BASELINE_E_J = 105597676.11285222
BASELINE_U_PCT = 148.69276459515976

# C1 is deliberately small. Override from PowerShell if needed:
#   $env:BFHC_N_TRIALS='30'
N_TRIALS = int(os.getenv("BFHC_N_TRIALS", "30"))
MIN_LIFETIME_RATIO = float(os.getenv("BFHC_MIN_LIFETIME_RATIO", "0.50"))
MIN_LIFETIME_H = BASELINE_LIFETIME_H * MIN_LIFETIME_RATIO
MAX_U_RATIO = float(os.getenv("BFHC_MAX_U_RATIO", "1.20"))
MAX_U_PCT = BASELINE_U_PCT * MAX_U_RATIO
U_PENALTY_WEIGHT = float(os.getenv("BFHC_U_PENALTY_WEIGHT", "0.25"))

runner: COMSOLRunner | None = None


CSV_HEADER = [
    "trial", "r1_mm", "r2_mm", "r3_mm", "r4_mm",
    "Vwork_V", "initialTmax_K", "Tmin_K", "Tmean_K", "U_pct",
    "maxErosionTmax_K", "lifetimeH", "R_L_pct", "eta_L_pct",
    "initialP03sphere_W", "initialPradSphere_W",
    "lifeAvgP03sphere_W", "lifeAvgPradSphere_W",
    "lifeTotalP03sphere_J", "eta_E_pct",
    "objectiveScore", "uPenalty_pctpt", "U_limit_pct",
    "selfViewLoss_pct", "failureReached", "erosionSteps",
    "overtempStep", "overtempTimeH", "overtempTmax_K",
    "runnerStatus", "status", "voltagePolicy", "voltageObjective",
    "elapsed_sec",
]


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def compute_r4(r1_mm, r2_mm, r3_mm):
    remainder = TOTAL_SQ - r1_mm**2 - r2_mm**2 - r3_mm**2
    if remainder < R_MIN_MM**2:
        return None
    r4 = math.sqrt(remainder)
    if r4 > R_MAX_MM:
        return None
    return r4


def init_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_csv(row_dict):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, "") for h in CSV_HEADER])


def compact_status(status, limit=240):
    return " ".join(str(status).split())[:limit]


def add_scores(row):
    lifetime_h = float(row.get("lifetimeH", float("nan")))
    energy_j = float(row.get("lifeTotalP03sphere_J", float("nan")))
    u_pct = float(row.get("U_pct", float("nan")))

    r_l_pct = lifetime_h / BASELINE_LIFETIME_H * 100.0
    eta_l_pct = (lifetime_h - BASELINE_LIFETIME_H) / BASELINE_LIFETIME_H * 100.0
    eta_e_pct = (energy_j - BASELINE_E_J) / BASELINE_E_J * 100.0
    u_penalty = max(0.0, u_pct - BASELINE_U_PCT)
    score = eta_e_pct - U_PENALTY_WEIGHT * u_penalty

    row.update({
        "R_L_pct": r_l_pct,
        "eta_L_pct": eta_l_pct,
        "eta_E_pct": eta_e_pct,
        "objectiveScore": score,
        "uPenalty_pctpt": u_penalty,
        "U_limit_pct": MAX_U_PCT,
    })
    return row


def prune_with_row(row, message):
    append_csv(row)
    raise optuna.TrialPruned(message)


def recover_interrupted_trials(study):
    """Put interrupted RUNNING trials back into WAITING so Optuna retries them.

    Optuna leaves a trial in RUNNING if the Python/COMSOL process is killed.
    Without this recovery, a resumed search can allocate a fresh trial and skip
    the interrupted geometry. Resetting to WAITING preserves the same trial
    number and parameters, so the next optimize call evaluates that point first.
    """
    running_trials = study.get_trials(
        deepcopy=False,
        states=(TrialState.RUNNING,),
    )
    if not running_trials:
        return 0

    recovered = 0
    for frozen in running_trials:
        if study._storage.set_trial_state_values(
                frozen._trial_id, TrialState.WAITING):
            recovered += 1
            print(f"Recovered interrupted trial #{frozen.number} "
                  "from RUNNING to WAITING.")
        else:
            print(f"WARN: failed to recover RUNNING trial #{frozen.number}.")
    return recovered


def count_finished_trials(study):
    return sum(
        1 for trial in study.get_trials(deepcopy=False)
        if trial.state.is_finished()
    )


def objective(trial):
    global runner
    if runner is None:
        raise RuntimeError("COMSOL runner is not initialized.")

    t_start = time.time()
    trial_num = trial.number

    r1 = trial.suggest_float("r1_mm", R_MIN_MM, R_MAX_MM)

    r2_max_sq = TOTAL_SQ - r1**2 - 2 * R_MIN_MM**2
    r2_min_sq = TOTAL_SQ - r1**2 - 2 * R_MAX_MM**2
    r2_lo = max(R_MIN_MM, math.sqrt(max(0.0, r2_min_sq)))
    r2_hi = min(R_MAX_MM, math.sqrt(max(R_MIN_MM**2, r2_max_sq)))
    r2 = trial.suggest_float("r2_mm", r2_lo, r2_hi)

    r3_max_sq = TOTAL_SQ - r1**2 - r2**2 - R_MIN_MM**2
    r3_min_sq = TOTAL_SQ - r1**2 - r2**2 - R_MAX_MM**2
    r3_lo = max(R_MIN_MM, math.sqrt(max(0.0, r3_min_sq)))
    r3_hi = min(R_MAX_MM, math.sqrt(max(R_MIN_MM**2, r3_max_sq)))
    r3 = trial.suggest_float("r3_mm", r3_lo, r3_hi)

    r4 = compute_r4(r1, r2, r3)
    if r4 is None:
        raise optuna.TrialPruned("Invalid volume-conserving geometry.")

    radii_mm = [r1, r2, r3, r4, r4, r3, r2, r1]
    radii_m = [r * 1e-3 for r in radii_mm]

    print("\n" + "=" * 60)
    print(f"Trial {trial_num}: r = "
          f"[{', '.join(f'{r:.3f}' for r in radii_mm)}] mm")
    print("=" * 60)

    base_row = {
        "trial": trial_num,
        "r1_mm": r1,
        "r2_mm": r2,
        "r3_mm": r3,
        "r4_mm": r4,
        "U_limit_pct": MAX_U_PCT,
    }

    try:
        result = runner.evaluate(radii_m)
    except Exception as exc:
        elapsed = time.time() - t_start
        row = dict(base_row)
        row.update({
            "runnerStatus": "EXCEPTION",
            "status": compact_status(f"ERROR: {exc}"),
            "elapsed_sec": round(elapsed, 1),
        })
        append_csv(row)
        raise optuna.TrialPruned(f"COMSOL error: {exc}")

    elapsed = time.time() - t_start
    row = dict(base_row)
    row.update(result)
    row["runnerStatus"] = compact_status(result.get("status", "UNKNOWN"))
    row["elapsed_sec"] = round(elapsed, 1)

    if result.get("status") != "OK":
        row["status"] = row["runnerStatus"]
        prune_with_row(row, row["runnerStatus"])

    required = [
        "lifetimeH", "lifeTotalP03sphere_J", "U_pct",
        "Vwork_V", "maxErosionTmax_K",
    ]
    invalid = [key for key in required if not finite_number(result.get(key))]
    if invalid:
        row["status"] = "PRUNE_INVALID_METRIC"
        prune_with_row(row, "Invalid metric(s): " + ", ".join(invalid))

    row = add_scores(row)
    lifetime_h = float(row["lifetimeH"])
    u_pct = float(row["U_pct"])
    score = float(row["objectiveScore"])

    if lifetime_h < MIN_LIFETIME_H:
        row["status"] = "PRUNE_LIFETIME"
        prune_with_row(
            row,
            f"R_L {row['R_L_pct']:.2f}% < "
            f"{MIN_LIFETIME_RATIO * 100.0:.2f}%",
        )

    if u_pct > MAX_U_PCT:
        row["status"] = "PRUNE_U"
        prune_with_row(row, f"U {u_pct:.2f}% > {MAX_U_PCT:.2f}%")

    row["status"] = "OK"
    append_csv(row)

    print(f"  Vwork={result['Vwork_V']:.4f}V  "
          f"L={lifetime_h:.2f}h  R_L={row['R_L_pct']:.2f}%  "
          f"eta_E={row['eta_E_pct']:.2f}%  U={u_pct:.2f}%  "
          f"score={score:.2f}  [{elapsed:.0f}s]")

    return score


def main():
    global runner

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_csv()

    print("BFHC cylinder rematch C1 search")
    print(f"CSV output:       {CSV_PATH}")
    print(f"Optuna DB:        sqlite:///{DB_PATH}")
    print(f"Study name:       {STUDY_NAME}")
    print(f"Trials target:    {N_TRIALS}")
    print(f"Lifetime floor:   {MIN_LIFETIME_H:.4f} h "
          f"({MIN_LIFETIME_RATIO * 100.0:.1f}% of baseline)")
    print(f"U limit:          {MAX_U_PCT:.4f}% "
          f"({MAX_U_RATIO:.2f}x baseline)")
    print(f"U penalty weight: {U_PENALTY_WEIGHT}")
    print()

    runner = COMSOLRunner()
    runner.start()

    try:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=f"sqlite:///{DB_PATH}",
            direction="maximize",
            load_if_exists=True,
        )

        recover_interrupted_trials(study)
        finished_trials = count_finished_trials(study)
        n_remaining = N_TRIALS - finished_trials
        if n_remaining <= 0:
            print(f"Already have {finished_trials} finished trials, "
                  f"target is {N_TRIALS}. Done.")
        else:
            print(f"Resuming from {finished_trials} finished trials, "
                  f"running {n_remaining} more...")
            study.optimize(objective, n_trials=n_remaining)

        print("\n" + "=" * 60)
        print("  C1 SEARCH COMPLETE")
        print("=" * 60)
        try:
            best = study.best_trial
        except ValueError:
            best = None
        if best is None:
            print("No completed trial yet. Check the CSV for prune reasons.")
        else:
            print(f"Best trial:     #{best.number}")
            print(f"Best score:     {best.value:.4f}")
            print(f"Best params:    r1={best.params['r1_mm']:.3f}  "
                  f"r2={best.params['r2_mm']:.3f}  "
                  f"r3={best.params['r3_mm']:.3f} mm")
            r4 = compute_r4(
                best.params["r1_mm"],
                best.params["r2_mm"],
                best.params["r3_mm"],
            )
            if r4 is not None:
                radii = [
                    best.params["r1_mm"], best.params["r2_mm"],
                    best.params["r3_mm"], r4, r4,
                    best.params["r3_mm"], best.params["r2_mm"],
                    best.params["r1_mm"],
                ]
                print("Full radii (mm): "
                      f"[{', '.join(f'{r:.3f}' for r in radii)}]")
        print(f"\nTotal trials:   {len(study.trials)}")
        print(f"Results saved:  {CSV_PATH}")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
