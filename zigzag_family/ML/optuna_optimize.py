"""C8 local Optuna search for the zigzag family.

This script intentionally does not reuse first-round or C6 trials/databases.
It searches the verified trial-19 neighborhood with the current rematch COMSOL
runner and scores against the official B2 cylinder baseline.
"""

import csv
import math
import os
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState

from zigzag_runner import COMSOLRunner


ML_DIR = Path(r"D:\VScode\project\BFHC\zigzag_family\ML")
DATA_DIR = ML_DIR / "data"

DB_PATH = DATA_DIR / os.getenv("BFHC_DB_FILE", "rematch_c8_local_optuna.db")
CSV_PATH = DATA_DIR / os.getenv("BFHC_CSV_FILE", "rematch_c8_local_trials.csv")
STUDY_NAME = os.getenv("BFHC_STUDY_NAME", "zigzag_rematch_c8_local")

N_RUNS_CHOICES = [
    int(item) for item in os.getenv(
        "BFHC_N_RUNS_CHOICES", "8,10,12,14").split(",")
    if item.strip()
]
L_RUN_MIN_MM = float(os.getenv("BFHC_L_RUN_MIN_MM", "70.0"))
L_RUN_MAX_MM = float(os.getenv("BFHC_L_RUN_MAX_MM", "120.0"))
Z_FIRST_MIN_MM = float(os.getenv("BFHC_Z_FIRST_MIN_MM", "1.6"))
Z_FIRST_MAX_MM = float(os.getenv("BFHC_Z_FIRST_MAX_MM", "2.5"))

# Verified C8 trial-19 candidate used as the first queued geometry.
REFERENCE_N_RUNS = int(os.getenv("BFHC_REFERENCE_N_RUNS", "12"))
REFERENCE_L_RUN_MM = float(os.getenv(
    "BFHC_REFERENCE_L_RUN_MM", "89.09934228871522"))
REFERENCE_Z_FIRST_MM = float(os.getenv(
    "BFHC_REFERENCE_Z_FIRST_MM", "2.10495899560028"))

# Current rematch official cylinder baseline.
BASELINE_LIFETIME_H = 242.07911958397654
BASELINE_E_J = 105597676.11285222
BASELINE_U_PCT = 148.69276459515976

# C8 local search. Override from PowerShell if needed:
#   $env:BFHC_N_TRIALS='30'
N_TRIALS = int(os.getenv("BFHC_N_TRIALS", "20"))
MAX_LIFETIME_H = float(os.getenv("BFHC_MAX_LIFETIME_H", "500.0"))
MIN_LIFETIME_RATIO = float(os.getenv("BFHC_MIN_LIFETIME_RATIO", "0.50"))
MIN_LIFETIME_H = BASELINE_LIFETIME_H * MIN_LIFETIME_RATIO
MAX_U_RATIO = float(os.getenv("BFHC_MAX_U_RATIO", "1.20"))
MAX_U_PCT = BASELINE_U_PCT * MAX_U_RATIO
U_PENALTY_WEIGHT = float(os.getenv("BFHC_U_PENALTY_WEIGHT", "0.25"))

runner: COMSOLRunner | None = None


CSV_HEADER = [
    "trial", "N_RUNS", "L_RUN_mm", "z_first_mm", "side_mm",
    "pathLength_mm", "maxLifetimeCap_h", "Vwork_V", "initialTmax_K",
    "Tmin_K", "Tmean_K", "U_pct", "maxErosionTmax_K", "lifetimeH",
    "R_L_pct", "eta_L_pct",
    "initialP03sphere_W", "initialPradSphere_W",
    "lifeAvgP03sphere_W", "lifeAvgPradSphere_W",
    "lifeTotalP03sphere_J", "eta_E_pct",
    "objectiveScore", "uPenalty_pctpt", "U_limit_pct",
    "selfViewLoss_pct", "failureReached", "capLimited", "erosionSteps",
    "overtempStep", "overtempTimeH", "overtempTmax_K",
    "runnerStatus", "status", "voltagePolicy", "voltageObjective",
    "voltageCandidateCount", "voltageMaxSafe_V", "voltageScanSummary",
    "elapsed_sec",
]


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


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


def bool_value(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def add_cap_status(row):
    lifetime_h = row.get("lifetimeH")
    cap_limited = False
    if finite_number(lifetime_h):
        cap_limited = (
            not bool_value(row.get("failureReached"))
            and float(lifetime_h) >= MAX_LIFETIME_H - 1e-6
        )
    row["capLimited"] = cap_limited
    row["maxLifetimeCap_h"] = MAX_LIFETIME_H
    return row


def prune_with_row(row, message):
    append_csv(row)
    raise optuna.TrialPruned(message)


def recover_interrupted_trials(study):
    """Put interrupted RUNNING trials back into WAITING so Optuna retries them."""
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

    n_runs = trial.suggest_categorical("N_RUNS", N_RUNS_CHOICES)
    l_run_mm = trial.suggest_float("L_RUN_mm", L_RUN_MIN_MM, L_RUN_MAX_MM)
    z_first_mm = trial.suggest_float(
        "z_first_mm", Z_FIRST_MIN_MM, Z_FIRST_MAX_MM)

    l_run_m = l_run_mm * 1e-3
    z_first_m = z_first_mm * 1e-3
    side, _, path_len = runner.compute_side_and_blocks(
        n_runs, l_run_m, z_first_m)

    print("\n" + "=" * 60)
    print(f"Trial {trial_num}: N={n_runs}  L={l_run_mm:.2f} mm  "
          f"zf={z_first_mm:.3f} mm  side={side * 1e3:.4f} mm  "
          f"path={path_len * 1e3:.2f} mm")
    print("=" * 60)

    base_row = {
        "trial": trial_num,
        "N_RUNS": n_runs,
        "L_RUN_mm": l_run_mm,
        "z_first_mm": z_first_mm,
        "side_mm": side * 1e3,
        "pathLength_mm": path_len * 1e3,
        "maxLifetimeCap_h": MAX_LIFETIME_H,
        "U_limit_pct": MAX_U_PCT,
    }

    try:
        result = runner.evaluate(
            N_RUNS=n_runs,
            L_RUN_m=l_run_m,
            z_first_m=z_first_m,
        )
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

    row = add_cap_status(row)
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

    print("BFHC zigzag rematch C8 local search")
    print(f"CSV output:       {CSV_PATH}")
    print(f"Optuna DB:        sqlite:///{DB_PATH}")
    print(f"Study name:       {STUDY_NAME}")
    print(f"Trials target:    {N_TRIALS}")
    print(f"Search space:     N={N_RUNS_CHOICES}, "
          f"L=[{L_RUN_MIN_MM}, {L_RUN_MAX_MM}] mm, "
          f"zf=[{Z_FIRST_MIN_MM}, {Z_FIRST_MAX_MM}] mm")
    print(f"Reference queued: N={REFERENCE_N_RUNS}, "
          f"L={REFERENCE_L_RUN_MM} mm, zf={REFERENCE_Z_FIRST_MM} mm")
    print(f"Lifetime cap:     {MAX_LIFETIME_H:.1f} h")
    print(f"Lifetime floor:   {MIN_LIFETIME_H:.4f} h "
          f"({MIN_LIFETIME_RATIO * 100.0:.1f}% of B2 baseline)")
    print(f"U limit:          {MAX_U_PCT:.4f}% "
          f"({MAX_U_RATIO:.2f}x B2 baseline)")
    print(f"U penalty weight: {U_PENALTY_WEIGHT}")
    print()

    runner = COMSOLRunner()
    runner.max_lifetime_h = MAX_LIFETIME_H
    runner.start()
    runner.max_lifetime_h = MAX_LIFETIME_H

    try:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=f"sqlite:///{DB_PATH}",
            direction="maximize",
            load_if_exists=True,
        )

        if len(study.trials) == 0:
            study.enqueue_trial({
                "N_RUNS": REFERENCE_N_RUNS,
                "L_RUN_mm": REFERENCE_L_RUN_MM,
                "z_first_mm": REFERENCE_Z_FIRST_MM,
            })

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
        print("  C8 LOCAL SEARCH COMPLETE")
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
            print(f"Best params:    N={best.params['N_RUNS']}  "
                  f"L={best.params['L_RUN_mm']:.2f} mm  "
                  f"zf={best.params['z_first_mm']:.3f} mm")
            side, _, path_len = runner.compute_side_and_blocks(
                best.params["N_RUNS"],
                best.params["L_RUN_mm"] * 1e-3,
                best.params["z_first_mm"] * 1e-3,
            )
            print(f"Best side:      {side * 1e3:.4f} mm")
            print(f"Best path len:  {path_len * 1e3:.2f} mm")
        print(f"\nTotal trials:   {len(study.trials)}")
        print(f"Results saved:  {CSV_PATH}")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
