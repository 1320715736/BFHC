"""
Rematch Optuna search for the 8-segment cylinder family.

This script intentionally does not reuse first-round trials or databases.
It writes a C4 local-expansion study under ``cylinder_family/ML/data`` and
evaluates every candidate with the current rematch COMSOL runner.
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

DB_PATH = DATA_DIR / os.getenv("BFHC_DB_FILE", "rematch_c4_optuna.db")
CSV_PATH = DATA_DIR / os.getenv("BFHC_CSV_FILE", "rematch_c4_trials.csv")
STUDY_NAME = os.getenv("BFHC_STUDY_NAME", "cylinder_rematch_c4")

R0_MM = 2.5
SEG_COUNT = 8
GEOM_R_MIN_MM = 0.8
GEOM_R_MAX_MM = 4.5
TOTAL_SQ = 4.0 * R0_MM**2

# C4 local expansion bounds from the C3 analysis.
R1_MIN_MM = float(os.getenv("BFHC_R1_MIN_MM", "1.8"))
R1_MAX_MM = float(os.getenv("BFHC_R1_MAX_MM", "2.8"))
R2_MIN_MM = float(os.getenv("BFHC_R2_MIN_MM", "1.6"))
R2_MAX_MM = float(os.getenv("BFHC_R2_MAX_MM", "2.2"))
R3_MIN_MM = float(os.getenv("BFHC_R3_MIN_MM", "3.0"))
R3_MAX_MM = float(os.getenv("BFHC_R3_MAX_MM", "3.8"))
R4_MIN_MM = float(os.getenv("BFHC_R4_MIN_MM", "1.5"))
R4_MAX_MM = float(os.getenv("BFHC_R4_MAX_MM", "2.5"))

# Current rematch official cylinder baseline.
BASELINE_LIFETIME_H = 242.07911958397654
BASELINE_E_J = 105597676.11285222
BASELINE_U_PCT = 148.69276459515976

# C4 is a local expansion run. Override from PowerShell if needed:
#   $env:BFHC_N_TRIALS='80'
N_TRIALS = int(os.getenv("BFHC_N_TRIALS", "80"))
MIN_LIFETIME_RATIO = float(os.getenv("BFHC_MIN_LIFETIME_RATIO", "0.50"))
MIN_LIFETIME_H = BASELINE_LIFETIME_H * MIN_LIFETIME_RATIO
MAX_U_RATIO = float(os.getenv("BFHC_MAX_U_RATIO", "1.20"))
MAX_U_PCT = BASELINE_U_PCT * MAX_U_RATIO
U_PENALTY_WEIGHT = float(os.getenv("BFHC_U_PENALTY_WEIGHT", "0.25"))

runner: COMSOLRunner | None = None


CSV_HEADER = [
    "trial", "r1_mm", "r2_mm", "r3_mm", "r4_mm",
    "Vwork_V", "initialTmax_K", "Tmin_K", "Tmean_K", "U_pct",
    "TmaxAll_K", "TminAll_K", "TmeanAll_K", "UAll_pct",
    "TmaxActive_K", "TminActive_K", "TmeanActive_K", "UActive_pct",
    "activeVolumeFraction", "TmaxFreeSurface_K", "TminFreeSurface_K",
    "TmeanFreeSurface_K", "UFreeSurface_pct", "freeSurfaceArea_m2",
    "electrodeTemperatureUndershoot_K", "temperatureFallbackUsed",
    "maxErosionTmax_K", "lifetimeH", "R_L_pct", "eta_L_pct",
    "initialP03sphere_W", "initialPradSphere_W",
    "lifeAvgP03sphere_W", "lifeAvgPradSphere_W",
    "lifeTotalP03sphere_J", "eta_E_pct",
    "initialP03gross_W", "initialP03escape_W",
    "initialP03selfAbsorbed_W", "lifeAvgP03gross_W",
    "lifeAvgP03escape_W", "lifeTotalP03gross_J",
    "lifeTotalP03escape_J", "lifeTotalP03selfAbsorbed_J",
    "selfViewLossRaw_pct", "radiationNumericalExcess_pct",
    "objectiveScore", "uPenalty_pctpt", "U_limit_pct",
    "selfViewLoss_pct", "failureReached", "capLimited", "stepLimited",
    "censored", "lifetimeExact", "terminationReason", "failureFeature",
    "failureIndex", "maxFeatureLoss_pct", "erosionSteps",
    "erosionAttemptedSteps", "erosionSolveRetries", "maxLifetimeCap_h",
    "maxErosionSteps", "initialCOMSOLVolume_m3",
    "initialExpectedVolume_m3", "initialGeometryVolumeError_rel",
    "initialTargetVolumeDeviation_rel", "initialSegmentMaskAreaRatioMin",
    "initialSegmentMaskAreaRatioMax", "initialVolume_m3", "finalVolume_m3",
    "volumeLoss_pct", "initialShoulderArea_m2",
    "finalShoulderArea_m2", "maxShoulderArea_m2",
    "overtempStep", "overtempTimeH", "overtempTmax_K",
    "runnerStatus", "status", "voltagePolicy", "voltageObjective",
    "operatingPointVersion", "ratedVoltageEligible",
    "ratedVoltageExactCandidateCount", "ratedVoltageSelectionReason",
    "ratedVoltageSourceStatus", "voltageCandidateCount",
    "voltageMaxSafe_V", "voltageCandidateRatios", "voltageScanSummary",
    "metricVersion", "physicsVersion", "geometryVersion",
    "lifecycleVersion", "erosionModel", "failureFraction",
    "maxErosionStep_s", "geometryVolumeTolerance_rel",
    "radiationEscapeMethod", "spectralSplit_um", "thermalAmbient_K",
    "scoreAmbientTarget_K", "temperatureStatisticVersion",
    "temperaturePrimaryDomain", "activeTemperatureTrim_mm",
    "electrodeBoundaryMode", "electrodeBoundaryVersion",
    "electrodeBoundaryApproximation", "electrodeTemperature_K",
    "copperThermalConductivity_W_mK", "electrodeContactRadiusIn_mm",
    "electrodeContactRadiusOut_mm", "electrodeSpreadingHIn_W_m2K",
    "electrodeSpreadingHOut_W_m2K",
    "elapsed_sec",
]


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def compute_r4(r1_mm, r2_mm, r3_mm):
    remainder = TOTAL_SQ - r1_mm**2 - r2_mm**2 - r3_mm**2
    if remainder < GEOM_R_MIN_MM**2:
        return None
    r4 = math.sqrt(remainder)
    if r4 > GEOM_R_MAX_MM:
        return None
    if r4 < R4_MIN_MM or r4 > R4_MAX_MM:
        return None
    return r4


def feasible_suggest_float(trial, name, lower, upper):
    if lower > upper:
        raise optuna.TrialPruned(f"No feasible range for {name}.")
    return trial.suggest_float(name, lower, upper)


def init_csv():
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != CSV_HEADER:
            raise RuntimeError(
                "Refusing to mix pre-D3 and D3 CSV rows. "
                "Use new BFHC_CSV_FILE, BFHC_DB_FILE, and BFHC_STUDY_NAME values."
            )
        return
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
    energy_j = float(row.get("lifeTotalP03escape_J", float("nan")))
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

    r1 = trial.suggest_float("r1_mm", R1_MIN_MM, R1_MAX_MM)

    r2_min_sq = TOTAL_SQ - r1**2 - (R3_MAX_MM**2 + R4_MAX_MM**2)
    r2_max_sq = TOTAL_SQ - r1**2 - (R3_MIN_MM**2 + R4_MIN_MM**2)
    r2_lo = max(R2_MIN_MM, math.sqrt(max(0.0, r2_min_sq)))
    r2_hi = min(R2_MAX_MM, math.sqrt(max(0.0, r2_max_sq)))
    r2 = feasible_suggest_float(trial, "r2_mm", r2_lo, r2_hi)

    r3_min_sq = TOTAL_SQ - r1**2 - r2**2 - R4_MAX_MM**2
    r3_max_sq = TOTAL_SQ - r1**2 - r2**2 - R4_MIN_MM**2
    r3_lo = max(R3_MIN_MM, math.sqrt(max(0.0, r3_min_sq)))
    r3_hi = min(R3_MAX_MM, math.sqrt(max(0.0, r3_max_sq)))
    r3 = feasible_suggest_float(trial, "r3_mm", r3_lo, r3_hi)

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
        result = runner.evaluate(
            radii_m,
            voltage_policy="rated_lifecycle_scan",
            voltage_objective="lifeTotalP03escape_J",
            electrode_boundary_mode="fixed_temperature",
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
        "lifetimeH", "lifeTotalP03escape_J", "U_pct",
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

    print("BFHC cylinder rematch C4 local search")
    print(f"CSV output:       {CSV_PATH}")
    print(f"Optuna DB:        sqlite:///{DB_PATH}")
    print(f"Study name:       {STUDY_NAME}")
    print(f"Trials target:    {N_TRIALS}")
    print("Radius bounds:    "
          f"r1=[{R1_MIN_MM}, {R1_MAX_MM}] mm, "
          f"r2=[{R2_MIN_MM}, {R2_MAX_MM}] mm, "
          f"r3=[{R3_MIN_MM}, {R3_MAX_MM}] mm, "
          f"r4=[{R4_MIN_MM}, {R4_MAX_MM}] mm")
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
        print("  C4 SEARCH COMPLETE")
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
