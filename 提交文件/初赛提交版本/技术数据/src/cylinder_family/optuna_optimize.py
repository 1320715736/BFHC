import math
import time
import csv
import optuna
from pathlib import Path
from comsol_runner import COMSOLRunner

# File locations and Optuna study settings.
ML_DIR = Path('D:\\VScode\\project\\BFHC\\cylinder_family\\ML')
DATA_DIR = ML_DIR / 'data'
DB_PATH = DATA_DIR / 'optuna.db'
STUDY_NAME = 'cylinder_bo_v2'
CSV_PATH = DATA_DIR / 'trials.csv'

# Symmetric cylinder design: [r1,r2,r3,r4,r4,r3,r2,r1].
R0_MM = 2.5
R0_M = R0_MM * 0.001
SEG_COUNT = 8
R_MIN_MM = 0.8
R_MAX_MM = 4.5
BASELINE_LIFETIME_H = 115.5
MIN_LIFETIME_H = 0.3 * BASELINE_LIFETIME_H
N_TRIALS = 151
runner: COMSOLRunner = None
TOTAL_SQ = 4.0 * R0_MM ** 2

# r4 is determined by volume conservation after r1-r3 are sampled.
def compute_r4(r1_mm, r2_mm, r3_mm):
    remainder = TOTAL_SQ - r1_mm ** 2 - r2_mm ** 2 - r3_mm ** 2
    if remainder < R_MIN_MM ** 2:
        return None
    r4 = math.sqrt(remainder)
    if r4 > R_MAX_MM:
        return None
    return r4
CSV_HEADER = ['trial', 'r1_mm', 'r2_mm', 'r3_mm', 'r4_mm', 'Vwork_V', 'initialTmax_K', 'lifetimeH', 'initialP03sphere_W', 'initialPradSphere_W', 'lifeAvgP03sphere_W', 'lifeAvgPradSphere_W', 'selfViewLoss_pct', 'failureReached', 'erosionSteps', 'status', 'elapsed_sec']

# CSV is the audit trail for every COMSOL trial.
def init_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_csv(row_dict):
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, '') for h in CSV_HEADER])

def compact_status(status, limit=240):
    text = ' '.join(str(status).split())
    return text[:limit]

# Optuna objective: run COMSOL and maximize lifecycle average P03 power.
def objective(trial):
    global runner
    t_start = time.time()
    trial_num = trial.number
    r1 = trial.suggest_float('r1_mm', R_MIN_MM, R_MAX_MM)
    r2_max_sq = TOTAL_SQ - r1 ** 2 - 2 * R_MIN_MM ** 2
    r2_min_sq = TOTAL_SQ - r1 ** 2 - 2 * R_MAX_MM ** 2
    r2_lo = max(R_MIN_MM, math.sqrt(max(0.0, r2_min_sq)))
    r2_hi = min(R_MAX_MM, math.sqrt(max(R_MIN_MM ** 2, r2_max_sq)))
    r2 = trial.suggest_float('r2_mm', r2_lo, r2_hi)
    r3_max_sq = TOTAL_SQ - r1 ** 2 - r2 ** 2 - R_MIN_MM ** 2
    r3_min_sq = TOTAL_SQ - r1 ** 2 - r2 ** 2 - R_MAX_MM ** 2
    r3_lo = max(R_MIN_MM, math.sqrt(max(0.0, r3_min_sq)))
    r3_hi = min(R_MAX_MM, math.sqrt(max(R_MIN_MM ** 2, r3_max_sq)))
    r3 = trial.suggest_float('r3_mm', r3_lo, r3_hi)
    r4_sq = TOTAL_SQ - r1 ** 2 - r2 ** 2 - r3 ** 2
    r4 = math.sqrt(max(R_MIN_MM ** 2, r4_sq))
    radii_mm = [r1, r2, r3, r4, r4, r3, r2, r1]
    radii_m = [r * 0.001 for r in radii_mm]
    print(f'\n{'=' * 60}')
    print(f'Trial {trial_num}: r = [{', '.join((f'{r:.3f}' for r in radii_mm))}] mm')
    print(f'{'=' * 60}')
    try:
        result = runner.evaluate(radii_m)
    except Exception as e:
        elapsed = time.time() - t_start
        print(f'  ERROR: {e}')
        append_csv({'trial': trial_num, 'r1_mm': r1, 'r2_mm': r2, 'r3_mm': r3, 'r4_mm': r4, 'status': compact_status(f'ERROR: {e}'), 'elapsed_sec': round(elapsed, 1)})
        raise optuna.TrialPruned(f'COMSOL error: {e}')
    elapsed = time.time() - t_start
    if result.get('status') != 'OK':
        print(f'  FAILED: {result.get('status')}')
        row = {'trial': trial_num, 'r1_mm': r1, 'r2_mm': r2, 'r3_mm': r3, 'r4_mm': r4}
        row.update(result)
        row['status'] = compact_status(result.get('status', 'UNKNOWN'))
        row['elapsed_sec'] = round(elapsed, 1)
        append_csv(row)
        raise optuna.TrialPruned(result.get('status'))
    lifetime = result['lifetimeH']
    target = result['lifeAvgP03sphere_W']
    print(f"  Vwork={result['Vwork_V']:.4f}V  Tmax={result['initialTmax_K']:.1f}K  Lifetime={lifetime:.2f}h  AvgP03={target:.2f}W  [{elapsed:.0f}s]")
    status = 'OK'
    if lifetime < MIN_LIFETIME_H:
        status = 'PRUNE_LIFETIME'
    row = {'trial': trial_num, 'r1_mm': r1, 'r2_mm': r2, 'r3_mm': r3, 'r4_mm': r4}
    row.update(result)
    row['status'] = status
    row['elapsed_sec'] = round(elapsed, 1)
    append_csv(row)
    if status != 'OK':
        raise optuna.TrialPruned(f'Lifetime {lifetime:.2f}h < {MIN_LIFETIME_H:.2f}h')
    return target

# Main entry point for a resumable local optimization run.
def main():
    global runner
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_csv()
    print(f'CSV output:     {CSV_PATH}')
    print(f'Optuna DB:      sqlite:///{DB_PATH}')
    print(f'Trials:         {N_TRIALS}')
    print(f'Search space:   r1,r2,r3 ∈ [{R_MIN_MM}, {R_MAX_MM}] mm')
    print(f'Baseline:       Lifetime={BASELINE_LIFETIME_H}h, min={MIN_LIFETIME_H:.2f}h')
    print()
    runner = COMSOLRunner()
    runner.start()
    try:
        storage = f'sqlite:///{DB_PATH}'
        study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction='maximize', load_if_exists=True)
        if len(study.trials) == 0:
            study.enqueue_trial({'r1_mm': R0_MM, 'r2_mm': R0_MM, 'r3_mm': R0_MM})
        n_remaining = N_TRIALS - len(study.trials)
        if n_remaining <= 0:
            print(f'Already have {len(study.trials)} trials, target is {N_TRIALS}. Done.')
        else:
            print(f'Resuming from {len(study.trials)} trials, running {n_remaining} more...')
            study.optimize(objective, n_trials=n_remaining)
        print('\n' + '=' * 60)
        print('  OPTIMIZATION COMPLETE')
        print('=' * 60)
        best = study.best_trial
        print(f'Best trial:     #{best.number}')
        print(f'Best value:     {best.value:.2f} W (lifeAvgP03sphere)')
        print(f"Best params:    r1={best.params['r1_mm']:.3f}  r2={best.params['r2_mm']:.3f}  r3={best.params['r3_mm']:.3f} mm")
        r4 = compute_r4(best.params['r1_mm'], best.params['r2_mm'], best.params['r3_mm'])
        if r4:
            radii = [best.params['r1_mm'], best.params['r2_mm'], best.params['r3_mm'], r4, r4, best.params['r3_mm'], best.params['r2_mm'], best.params['r1_mm']]
            print(f'Full radii (mm): [{', '.join((f'{r:.3f}' for r in radii))}]')
        print(f'\nTotal trials:   {len(study.trials)}')
        print(f'Results saved:  {CSV_PATH}')
    finally:
        runner.stop()
if __name__ == '__main__':
    main()
