import json
import math
import time
import csv
import optuna
from optuna.trial import TrialState
from pathlib import Path
from zigzag_runner import COMSOLRunner, ServerDisconnectError

# File locations and Optuna study settings.
ML_DIR = Path('D:\\VScode\\project\\BFHC\\zigzag_family\\ML')
DATA_DIR = ML_DIR / 'data'
DB_PATH = DATA_DIR / 'optuna.db'
STUDY_NAME = 'zigzag_bo_v3_cylinder_life'
CSV_PATH = DATA_DIR / 'trials.csv'
FAILED_TRIAL_PATH = DATA_DIR / 'failed_trial.json'

# Free variables for the zigzag family. Side length is computed by the runner.
N_RUNS_CHOICES = [4, 6, 8, 10, 12, 14, 16]
L_RUN_MIN_MM = 20.0
L_RUN_MAX_MM = 300.0
Z_FIRST_MIN_MM = 0.6
Z_FIRST_MAX_MM = 3.0
BASELINE_N_RUNS = 8
BASELINE_L_RUN_MM = 104.0
BASELINE_Z_FIRST_MM = 0.8
CYLINDER_BASELINE_LIFETIME_H = 115.5037
MIN_LIFETIME_H = 0.3 * CYLINDER_BASELINE_LIFETIME_H
ZIGZAG_JAVA_BASELINE_LIFETIME_H = 7.1277
N_TRIALS = 150
runner: COMSOLRunner = None

# Keep exception text short enough for CSV storage.
def safe_exception_text(exc):
    try:
        return str(exc)
    except Exception:
        return exc.__class__.__name__
CSV_HEADER = ['trial', 'N_RUNS', 'L_RUN_mm', 'z_first_mm', 'side_mm', 'Vwork_V', 'initialTmax_K', 'lifetimeH', 'initialP03sphere_W', 'initialPradSphere_W', 'lifeAvgP03sphere_W', 'lifeAvgPradSphere_W', 'selfViewLoss_pct', 'failureReached', 'erosionSteps', 'status', 'elapsed_sec']

# CSV rows are rewritten when an interrupted trial is resumed.
def init_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_csv(row_dict):
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, '') for h in CSV_HEADER])

def finalize_csv(row_dict):
    trial_id = str(row_dict.get('trial', ''))
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
    full_row = {h: row_dict.get(h, '') for h in CSV_HEADER}
    replace_idx = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get('trial') == trial_id and rows[i].get('status') in ('RUNNING', 'SERVER_DISCONNECT'):
            replace_idx = i
            break
    if replace_idx is None:
        rows.append(full_row)
    else:
        rows[replace_idx] = full_row
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

# Optuna objective: run COMSOL and maximize lifecycle average P03 power.
def objective(trial):
    global runner
    t_start = time.time()
    trial_num = trial.number
    N_RUNS = trial.suggest_categorical('N_RUNS', N_RUNS_CHOICES)
    L_RUN_mm = trial.suggest_float('L_RUN_mm', L_RUN_MIN_MM, L_RUN_MAX_MM)
    z_first_mm = trial.suggest_float('z_first_mm', Z_FIRST_MIN_MM, Z_FIRST_MAX_MM)
    L_RUN_m = L_RUN_mm * 0.001
    z_first_m = z_first_mm * 0.001
    side, _, plen = runner.compute_side_and_blocks(N_RUNS, L_RUN_m, z_first_m)
    side_mm = side * 1000.0
    print(f'\n{'=' * 60}')
    print(f'Trial {trial_num}: N_RUNS={N_RUNS}  L_RUN={L_RUN_mm:.1f}mm  z_first={z_first_mm:.2f}mm  side={side_mm:.4f}mm  path={plen * 1000.0:.1f}mm')
    print(f'{'=' * 60}')
    append_csv({'trial': trial_num, 'N_RUNS': N_RUNS, 'L_RUN_mm': L_RUN_mm, 'z_first_mm': z_first_mm, 'side_mm': side_mm, 'status': 'RUNNING'})
    MAX_RETRIES = 2
    result = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = runner.evaluate(N_RUNS=N_RUNS, L_RUN_m=L_RUN_m, z_first_m=z_first_m)
            break
        except ServerDisconnectError as e:
            elapsed = time.time() - t_start
            finalize_csv({'trial': trial_num, 'N_RUNS': N_RUNS, 'L_RUN_mm': L_RUN_mm, 'z_first_mm': z_first_mm, 'side_mm': side_mm, 'status': 'SERVER_DISCONNECT', 'elapsed_sec': round(elapsed, 1)})
            with open(FAILED_TRIAL_PATH, 'w') as fp:
                json.dump({'N_RUNS': N_RUNS, 'L_RUN_mm': L_RUN_mm, 'z_first_mm': z_first_mm}, fp)
            print(f'  SERVER_DISCONNECT: params saved to {FAILED_TRIAL_PATH}')
            raise
        except Exception as e:
            err_msg = safe_exception_text(e)
            print(f'  ERROR (attempt {attempt + 1}/{MAX_RETRIES + 1}): {err_msg}')
            if attempt < MAX_RETRIES:
                print('  -> Restarting COMSOL server...')
                try:
                    runner.stop()
                except Exception:
                    pass
                time.sleep(5)
                runner.start()
                print('  -> COMSOL reconnected, retrying...')
            else:
                elapsed = time.time() - t_start
                finalize_csv({'trial': trial_num, 'N_RUNS': N_RUNS, 'L_RUN_mm': L_RUN_mm, 'z_first_mm': z_first_mm, 'side_mm': side_mm, 'status': f'ERROR: {err_msg}', 'elapsed_sec': round(elapsed, 1)})
                raise optuna.TrialPruned(f'COMSOL error after {MAX_RETRIES + 1} attempts: {err_msg}')
    elapsed = time.time() - t_start
    if result.get('status') != 'OK':
        print(f'  FAILED: {result.get('status')}')
        finalize_csv({'trial': trial_num, 'N_RUNS': N_RUNS, 'L_RUN_mm': L_RUN_mm, 'z_first_mm': z_first_mm, 'side_mm': side_mm, 'status': result.get('status', 'UNKNOWN'), 'elapsed_sec': round(elapsed, 1)})
        raise optuna.TrialPruned(result.get('status'))
    lifetime = result['lifetimeH']
    target = result['lifeAvgP03sphere_W']
    print(f"  Vwork={result['Vwork_V']:.4f}V  Tmax={result['initialTmax_K']:.1f}K  Lifetime={lifetime:.2f}h  AvgP03={target:.2f}W  [{elapsed:.0f}s]")
    status = 'OK'
    if lifetime < MIN_LIFETIME_H:
        status = 'PRUNE_LIFETIME'
    row = {'trial': trial_num, 'N_RUNS': N_RUNS, 'L_RUN_mm': L_RUN_mm, 'z_first_mm': z_first_mm, 'side_mm': side_mm, 'status': status, 'elapsed_sec': round(elapsed, 1)}
    row.update(result)
    row['status'] = status
    finalize_csv(row)
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
    print(f'Search space:   N_RUNS ∈ {N_RUNS_CHOICES}')
    print(f'                L_RUN ∈ [{L_RUN_MIN_MM}, {L_RUN_MAX_MM}] mm')
    print(f'                z_first ∈ [{Z_FIRST_MIN_MM}, {Z_FIRST_MAX_MM}] mm')
    print(f'Zigzag ref:     N={BASELINE_N_RUNS} L={BASELINE_L_RUN_MM}mm zf={BASELINE_Z_FIRST_MM}mm')
    print(f'Life constraint: cylinder baseline {CYLINDER_BASELINE_LIFETIME_H:.4f}h × 30% = {MIN_LIFETIME_H:.2f}h')
    print(f'Zigzag Java ref: lifetime={ZIGZAG_JAVA_BASELINE_LIFETIME_H:.4f}h (reference only, not the constraint)')
    print()
    runner = COMSOLRunner()
    runner.start()
    try:
        storage = f'sqlite:///{DB_PATH}'
        study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction='maximize', load_if_exists=True)
        if len(study.trials) == 0:
            study.enqueue_trial({'N_RUNS': BASELINE_N_RUNS, 'L_RUN_mm': BASELINE_L_RUN_MM, 'z_first_mm': BASELINE_Z_FIRST_MM})
        if FAILED_TRIAL_PATH.exists():
            with open(FAILED_TRIAL_PATH) as fp:
                failed_params = json.load(fp)
            study.enqueue_trial(failed_params)
            FAILED_TRIAL_PATH.unlink()
            print(f'  Re-enqueued disconnected trial: {failed_params}')
        for t in study.trials:
            if t.state == TrialState.RUNNING and t.params:
                study.enqueue_trial(t.params)
                print(f'  Re-enqueued interrupted trial #{t.number}: {t.params}')
        n_remaining = N_TRIALS - len(study.trials)
        if n_remaining <= 0:
            print(f'Already have {len(study.trials)} trials, target is {N_TRIALS}. Done.')
        else:
            print(f'Resuming from {len(study.trials)} trials, running {n_remaining} more...')
            try:
                study.optimize(objective, n_trials=n_remaining)
            except ServerDisconnectError:
                print('\nCOMSOL server disconnected. Optimization paused.')
                if FAILED_TRIAL_PATH.exists():
                    print(f'Retry params saved to: {FAILED_TRIAL_PATH}')
                print('Re-run the script to resume from the interrupted trial.')
        print('\n' + '=' * 60)
        print('  OPTIMIZATION COMPLETE')
        print('=' * 60)
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
        if complete_trials:
            best = study.best_trial
            print(f'Best trial:     #{best.number}')
            print(f'Best value:     {best.value:.2f} W (lifeAvgP03sphere)')
            print(f"Best params:    N_RUNS={best.params['N_RUNS']}  L_RUN={best.params['L_RUN_mm']:.1f}mm  z_first={best.params['z_first_mm']:.2f}mm")
            side_best, _, plen_best = runner.compute_side_and_blocks(best.params['N_RUNS'], best.params['L_RUN_mm'] * 0.001, best.params['z_first_mm'] * 0.001)
            print(f'Best side:      {side_best * 1000.0:.4f} mm')
            print(f'Best path len:  {plen_best * 1000.0:.1f} mm')
        else:
            print('No feasible COMPLETE trial yet under the cylinder-based lifetime constraint.')
            print(f'Life floor:     {MIN_LIFETIME_H:.2f} h')
        print(f'\nTotal trials:   {len(study.trials)}')
        print(f'Results saved:  {CSV_PATH}')
    finally:
        runner.stop()
if __name__ == '__main__':
    main()
