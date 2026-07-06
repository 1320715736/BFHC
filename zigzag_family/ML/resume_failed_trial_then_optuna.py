"""
Resume a disconnected zigzag trial without skipping its CSV trial number.

This script is intentionally conservative:
1. Read data/failed_trial.json.
2. Find the latest SERVER_DISCONNECT row in data/trials.csv with the same params.
3. Re-evaluate that row and overwrite the same CSV trial number.
4. Remove failed_trial.json only after the row is finalized.
5. Exit after recovery. Start a fresh Python process for the normal Optuna run.
"""

import csv
import json
import math
import time

import optuna

import optuna_optimize as optz
from zigzag_runner import COMSOLRunner, ServerDisconnectError


def _same_float(a, b, tol=1.0e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _find_disconnected_trial(params):
    rows = []
    with open(optz.CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for row in reversed(rows):
        if row.get("status") != "SERVER_DISCONNECT":
            continue
        if str(row.get("N_RUNS")) != str(params["N_RUNS"]):
            continue
        if not _same_float(row.get("L_RUN_mm"), params["L_RUN_mm"]):
            continue
        if not _same_float(row.get("z_first_mm"), params["z_first_mm"]):
            continue
        return int(float(row["trial"]))
    raise RuntimeError("No matching SERVER_DISCONNECT row found in trials.csv.")


def _side_mm_for_params(runner, params):
    side, _, _ = runner.compute_side_and_blocks(
        int(params["N_RUNS"]),
        float(params["L_RUN_mm"]) * 1.0e-3,
        float(params["z_first_mm"]) * 1.0e-3,
    )
    return side * 1.0e3


def _resume_one_failed_trial():
    if not optz.FAILED_TRIAL_PATH.exists():
        print("No failed_trial.json found; continuing normal Optuna run.")
        return

    with open(optz.FAILED_TRIAL_PATH, encoding="utf-8") as f:
        params = json.load(f)

    trial_num = _find_disconnected_trial(params)
    print(f"Resuming disconnected CSV trial #{trial_num}: {params}")

    runner = COMSOLRunner()
    t_start = time.time()
    runner.start()
    try:
        side_mm = _side_mm_for_params(runner, params)
        result = runner.evaluate(
            N_RUNS=int(params["N_RUNS"]),
            L_RUN_m=float(params["L_RUN_mm"]) * 1.0e-3,
            z_first_m=float(params["z_first_mm"]) * 1.0e-3,
        )
        elapsed = time.time() - t_start

        row = {
            "trial": trial_num,
            "N_RUNS": params["N_RUNS"],
            "L_RUN_mm": params["L_RUN_mm"],
            "z_first_mm": params["z_first_mm"],
            "side_mm": side_mm,
            "elapsed_sec": round(elapsed, 1),
        }

        if result.get("status") != "OK":
            row["status"] = result.get("status", "UNKNOWN")
            optz.finalize_csv(row)
            raise RuntimeError(f"Recovered trial #{trial_num} failed: {row['status']}")

        lifetime = float(result["lifetimeH"])
        status = "OK" if lifetime >= optz.MIN_LIFETIME_H else "PRUNE_LIFETIME"
        row.update(result)
        row["status"] = status
        row["elapsed_sec"] = round(elapsed, 1)
        optz.finalize_csv(row)

        optz.FAILED_TRIAL_PATH.unlink()
        print(
            f"Recovered trial #{trial_num}: status={status}, "
            f"lifetime={lifetime:.2f}h, "
            f"AvgP03={float(result['lifeAvgP03sphere_W']):.2f}W"
        )
    except ServerDisconnectError:
        print("Server disconnected again while recovering the failed trial.")
        raise
    finally:
        try:
            runner.stop()
        except Exception:
            pass


def main():
    _resume_one_failed_trial()
    print(
        "Recovered failed trial. Start a fresh process for the remaining Optuna "
        "trials to avoid stale COMSOL/mph client state."
    )


if __name__ == "__main__":
    main()
