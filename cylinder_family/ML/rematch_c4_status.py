"""Status report for the cylinder rematch C4 local search.

This script is read-only. It checks the C4 CSV/SQLite outputs and prints a
compact progress summary for manual wake-up checks.
"""

import csv
import math
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path


ML_DIR = Path(r"D:\VScode\project\BFHC\cylinder_family\ML")
DATA_DIR = ML_DIR / "data"
DB_PATH = DATA_DIR / "rematch_c4_optuna.db"
CSV_PATH = DATA_DIR / "rematch_c4_trials.csv"


def finite_float(value):
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def read_csv_rows():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_process_status():
    command = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -and ("
        "($_.Name -like 'python*' -and "
        "$_.CommandLine -like '*optuna_optimize.py*') -or "
        "($_.Name -like 'comsol*' -or "
        "$_.Name -like 'comsolmphserver*')) } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        print(f"process_check_error={exc}")
        return

    output = completed.stdout.strip()
    if not output:
        print("process=not_running")
    else:
        print("process=running")
        print(output)


def print_db_status():
    if not DB_PATH.exists():
        print("db=missing")
        return

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        states = Counter(
            state for (state,) in cur.execute("select state from trials")
        )
        print("db_trial_states=" + ", ".join(
            f"{key}:{states[key]}" for key in sorted(states)
        ))
        running = list(cur.execute(
            "select number from trials where state='RUNNING' order by number"
        ))
        waiting = list(cur.execute(
            "select number from trials where state='WAITING' order by number"
        ))
        if running:
            print("db_running_trials=" + ",".join(str(x[0]) for x in running))
        if waiting:
            print("db_waiting_trials=" + ",".join(str(x[0]) for x in waiting))
    finally:
        con.close()


def print_csv_status():
    rows = read_csv_rows()
    print(f"csv_rows={len(rows)}")
    if not rows:
        return

    statuses = Counter(row.get("status", "") for row in rows)
    print("csv_status=" + ", ".join(
        f"{key}:{statuses[key]}" for key in sorted(statuses)
    ))

    ok_rows = [
        row for row in rows
        if row.get("status") == "OK"
        and finite_float(row.get("objectiveScore")) is not None
    ]
    if ok_rows:
        best = max(ok_rows, key=lambda row: finite_float(row["objectiveScore"]))
        keys = [
            "trial", "objectiveScore", "R_L_pct", "eta_E_pct", "U_pct",
            "lifetimeH", "lifeTotalP03sphere_J", "Vwork_V",
        ]
        print("best_ok=" + "; ".join(
            f"{key}={best.get(key, '')}" for key in keys
        ))

    last = rows[-1]
    print("last_row=" + "; ".join(
        f"{key}={last.get(key, '')}"
        for key in ("trial", "status", "runnerStatus", "elapsed_sec")
    ))


def main():
    print(f"csv={CSV_PATH}")
    print(f"db={DB_PATH}")
    print_process_status()
    print_db_status()
    print_csv_status()


if __name__ == "__main__":
    main()
