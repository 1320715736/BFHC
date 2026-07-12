"""Status report for the zigzag C8 trial-19 verification run."""

import csv
import subprocess
from pathlib import Path


ML_DIR = Path(r"D:\VScode\project\BFHC\zigzag_family\ML")
DATA_DIR = ML_DIR / "data"
CSV_PATH = DATA_DIR / "rematch_c8_trial19_verify.csv"
STDOUT_PATH = DATA_DIR / "rematch_c8_trial19_verify_stdout.log"
STDERR_PATH = DATA_DIR / "rematch_c8_trial19_verify_stderr.log"


def read_row():
    if not CSV_PATH.exists():
        return None
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def print_process_status():
    command = (
        "Get-CimInstance Win32_Process -Filter "
        "\"Name = 'python.exe' OR Name = 'pythonw.exe' OR "
        "Name LIKE 'comsol%'\" | Where-Object { "
        "($_.Name -like 'python*' -and $_.CommandLine -and "
        "$_.CommandLine -like '*rematch_c8_verify_trial19.py*') -or "
        "($_.Name -like 'comsol*') } | "
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


def print_csv_status():
    print(f"csv={CSV_PATH}")
    row = read_row()
    if row is None:
        print("csv_rows=0")
        return

    keys = [
        "status", "runnerStatus", "source_trial", "N_RUNS", "L_RUN_mm",
        "z_first_mm", "side_mm", "maxLifetimeCap_h", "lifetimeH",
        "R_L_pct", "eta_E_pct", "U_pct", "lifeTotalP03sphere_J",
        "failureReached", "capLimited", "erosionSteps", "elapsed_sec",
    ]
    print("csv_last=" + "; ".join(
        f"{key}={row.get(key, '')}" for key in keys
    ))


def print_log_tails():
    for label, path in (("stdout", STDOUT_PATH), ("stderr", STDERR_PATH)):
        print(f"{label}={path}")
        if not path.exists():
            print(f"{label}_missing=true")
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            print(f"{label}_read_error={exc}")
            continue
        for line in lines[-20:]:
            print(line)


def main():
    print_process_status()
    print_csv_status()
    print_log_tails()


if __name__ == "__main__":
    main()
