"""C7 analysis for the zigzag rematch C6 small search.

Reads the completed C6 CSV and writes two read-only analysis artifacts:

* data/rematch_c7_candidate_ranking.csv
* data/rematch_c7_analysis.csv
"""

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


ML_DIR = Path(r"D:\VScode\project\BFHC\zigzag_family\ML")
DATA_DIR = ML_DIR / "data"
C6_CSV_PATH = DATA_DIR / "rematch_c6_trials.csv"
SUMMARY_CSV_PATH = DATA_DIR / "rematch_c7_analysis.csv"
RANKING_CSV_PATH = DATA_DIR / "rematch_c7_candidate_ranking.csv"

BASELINE_LIFETIME_H = 242.07911958397654
BASELINE_E_J = 105597676.11285222
BASELINE_U_PCT = 148.69276459515976
MAX_TEMP_K = 3273.15

# C5 cylinder lead for family-to-family comparison.
CYLINDER_LEAD_TRIAL = 68
CYLINDER_LEAD_LIFETIME_H = 378.33572614862254
CYLINDER_LEAD_E_J = 243679111.41385093
CYLINDER_LEAD_U_PCT = 127.03670684533598
CYLINDER_LEAD_INITIAL_P03_W = 215.95999424026604
CYLINDER_LEAD_AVG_P03_W = 178.9115787410802


def finite_float(value):
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def parse_row(row):
    parsed = dict(row)
    for key, value in row.items():
        if value == "":
            parsed[key] = None
            continue
        if key in {
                "runnerStatus", "status", "voltagePolicy",
                "voltageObjective", "voltageScanSummary"}:
            continue
        if key == "failureReached":
            parsed[key] = parse_bool(value)
            continue
        number = finite_float(value)
        if number is not None:
            parsed[key] = number
    for key in ("trial", "N_RUNS", "erosionSteps", "overtempStep"):
        if parsed.get(key) is not None:
            parsed[key] = int(parsed[key])
    return parsed


def read_c6_rows():
    with open(C6_CSV_PATH, newline="", encoding="utf-8") as f:
        return [parse_row(row) for row in csv.DictReader(f)]


def pct_change(value, baseline):
    return (value - baseline) / baseline * 100.0


def relative_reduction(value, baseline):
    return (baseline - value) / baseline * 100.0


def format_geometry(row):
    return (
        f"N={row['N_RUNS']}; "
        f"L_RUN_mm={row['L_RUN_mm']:.4f}; "
        f"z_first_mm={row['z_first_mm']:.4f}; "
        f"side_mm={row['side_mm']:.4f}"
    )


def is_lifetime_cap_candidate(row):
    return row.get("status") == "OK" and row.get("failureReached") is False


def is_pareto_candidate(candidate, rows):
    for other in rows:
        if other is candidate:
            continue
        dominates = (
            other["eta_E_pct"] >= candidate["eta_E_pct"]
            and other["R_L_pct"] >= candidate["R_L_pct"]
            and other["U_pct"] <= candidate["U_pct"]
            and (
                other["eta_E_pct"] > candidate["eta_E_pct"]
                or other["R_L_pct"] > candidate["R_L_pct"]
                or other["U_pct"] < candidate["U_pct"]
            )
        )
        if dominates:
            return False
    return True


def numeric_values(rows, key):
    return [
        value for value in (finite_float(row.get(key)) for row in rows)
        if value is not None
    ]


def stats_text(rows, key, digits=4):
    values = numeric_values(rows, key)
    if not values:
        return ""
    return (
        f"min={min(values):.{digits}f}; "
        f"median={median(values):.{digits}f}; "
        f"mean={mean(values):.{digits}f}; "
        f"max={max(values):.{digits}f}"
    )


def write_candidate_ranking(ok_rows):
    pareto_trials = {
        row["trial"] for row in ok_rows
        if is_pareto_candidate(row, ok_rows)
    }
    ranked = sorted(
        ok_rows,
        key=lambda row: (
            row["objectiveScore"], row["eta_E_pct"], -row["U_pct"]),
        reverse=True,
    )

    header = [
        "rank", "trial", "pareto", "cap_limited", "geometry",
        "N_RUNS", "L_RUN_mm", "z_first_mm", "side_mm", "pathLength_mm",
        "Vwork_V", "objectiveScore", "eta_E_pct", "R_L_pct", "eta_L_pct",
        "lifetimeH", "lifeTotalP03sphere_J", "U_pct",
        "U_delta_pctpt_vs_B2", "U_relative_reduction_pct_vs_B2",
        "initialP03sphere_W", "lifeAvgP03sphere_W",
        "maxErosionTmax_K", "overtemp_margin_K", "failureReached",
        "erosionSteps", "E_gain_pct_vs_cylinder_trial68",
        "life_ratio_pct_vs_cylinder_trial68",
        "U_reduction_pct_vs_cylinder_trial68", "note",
    ]

    with open(RANKING_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            margin = MAX_TEMP_K - row["maxErosionTmax_K"]
            note = ""
            if rank == 1:
                note = "C7 lead zigzag candidate; verify with higher lifetime cap"
            elif row["trial"] in pareto_trials:
                note = "Pareto candidate"
            writer.writerow({
                "rank": rank,
                "trial": row["trial"],
                "pareto": row["trial"] in pareto_trials,
                "cap_limited": is_lifetime_cap_candidate(row),
                "geometry": format_geometry(row),
                "N_RUNS": row["N_RUNS"],
                "L_RUN_mm": row["L_RUN_mm"],
                "z_first_mm": row["z_first_mm"],
                "side_mm": row["side_mm"],
                "pathLength_mm": row["pathLength_mm"],
                "Vwork_V": row["Vwork_V"],
                "objectiveScore": row["objectiveScore"],
                "eta_E_pct": row["eta_E_pct"],
                "R_L_pct": row["R_L_pct"],
                "eta_L_pct": row["eta_L_pct"],
                "lifetimeH": row["lifetimeH"],
                "lifeTotalP03sphere_J": row["lifeTotalP03sphere_J"],
                "U_pct": row["U_pct"],
                "U_delta_pctpt_vs_B2": row["U_pct"] - BASELINE_U_PCT,
                "U_relative_reduction_pct_vs_B2": relative_reduction(
                    row["U_pct"], BASELINE_U_PCT),
                "initialP03sphere_W": row["initialP03sphere_W"],
                "lifeAvgP03sphere_W": row["lifeAvgP03sphere_W"],
                "maxErosionTmax_K": row["maxErosionTmax_K"],
                "overtemp_margin_K": margin,
                "failureReached": row["failureReached"],
                "erosionSteps": row["erosionSteps"],
                "E_gain_pct_vs_cylinder_trial68": pct_change(
                    row["lifeTotalP03sphere_J"], CYLINDER_LEAD_E_J),
                "life_ratio_pct_vs_cylinder_trial68": (
                    row["lifetimeH"] / CYLINDER_LEAD_LIFETIME_H * 100.0),
                "U_reduction_pct_vs_cylinder_trial68": relative_reduction(
                    row["U_pct"], CYLINDER_LEAD_U_PCT),
                "note": note,
            })

    return ranked, pareto_trials


def add_summary(summary, section, key, value, notes=""):
    summary.append({
        "section": section,
        "key": key,
        "value": value,
        "notes": notes,
    })


def write_summary(rows, ok_rows, ranked, pareto_trials):
    summary = []
    status_counts = Counter(row.get("status", "") for row in rows)
    best = ranked[0] if ranked else None
    cap_limited = [row for row in ok_rows if is_lifetime_cap_candidate(row)]

    add_summary(summary, "summary", "total_trials", len(rows),
                "Completed C6 small-search trials")
    for status in sorted(status_counts):
        add_summary(summary, "summary", status, status_counts[status],
                    "C6 status count")
    add_summary(summary, "summary", "ok_rate_pct",
                len(ok_rows) / len(rows) * 100.0,
                "OK trials divided by total trials")
    add_summary(summary, "summary", "pareto_count", len(pareto_trials),
                "Non-dominated by eta_E/R_L/U among OK trials")
    add_summary(summary, "summary", "cap_limited_ok_count",
                len(cap_limited),
                "OK candidates that stopped at runner max_lifetime_h")

    if best is not None:
        add_summary(summary, "best", "trial", best["trial"],
                    "C7 lead zigzag candidate by objectiveScore")
        add_summary(summary, "best", "geometry", format_geometry(best),
                    "Main zigzag geometry parameters")
        add_summary(summary, "best", "Vwork_V", best["Vwork_V"],
                    "Selected by max_safe voltage policy")
        add_summary(summary, "best", "objectiveScore", best["objectiveScore"],
                    "Current BO scalar score")
        add_summary(summary, "best", "lifetimeH", best["lifetimeH"],
                    "Current lifecycle length; cap-limited if failureReached=False")
        add_summary(summary, "best", "R_L_pct", best["R_L_pct"],
                    "L_opt / L_ini * 100")
        add_summary(summary, "best", "eta_L_pct", best["eta_L_pct"],
                    "(L_opt - L_ini) / L_ini * 100")
        add_summary(summary, "best", "lifeTotalP03sphere_J",
                    best["lifeTotalP03sphere_J"],
                    "Lifecycle cumulative 0-3 um effective radiation energy")
        add_summary(summary, "best", "eta_E_pct", best["eta_E_pct"],
                    "(E_opt - E_ini) / E_ini * 100")
        add_summary(summary, "best", "U_pct", best["U_pct"],
                    "Full-domain steady-state temperature uniformity")
        add_summary(summary, "best", "U_relative_reduction_pct_vs_B2",
                    relative_reduction(best["U_pct"], BASELINE_U_PCT),
                    "Relative U reduction against B2")
        add_summary(summary, "best", "maxErosionTmax_K",
                    best["maxErosionTmax_K"],
                    "Maximum temperature during full erosion loop")
        add_summary(summary, "best", "overtemp_margin_K",
                    MAX_TEMP_K - best["maxErosionTmax_K"],
                    "3273.15 K - maxErosionTmax_K")
        add_summary(summary, "best", "failureReached",
                    best["failureReached"],
                    "False means lifetime is truncated by runner cap")
        add_summary(summary, "compare_cylinder68", "E_gain_pct",
                    pct_change(best["lifeTotalP03sphere_J"],
                               CYLINDER_LEAD_E_J),
                    "Against C5 cylinder lead trial 68")
        add_summary(summary, "compare_cylinder68", "life_ratio_pct",
                    best["lifetimeH"] / CYLINDER_LEAD_LIFETIME_H * 100.0,
                    "Against C5 cylinder lead trial 68")
        add_summary(summary, "compare_cylinder68", "U_reduction_pct",
                    relative_reduction(best["U_pct"],
                                       CYLINDER_LEAD_U_PCT),
                    "Against C5 cylinder lead trial 68")

    for rank, row in enumerate(ranked[:5], start=1):
        add_summary(
            summary,
            "top_objective",
            f"rank_{rank}",
            row["trial"],
            (f"obj={row['objectiveScore']:.4f}; "
             f"eta_E={row['eta_E_pct']:.4f}%; "
             f"R_L={row['R_L_pct']:.4f}%; "
             f"U={row['U_pct']:.4f}%; "
             f"cap_limited={is_lifetime_cap_candidate(row)}; "
             f"{format_geometry(row)}")
        )

    for status in sorted(status_counts):
        group = [row for row in rows if row.get("status") == status]
        add_summary(summary, "status_ranges", status + "_N_RUNS",
                    stats_text(group, "N_RUNS"),
                    "Distribution by status")
        add_summary(summary, "status_ranges", status + "_L_RUN_mm",
                    stats_text(group, "L_RUN_mm"),
                    "Distribution by status")
        add_summary(summary, "status_ranges", status + "_side_mm",
                    stats_text(group, "side_mm"),
                    "Distribution by status")

    add_summary(summary, "decision", "current_lead", "trial_19",
                "Zigzag now beats cylinder trial 68 on energy and U")
    add_summary(summary, "decision", "not_final_without_recheck", "true",
                "trial 19 is cap-limited; actual failure lifetime is not measured")
    add_summary(summary, "next_step", "recommended", "C8",
                "Re-evaluate trial 19 with higher max_lifetime_h, then local BO")
    add_summary(summary, "next_bounds", "N_RUNS", "8,10,12,14",
                "Keep around C6 OK cluster and trial 19")
    add_summary(summary, "next_bounds", "L_RUN_mm", "70-120",
                "Center around trial 19 L=89.10 mm")
    add_summary(summary, "next_bounds", "z_first_mm", "1.6-2.5",
                "Center around trial 19 z_first=2.105 mm")
    add_summary(summary, "next_bounds", "max_lifetime_h", "500",
                "Needed to measure actual failure or stronger lower bound")

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "key", "value", "notes"])
        writer.writeheader()
        writer.writerows(summary)


def main():
    rows = read_c6_rows()
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    ranked, pareto_trials = write_candidate_ranking(ok_rows)
    write_summary(rows, ok_rows, ranked, pareto_trials)

    print(f"rows={len(rows)}")
    print(f"ok={len(ok_rows)}")
    print(f"best_trial={ranked[0]['trial'] if ranked else ''}")
    print(f"ranking_csv={RANKING_CSV_PATH}")
    print(f"summary_csv={SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
