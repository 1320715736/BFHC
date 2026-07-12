"""C5 analysis for the cylinder rematch C4 local expansion.

Reads the completed C4 CSV and writes two read-only analysis artifacts:

* data/rematch_c5_candidate_ranking.csv
* data/rematch_c5_analysis.csv
"""

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


ML_DIR = Path(r"D:\VScode\project\BFHC\cylinder_family\ML")
DATA_DIR = ML_DIR / "data"
C4_CSV_PATH = DATA_DIR / "rematch_c4_trials.csv"
SUMMARY_CSV_PATH = DATA_DIR / "rematch_c5_analysis.csv"
RANKING_CSV_PATH = DATA_DIR / "rematch_c5_candidate_ranking.csv"

BASELINE_LIFETIME_H = 242.07911958397654
BASELINE_E_J = 105597676.11285222
BASELINE_U_PCT = 148.69276459515976
MAX_TEMP_K = 3273.15


def finite_float(value):
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def parse_row(row):
    parsed = dict(row)
    for key, value in row.items():
        if value == "":
            parsed[key] = None
            continue
        if key in {
                "runnerStatus", "status", "voltagePolicy",
                "voltageObjective", "failureReached"}:
            continue
        number = finite_float(value)
        if number is not None:
            parsed[key] = number
    if parsed.get("trial") is not None:
        parsed["trial"] = int(parsed["trial"])
    if parsed.get("erosionSteps") is not None:
        parsed["erosionSteps"] = int(parsed["erosionSteps"])
    if parsed.get("overtempStep") is not None:
        parsed["overtempStep"] = int(parsed["overtempStep"])
    return parsed


def read_c4_rows():
    with open(C4_CSV_PATH, newline="", encoding="utf-8") as f:
        return [parse_row(row) for row in csv.DictReader(f)]


def radii_full(row):
    return [
        row["r1_mm"], row["r2_mm"], row["r3_mm"], row["r4_mm"],
        row["r4_mm"], row["r3_mm"], row["r2_mm"], row["r1_mm"],
    ]


def format_list(values, digits=4):
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def pct_delta(value, baseline):
    return (value - baseline) / baseline * 100.0


def relative_reduction(value, baseline):
    return (baseline - value) / baseline * 100.0


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
        "rank", "trial", "pareto", "radii_full_mm",
        "r1_mm", "r2_mm", "r3_mm", "r4_mm", "Vwork_V",
        "objectiveScore", "eta_E_pct", "R_L_pct", "eta_L_pct",
        "lifetimeH", "lifeTotalP03sphere_J", "U_pct",
        "U_delta_pctpt_vs_B2", "U_relative_reduction_pct_vs_B2",
        "initialP03sphere_W", "lifeAvgP03sphere_W",
        "maxErosionTmax_K", "overtemp_margin_K", "erosionSteps",
        "note",
    ]

    with open(RANKING_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            u_delta = row["U_pct"] - BASELINE_U_PCT
            u_relative_reduction = (
                (BASELINE_U_PCT - row["U_pct"]) / BASELINE_U_PCT * 100.0
            )
            margin = MAX_TEMP_K - row["maxErosionTmax_K"]
            note = ""
            if rank == 1:
                note = "C5 lead cylinder candidate"
            elif row["trial"] in pareto_trials:
                note = "Pareto candidate"
            writer.writerow({
                "rank": rank,
                "trial": row["trial"],
                "pareto": row["trial"] in pareto_trials,
                "radii_full_mm": format_list(radii_full(row)),
                "r1_mm": row["r1_mm"],
                "r2_mm": row["r2_mm"],
                "r3_mm": row["r3_mm"],
                "r4_mm": row["r4_mm"],
                "Vwork_V": row["Vwork_V"],
                "objectiveScore": row["objectiveScore"],
                "eta_E_pct": row["eta_E_pct"],
                "R_L_pct": row["R_L_pct"],
                "eta_L_pct": row["eta_L_pct"],
                "lifetimeH": row["lifetimeH"],
                "lifeTotalP03sphere_J": row["lifeTotalP03sphere_J"],
                "U_pct": row["U_pct"],
                "U_delta_pctpt_vs_B2": u_delta,
                "U_relative_reduction_pct_vs_B2": u_relative_reduction,
                "initialP03sphere_W": row["initialP03sphere_W"],
                "lifeAvgP03sphere_W": row["lifeAvgP03sphere_W"],
                "maxErosionTmax_K": row["maxErosionTmax_K"],
                "overtemp_margin_K": margin,
                "erosionSteps": row["erosionSteps"],
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


def write_summary(rows, ok_rows, fail_rows, ranked, pareto_trials):
    summary = []
    status_counts = Counter(row.get("status", "") for row in rows)
    best = ranked[0]
    best_u = min(ok_rows, key=lambda row: row["U_pct"])
    best_life = max(ok_rows, key=lambda row: row["R_L_pct"])

    add_summary(summary, "summary", "total_trials", len(rows),
                "Completed C4 local-expansion trials")
    for status in sorted(status_counts):
        add_summary(summary, "summary", status, status_counts[status],
                    "C4 status count")
    add_summary(summary, "summary", "ok_rate_pct",
                len(ok_rows) / len(rows) * 100.0,
                "OK trials divided by total trials")
    add_summary(summary, "summary", "pareto_count", len(pareto_trials),
                "Non-dominated by eta_E/R_L/U among OK trials")

    add_summary(summary, "best", "trial", best["trial"],
                "C5 lead cylinder candidate by objectiveScore")
    add_summary(summary, "best", "radii_mm", format_list(radii_full(best)),
                "Eight mirrored segment radii")
    add_summary(summary, "best", "Vwork_V", best["Vwork_V"],
                "Selected by max_safe voltage policy")
    add_summary(summary, "best", "objectiveScore", best["objectiveScore"],
                "Current BO scalar score")
    add_summary(summary, "best", "lifetimeH", best["lifetimeH"],
                "Full erosion lifetime")
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
    add_summary(summary, "best", "U_delta_pctpt_vs_B2",
                best["U_pct"] - BASELINE_U_PCT,
                "Absolute U change against B2")
    add_summary(summary, "best", "U_relative_reduction_pct_vs_B2",
                relative_reduction(best["U_pct"], BASELINE_U_PCT),
                "Relative U reduction against B2")
    add_summary(summary, "best", "initialP03sphere_W",
                best["initialP03sphere_W"],
                "Initial 0-3 um effective power")
    add_summary(summary, "best", "lifeAvgP03sphere_W",
                best["lifeAvgP03sphere_W"],
                "Lifecycle average 0-3 um effective power")
    add_summary(summary, "best", "maxErosionTmax_K",
                best["maxErosionTmax_K"],
                "Maximum temperature during full erosion")
    add_summary(summary, "best", "overtemp_margin_K",
                MAX_TEMP_K - best["maxErosionTmax_K"],
                "3273.15 K - maxErosionTmax_K")

    add_summary(summary, "best_tradeoff", "lowest_U_trial",
                best_u["trial"],
                (f"U={best_u['U_pct']:.4f}%; "
                 f"eta_E={best_u['eta_E_pct']:.4f}%; "
                 f"R_L={best_u['R_L_pct']:.4f}%"))
    add_summary(summary, "best_tradeoff", "longest_life_trial",
                best_life["trial"],
                (f"R_L={best_life['R_L_pct']:.4f}%; "
                 f"eta_E={best_life['eta_E_pct']:.4f}%; "
                 f"U={best_life['U_pct']:.4f}%"))

    for rank, row in enumerate(ranked[:10], start=1):
        add_summary(
            summary,
            "top_objective",
            f"rank_{rank}",
            row["trial"],
            (f"obj={row['objectiveScore']:.4f}; "
             f"eta_E={row['eta_E_pct']:.4f}%; "
             f"R_L={row['R_L_pct']:.4f}%; "
             f"U={row['U_pct']:.4f}%; "
             f"r={format_list([row[f'r{i}_mm'] for i in range(1, 5)])}")
        )

    add_summary(summary, "ok_ranges", "r1_mm", stats_text(ok_rows, "r1_mm"),
                "OK trial distribution")
    add_summary(summary, "ok_ranges", "r2_mm", stats_text(ok_rows, "r2_mm"),
                "OK trial distribution")
    add_summary(summary, "ok_ranges", "r3_mm", stats_text(ok_rows, "r3_mm"),
                "OK trial distribution")
    add_summary(summary, "ok_ranges", "r4_mm", stats_text(ok_rows, "r4_mm"),
                "OK trial distribution")
    add_summary(summary, "ok_ranges", "U_pct", stats_text(ok_rows, "U_pct"),
                "OK trial distribution")
    add_summary(summary, "ok_ranges", "eta_E_pct",
                stats_text(ok_rows, "eta_E_pct"),
                "OK trial distribution")

    add_summary(summary, "failure_pattern",
                "FAIL_OVERTEMP_DURING_EROSION", len(fail_rows),
                "Only failure class observed in C4")
    add_summary(summary, "failure_pattern", "overtempTimeH",
                stats_text(fail_rows, "overtempTimeH"),
                "Failed trials exceed 3273.15 K during erosion")
    add_summary(summary, "failure_pattern", "overtempTmax_K",
                stats_text(fail_rows, "overtempTmax_K"),
                "Failed trials exceed 3273.15 K during erosion")
    add_summary(summary, "failure_pattern", "interpretation",
                "late-life overheating dominates C4 failures",
                "Keep full lifecycle overtemperature checks")

    add_summary(summary, "decision", "expand_to_150_now", "no",
                "Archive trial 68 as cylinder lead, then compare zigzag in C6")
    add_summary(summary, "decision", "reason",
                "C4 already gives a strong cylinder lead and stable OK cluster",
                "Best trial appears late, so later cylinder refinement remains useful")
    add_summary(summary, "next_step", "recommended", "C6",
                "Run zigzag rematch objective on current physics before more cylinder-only budget")
    add_summary(summary, "next_step", "cylinder_followup",
                "Use trial 68 neighborhood for 150-run refinement if zigzag underperforms",
                "Candidate center: r=[1.9064,1.9424,3.6475,2.0709] mm")

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "key", "value", "notes"])
        writer.writeheader()
        writer.writerows(summary)


def main():
    rows = read_c4_rows()
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    fail_rows = [row for row in rows if row.get("status") != "OK"]
    ranked, pareto_trials = write_candidate_ranking(ok_rows)
    write_summary(rows, ok_rows, fail_rows, ranked, pareto_trials)

    print(f"rows={len(rows)}")
    print(f"ok={len(ok_rows)}")
    print(f"fail={len(fail_rows)}")
    print(f"best_trial={ranked[0]['trial']}")
    print(f"ranking_csv={RANKING_CSV_PATH}")
    print(f"summary_csv={SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
