"""C9 analysis for the zigzag C8 local search.

Reads the completed C8 local-search CSV and writes three read-only artifacts:

* data/rematch_c9_candidate_ranking.csv
* data/rematch_c9_analysis.csv
* data/rematch_c9_report.md
"""

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


ML_DIR = Path(r"D:\VScode\project\BFHC\zigzag_family\ML")
DATA_DIR = ML_DIR / "data"

C8_LOCAL_CSV_PATH = DATA_DIR / "rematch_c8_local_trials.csv"
C8_VERIFY_CSV_PATH = DATA_DIR / "rematch_c8_trial19_verify.csv"
SUMMARY_CSV_PATH = DATA_DIR / "rematch_c9_analysis.csv"
RANKING_CSV_PATH = DATA_DIR / "rematch_c9_candidate_ranking.csv"
REPORT_MD_PATH = DATA_DIR / "rematch_c9_report.md"

BASELINE_LIFETIME_H = 242.07911958397654
BASELINE_E_J = 105597676.11285222
BASELINE_U_PCT = 148.69276459515976
MAX_TEMP_K = 3273.15

# C5 cylinder lead for family-to-family comparison.
CYLINDER_LEAD_TRIAL = 68
CYLINDER_LEAD_LIFETIME_H = 378.33572614862254
CYLINDER_LEAD_E_J = 243679111.41385093
CYLINDER_LEAD_U_PCT = 127.03670684533598


TEXT_KEYS = {
    "runnerStatus", "status", "voltagePolicy", "voltageObjective",
    "voltageScanSummary", "case_id",
}
BOOL_KEYS = {"failureReached", "capLimited"}
INT_KEYS = {
    "trial", "source_trial", "N_RUNS", "erosionSteps",
    "overtempStep", "voltageCandidateCount",
}


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
        if key in TEXT_KEYS:
            continue
        if key in BOOL_KEYS:
            parsed[key] = parse_bool(value)
            continue
        number = finite_float(value)
        if number is not None:
            parsed[key] = number
    for key in INT_KEYS:
        if parsed.get(key) is not None:
            parsed[key] = int(parsed[key])
    return parsed


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [parse_row(row) for row in csv.DictReader(f)]


def pct_change(value, baseline):
    return (value - baseline) / baseline * 100.0


def ratio_pct(value, baseline):
    return value / baseline * 100.0


def relative_reduction(value, baseline):
    return (baseline - value) / baseline * 100.0


def format_geometry(row):
    return (
        f"N={row['N_RUNS']}; "
        f"L_RUN_mm={row['L_RUN_mm']:.4f}; "
        f"z_first_mm={row['z_first_mm']:.4f}; "
        f"side_mm={row['side_mm']:.4f}"
    )


def is_ok(row):
    return row.get("status") == "OK"


def is_cap_limited(row):
    return is_ok(row) and row.get("capLimited") is True


def is_measured_failure(row):
    return is_ok(row) and row.get("failureReached") is True


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


def ranked_ok_rows(ok_rows):
    return sorted(
        ok_rows,
        key=lambda row: (
            row["objectiveScore"], row["eta_E_pct"],
            row["R_L_pct"], -row["U_pct"],
        ),
        reverse=True,
    )


def write_candidate_ranking(ok_rows, verify_best):
    ranked = ranked_ok_rows(ok_rows)
    pareto_trials = {
        row["trial"] for row in ok_rows
        if is_pareto_candidate(row, ok_rows)
    }

    header = [
        "rank", "trial", "pareto", "cap_limited", "measured_failure",
        "geometry", "N_RUNS", "L_RUN_mm", "z_first_mm", "side_mm",
        "pathLength_mm", "Vwork_V", "objectiveScore", "eta_E_pct",
        "R_L_pct", "eta_L_pct", "lifetimeH", "lifeTotalP03sphere_J",
        "U_pct", "U_delta_pctpt_vs_B2", "U_relative_reduction_pct_vs_B2",
        "initialP03sphere_W", "lifeAvgP03sphere_W", "maxErosionTmax_K",
        "overtemp_margin_K", "failureReached", "capLimited",
        "erosionSteps", "E_gain_pct_vs_cylinder_trial68",
        "life_ratio_pct_vs_cylinder_trial68",
        "U_reduction_pct_vs_cylinder_trial68",
        "E_gain_pct_vs_c8_verified_trial19",
        "life_ratio_pct_vs_c8_verified_trial19",
        "U_reduction_pct_vs_c8_verified_trial19", "note",
    ]

    with open(RANKING_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            margin = MAX_TEMP_K - row["maxErosionTmax_K"]
            note = ""
            if rank == 1 and is_cap_limited(row):
                note = (
                    "C9 lead lower-bound candidate; verify with higher "
                    "max_lifetime_h before treating lifetime as final"
                )
            elif rank == 1:
                note = "C9 lead measured-failure candidate"
            elif row["trial"] in pareto_trials:
                note = "Pareto candidate"

            writer.writerow({
                "rank": rank,
                "trial": row["trial"],
                "pareto": row["trial"] in pareto_trials,
                "cap_limited": is_cap_limited(row),
                "measured_failure": is_measured_failure(row),
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
                "capLimited": row["capLimited"],
                "erosionSteps": row["erosionSteps"],
                "E_gain_pct_vs_cylinder_trial68": pct_change(
                    row["lifeTotalP03sphere_J"], CYLINDER_LEAD_E_J),
                "life_ratio_pct_vs_cylinder_trial68": ratio_pct(
                    row["lifetimeH"], CYLINDER_LEAD_LIFETIME_H),
                "U_reduction_pct_vs_cylinder_trial68": relative_reduction(
                    row["U_pct"], CYLINDER_LEAD_U_PCT),
                "E_gain_pct_vs_c8_verified_trial19": pct_change(
                    row["lifeTotalP03sphere_J"],
                    verify_best["lifeTotalP03sphere_J"]),
                "life_ratio_pct_vs_c8_verified_trial19": ratio_pct(
                    row["lifetimeH"], verify_best["lifetimeH"]),
                "U_reduction_pct_vs_c8_verified_trial19": relative_reduction(
                    row["U_pct"], verify_best["U_pct"]),
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


def write_summary(rows, ok_rows, ranked, pareto_trials, verify_best):
    summary = []
    status_counts = Counter(row.get("status", "") for row in rows)
    best = ranked[0] if ranked else None
    cap_limited = [row for row in ok_rows if is_cap_limited(row)]
    measured = [row for row in ok_rows if is_measured_failure(row)]
    measured_ranked = ranked_ok_rows(measured)
    measured_best = measured_ranked[0] if measured_ranked else None

    add_summary(summary, "summary", "total_trials", len(rows),
                "Completed C8 local-search trials")
    for status in sorted(status_counts):
        add_summary(summary, "summary", status, status_counts[status],
                    "C8 local status count")
    add_summary(summary, "summary", "ok_rate_pct",
                len(ok_rows) / len(rows) * 100.0 if rows else 0.0,
                "OK trials divided by total trials")
    add_summary(summary, "summary", "pareto_count", len(pareto_trials),
                "Non-dominated by eta_E/R_L/U among OK trials")
    add_summary(summary, "summary", "cap_limited_ok_count",
                len(cap_limited),
                "OK candidates stopped by max_lifetime_h")
    add_summary(summary, "summary", "measured_failure_ok_count",
                len(measured),
                "OK candidates that actually reached erosion failure")

    if best is not None:
        add_summary(summary, "best_lower_bound", "trial", best["trial"],
                    "Best C8 local candidate by objectiveScore")
        add_summary(summary, "best_lower_bound", "geometry",
                    format_geometry(best), "Main zigzag geometry parameters")
        add_summary(summary, "best_lower_bound", "objectiveScore",
                    best["objectiveScore"], "BO scalar score")
        add_summary(summary, "best_lower_bound", "lifetimeH",
                    best["lifetimeH"],
                    "Cap-limited if capLimited=True")
        add_summary(summary, "best_lower_bound", "R_L_pct",
                    best["R_L_pct"], "L_opt / L_ini * 100")
        add_summary(summary, "best_lower_bound", "eta_L_pct",
                    best["eta_L_pct"],
                    "(L_opt - L_ini) / L_ini * 100")
        add_summary(summary, "best_lower_bound", "lifeTotalP03sphere_J",
                    best["lifeTotalP03sphere_J"],
                    "Lifecycle cumulative 0-3 um effective radiation energy")
        add_summary(summary, "best_lower_bound", "eta_E_pct",
                    best["eta_E_pct"],
                    "(E_opt - E_ini) / E_ini * 100")
        add_summary(summary, "best_lower_bound", "U_pct",
                    best["U_pct"],
                    "Full-domain steady-state temperature uniformity")
        add_summary(summary, "best_lower_bound", "maxErosionTmax_K",
                    best["maxErosionTmax_K"],
                    "Maximum temperature during erosion loop")
        add_summary(summary, "best_lower_bound", "overtemp_margin_K",
                    MAX_TEMP_K - best["maxErosionTmax_K"],
                    "3273.15 K - maxErosionTmax_K")
        add_summary(summary, "best_lower_bound", "failureReached",
                    best["failureReached"],
                    "False means true failure lifetime is not measured")
        add_summary(summary, "best_lower_bound", "capLimited",
                    best["capLimited"], "True means lower-bound result")
        add_summary(summary, "compare_cylinder68", "E_gain_pct",
                    pct_change(best["lifeTotalP03sphere_J"],
                               CYLINDER_LEAD_E_J),
                    "Against C5 cylinder lead trial 68")
        add_summary(summary, "compare_cylinder68", "life_ratio_pct",
                    ratio_pct(best["lifetimeH"],
                              CYLINDER_LEAD_LIFETIME_H),
                    "Against C5 cylinder lead trial 68")
        add_summary(summary, "compare_cylinder68", "U_reduction_pct",
                    relative_reduction(best["U_pct"],
                                       CYLINDER_LEAD_U_PCT),
                    "Against C5 cylinder lead trial 68")
        add_summary(summary, "compare_c8_verified", "E_gain_pct",
                    pct_change(best["lifeTotalP03sphere_J"],
                               verify_best["lifeTotalP03sphere_J"]),
                    "Against fixed C8 verified trial 19")
        add_summary(summary, "compare_c8_verified", "life_ratio_pct",
                    ratio_pct(best["lifetimeH"],
                              verify_best["lifetimeH"]),
                    "Against fixed C8 verified trial 19")

    if measured_best is not None:
        add_summary(summary, "best_measured_failure", "trial",
                    measured_best["trial"],
                    "Best candidate whose failureReached=True")
        add_summary(summary, "best_measured_failure", "geometry",
                    format_geometry(measured_best),
                    "This is the conservative measured-failure candidate")
        add_summary(summary, "best_measured_failure", "objectiveScore",
                    measured_best["objectiveScore"], "")
        add_summary(summary, "best_measured_failure", "lifetimeH",
                    measured_best["lifetimeH"], "")
        add_summary(summary, "best_measured_failure", "eta_E_pct",
                    measured_best["eta_E_pct"], "")
        add_summary(summary, "best_measured_failure", "R_L_pct",
                    measured_best["R_L_pct"], "")
        add_summary(summary, "best_measured_failure", "U_pct",
                    measured_best["U_pct"], "")

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
             f"cap_limited={is_cap_limited(row)}; "
             f"measured_failure={is_measured_failure(row)}; "
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
        add_summary(summary, "status_ranges", status + "_z_first_mm",
                    stats_text(group, "z_first_mm"),
                    "Distribution by status")
        add_summary(summary, "status_ranges", status + "_side_mm",
                    stats_text(group, "side_mm"),
                    "Distribution by status")

    add_summary(summary, "decision", "zigzag_direction", "continue",
                "C8 local best strongly exceeds cylinder lead on eta_E and U")
    add_summary(summary, "decision", "current_best_is_lower_bound",
                bool(best and is_cap_limited(best)),
                "Do not treat cap-limited lifetime as final failure lifetime")
    add_summary(summary, "next_step", "recommended", "C10",
                "Verify C8-local trial 19 with higher max_lifetime_h")
    add_summary(summary, "next_step", "max_lifetime_h", "800 or 1000",
                "Needed to measure actual failure or a stronger lower bound")

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "section", "key", "value", "notes",
        ])
        writer.writeheader()
        writer.writerows(summary)

    return measured_best


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    number = finite_float(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}f}"


def write_report(rows, ranked, measured_best, verify_best):
    status_counts = Counter(row.get("status", "") for row in rows)
    best = ranked[0] if ranked else None
    ok_rows = [row for row in rows if is_ok(row)]

    report = []
    report.append("# C9 zigzag 局部搜索结果分析\n")
    report.append("## 输入文件\n")
    report.append(f"- `zigzag_family/ML/data/{C8_LOCAL_CSV_PATH.name}`")
    report.append(f"- `zigzag_family/ML/data/{C8_VERIFY_CSV_PATH.name}`\n")

    report.append("## 运行状态\n")
    report.append(md_table(
        ["项目", "数值"],
        [
            ["总 trial", str(len(rows))],
            ["OK", str(status_counts.get("OK", 0))],
            ["PRUNE_LIFETIME", str(status_counts.get("PRUNE_LIFETIME", 0))],
            ["FAIL_EROSION_SOLVE", str(status_counts.get("FAIL_EROSION_SOLVE", 0))],
            ["OK 比例", f"{len(ok_rows) / len(rows) * 100.0:.2f}%"],
        ],
    ))
    report.append("")

    if best is not None:
        report.append("## 当前最优候选\n")
        report.append(md_table(
            ["字段", "数值"],
            [
                ["trial", str(best["trial"])],
                ["geometry", format_geometry(best)],
                ["Vwork_V", fmt(best["Vwork_V"], 4)],
                ["lifetimeH", fmt(best["lifetimeH"], 4)],
                ["R_L_pct", fmt(best["R_L_pct"], 4)],
                ["eta_L_pct", fmt(best["eta_L_pct"], 4)],
                ["lifeTotalP03sphere_J", f"{best['lifeTotalP03sphere_J']:.6e}"],
                ["eta_E_pct", fmt(best["eta_E_pct"], 4)],
                ["U_pct", fmt(best["U_pct"], 4)],
                ["maxErosionTmax_K", fmt(best["maxErosionTmax_K"], 4)],
                ["failureReached", str(best["failureReached"])],
                ["capLimited", str(best["capLimited"])],
            ],
        ))
        report.append("")

    report.append("## 对比结论\n")
    comparison_rows = []
    if best is not None:
        comparison_rows.append([
            "C8-local 最优 trial " + str(best["trial"]),
            fmt(best["lifetimeH"], 4),
            f"{best['lifeTotalP03sphere_J']:.6e}",
            fmt(best["eta_E_pct"], 4),
            fmt(best["U_pct"], 4),
            str(is_cap_limited(best)),
        ])
    if measured_best is not None:
        comparison_rows.append([
            "C8-local 已失效最优 trial " + str(measured_best["trial"]),
            fmt(measured_best["lifetimeH"], 4),
            f"{measured_best['lifeTotalP03sphere_J']:.6e}",
            fmt(measured_best["eta_E_pct"], 4),
            fmt(measured_best["U_pct"], 4),
            str(is_cap_limited(measured_best)),
        ])
    comparison_rows.append([
        "C8 固定复核 trial 19",
        fmt(verify_best["lifetimeH"], 4),
        f"{verify_best['lifeTotalP03sphere_J']:.6e}",
        fmt(verify_best["eta_E_pct"], 4),
        fmt(verify_best["U_pct"], 4),
        str(verify_best["capLimited"]),
    ])
    comparison_rows.append([
        "C5 圆柱 trial 68",
        fmt(CYLINDER_LEAD_LIFETIME_H, 4),
        f"{CYLINDER_LEAD_E_J:.6e}",
        fmt(pct_change(CYLINDER_LEAD_E_J, BASELINE_E_J), 4),
        fmt(CYLINDER_LEAD_U_PCT, 4),
        "False",
    ])
    report.append(md_table(
        ["候选", "lifetimeH", "lifeTotalP03sphere_J", "eta_E_pct", "U_pct", "capLimited"],
        comparison_rows,
    ))
    report.append("")

    report.append("## C9 判断\n")
    report.append("- zigzag 方向继续作为主推方向：C8-local 最优候选在累计有效辐射能量和温度均匀性上明显强于圆柱 trial 68。")
    report.append("- 当前最优 trial 19 是 500 h 上限截断结果，`failureReached=False`、`capLimited=True`，因此 `lifetimeH=500 h` 和 `lifeTotalP03sphere_J` 只能作为保守下限，不能当作真实失效寿命。")
    report.append("- 已真实退蚀到失效的保守候选仍可使用 C8 固定复核 trial 19 / C8-local trial 0，寿命约 384.8 h，`eta_E` 约 2522%。")
    report.append("- C8-local 中 `FAIL_EROSION_SOLVE=9`，说明局部空间仍有较多重建/网格失败点；下一步不宜直接扩大 BO，应先对当前最优 trial 19 做更高寿命上限复核。\n")

    report.append("## 下一步\n")
    report.append("- C10：固定 C8-local trial 19 几何，把 `max_lifetime_h` 提高到 `800 h` 或 `1000 h`，测真实失效寿命或得到更强下限。")
    report.append("- 若 C10 达到失效：用真实 `lifetimeH` / `lifeTotalP03sphere_J` 作为最终 zigzag 主推候选。")
    report.append("- 若 C10 仍 cap-limited：可按 `>= 上限寿命` 报告保守下限，或继续提高上限做最终复核。")

    REPORT_MD_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    rows = read_rows(C8_LOCAL_CSV_PATH)
    verify_rows = read_rows(C8_VERIFY_CSV_PATH)
    if not verify_rows:
        raise RuntimeError(f"No rows found in {C8_VERIFY_CSV_PATH}")
    verify_best = verify_rows[0]

    ok_rows = [row for row in rows if is_ok(row)]
    ranked, pareto_trials = write_candidate_ranking(ok_rows, verify_best)
    measured_best = write_summary(
        rows, ok_rows, ranked, pareto_trials, verify_best)
    write_report(rows, ranked, measured_best, verify_best)

    best = ranked[0] if ranked else None
    print(f"rows={len(rows)}")
    print(f"ok={len(ok_rows)}")
    print(f"best_trial={best['trial'] if best else ''}")
    print(f"best_cap_limited={is_cap_limited(best) if best else ''}")
    print(f"ranking_csv={RANKING_CSV_PATH}")
    print(f"summary_csv={SUMMARY_CSV_PATH}")
    print(f"report_md={REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
