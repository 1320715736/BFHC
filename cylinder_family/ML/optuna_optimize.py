"""
optuna_optimize.py — 贝叶斯优化主脚本
=====================================
用 Optuna TPE 搜索 8 段圆柱钨棒的最优半径配置。

搜索空间（3 个自由维度，条件边界保证体积守恒）:
  r1 ∈ [R_MIN, R_MAX]
  r2 ∈ [r2_lo, r2_hi]  （条件边界：留余量给 r3, r4 ≥ R_MIN）
  r3 ∈ [r3_lo, r3_hi]  （条件边界：留余量给 r4 ≥ R_MIN）
  r4 = sqrt(4·R0² - r1² - r2² - r3²)
  对称: [r1, r2, r3, r4, r4, r3, r2, r1]

失效判据（按赛题要求 per-segment）:
  任意 (r_i(0) - r_i(t)) / r_i(0) ≥ 20% 即失效

优化目标: maximize lifeAvgP03sphere_W
硬约束:   lifetime >= 34.65 h  (30% × 115.50 h baseline)

调用方式:
  conda activate BFHC
  python optuna_optimize.py
"""

import math
import time
import csv
import optuna
from pathlib import Path
from comsol_runner import COMSOLRunner

# ============================================================
#  配置
# ============================================================

# 项目路径
ML_DIR = Path(r"D:\VScode\project\BFHC\cylinder_family\ML")
DATA_DIR = ML_DIR / "data"

# Optuna 持久化；STUDY_NAME 保留 v2 后缀以兼容既有 optuna.db
DB_PATH = DATA_DIR / "optuna.db"
STUDY_NAME = "cylinder_bo_v2"
CSV_PATH = DATA_DIR / "trials.csv"

# 物理常数
R0_MM = 2.5          # baseline 半径 (mm)
R0_M = R0_MM * 1e-3  # baseline 半径 (m)
SEG_COUNT = 8
R_MIN_MM = 0.8       # 最小允许半径 (mm)
R_MAX_MM = 4.5       # 最大允许半径 (mm)

# 约束（基线来自 cylinder_baseline.java 均匀圆柱，Fix-1 IntSurface 段温度已验证）
BASELINE_LIFETIME_H = 115.50
MIN_LIFETIME_H = 0.30 * BASELINE_LIFETIME_H  # 34.65 h

# 优化预算
N_TRIALS = 151

# 全局 runner（在 main 中初始化）
runner: COMSOLRunner = None

# ============================================================
#  体积守恒：计算 r4
# ============================================================

# 总平方半径预算 r1²+r2²+r3²+r4² = 4 × R0² = 25.0
TOTAL_SQ = 4.0 * R0_MM**2  # 25.0

def compute_r4(r1_mm, r2_mm, r3_mm):
    """从 r1,r2,r3 (mm) 计算 r4 (mm)，满足体积守恒。
    返回 r4_mm 或 None（无解/太小）。
    """
    remainder = TOTAL_SQ - r1_mm**2 - r2_mm**2 - r3_mm**2
    if remainder < R_MIN_MM**2:
        return None
    r4 = math.sqrt(remainder)
    if r4 > R_MAX_MM:
        return None
    return r4

# ============================================================
#  CSV 日志
# ============================================================

CSV_HEADER = [
    "trial", "r1_mm", "r2_mm", "r3_mm", "r4_mm",
    "Vwork_V", "initialTmax_K", "lifetimeH",
    "initialP03sphere_W", "initialPradSphere_W",
    "lifeAvgP03sphere_W", "lifeAvgPradSphere_W",
    "selfViewLoss_pct", "failureReached", "erosionSteps",
    "status", "elapsed_sec"
]

def init_csv():
    """如果 CSV 不存在则创建并写入表头。"""
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_csv(row_dict):
    """追加一行到 CSV。"""
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, "") for h in CSV_HEADER])


def compact_status(status, limit=240):
    """压缩异常状态，避免 COMSOL 多行错误把 CSV 撑得难以阅读。"""
    text = " ".join(str(status).split())
    return text[:limit]

# ============================================================
#  Optuna 目标函数
# ============================================================

def objective(trial):
    global runner
    t_start = time.time()
    trial_num = trial.number

    # 1) 采样 r1（全范围）
    r1 = trial.suggest_float("r1_mm", R_MIN_MM, R_MAX_MM)

    # 2) 采样 r2（条件边界：留余量给 r3, r4 ≥ R_MIN）
    r2_max_sq = TOTAL_SQ - r1**2 - 2 * R_MIN_MM**2
    r2_min_sq = TOTAL_SQ - r1**2 - 2 * R_MAX_MM**2
    r2_lo = max(R_MIN_MM, math.sqrt(max(0.0, r2_min_sq)))
    r2_hi = min(R_MAX_MM, math.sqrt(max(R_MIN_MM**2, r2_max_sq)))
    r2 = trial.suggest_float("r2_mm", r2_lo, r2_hi)

    # 3) 采样 r3（条件边界：留余量给 r4 ≥ R_MIN）
    r3_max_sq = TOTAL_SQ - r1**2 - r2**2 - R_MIN_MM**2
    r3_min_sq = TOTAL_SQ - r1**2 - r2**2 - R_MAX_MM**2
    r3_lo = max(R_MIN_MM, math.sqrt(max(0.0, r3_min_sq)))
    r3_hi = min(R_MAX_MM, math.sqrt(max(R_MIN_MM**2, r3_max_sq)))
    r3 = trial.suggest_float("r3_mm", r3_lo, r3_hi)

    # 4) r4 由体积守恒确定（边界保证有效）
    r4_sq = TOTAL_SQ - r1**2 - r2**2 - r3**2
    r4 = math.sqrt(max(R_MIN_MM**2, r4_sq))

    # 对称排列
    radii_mm = [r1, r2, r3, r4, r4, r3, r2, r1]
    radii_m = [r * 1e-3 for r in radii_mm]

    print(f"\n{'=' * 60}")
    print(f"Trial {trial_num}: r = "
          f"[{', '.join(f'{r:.3f}' for r in radii_mm)}] mm")
    print(f"{'=' * 60}")

    # 5) 调用 COMSOL 求解
    try:
        result = runner.evaluate(radii_m)
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  ERROR: {e}")
        append_csv({"trial": trial_num, "r1_mm": r1, "r2_mm": r2,
                     "r3_mm": r3, "r4_mm": r4,
                     "status": compact_status(f"ERROR: {e}"),
                     "elapsed_sec": round(elapsed, 1)})
        raise optuna.TrialPruned(f"COMSOL error: {e}")

    elapsed = time.time() - t_start

    if result.get("status") != "OK":
        print(f"  FAILED: {result.get('status')}")
        row = {"trial": trial_num, "r1_mm": r1, "r2_mm": r2,
               "r3_mm": r3, "r4_mm": r4}
        row.update(result)
        row["status"] = compact_status(result.get("status", "UNKNOWN"))
        row["elapsed_sec"] = round(elapsed, 1)
        append_csv(row)
        raise optuna.TrialPruned(result.get("status"))

    lifetime = result["lifetimeH"]
    target = result["lifeAvgP03sphere_W"]

    print(f"  Vwork={result['Vwork_V']:.4f}V  "
          f"Tmax={result['initialTmax_K']:.1f}K  "
          f"Lifetime={lifetime:.2f}h  AvgP03={target:.2f}W  "
          f"[{elapsed:.0f}s]")

    # 5) 硬约束：寿命
    status = "OK"
    if lifetime < MIN_LIFETIME_H:
        status = "PRUNE_LIFETIME"

    # 记录 CSV
    row = {
        "trial": trial_num, "r1_mm": r1, "r2_mm": r2,
        "r3_mm": r3, "r4_mm": r4,
    }
    row.update(result)
    # result["status"] 是 runner 物理求解状态；这里要保留优化层剪枝状态。
    row["status"] = status
    row["elapsed_sec"] = round(elapsed, 1)
    append_csv(row)

    if status != "OK":
        raise optuna.TrialPruned(
            f"Lifetime {lifetime:.2f}h < {MIN_LIFETIME_H:.2f}h")

    return target  # maximize

# ============================================================
#  主入口
# ============================================================

def main():
    global runner

    # 创建目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_csv()

    print(f"CSV output:     {CSV_PATH}")
    print(f"Optuna DB:      sqlite:///{DB_PATH}")
    print(f"Trials:         {N_TRIALS}")
    print(f"Search space:   r1,r2,r3 ∈ [{R_MIN_MM}, {R_MAX_MM}] mm")
    print(f"Baseline:       Lifetime={BASELINE_LIFETIME_H}h, "
          f"min={MIN_LIFETIME_H:.2f}h")
    print()

    # 启动 COMSOL 服务器
    runner = COMSOLRunner()
    runner.start()

    try:
        # Optuna study（SQLite 持久化支持断点续跑）
        storage = f"sqlite:///{DB_PATH}"
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )

        # 如果是全新 study，先注入 baseline 作为第一个 trial
        if len(study.trials) == 0:
            study.enqueue_trial({
                "r1_mm": R0_MM,
                "r2_mm": R0_MM,
                "r3_mm": R0_MM,
            })

        n_remaining = N_TRIALS - len(study.trials)
        if n_remaining <= 0:
            print(f"Already have {len(study.trials)} trials, "
                  f"target is {N_TRIALS}. Done.")
        else:
            print(f"Resuming from {len(study.trials)} trials, "
                  f"running {n_remaining} more...")
            study.optimize(objective, n_trials=n_remaining)

        # 打印最佳结果
        print("\n" + "=" * 60)
        print("  OPTIMIZATION COMPLETE")
        print("=" * 60)
        best = study.best_trial
        print(f"Best trial:     #{best.number}")
        print(f"Best value:     {best.value:.2f} W (lifeAvgP03sphere)")
        print(f"Best params:    r1={best.params['r1_mm']:.3f}  "
              f"r2={best.params['r2_mm']:.3f}  "
              f"r3={best.params['r3_mm']:.3f} mm")
        r4 = compute_r4(best.params["r1_mm"],
                         best.params["r2_mm"],
                         best.params["r3_mm"])
        if r4:
            radii = [best.params["r1_mm"], best.params["r2_mm"],
                     best.params["r3_mm"], r4, r4,
                     best.params["r3_mm"], best.params["r2_mm"],
                     best.params["r1_mm"]]
            print(f"Full radii (mm): "
                  f"[{', '.join(f'{r:.3f}' for r in radii)}]")
        print(f"\nTotal trials:   {len(study.trials)}")
        print(f"Results saved:  {CSV_PATH}")

    finally:
        runner.stop()


if __name__ == "__main__":
    main()
