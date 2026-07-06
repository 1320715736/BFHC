"""
optuna_optimize.py — 折线构型贝叶斯优化主脚本（v2，MultipleSpectralBands）
==========================================================================
用 Optuna TPE 搜索平面折线钨丝的最优几何配置。
S2S 辐射模型：MultipleSpectralBands（ε₀₃=0.35，ε_rest=0.15），对应 zigzag_runner.py。

搜索空间（3 个自由维度）:
  N_RUNS    ∈ {4, 6, 8, 10, 12, 14, 16}  (偶数, 保证首尾电极对齐 x=0)
  L_RUN_mm  ∈ [20, 300]                   (每段水平长度)
  z_first_mm ∈ [0.6, 3.0]                 (第一段 z 坐标, z_last = L0 - z_first)

体积守恒: side = sqrt(flexVol / pathLength)  自动满足
失效判据: 任意 block 边长损失 ≥ 20% (与赛题要求一致)

优化目标: maximize lifeAvgP03sphere_W
硬约束:   lifetime >= 0.30 × CYLINDER_BASELINE_LIFETIME_H

注意：赛题中的“初始形状”指直径 5 mm、高 15 mm 的圆柱钨坯料。
      因此 zigzag 族的寿命硬约束必须对齐圆柱簇 baseline，而不是
      zigzag trial 0 的 30%。

调用方式:
  conda activate BFHC
  python optuna_optimize.py
"""

import json
import math
import time
import csv
import optuna
from optuna.trial import TrialState
from pathlib import Path
from zigzag_runner import COMSOLRunner, ServerDisconnectError

# ============================================================
#  配置
# ============================================================

ML_DIR = Path(r"D:\VScode\project\BFHC\zigzag_family\ML")
DATA_DIR = ML_DIR / "data"

# Optuna 持久化
DB_PATH = DATA_DIR / "optuna.db"
STUDY_NAME = "zigzag_bo_v3_cylinder_life"   # v3：寿命硬约束对齐圆柱初始坯料
CSV_PATH = DATA_DIR / "trials.csv"
FAILED_TRIAL_PATH = DATA_DIR / "failed_trial.json"  # 断线时保存待重跑的 trial 参数

# 搜索空间
N_RUNS_CHOICES = [4, 6, 8, 10, 12, 14, 16]  # 偶数
L_RUN_MIN_MM = 20.0
L_RUN_MAX_MM = 300.0
Z_FIRST_MIN_MM = 0.6
Z_FIRST_MAX_MM = 3.0

# Zigzag 几何参考参数（仅用于注入第一个参考 trial，不用于寿命门槛）
BASELINE_N_RUNS = 8
BASELINE_L_RUN_MM = 104.0
BASELINE_Z_FIRST_MM = 0.8

# 约束
# 赛题口径：优化后寿命不得低于“圆柱初始坯料”寿命的 30%。
# 圆柱簇提交版 baseline（cylinder_family/src/cylinder_baseline.java / trials.csv trial 0）：
# RESULT lifetimeH = 115.5037 h
CYLINDER_BASELINE_LIFETIME_H = 115.5037
MIN_LIFETIME_H = 0.30 * CYLINDER_BASELINE_LIFETIME_H  # ≈ 34.65 h

# Zigzag Java baseline 只作为折线族参考，不作为硬约束门槛：
# RESULT=100.0000,3209.3,7.1277,3711.29,3881.58,3539.04,3703.73,-3.77,true,11
ZIGZAG_JAVA_BASELINE_LIFETIME_H = 7.1277

# 优化预算
N_TRIALS = 150

# 全局 runner
runner: COMSOLRunner = None


def safe_exception_text(exc):
    """COMSOL/JVM 崩溃后 Java 异常字符串化也可能失败。"""
    try:
        return str(exc)
    except Exception:
        return exc.__class__.__name__

# ============================================================
#  CSV 日志
# ============================================================

CSV_HEADER = [
    "trial", "N_RUNS", "L_RUN_mm", "z_first_mm", "side_mm",
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


def finalize_csv(row_dict):
    """Replace the latest unfinished row for this trial; append if none exists."""
    trial_id = str(row_dict.get("trial", ""))
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    full_row = {h: row_dict.get(h, "") for h in CSV_HEADER}
    replace_idx = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get("trial") == trial_id and rows[i].get("status") in ("RUNNING", "SERVER_DISCONNECT"):
            replace_idx = i
            break

    if replace_idx is None:
        rows.append(full_row)
    else:
        rows[replace_idx] = full_row

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
#  Optuna 目标函数
# ============================================================

def objective(trial):
    global runner
    t_start = time.time()
    trial_num = trial.number

    # 1) 采样 N_RUNS（离散偶数）
    N_RUNS = trial.suggest_categorical("N_RUNS", N_RUNS_CHOICES)

    # 2) 采样 L_RUN
    L_RUN_mm = trial.suggest_float("L_RUN_mm", L_RUN_MIN_MM, L_RUN_MAX_MM)

    # 3) 采样 z_first
    z_first_mm = trial.suggest_float(
        "z_first_mm", Z_FIRST_MIN_MM, Z_FIRST_MAX_MM)

    # 转换为 SI 单位
    L_RUN_m = L_RUN_mm * 1e-3
    z_first_m = z_first_mm * 1e-3

    # 预计算 side 用于日志
    side, _, plen = runner.compute_side_and_blocks(
        N_RUNS, L_RUN_m, z_first_m)
    side_mm = side * 1e3

    print(f"\n{'=' * 60}")
    print(f"Trial {trial_num}: N_RUNS={N_RUNS}  "
          f"L_RUN={L_RUN_mm:.1f}mm  z_first={z_first_mm:.2f}mm  "
          f"side={side_mm:.4f}mm  path={plen * 1e3:.1f}mm")
    print(f"{'=' * 60}")

    # 立即写入 RUNNING 记录（确保 CSV 序号连续，中断的 trial 也有记录）
    append_csv({
        "trial": trial_num, "N_RUNS": N_RUNS,
        "L_RUN_mm": L_RUN_mm, "z_first_mm": z_first_mm,
        "side_mm": side_mm, "status": "RUNNING",
    })

    # 4) 调用 COMSOL 求解（网格/几何失败不重试服务器；断线立即退出）
    MAX_RETRIES = 2
    result = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = runner.evaluate(
                N_RUNS=N_RUNS, L_RUN_m=L_RUN_m, z_first_m=z_first_m)
            break  # 成功，跳出重试循环
        except ServerDisconnectError as e:
            # 服务器断线：保存参数到文件，立即退出（不重试）
            elapsed = time.time() - t_start
            finalize_csv({
                "trial": trial_num, "N_RUNS": N_RUNS,
                "L_RUN_mm": L_RUN_mm, "z_first_mm": z_first_mm,
                "side_mm": side_mm,
                "status": "SERVER_DISCONNECT",
                "elapsed_sec": round(elapsed, 1),
            })
            with open(FAILED_TRIAL_PATH, "w") as fp:
                json.dump({"N_RUNS": N_RUNS, "L_RUN_mm": L_RUN_mm,
                           "z_first_mm": z_first_mm}, fp)
            print(f"  SERVER_DISCONNECT: params saved to {FAILED_TRIAL_PATH}")
            raise  # 传播到 study.optimize() → main() 捕获并退出
        except Exception as e:
            err_msg = safe_exception_text(e)
            print(f"  ERROR (attempt {attempt + 1}/{MAX_RETRIES + 1}): {err_msg}")
            if attempt < MAX_RETRIES:
                print("  -> Restarting COMSOL server...")
                try:
                    runner.stop()
                except Exception:
                    pass
                time.sleep(5)
                runner.start()
                print("  -> COMSOL reconnected, retrying...")
            else:
                elapsed = time.time() - t_start
                finalize_csv({
                    "trial": trial_num, "N_RUNS": N_RUNS,
                    "L_RUN_mm": L_RUN_mm, "z_first_mm": z_first_mm,
                    "side_mm": side_mm,
                    "status": f"ERROR: {err_msg}",
                    "elapsed_sec": round(elapsed, 1),
                })
                raise optuna.TrialPruned(f"COMSOL error after {MAX_RETRIES + 1} attempts: {err_msg}")

    elapsed = time.time() - t_start

    if result.get("status") != "OK":
        print(f"  FAILED: {result.get('status')}")
        finalize_csv({
            "trial": trial_num, "N_RUNS": N_RUNS,
            "L_RUN_mm": L_RUN_mm, "z_first_mm": z_first_mm,
            "side_mm": side_mm,
            "status": result.get("status", "UNKNOWN"),
            "elapsed_sec": round(elapsed, 1),
        })
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
        "trial": trial_num, "N_RUNS": N_RUNS,
        "L_RUN_mm": L_RUN_mm, "z_first_mm": z_first_mm,
        "side_mm": side_mm, "status": status,
        "elapsed_sec": round(elapsed, 1),
    }
    row.update(result)
    row["status"] = status
    finalize_csv(row)

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
    print(f"Search space:   N_RUNS ∈ {N_RUNS_CHOICES}")
    print(f"                L_RUN ∈ [{L_RUN_MIN_MM}, {L_RUN_MAX_MM}] mm")
    print(f"                z_first ∈ [{Z_FIRST_MIN_MM}, {Z_FIRST_MAX_MM}] mm")
    print(f"Zigzag ref:     N={BASELINE_N_RUNS} L={BASELINE_L_RUN_MM}mm "
          f"zf={BASELINE_Z_FIRST_MM}mm")
    print(f"Life constraint: cylinder baseline {CYLINDER_BASELINE_LIFETIME_H:.4f}h "
          f"× 30% = {MIN_LIFETIME_H:.2f}h")
    print(f"Zigzag Java ref: lifetime={ZIGZAG_JAVA_BASELINE_LIFETIME_H:.4f}h "
          "(reference only, not the constraint)")
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
                "N_RUNS": BASELINE_N_RUNS,
                "L_RUN_mm": BASELINE_L_RUN_MM,
                "z_first_mm": BASELINE_Z_FIRST_MM,
            })

        # 恢复被中断的 trial（断线保存的参数 / Ctrl+C 中断的 RUNNING trial）
        if FAILED_TRIAL_PATH.exists():
            with open(FAILED_TRIAL_PATH) as fp:
                failed_params = json.load(fp)
            study.enqueue_trial(failed_params)
            FAILED_TRIAL_PATH.unlink()
            print(f"  Re-enqueued disconnected trial: {failed_params}")

        for t in study.trials:
            if t.state == TrialState.RUNNING and t.params:
                study.enqueue_trial(t.params)
                print(f"  Re-enqueued interrupted trial #{t.number}: {t.params}")

        n_remaining = N_TRIALS - len(study.trials)
        if n_remaining <= 0:
            print(f"Already have {len(study.trials)} trials, "
                  f"target is {N_TRIALS}. Done.")
        else:
            print(f"Resuming from {len(study.trials)} trials, "
                  f"running {n_remaining} more...")
            try:
                study.optimize(objective, n_trials=n_remaining)
            except ServerDisconnectError:
                print("\nCOMSOL server disconnected. Optimization paused.")
                if FAILED_TRIAL_PATH.exists():
                    print(f"Retry params saved to: {FAILED_TRIAL_PATH}")
                print("Re-run the script to resume from the interrupted trial.")

        # 打印最佳结果
        print("\n" + "=" * 60)
        print("  OPTIMIZATION COMPLETE")
        print("=" * 60)
        complete_trials = [
            t for t in study.trials
            if t.state == TrialState.COMPLETE and t.value is not None
        ]
        if complete_trials:
            best = study.best_trial
            print(f"Best trial:     #{best.number}")
            print(f"Best value:     {best.value:.2f} W (lifeAvgP03sphere)")
            print(f"Best params:    N_RUNS={best.params['N_RUNS']}  "
                  f"L_RUN={best.params['L_RUN_mm']:.1f}mm  "
                  f"z_first={best.params['z_first_mm']:.2f}mm")

            # 计算 side
            side_best, _, plen_best = runner.compute_side_and_blocks(
                best.params["N_RUNS"],
                best.params["L_RUN_mm"] * 1e-3,
                best.params["z_first_mm"] * 1e-3)
            print(f"Best side:      {side_best * 1e3:.4f} mm")
            print(f"Best path len:  {plen_best * 1e3:.1f} mm")
        else:
            print("No feasible COMPLETE trial yet under the cylinder-based lifetime constraint.")
            print(f"Life floor:     {MIN_LIFETIME_H:.2f} h")
        print(f"\nTotal trials:   {len(study.trials)}")
        print(f"Results saved:  {CSV_PATH}")

    finally:
        runner.stop()


if __name__ == "__main__":
    main()
