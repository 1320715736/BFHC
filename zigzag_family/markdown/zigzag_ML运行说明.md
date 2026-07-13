# 蛇形簇（Zigzag Family）ML 优化运行说明

## 文件结构

```
zigzag_family/
├── ML/
│   ├── zigzag_runner.py       # COMSOL 求解器封装（单次 trial 评估）
│   ├── optuna_optimize.py     # Optuna TPE 贝叶斯优化主脚本
│   └── data/
│       ├── optuna.db          # Optuna SQLite 持久化数据库（自动创建）
│       ├── trials.csv         # 每次 trial 结果日志（自动追加）
│       └── failed_trial.json  # 断线时保存的待重跑参数（自动创建/删除）
├── zigzag_baseline.java       # COMSOL Desktop Java Shell 验证脚本
└── markdown/
    ├── zigzag_baseline_frozen.md   # 基准参数 & 实测结果（冻结）
    └── zigzag_ML运行说明.md        # 本文件
```

---

## 搜索空间

| 参数 | 范围 | 说明 |
|---|---|---|
| N_RUNS | {4,6,8,10,12,14,16} | 水平折线段数（偶数，保证电极对齐 x=0） |
| L_RUN_mm | [20, 300] mm | 每段水平长度 |
| z_first_mm | [0.6, 3.0] mm | 第一段 z 坐标（z_last = L0 - z_first，关于 L0/2 对称） |
| side | 自动 | 由体积守恒导出：side = sqrt(flexVol / pathLength) |

**优化目标**：最大化 `lifeAvgP03sphere_W`（寿命平均外接球 0-3μm 辐射功率）

**硬约束**：`lifetimeH >= 0.30 × 6.38 h ≈ 1.91 h`

**Trial 预算**：150 次（含 baseline Trial 0）

---

## 首次启动

```powershell
# 激活环境
conda activate BFHC

# 进入 ML 目录
Set-Location "d:\VScode\project\BFHC\zigzag_family\ML"

# 启动优化（自动建库、启动 COMSOL 服务器、注入 baseline 作为 Trial 0）
& "D:\conda_envs\BFHC\python.exe" optuna_optimize.py
```

首次运行会自动：
1. 创建 `data/` 目录和 `trials.csv`（含表头）
2. 启动 `comsolmphserver`
3. 以 baseline 参数（N=8, L=104mm, zf=0.8mm）作为 Trial 0 热启动
4. 随后运行 TPE 贝叶斯优化

---

## 断点续跑

直接重新执行相同命令即可，无需任何额外操作：

```powershell
Set-Location "d:\VScode\project\BFHC\zigzag_family\ML"
& "D:\conda_envs\BFHC\python.exe" optuna_optimize.py
```

脚本在 `study.create(..., load_if_exists=True)` 时自动加载已有数据库，并从上次中断处继续。

**中断情形的自动恢复逻辑**：

| 中断原因 | 自动处理 |
|---|---|
| 手动关闭终端（Ctrl+C / 关窗口） | 启动时扫描 DB 中 `RUNNING` 状态的 trial，重新入队其参数 |
| COMSOL 服务器断线 | 断线时将 trial 参数写入 `failed_trial.json`，下次启动时自动读取并重新入队 |
| 网格生成失败（FAIL_MESH） | 返回 FAIL_MESH 状态，标记 PRUNED，不重启服务器，继续下一 trial |

---

## 监控进度

**查看当前 trial 数和最优结果（另开一个 PowerShell）**：

```powershell
conda activate BFHC
python -c "
import optuna
study = optuna.load_study(
    study_name='zigzag_bo_v2',
    storage='sqlite:///d:/VScode/project/BFHC/zigzag_family/ML/data/optuna.db'
)
print(f'Trials: {len(study.trials)}')
print(f'Best value: {study.best_value:.2f} W')
print(f'Best params: {study.best_params}')
"
```

**查看 CSV 日志**（Excel 或 pandas）：

```python
import pandas as pd
df = pd.read_csv(r"d:\VScode\project\BFHC\zigzag_family\ML\data\trials.csv")
# 过滤掉 RUNNING 标记行（每个 trial 开始时写入，中断时保留为记录）
df = df[df["status"] != "RUNNING"]
print(df.sort_values("lifeAvgP03sphere_W", ascending=False).head(10))
```

---

## CSV 字段说明

| 列名 | 说明 |
|---|---|
| trial | Optuna trial 编号（从 0 开始） |
| N_RUNS | 折线段数 |
| L_RUN_mm | 每段水平长度 (mm) |
| z_first_mm | 第一段 z 坐标 (mm) |
| side_mm | 方截面边长 (mm，由体积守恒导出) |
| Vwork_V | 工作电压 (V) |
| initialTmax_K | 初始最高温度 (K) |
| lifetimeH | 寿命 (h) |
| initialP03sphere_W | 初始外接球 0-3μm 辐射功率 (W) |
| initialPradSphere_W | 初始外接球全波段辐射功率 (W) |
| lifeAvgP03sphere_W | **优化目标**：寿命平均外接球 0-3μm 功率 (W) |
| lifeAvgPradSphere_W | 寿命平均外接球全波段功率 (W) |
| selfViewLoss_pct | 自遮挡损失 = (1 - P03sphere/P03emit)×100% |
| failureReached | 是否达到 20% 失效准则 |
| erosionSteps | 侵蚀循环步数 |
| status | OK / PRUNE_LIFETIME / FAIL_MESH / FAIL_Z_OVERLAP / SERVER_DISCONNECT / RUNNING |
| elapsed_sec | 该 trial 耗时 (s) |

> **注**：每个 trial 在开始时写一行 `status=RUNNING`，完成后再写一行最终状态。
> 分析时过滤掉 `status=="RUNNING"` 的行，或用 `drop_duplicates(subset=["trial"], keep="last")`。

---

## 失败状态说明

| status | 原因 | 处理方式 |
|---|---|---|
| PRUNE_LIFETIME | 寿命 < 1.91h | PRUNED，不计入最优 |
| FAIL_MESH | 网格在 hauto=5..9 均失败 | PRUNED，不重启服务器 |
| FAIL_Z_OVERLAP | z_step < 1.2×side（blocks 重叠） | 几何预检直接返回 |
| FAIL_SIDE_TOO_SMALL | side < 0.1mm | 几何预检直接返回 |
| SERVER_DISCONNECT | COMSOL 服务器断线 | 立即退出，参数保存待续跑 |
| ERROR: ... | COMSOL 内部异常 | 重试 2 次，仍失败则 PRUNED |

---

## 参考基准（Baseline Trial 0）

| 参数 | 值 |
|---|---|
| N_RUNS | 8 |
| L_RUN | 104 mm |
| z_first | 0.8 mm |
| side | 0.5700 mm |
| lifetimeH | 6.38 h（Python/mph） |
| lifeAvgP03sphere_W | 3549.5 W |

最优结果应超过 baseline 的 `lifeAvgP03sphere_W = 3549.5 W`。
