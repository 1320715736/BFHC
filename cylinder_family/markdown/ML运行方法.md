# ML 优化运行方法 — cylinder_family

> 提交版流程：`optuna_optimize.py` 调用 `comsol_runner.py` 生成记录；优化结束后依次运行 `parse_results.py` 和 `surrogate_analysis.py` 生成图表与可解释性分析。
> COMSOL 物理模板对齐 `src/cylinder_baseline.java`，基线寿命 115.50 h。

---

## 1. 启动圆柱簇优化

```powershell
cd "d:\VScode\project\BFHC\cylinder_family\ML"
conda activate BFHC
python optuna_optimize.py
```

运行逻辑：

- `optuna_optimize.py` 负责 Optuna TPE 贝叶斯采样、体积守恒条件边界、断点续跑和 CSV/SQLite 记录。
- `comsol_runner.py` 负责连接 COMSOL server，从空白模型复现 `cylinder_baseline.java` 的建模、电压搜索和侵蚀寿命循环。
- 运行后会在 `data/` 下生成或更新 `trials.csv` 与 `optuna.db`。

Trial 0 是均匀圆柱 baseline，结果应接近：

```text
Vwork=0.6348 V
initialTmax=3196.2 K
lifetimeH=115.50 h
lifeAvgP03sphere=428.57 W
```

当前提交数据为 150 组历史 trial，其中 148 组正常完成、2 组因 COMSOL 网格失败记录为 ERROR；最优可行结果为 Trial 122。新版脚本后续会进一步区分 `OK` 与 `PRUNE_LIFETIME`，寿命不足但成功求解的 trial 不再伪装为可行最优。

---

## 2. 断点续跑

优化中途中断后，重新执行同一命令即可：

```powershell
python optuna_optimize.py
```

Optuna 会从 `data/optuna.db` 读取已有 trial，`trials.csv` 保留已完成记录。若某个 trial 在 COMSOL 网格、server 心跳或侵蚀步求解阶段失败，会记录异常状态，不参与最优筛选；若成功求解但寿命低于 34.65 h，会记录为 `PRUNE_LIFETIME`。

---

## 3. 解析优化结果

```powershell
python parse_results.py
```

输出：

- `figures/optimization_history.png`：优化历史曲线
- `figures/pareto_front.png`：功率-寿命权衡关系
- `figures/parallel_coordinate.png`：半径参数与目标的平行坐标图
- `figures/scatter_r1_r3.png`：r1-r3 设计空间散点图

---

## 4. 代理模型与 SHAP 分析

```powershell
python surrogate_analysis.py
```

输出：

- `figures/feature_importance.png`：随机森林特征重要性
- `figures/shap_summary_p03.png`：AvgP03 的 SHAP 全局解释
- `figures/shap_summary_life.png`：寿命的 SHAP 全局解释
- `figures/shap_dependence_r3.png`：固定展示 r3 的 SHAP 依赖关系
- `figures/response_surface.png`：代理模型响应面
- `figures/surrogate_optimal.png`：代理模型候选与实测最优对比
- `figures/shap_force_optimal.png`：最优样本的局部解释

---

## 5. 关键参数

| 项目 | 当前值 |
|---|---:|
| 总 trial 数 | 150 |
| baseline 寿命 | 115.50 h |
| 寿命硬约束下限 | 34.65 h |
| baseline lifeAvgP03sphere | 428.57 W |
| 当前最优 trial | 122 |
| 当前最优 lifeAvgP03sphere | 559.64 W |
| 当前最优寿命 | 35.27 h |
| 相对 baseline 提升 | 30.58% |

---

## 6. 文件结构

```text
cylinder_family/ML/
├── optuna_optimize.py     # 贝叶斯优化主脚本
├── comsol_runner.py       # COMSOL mph 接口与物理求解器
├── parse_results.py       # 优化结果解析与基础可视化
├── surrogate_analysis.py  # 随机森林代理模型与 SHAP 可解释性分析
├── data/
│   ├── trials.csv         # trial 结果日志
│   └── optuna.db          # Optuna 持久化数据库
└── figures/               # parse/surrogate 生成的图表
```
