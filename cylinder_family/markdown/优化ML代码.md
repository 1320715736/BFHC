# ML 运行问题记录（内部）

> 本文件是开发过程中的问题记录，不作为提交版技术说明。提交版运行流程以 `运行方法.md`、`baseline_final.md` 和 `ML/data/trials.csv` 为准。

---

## 当前处理状态

| 问题 | 当前状态 | 处理方式 |
|---|---|---|
| 中断后续跑跳过未完成 trial | 已通过 Optuna SQLite 持久化缓解 | `optuna_optimize.py` 从 `data/optuna.db` 读取已有 trial，继续剩余配额 |
| 个别构型 COMSOL 网格失败 | 已增强处理 | `comsol_runner.py` 已将网格降级重试扩展为 `hauto=4/5/6/7/8`，且初始建模和侵蚀几何更新共用同一套重试逻辑；全部失败时记录异常状态，不参与最优筛选 |
| COMSOL server 断联后产生无效结果 | 已增强处理 | `comsol_runner.py` 在每个 trial 开始前做 server/model 心跳检查，并对求解结果做有限值校验；侵蚀步求解失败不再返回 `OK` |
| 寿命剪枝状态被 `row.update(result)` 覆盖 | 已修复 | `optuna_optimize.py` 现在先合并 runner 结果，再写回优化层状态，确保寿命不足写为 `PRUNE_LIFETIME` |

---

## 当前提交数据

- `data/trials.csv`：当前提交数据为 150 组历史 trial 记录，其中 148 组按旧记录口径为 `OK`、2 组 `ERROR`；新版脚本后续会显式区分 `OK` 与 `PRUNE_LIFETIME`。
- `data/optuna.db`：Optuna 断点续跑数据库。
- `figures/`：2026-05-05 重新生成的 parse 与 surrogate 图表。

当前最优可行 trial 为 Trial 122，`lifeAvgP03sphere_W=559.64 W`，相对 baseline `428.57 W` 提升 `30.58%`。

---

## 后续可改进

1. 对网格失败构型增加自动几何诊断，记录失败边/域对应的段半径跳变。
2. 若需要严格补齐每一个 trial，可增加失败点重排队机制，而不是只记录 `ERROR`。
3. 若后续重新大规模运行，可考虑将历史 CSV 按新版状态口径重写，把寿命不足的旧 `OK` 行改标为 `PRUNE_LIFETIME`。
