# C9 zigzag 局部搜索结果分析

## 输入文件

- `zigzag_family/ML/data/rematch_c8_local_trials.csv`
- `zigzag_family/ML/data/rematch_c8_trial19_verify.csv`

## 运行状态

| 项目 | 数值 |
| --- | --- |
| 总 trial | 20 |
| OK | 3 |
| PRUNE_LIFETIME | 8 |
| FAIL_EROSION_SOLVE | 9 |
| OK 比例 | 15.00% |

## 当前最优候选

| 字段 | 数值 |
| --- | --- |
| trial | 19 |
| geometry | N=12; L_RUN_mm=92.0111; z_first_mm=2.2666; side_mm=0.4958 |
| Vwork_V | 100.0000 |
| lifetimeH | 500.0000 |
| R_L_pct | 206.5440 |
| eta_L_pct | 106.5440 |
| lifeTotalP03sphere_J | 3.501654e+09 |
| eta_E_pct | 3216.0330 |
| U_pct | 97.0981 |
| maxErosionTmax_K | 2868.8820 |
| failureReached | False |
| capLimited | True |

## 对比结论

| 候选 | lifetimeH | lifeTotalP03sphere_J | eta_E_pct | U_pct | capLimited |
| --- | --- | --- | --- | --- | --- |
| C8-local 最优 trial 19 | 500.0000 | 3.501654e+09 | 3216.0330 | 97.0981 | True |
| C8-local 已失效最优 trial 0 | 384.8174 | 2.768882e+09 | 2522.1048 | 97.2924 | False |
| C8 固定复核 trial 19 | 384.8672 | 2.769180e+09 | 2522.3875 | 97.2924 | False |
| C5 圆柱 trial 68 | 378.3357 | 2.436791e+08 | 130.7618 | 127.0367 | False |

## C9 判断

- zigzag 方向继续作为主推方向：C8-local 最优候选在累计有效辐射能量和温度均匀性上明显强于圆柱 trial 68。
- 当前最优 trial 19 是 500 h 上限截断结果，`failureReached=False`、`capLimited=True`，因此 `lifetimeH=500 h` 和 `lifeTotalP03sphere_J` 只能作为保守下限，不能当作真实失效寿命。
- 已真实退蚀到失效的保守候选仍可使用 C8 固定复核 trial 19 / C8-local trial 0，寿命约 384.8 h，`eta_E` 约 2522%。
- C8-local 中 `FAIL_EROSION_SOLVE=9`，说明局部空间仍有较多重建/网格失败点；下一步不宜直接扩大 BO，应先对当前最优 trial 19 做更高寿命上限复核。

## 下一步

- C10：固定 C8-local trial 19 几何，把 `max_lifetime_h` 提高到 `800 h` 或 `1000 h`，测真实失效寿命或得到更强下限。
- 若 C10 达到失效：用真实 `lifetimeH` / `lifeTotalP03sphere_J` 作为最终 zigzag 主推候选。
- 若 C10 仍 cap-limited：可按 `>= 上限寿命` 报告保守下限，或继续提高上限做最终复核。
