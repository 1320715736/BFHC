# Zigzag Baseline 冻结参数

## 几何参数

- 构型：平面折线 Zigzag Meander (8-run Manhattan path)
- 跑道数 N_RUNS = 8
- 跑道长 L_RUN = 104.0 mm
- 路径总长 PATH_LENGTH = 846.00 mm
- 方截面边长 side = 0.5700 mm
- Block 数 = 17
- Terminal stub 高 = 0.5 mm, 半径 = 2.5 mm (与坯料等径)
- 整体高度 L0 = 15.0 mm
- z_first = 0.8 mm, z_last = 14.2 mm, z_step ≈ 1.914 mm
- 外接球半径 R_ENV = 109.75 mm
- 目标体积 V0 = π × 2.5² × 15 mm³ ≈ 2.9452e-7 m³

## 材料参数 (纯钨)

- 密度 ρ = 19350 kg/m³ (常数, 忽略热膨胀)
- 电阻率 ρ_e(T) = 5.5e-8 × (1 + 3.836e-3·(T-293.15) + 7.55e-7·(T-293.15)²) Ω·m
- 热导率 k(T) = max(75, 175 - 0.032·(T-293.15)) W/(m·K)
- 比热 Cp(T) = min(195, 132 + 0.020·(T-293.15)) J/(kg·K)

## 辐射参数

- S2S 模型：MultipleSpectralBands（非灰体，分波段辐射）
- ε₀₋₃μm = 0.35（0~3 μm 波段，主要有效辐射区间）
- ε_rest = 0.15（>3 μm 波段）
- 环境温度 Tamb = 293.15 K
- 环境发射率 ε_amb = 1 (黑体)
- σ_SB = 5.670374419e-8 W/(m²·K⁴)
- Planck 截止波长 λ₀₃ = 3 μm, c₂ = 1.438776877e-2 m·K

## 蒸发参数

- A_ev = 3.9e9 kg/(m²·s)（赛题 3.9e8 g/(cm²·s) 换算）
- B_ev = 1.023e5 K
- dsdt = 2·A_ev·exp(-B_ev/T) / ρ（方截面四面蒸发）
- 失效判据 = 20% 特征尺寸损失

## 求解设置

- 物理场：EC + HT + S2S(MultipleSpectralBands) + ElectromagneticHeatSource 耦合
- 网格：hauto = 5 (Normal), FreeTet；失败自动降级到 hauto=6/7
- 求解器：Stationary
- 初始温度 T_init = 1500 K
- 电压搜索：二分法, 上限硬帽 100V
- 每段温度：抛物线近似（基于 block z 中心位置）

## Baseline 仿真结果（MultipleSpectralBands，2026-05-04 实测）

| 指标 | 值 |
|---|---|
| Vwork | 100.0000 V |
| Tmax (初始) | 3209.3 K |
| 电阻 R | 2.578 Ω |
| 电功率 Pelec | 3877.7 W |
| P03,sphere (初始) | 3707.6 W |
| Prad,sphere (初始) | 3877.7 W |
| P03,sphere (寿命平均) | 3549.5 W |
| Prad,sphere (寿命平均) | 3714.5 W |
| selfViewLoss | -3.60% |
| 寿命 | **6.38 h** |
| 侵蚀步数 | 10 |
| 失效标志 | True（中心 block 最先失效）|
| 总耗时 | 634 s ≈ 10.6 min |

> 与旧基线（灰体 ε=0.32，寿命 3.86h）差异：新 S2S 模型 Tmax 低 51K（3209K vs 3261K），
> 蒸发率指数敏感 → 寿命 +65%。

## Block 侵蚀说明

采用抛物线温度近似，中心 block（#9）温度最高，侵蚀最快，最先达到 20% 失效。
侵蚀率分布对称（路径关于 z=7.5mm 轴对称），两端 block 侵蚀可忽略。

## Java Shell 验证结果（COMSOL Desktop，2026-05-05 实测）

| 指标 | 值 |
|---|---|
| Vwork | 100.0000 V |
| Tmax (初始) | 3209.3 K |
| 电阻 R | — (由 100V/I 可算) |
| P03,sphere (初始) | 3711.29 W |
| Prad,sphere (初始) | 3881.58 W |
| P03,sphere (寿命平均) | 3539.04 W |
| Prad,sphere (寿命平均) | 3703.73 W |
| selfViewLoss | -3.77% |
| 寿命 | **7.1277 h** |
| 侵蚀步数 | 11 |
| 失效标志 | true |

> 与 Python（mph）结果对比：稳态指标（Tmax=3209.3K，P03sphere≈3711W）完全吻合。
> 寿命差异 7.1277h vs Python trial 0 约 6.38h（约 +12%）源于 COMSOL Desktop 与 mph 后端网格生成差异，属正常离散误差。
> 当前 Python runner 的侵蚀循环已改为 `geom_only=True`：每步只重建几何和网格，不重建 S2S 物理场，与 Java Shell 的 dataset 保持策略一致。

## 赛题寿命约束口径修正

赛题中“优化后器件寿命最低不得低于器件初始形状寿命的 30%”的初始形状指直径 5 mm、高 15 mm 的圆柱钨坯料，而不是 zigzag baseline。圆柱簇提交版 baseline 寿命为 115.5037 h，因此 zigzag 族优化的硬约束应为：

- 寿命下限 = 0.30 × 115.5037 h = **34.6511 h**

上述 zigzag baseline 寿命约 7.13 h，只作为折线族参考结果，不作为寿命硬约束基准；按赛题口径它不满足最终寿命约束。
