"""
zigzag_runner.py — 折线族 COMSOL 求解器（MultipleSpectralBands S2S）
====================================================================
对应 zigzag_baseline.java（已修复版：MultipleSpectralBands + rhoeExpr guard）

S2S 辐射模型：MultipleSpectralBands，ε₀₃=0.35（0-3μm），ε_rest=0.15（>3μm）
与 cylinder_family/src/cylinder_baseline.java 保持一致。

搜索空间参数化：
  N_RUNS    — 水平段数（偶数，4~16）
  L_RUN_m   — 每段水平长度 (m)
  z_first_m — 第一段 z 坐标 (m)，关于 L0/2 对称（z_last = L0 - z_first）
  side      — 由体积守恒自动确定：side = sqrt(flexVol / pathLength)

体积守恒：
  总体积 = 2×π×R0²×STUB_LEN（terminal stubs）+ side²×pathLength（折线段）
  = π×R0²×L0（与圆柱基准体积相等）

侵蚀模型：
  dside/dt = 2γ/ρ，γ = Aev·exp(-Bev/T)（方截面 4 面蒸发）
  失效准则：任意 block 边长损失 ≥ 20%

用法：
    runner = COMSOLRunner()
    runner.start()
    result = runner.evaluate(N_RUNS=8, L_RUN_m=104e-3, z_first_m=0.8e-3)
    runner.stop()
"""

import math
import time
import mph
from jpype.types import JInt  # 避免 set(str, int) 与 set(str, boolean) 歧义


class MeshError(RuntimeError):
    """网格生成在所有 hauto 级别均失败。"""


class ServerDisconnectError(RuntimeError):
    """COMSOL 服务器连接已断开。"""


class COMSOLRunner:
    """通过 mph 库控制 COMSOL 服务器，评估平面折线构型。"""

    def __init__(self):
        self.client = None
        self.model = None
        self.j = None

        # ---- 固定参数（与 zigzag_baseline.java 完全一致）----
        self.R0 = 2.5e-3            # terminal stub 半径 (m)
        self.L0 = 15e-3             # 器件全高 (m)
        self.V0 = math.pi * self.R0 ** 2 * self.L0  # 目标体积 (m³)
        self.STUB_LEN = 0.5e-3      # terminal stub 长度 (m)
        self.temp_limit_K = 3273.15
        self.rho_mass = 19350.0
        self.vol_tol = 0.03
        self.current_tol = 1e-9
        self.sphere_margin = 1.05
        self.voltage_upper = 100.0
        self.voltage_floor = 1e-3
        self.voltage_tol = 0.05
        self.max_voltage_iters = 20
        self.voltage_policy = "max_safe"
        self.voltage_objective = "lifeTotalP03sphere_J"
        self.voltage_candidate_ratios = (
            1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60)
        self.Aev = 3.9e9
        self.Bev = 1.023e5
        self.failure_fraction = 0.20
        self.MAX_BLOCK_SLOTS = 64
        self.max_erosion_steps = 50
        self.max_lifetime_h = 200.0   # 侵蚀循环最大仿真时长，避免低蒸发率构型无限运行

        # 材料属性表达式（与 zigzag_baseline.java 一致，含 max guard）
        self.rhoe_expr = (
            "max(1e-10[ohm*m], 5.5e-8[ohm*m]*(1+0.003836*(T-293.15[K])/1[K]"
            "+7.55e-7*((T-293.15[K])/1[K])^2))")
        self.k_expr = ("max(75[W/(m*K)],175[W/(m*K)]"
                       "-0.032[W/(m*K^2)]*(T-293.15[K]))")
        self.cp_expr = ("min(195[J/(kg*K)],132[J/(kg*K)]"
                        "+0.020[J/(kg*K^2)]*(T-293.15[K]))")

        # Planck f03 表达式（在 _build_expressions 中构建）
        self.q03_expr = None
        self.qrad_expr = None
        self._build_expressions()

        # 运行时状态（每次 evaluate 设置）
        self._blocks0 = None
        self._init_side = None
        self._n_blocks = None

    # ================================================================
    #  表达式构建
    # ================================================================

    def _build_expressions(self):
        """构建 Planck f03 黑体谱分数表达式（6 项级数，对应 zigzag_baseline.java）。"""
        x03T = "(c2bb/(lam03*T))"
        parts_T = []
        for n in range(1, 7):
            n2, n3, n4 = n * n, n ** 3, n ** 4
            parts_T.append(
                f"exp(-{n}*{x03T})*({x03T}^3/{n}+3*{x03T}^2/{n2}"
                f"+6*{x03T}/{n3}+6/{n4})")
        series_T = "+".join(parts_T)
        f03T = f"min(1,max(0,(15/pi^4)*({series_T})))"
        # Effective radiation is scored against a 0 K black surface.
        self.q03_expr = f"eps03*sigmaSB*(({f03T})*T^4)"
        self.qrad_expr = (
            f"sigmaSB*(epsRest*T^4+(eps03-epsRest)*(({f03T})*T^4))")

    # ================================================================
    #  服务器管理
    # ================================================================

    def start(self):
        """启动 comsolmphserver 并连接。"""
        print("Starting COMSOL server...")
        self.client = mph.start()
        print("COMSOL server connected. Running warm-up...")
        try:
            dummy = self.client.create("warmup_dummy")
            dummy.java.component().create("comp1")
            self.client.remove(dummy)
            print("Warm-up done.")
        except Exception as e:
            print(f"Warm-up failed (non-fatal): {self._safe_exception_text(e)}")

    # ================================================================
    #  几何计算（纯 Python）
    # ================================================================

    @staticmethod
    def build_full_path(N_RUNS, L_RUN, z_first, z_last, stub_len, L0):
        """生成 Manhattan 路径点序列 [(x, z), ...]。"""
        z_step = (z_last - z_first) / max(N_RUNS - 1, 1)
        x_targets = [L_RUN if i % 2 == 0 else 0.0 for i in range(N_RUNS)]
        z_levels = [z_first + i * z_step for i in range(N_RUNS)]
        pts = [(0.0, stub_len)]
        cur_x = 0.0
        for i in range(N_RUNS):
            pts.append((cur_x, z_levels[i]))
            cur_x = x_targets[i]
            pts.append((cur_x, z_levels[i]))
        pts.append((cur_x, L0 - stub_len))
        return pts

    @staticmethod
    def path_length(pts):
        return sum(
            abs(pts[i + 1][0] - pts[i][0]) + abs(pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1))

    @staticmethod
    def build_blocks(pts, side):
        """将 Manhattan 路径分解为 Block 数据列表。
        返回 [(tag, x0, y0, z0, sx, sy, sz), ...]
        """
        blocks = []
        half = 0.5 * side
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            dx, dz = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1e-12 and abs(dz) < 1e-12:
                continue
            ext_s = 0.0 if i == 0 else half
            ext_e = 0.0 if i == len(pts) - 2 else half
            tag = f"blk_{len(blocks) + 1}"
            if abs(dx) > 1e-12:  # 水平段
                d = 1.0 if dx > 0 else -1.0
                xa = p0[0] - d * ext_s
                xb = p1[0] + d * ext_e
                x_lo, x_hi = min(xa, xb), max(xa, xb)
                blocks.append((tag, x_lo, -half, p0[1] - half,
                               x_hi - x_lo, side, side))
            else:  # 垂直段
                d = 1.0 if dz > 0 else -1.0
                za = p0[1] - d * ext_s
                zb = p1[1] + d * ext_e
                z_lo, z_hi = min(za, zb), max(za, zb)
                blocks.append((tag, p0[0] - half, -half, z_lo,
                               side, side, z_hi - z_lo))
        return blocks

    def compute_side_and_blocks(self, N_RUNS, L_RUN_m, z_first_m):
        """从参数计算 side、blocks、path 信息。"""
        z_last = self.L0 - z_first_m  # 关于 L0/2 对称
        pts = self.build_full_path(
            N_RUNS, L_RUN_m, z_first_m, z_last, self.STUB_LEN, self.L0)
        plen = self.path_length(pts)
        fixed_vol = 2 * math.pi * self.R0 ** 2 * self.STUB_LEN
        flex_vol = self.V0 - fixed_vol
        side = math.sqrt(flex_vol / max(plen, 1e-300))
        blocks = self.build_blocks(pts, side)
        return side, blocks, plen

    def compute_envelope(self, blocks):
        """计算包含所有 blocks + terminal stubs 的外接球半径。"""
        max_dist = 0.0
        for _, x0, y0, z0, sx, sy, sz in blocks:
            for x in (x0, x0 + sx):
                for y in (y0, y0 + sy):
                    for z in (z0, z0 + sz):
                        max_dist = max(max_dist, math.sqrt(
                            x ** 2 + y ** 2 + (z - 0.5 * self.L0) ** 2))
        for rx in (-self.R0, self.R0):
            for ry in (-self.R0, self.R0):
                for zz in (0, self.STUB_LEN,
                           self.L0 - self.STUB_LEN, self.L0):
                    max_dist = max(max_dist, math.sqrt(
                        rx ** 2 + ry ** 2 + (zz - 0.5 * self.L0) ** 2))
        return self.sphere_margin * max_dist

    # ================================================================
    #  侵蚀后几何
    # ================================================================

    @staticmethod
    def eroded_blocks(blocks0, init_side, geom_side):
        """按统一侵蚀后边长重新计算块位置/尺寸。"""
        s0 = init_side
        s_new = geom_side
        shrink = (s0 - s_new) * 0.5
        new_blocks = []
        for idx, (_, x0, y0, z0, sx, sy, sz) in enumerate(blocks0):
            tag = f"blk_{idx + 1}"
            is_horiz = abs(sz - s0) < 1e-6 * s0
            if is_horiz:
                new_blocks.append((tag, x0, y0 + shrink, z0 + shrink,
                                   sx, s_new, s_new))
            else:
                new_blocks.append((tag, x0 + shrink, y0 + shrink, z0,
                                   s_new, s_new, sz))
        return new_blocks

    # ================================================================
    #  工具
    # ================================================================

    def _remove_safe(self, container, tag):
        try:
            container.remove(tag)
        except Exception:
            pass

    @staticmethod
    def _safe_exception_text(exc):
        """COMSOL/JVM 崩溃后某些 Java 异常连 str(exc) 都会再抛异常。"""
        try:
            return str(exc)
        except Exception:
            return exc.__class__.__name__

    @staticmethod
    def _finite_number(value):
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    def _is_server_alive(self):
        """Quick check: returns False if COMSOL server is disconnected."""
        if self.client is None:
            return False
        try:
            self.client.names()
            return True
        except Exception:
            return False

    def _clear_solutions(self, remove=False):
        """清理求解状态；初始建模可 remove，后续求解只 clearSolution。

        Java 基线的关键经验是：初始建模阶段可以移除旧 solver，让 COMSOL
        重新生成 S2S-aware solver；数值算子创建后只 clearSolution，避免
        result dataset 引用失效。
        """
        try:
            for st in list(self.j.sol().tags()):
                try:
                    self.j.sol(st).clearSolution()
                except Exception:
                    pass
                if remove:
                    try:
                        self.j.sol().remove(st)
                    except Exception:
                        pass
        except Exception:
            pass

    # ================================================================
    #  COMSOL 模型构建
    # ================================================================

    def _init_model(self, N_RUNS, L_RUN_m, z_first_m):
        """从空白创建完整 COMSOL 模型（含 1V 预热求解）。"""
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass

        side, blocks, _ = self.compute_side_and_blocks(
            N_RUNS, L_RUN_m, z_first_m)
        self._blocks0 = blocks
        self._init_side = side
        self._n_blocks = len(blocks)
        R_env = self.compute_envelope(blocks)

        self.model = self.client.create("zigzag_opt")
        j = self.model.java
        self.j = j

        # 组件 + 3D 几何
        j.component().create("comp1")
        j.component("comp1").geom().create("geom1", 3)
        j.component("comp1").geom("geom1").lengthUnit("mm")

        # 物理场
        j.component("comp1").physics().create("ec", "ConductiveMedia", "geom1")
        j.component("comp1").physics().create("ht", "HeatTransfer", "geom1")

        # 材料
        j.component("comp1").material().create("mat1", "Common")
        j.component("comp1").material("mat1").label("Tungsten")
        j.component("comp1").material("mat1").selection().all()

        # 网格容器 + 稳态研究
        j.component("comp1").mesh().create("mesh1", "geom1")
        j.study().create("std1")
        j.study("std1").create("stat", "Stationary")

        # 全局参数
        j.param().set("sigmaSB", "5.670374419e-8[W/(m^2*K^4)]",
                      "Stefan-Boltzmann constant")
        j.param().set("eps03",   "0.35",  "Emissivity 0-3um")
        j.param().set("epsRest", "0.15",  "Emissivity >3um")
        j.param().set("rhoMassW", "19350[kg/m^3]", "Tungsten density")
        j.param().set("Tamb",  "293.15[K]", "Ambient temperature for S2S solve")
        j.param().set("Telectrode", "293.15[K]", "Copper electrode temperature")
        j.param().set("Vapp",  "1[V]",      "Applied DC voltage")
        j.param().set("lam03", "3[um]",     "Upper wavelength bound")
        j.param().set("c2bb",  "1.438776877e-2[m*K]",
                      "Second radiation constant")
        j.param().set("r0",  f"{self.R0}[m]", "Reference radius")
        j.param().set("L0",  "15[mm]",        "Reference length")
        j.param().set("Aev", "3.9e9[kg/(m^2*s)]",
                      "Evaporation prefactor (SI)")
        j.param().set("Bev", "1.023e5[K]",
                      "Evaporation temperature coefficient")
        j.param().set("RenvZZ", f"{R_env}[m]",
                      "Enclosing sphere radius")
        j.param().set("AenvZZ", f"{4 * math.pi * R_env ** 2}[m^2]",
                      "Enclosing sphere area")

        # 几何 + 物理 + 网格
        self._rebuild(blocks, side, R_env)

        # 焦耳热多物理耦合
        j.multiphysics().create("emh1", "ElectromagneticHeatSource",
                                "geom1", 3)
        j.multiphysics("emh1").selection().all()
        j.multiphysics("emh1").set("EMHeat_physics", "ec")
        j.multiphysics("emh1").set("Heat_physics", "ht")

        # 收敛辅助：初始温度猜测
        try:
            j.component("comp1").physics("ht").feature("init1").set(
                "Tinit", "1500[K]")
        except Exception:
            pass

        # 初始建模阶段清旧 solver（含 remove），对应 zigzag_baseline.java Phase 0
        self._clear_solutions(remove=True)

        # Phase 0: 预热求解（触发 COMSOL 自动生成 S2S solver）
        # Java baseline 使用 1V；随机 zigzag 构型有时在 1V 下 S2S 内部
        # Tu_band1 初始化为 0，导致 MultipleSpectralBands 的 Planck 积分除零。
        # 因此保留 1V 优先口径，失败后移除 solver 并用 10V/100V 重新初始化。
        last_phase0_error = None
        for warmup_voltage in (1.0, 10.0, 100.0):
            j.param().set("Vapp", f"{warmup_voltage}[V]")
            self._clear_solutions(remove=True)
            try:
                j.study("std1").run()
                if warmup_voltage != 1.0:
                    print(f"  NOTE: Phase-0 warm-up succeeded at {warmup_voltage:g}V")
                break  # 成功
            except Exception as exc:
                last_phase0_error = exc
        else:
            raise RuntimeError(
                "Phase-0 warm-up failed at 1V/10V/100V: "
                + self._safe_exception_text(last_phase0_error))

        # 创建数值算子
        j.result().numerical().create("maxTZZ", "MaxVolume")
        j.result().numerical("maxTZZ").selection().all()
        j.result().numerical("maxTZZ").set("expr", ["T"])

        j.result().numerical().create("minTZZ", "MinVolume")
        j.result().numerical("minTZZ").selection().all()
        j.result().numerical("minTZZ").set("expr", ["T"])

        j.result().numerical().create("volZZ", "IntVolume")
        j.result().numerical("volZZ").selection().all()
        j.result().numerical("volZZ").set("expr", ["1"])

        j.result().numerical().create("TintVolZZ", "IntVolume")
        j.result().numerical("TintVolZZ").selection().all()
        j.result().numerical("TintVolZZ").set("expr", ["T"])

        j.result().numerical().create("IinZZ", "IntSurface")
        j.result().numerical("IinZZ").selection().named("selInZZ")
        j.result().numerical("IinZZ").set("expr",
                                          ["ec.Jx*nx+ec.Jy*ny+ec.Jz*nz"])

        j.result().numerical().create("AsurfZZ", "IntSurface")
        j.result().numerical("AsurfZZ").selection().named("selFreeZZ")
        j.result().numerical("AsurfZZ").set("expr", ["1"])

        j.result().numerical().create("P03emitZZ", "IntSurface")
        j.result().numerical("P03emitZZ").selection().named("selFreeZZ")
        j.result().numerical("P03emitZZ").set("expr", [self.q03_expr])

        j.result().numerical().create("PradEmitZZ", "IntSurface")
        j.result().numerical("PradEmitZZ").selection().named("selFreeZZ")
        j.result().numerical("PradEmitZZ").set("expr", [self.qrad_expr])

        # 健全性检查
        san_T = float(j.result().numerical("maxTZZ").getReal()[0][0])
        san_I = abs(float(j.result().numerical("IinZZ").getReal()[0][0]))
        if san_T < 200.0:
            raise RuntimeError(f"S2S coupling failure: Tmax={san_T:.1f}K")
        print(f"  Model init OK: sanity Tmax={san_T:.1f}K I={san_I:.4f}A")

    # ================================================================
    #  几何 + 物理场构建
    # ================================================================

    def _rebuild(self, blocks, side, R_env, geom_only=False):
        """重建几何+网格，可选重建物理场。

        geom_only=True：只重建几何+网格（侵蚀循环专用），保留 S2S/EC/材料不动。
        geom_only=False：完整重建（初始建模专用）。

        Box 选择（selInZZ/selOutZZ）是坐标驱动的，几何重建后自动刷新，
        无需在 geom_only 模式下重建。S2S 视角因子由 study.run() 自动重算。
        """
        j = self.j
        geom = j.component("comp1").geom("geom1")

        # 清旧几何
        self._remove_safe(geom.feature(), "uniZZ")
        self._remove_safe(geom.feature(), "term_in")
        self._remove_safe(geom.feature(), "term_out")
        for i in range(self.MAX_BLOCK_SLOTS):
            self._remove_safe(geom.feature(), f"blk_{i + 1}")

        # 创建 blocks
        tags = []
        for tag, x0, y0, z0, sx, sy, sz in blocks:
            tags.append(tag)
            geom.create(tag, "Block")
            geom.feature(tag).set("size",
                                  [f"{sx}[m]", f"{sy}[m]", f"{sz}[m]"])
            geom.feature(tag).set("pos",
                                  [f"{x0}[m]", f"{y0}[m]", f"{z0}[m]"])

        # Terminal stubs（圆柱）
        geom.create("term_in", "Cylinder")
        geom.feature("term_in").set("r", f"{self.R0}[m]")
        geom.feature("term_in").set("h", f"{self.STUB_LEN}[m]")
        geom.feature("term_in").set("pos", ["0[m]", "0[m]", "0[m]"])

        geom.create("term_out", "Cylinder")
        geom.feature("term_out").set("r", f"{self.R0}[m]")
        geom.feature("term_out").set("h", f"{self.STUB_LEN}[m]")
        geom.feature("term_out").set("pos",
            ["0[m]", "0[m]", f"{self.L0 - self.STUB_LEN}[m]"])

        # Union
        geom.create("uniZZ", "Union")
        geom.feature("uniZZ").selection("input").set(tags + ["term_in", "term_out"])
        geom.feature("uniZZ").set("intbnd", False)
        geom.run()

        if not geom_only:
            # Box selections — 电极面（坐标驱动，geom_only 时无需重建）
            self._remove_safe(j.component("comp1").selection(), "selInZZ")
            self._remove_safe(j.component("comp1").selection(), "selOutZZ")
            self._remove_safe(j.component("comp1").selection(), "selFreeZZ")

            j.component("comp1").selection().create("selInZZ", "Box")
            j.component("comp1").selection("selInZZ").geom("geom1", 2)
            j.component("comp1").selection("selInZZ").set("condition", "inside")
            j.component("comp1").selection("selInZZ").set("xmin", -10.0)
            j.component("comp1").selection("selInZZ").set("xmax",  10.0)
            j.component("comp1").selection("selInZZ").set("ymin", -10.0)
            j.component("comp1").selection("selInZZ").set("ymax",  10.0)
            j.component("comp1").selection("selInZZ").set("zmin", -1e-6)
            j.component("comp1").selection("selInZZ").set("zmax",  1e-6)

            j.component("comp1").selection().create("selOutZZ", "Box")
            j.component("comp1").selection("selOutZZ").geom("geom1", 2)
            j.component("comp1").selection("selOutZZ").set("condition", "inside")
            j.component("comp1").selection("selOutZZ").set("xmin", -10.0)
            j.component("comp1").selection("selOutZZ").set("xmax",  10.0)
            j.component("comp1").selection("selOutZZ").set("ymin", -10.0)
            j.component("comp1").selection("selOutZZ").set("ymax",  10.0)
            j.component("comp1").selection("selOutZZ").set("zmin", 14.999999)
            j.component("comp1").selection("selOutZZ").set("zmax", 15.000001)

            xs = [-self.R0, self.R0]
            ys = [-self.R0, self.R0]
            for _, x0, y0, _, sx, sy, _ in blocks:
                xs.extend([x0, x0 + sx])
                ys.extend([y0, y0 + sy])
            x_min_mm = min(xs) * 1e3 - 1.0
            x_max_mm = max(xs) * 1e3 + 1.0
            y_min_mm = min(ys) * 1e3 - 1.0
            y_max_mm = max(ys) * 1e3 + 1.0
            j.component("comp1").selection().create("selFreeZZ", "Box")
            j.component("comp1").selection("selFreeZZ").geom("geom1", 2)
            j.component("comp1").selection("selFreeZZ").set("condition", "intersects")
            j.component("comp1").selection("selFreeZZ").set("xmin", x_min_mm)
            j.component("comp1").selection("selFreeZZ").set("xmax", x_max_mm)
            j.component("comp1").selection("selFreeZZ").set("ymin", y_min_mm)
            j.component("comp1").selection("selFreeZZ").set("ymax", y_max_mm)
            j.component("comp1").selection("selFreeZZ").set("zmin", 1e-6)
            j.component("comp1").selection("selFreeZZ").set("zmax", 14.999999)

            # EC 边界条件
            ec = j.component("comp1").physics("ec")
            self._remove_safe(ec.feature(), "potZZ")
            self._remove_safe(ec.feature(), "gndZZ")
            ec.create("potZZ", "ElectricPotential", 2)
            ec.feature("potZZ").selection().named("selInZZ")
            ec.feature("potZZ").set("V0", "Vapp")
            ec.create("gndZZ", "Ground", 2)
            ec.feature("gndZZ").selection().named("selOutZZ")

            # Electrode contact faces are held at copper room temperature.
            ht = j.component("comp1").physics("ht")
            self._remove_safe(ht.feature(), "tempInZZ")
            self._remove_safe(ht.feature(), "tempOutZZ")
            ht.create("tempInZZ", "TemperatureBoundary", 2)
            ht.feature("tempInZZ").selection().named("selInZZ")
            ht.feature("tempInZZ").set("T0", "Telectrode")
            ht.create("tempOutZZ", "TemperatureBoundary", 2)
            ht.feature("tempOutZZ").selection().named("selOutZZ")
            ht.feature("tempOutZZ").set("T0", "Telectrode")

            # S2S 面-面辐射（MultipleSpectralBands）
            # ★ 只在初始建模时创建，侵蚀循环不重建 ★
            # 视角因子由 study.run() 根据当前几何自动重算
            self._setup_s2s()

            # 材料属性
            mp = j.component("comp1").material("mat1").propertyGroup("def")
            mp.set("density",              ["rhoMassW"])
            mp.set("electricconductivity", [f"1/({self.rhoe_expr})"])
            mp.set("thermalconductivity",  [self.k_expr])
            mp.set("heatcapacity",         [self.cp_expr])

        # 网格（初始建模和侵蚀循环都需要）
        try:
            j.component("comp1").mesh("mesh1").feature("ftet1")
        except Exception:
            j.component("comp1").mesh("mesh1").create("ftet1", "FreeTet")
        for hauto in [5, 6, 7, 8, 9]:
            try:
                # JInt 显式指定 Java int，避免 JPype 与 boolean 歧义
                j.component("comp1").mesh("mesh1").feature("size").set(
                    "hauto", JInt(hauto))
                j.component("comp1").mesh("mesh1").run()
                if hauto > 5:
                    print(f"  NOTE: mesh OK with hauto={hauto} (fallback)")
                break
            except Exception as mesh_err:
                if hauto < 9:
                    print(f"  WARN: mesh hauto={hauto} failed, "
                          f"retrying hauto={hauto + 1}...")
                else:
                    raise MeshError(
                        f"Mesh failed at all levels (hauto=5..9): "
                        f"{mesh_err}") from mesh_err

        # 更新外接球参数
        j.param().set("RenvZZ", f"{R_env}[m]")
        j.param().set("AenvZZ", f"{4 * math.pi * R_env ** 2}[m^2]")

    def _setup_s2s(self):
        """设置 S2S 面-面辐射（MultipleSpectralBands，与 zigzag_baseline.java 一致）。"""
        j = self.j

        self._remove_safe(j.component("comp1").physics(), "rad")
        self._remove_safe(j.multiphysics(), "htradZZ")

        j.component("comp1").physics().create(
            "rad", "SurfaceToSurfaceRadiation", "geom1")
        j.component("comp1").physics("rad").prop("RadiationSettings").set(
            "wavelengthDependenceOfSurfaceProperties", "MultipleSpectralBands")
        j.component("comp1").physics("rad").prop("RadiationSettings").set(
            "lambda_r", "lam03")

        j.component("comp1").physics("rad").create("dsZZ", "DiffuseSurface", 2)
        ds = j.component("comp1").physics("rad").feature("dsZZ")
        eps_rad_multi = (
            "if(z<1e-9[m],0,"
            "if(z>L0-1e-9[m],0,"
            "if(comp1.rad.lambda<lam03,eps03,epsRest)))")
        ds.set("defineSurfaceEmissivityOnEachSide", "0")
        ds.set("epsilon_radMulti_mat", "userdef")
        ds.set("epsilon_radMulti", eps_rad_multi)
        ds.set("spectralBandNameAmbientEmissivityMulti",
               [["[0, 3["], ["[3, +inf["]])
        ds.set("Tamb",  "Tamb")
        ds.set("Tambu", "Tamb")
        ds.set("Tambd", "Tamb")
        ds.set("ambientEmissivity", "userdef")
        ds.set("epsilon_amb",  "1")
        ds.set("epsilon_ambu", "1")
        ds.set("epsilon_ambd", "1")
        ds.selection().all()

        j.multiphysics().create(
            "htradZZ", "HeatTransferWithSurfaceToSurfaceRadiation", "geom1", 2)
        j.multiphysics("htradZZ").selection().all()

    # ================================================================
    #  求解器
    # ================================================================

    def _solve_prepared(self, voltage):
        """设电压、求解、提取结果。返回 dict。"""
        j = self.j
        j.param().set("Vapp", f"{voltage}[V]")

        result = {
            "solve_ok": False, "applied_V": voltage,
            "Tmax": float('nan'), "Tmin": float('nan'),
            "Tmean": float('nan'), "U_pct": float('nan'),
            "I": float('nan'), "P03steady": float('nan'),
            "PradSteady": float('nan'), "P03sphere": float('nan'),
            "PradSphere": float('nan'), "vol_err": float('nan'),
            "temp_ok": False, "volume_ok": False, "current_ok": False,
            "block_Tavg": [0.0] * self._n_blocks,
        }

        try:
            # 对齐 Java 基线：每次 study.run() 前只 clearSolution，不 remove solver，
            # 保持 result dataset 和数值算子引用有效。
            self._clear_solutions(remove=False)

            try:
                j.result().numerical("IinZZ").selection().named("selInZZ")
            except Exception:
                pass

            j.study("std1").run()

            Tmax = float(j.result().numerical("maxTZZ").getReal()[0][0])
            try:
                Tmin = float(j.result().numerical("minTZZ").getReal()[0][0])
            except Exception:
                Tmin = Tmax * 0.95

            V = float(j.result().numerical("volZZ").getReal()[0][0])
            TintVol = float(j.result().numerical("TintVolZZ").getReal()[0][0])
            Tmean = TintVol / V if V > 1e-20 else float('nan')
            U_pct = ((Tmax - Tmin) / Tmean * 100.0
                     if Tmean > 1e-20 else float('nan'))
            I = abs(float(j.result().numerical("IinZZ").getReal()[0][0]))
            P03  = float(j.result().numerical("P03emitZZ").getReal()[0][0])
            Prad = float(j.result().numerical("PradEmitZZ").getReal()[0][0])

            PradSphere = Prad
            P03sphere = P03

            # Per-block 温度：抛物线近似（基于 block z-中心）
            # 折线 block 的矩形截面内径向温度差较小（~20K），
            # 轴向分布用抛物线描述，精度与均匀圆柱相当
            block_Tavg = []
            for _, x0, y0, z0, sx, sy, sz in self._blocks0:
                zc = z0 + 0.5 * sz
                eta = zc / self.L0
                block_Tavg.append(
                    Tmin + (Tmax - Tmin) * 4.0 * eta * (1.0 - eta))

            vol_err = abs(V - self.V0) / self.V0

            result.update({
                "solve_ok": True,
                "Tmax": Tmax, "Tmin": Tmin,
                "Tmean": Tmean, "U_pct": U_pct,
                "I": I,
                "R": (voltage / I) if I > self.current_tol else float('nan'),
                "Pelec": voltage * I,
                "P03steady": P03, "PradSteady": Prad,
                "P03sphere": P03sphere, "PradSphere": PradSphere,
                "vol_err": vol_err,
                "temp_ok":    Tmax < self.temp_limit_K,
                "volume_ok":  vol_err <= self.vol_tol,
                "current_ok": I > self.current_tol,
                "block_Tavg": block_Tavg,
            })
        except Exception as e:
            # 清空损坏的解状态，确保下次从 Tinit 重新初始化，
            # 避免 Planck 积分除零（Tu_band1→0）在同一 solver 状态中反复出现。
            self._clear_solutions(remove=False)
            if not self._is_server_alive():
                raise ServerDisconnectError(self._safe_exception_text(e))
            result["failure"] = self._safe_exception_text(e)
            print(f"  WARN solve failed: {result['failure']}")

        return result

    def _meets_constraint(self, r):
        return (r["solve_ok"] and r["current_ok"]
                and r["volume_ok"] and r["temp_ok"])

    def _search_best_voltage(self):
        """电压二分搜索（上限 100V）。
        注：折线 baseline（N=8, L=104mm）在 100V 下 Tmax≈3209K < 3273K，
        可直接返回 100V。对 ML 中非均匀构型，仍需此搜索以防超温。
        """
        steps = 0

        high_res = self._solve_prepared(self.voltage_upper)
        high_V = self.voltage_upper
        steps += 1

        if self._meets_constraint(high_res):
            high_res["search_ok"] = True
            high_res["search_steps"] = steps
            return high_res

        if (high_res["solve_ok"]
                and (not high_res["current_ok"]
                     or not high_res["volume_ok"])):
            high_res["search_ok"] = False
            high_res["search_steps"] = steps
            return high_res

        low_V, low_res = None, None

        if (high_res["solve_ok"] and high_res["current_ok"]
                and high_res["volume_ok"]
                and not math.isnan(high_res["Tmax"])
                and high_res["Tmax"] > 0):
            guess_V = self.voltage_upper * math.sqrt(
                self.temp_limit_K / max(high_res["Tmax"], 1e-300))
            guess_V = max(self.voltage_floor,
                          min(0.98 * self.voltage_upper, guess_V))
            if guess_V < high_V - 1e-12:
                guess_res = self._solve_prepared(guess_V)
                steps += 1
                if self._meets_constraint(guess_res):
                    low_V, low_res = guess_V, guess_res
                else:
                    high_V, high_res = guess_V, guess_res

        while low_res is None and high_V > self.voltage_floor + 1e-12:
            next_V = max(self.voltage_floor, 0.5 * high_V)
            if abs(next_V - high_V) <= 1e-12:
                break
            next_res = self._solve_prepared(next_V)
            steps += 1
            if self._meets_constraint(next_res):
                low_V, low_res = next_V, next_res
            else:
                high_V, high_res = next_V, next_res

        if low_res is None:
            high_res["search_ok"] = False
            high_res["search_steps"] = steps
            return high_res

        for _ in range(self.max_voltage_iters):
            if (high_V - low_V) <= self.voltage_tol:
                break
            mid_V = 0.5 * (low_V + high_V)
            mid_res = self._solve_prepared(mid_V)
            steps += 1
            if self._meets_constraint(mid_res):
                low_V, low_res = mid_V, mid_res
            else:
                high_V, high_res = mid_V, mid_res

        low_res["search_ok"] = True
        low_res["search_steps"] = steps
        return low_res

    def _build_voltage_candidates(self, max_voltage, voltage_candidates=None):
        """Build a descending, de-duplicated voltage candidate list."""
        if voltage_candidates is None:
            raw = [max_voltage * r for r in self.voltage_candidate_ratios]
        else:
            raw = list(voltage_candidates)
            raw.append(max_voltage)

        candidates = []
        for value in raw:
            try:
                voltage = float(value)
            except Exception:
                continue
            if not math.isfinite(voltage):
                continue
            if voltage < self.voltage_floor:
                continue
            if voltage > max_voltage + self.voltage_tol:
                continue
            voltage = min(voltage, max_voltage)
            if any(abs(voltage - existing) <= self.voltage_tol
                   for existing in candidates):
                continue
            candidates.append(voltage)

        return sorted(candidates, reverse=True)

    def _voltage_score(self, result, objective):
        """Return the scalar score used by A4 voltage candidate selection."""
        value = result.get(objective, float('nan'))
        if self._finite_number(value):
            return float(value)

        if objective == "lifeTotalP03sphere_J":
            avg = result.get("lifeAvgP03sphere_W", float('nan'))
            life_h = result.get("lifetimeH", float('nan'))
            if self._finite_number(avg) and self._finite_number(life_h):
                return float(avg) * float(life_h) * 3600.0

        return float('nan')

    def _voltage_scan_summary(self, results, objective):
        parts = []
        for item in results:
            voltage = item.get("Vwork_V", float('nan'))
            status = item.get("status", "UNKNOWN")
            score = self._voltage_score(item, objective)
            v_txt = f"{voltage:.4g}V" if self._finite_number(voltage) else "nanV"
            s_txt = f"{score:.4g}" if self._finite_number(score) else "nan"
            parts.append(f"{v_txt}:{status}:{s_txt}")
        return "; ".join(parts)

    def _annotate_voltage_result(self, result, policy, objective,
                                 max_safe_v=None, candidate_count=1,
                                 scan_summary=""):
        result["voltagePolicy"] = policy
        result["voltageObjective"] = objective
        result["voltageCandidateCount"] = candidate_count
        if max_safe_v is not None:
            result["voltageMaxSafe_V"] = max_safe_v
        if scan_summary:
            result["voltageScanSummary"] = scan_summary
        return result

    def _select_voltage_scan_result(self, results, objective):
        scored = []
        for item in results:
            score = self._voltage_score(item, objective)
            if self._finite_number(score):
                ok = item.get("status") == "OK"
                scored.append((ok, score, item))

        if not scored:
            return results[0]

        eligible = [item for item in scored if item[0]]
        pool = eligible if eligible else scored
        return max(pool, key=lambda item: item[1])[2]

    def evaluate_voltage_candidates(self, N_RUNS, L_RUN_m, z_first_m,
                                    voltage_candidates=None, objective=None):
        """Run full lifecycle evaluations for candidate working voltages."""
        objective = objective or self.voltage_objective
        scan_start = time.time()

        print("  A4 voltage scan: evaluating max-safe voltage first...")
        first = self.evaluate(
            N_RUNS=N_RUNS,
            L_RUN_m=L_RUN_m,
            z_first_m=z_first_m,
            voltage_policy="max_safe",
            voltage_objective=objective,
        )
        if not self._finite_number(first.get("Vwork_V", float('nan'))):
            first["voltageScanElapsed_sec"] = round(time.time() - scan_start, 1)
            return self._annotate_voltage_result(
                first, "full_scan", objective, None, 0)

        max_safe_v = float(first["Vwork_V"])
        candidates = self._build_voltage_candidates(
            max_safe_v, voltage_candidates)

        results = [first]
        for voltage in candidates:
            if abs(voltage - max_safe_v) <= self.voltage_tol:
                continue
            print(f"  A4 voltage scan: evaluating {voltage:.4f}V...")
            result = self.evaluate(
                N_RUNS=N_RUNS,
                L_RUN_m=L_RUN_m,
                z_first_m=z_first_m,
                voltage_policy="fixed",
                voltage_objective=objective,
                voltage_override=voltage,
            )
            results.append(result)

        selected = self._select_voltage_scan_result(results, objective)
        selected["voltageScanElapsed_sec"] = round(time.time() - scan_start, 1)
        return self._annotate_voltage_result(
            selected,
            "full_scan",
            objective,
            max_safe_v=max_safe_v,
            candidate_count=len(results),
            scan_summary=self._voltage_scan_summary(results, objective),
        )

    # ================================================================
    #  主评估入口
    # ================================================================

    def evaluate(self, N_RUNS, L_RUN_m, z_first_m, voltage_policy=None,
                 voltage_candidates=None, voltage_objective=None,
                 voltage_override=None):
        """
        完整评估流程：几何预检 → 建模 → 电压搜索 → 侵蚀循环。

        Args:
            N_RUNS:      水平段数 (int, 偶数)
            L_RUN_m:     每段水平长度 (m)
            z_first_m:   第一段 z 坐标 (m)

        Returns:
            dict: 包含所有赛题指标；status != "OK" 表示失败
        """
        policy = voltage_policy or self.voltage_policy
        objective = voltage_objective or self.voltage_objective
        if policy in ("full_scan", "scan") and voltage_override is None:
            return self.evaluate_voltage_candidates(
                N_RUNS=N_RUNS,
                L_RUN_m=L_RUN_m,
                z_first_m=z_first_m,
                voltage_candidates=voltage_candidates,
                objective=objective,
            )

        t_start = time.time()

        # ---- 几何预检 ----
        side, blocks, plen = self.compute_side_and_blocks(
            N_RUNS, L_RUN_m, z_first_m)

        if side < 0.1e-3:
            return self._annotate_voltage_result({
                "status": "FAIL_SIDE_TOO_SMALL",
                "elapsed_sec": round(time.time() - t_start, 1),
            }, policy, objective)
        if side > 1.2e-3:
            # 截面过厚（>1.2mm）：路径极短，辐射面积远小于 baseline，不具竞争力
            # 且视角因子计算量 O(n_faces²) 过大，易造成 COMSOL JVM 内存崩溃
            return self._annotate_voltage_result({
                "status": "FAIL_SIDE_TOO_LARGE",
                "elapsed_sec": round(time.time() - t_start, 1),
            }, policy, objective)

        z_last = self.L0 - z_first_m
        z_step = (z_last - z_first_m) / max(N_RUNS - 1, 1)
        if z_step < side * 1.2:
            return self._annotate_voltage_result({
                "status": "FAIL_Z_OVERLAP",
                "elapsed_sec": round(time.time() - t_start, 1),
            }, policy, objective)

        n_blocks = len(blocks)
        resolve_thr = 0.01 * side  # geometry update limit for solver stability

        # ---- 建模 ----
        print(f"  Building model: N={N_RUNS} L={L_RUN_m * 1e3:.1f}mm "
              f"zf={z_first_m * 1e3:.2f}mm side={side * 1e3:.4f}mm "
              f"nblk={n_blocks}")
        try:
            self._init_model(N_RUNS, L_RUN_m, z_first_m)
        except MeshError as e:
            return self._annotate_voltage_result({
                "status": "FAIL_MESH: " + self._safe_exception_text(e),
                "elapsed_sec": round(time.time() - t_start, 1),
            }, policy, objective)
        except ServerDisconnectError:
            raise
        except Exception as e:
            if not self._is_server_alive():
                raise ServerDisconnectError(self._safe_exception_text(e))
            return self._annotate_voltage_result({
                "status": "FAIL_INIT: " + self._safe_exception_text(e),
                "elapsed_sec": round(time.time() - t_start, 1),
            }, policy, objective)

        # ---- Phase 1: 电压搜索 ----
        if voltage_override is None:
            print("  Phase 1: voltage search...")
            r0_res = self._search_best_voltage()
        else:
            print(f"  Phase 1: fixed voltage {voltage_override:.4f}V...")
            r0_res = self._solve_prepared(voltage_override)
            r0_res["search_ok"] = self._meets_constraint(r0_res)
            r0_res["search_steps"] = 1
        Vwork = r0_res["applied_V"]
        max_safe_v = Vwork if (
            voltage_override is None and r0_res.get("search_ok", False)
        ) else None
        print(f"  PHASE1: Vwork={Vwork:.4f}V Tmax={r0_res['Tmax']:.1f}K "
              f"P03sph={r0_res['P03sphere']:.1f}W "
              f"steps={r0_res['search_steps']}")

        if not r0_res.get("search_ok", False):
            return self._annotate_voltage_result({
                "status": "FAIL_VOLTAGE_SEARCH",
                "failure": r0_res.get("failure", ""),
                "elapsed_sec": round(time.time() - t_start, 1),
            }, policy, objective, max_safe_v)

        # ---- Phase 2: 侵蚀循环 ----
        print("  Phase 2: erosion loop...")
        time_s = 0.0
        p03_int, prad_int = 0.0, 0.0
        p03s_int, prads_int = 0.0, 0.0
        macro = 0
        failed = False
        side0 = self._init_side
        block_sides = [side0] * n_blocks  # per-block 当前边长
        side_min = side0 * (1.0 - self.failure_fraction)

        prev_p03  = r0_res["P03steady"]
        prev_prad = r0_res["PradSteady"]
        prev_p03s = r0_res["P03sphere"]
        prev_prads = r0_res["PradSphere"]
        block_tavg = r0_res["block_Tavg"]
        max_erosion_tmax = r0_res["Tmax"]

        while macro < self.max_erosion_steps and not failed:
            macro += 1

            # (a) 蒸发速率：方截面 4 面蒸发 dside/dt = 2γ/ρ
            dsdt = [
                2.0 * self.Aev * math.exp(-self.Bev / block_tavg[i])
                / self.rho_mass
                for i in range(n_blocks)]
            max_dsdt = max(dsdt)

            if max_dsdt < 1e-15:
                print("  Evaporation negligible. Infinite lifetime.")
                break

            # (b) 宏步时长
            dt = float('inf')
            for i in range(n_blocks):
                if dsdt[i] > 1e-20:
                    dt = min(dt, resolve_thr / dsdt[i])
                    t_fail = (block_sides[i] - side_min) / dsdt[i]
                    if t_fail > 0:
                        dt = min(dt, t_fail)
            dt = max(1.0, min(36000.0, dt))

            # (c) 推进 per-block 边长
            max_loss = 0.0
            for i in range(n_blocks):
                block_sides[i] = max(1e-6, block_sides[i] - dsdt[i] * dt)
                loss = (side0 - block_sides[i]) / side0
                max_loss = max(max_loss, loss)
            time_s += dt

            if max_loss >= self.failure_fraction:
                failed = True

            # 最大仿真时长保护：低蒸发率构型不值得跑超过 max_lifetime_h
            if not failed and time_s / 3600.0 >= self.max_lifetime_h:
                print(f"  Lifetime cap {self.max_lifetime_h:.0f}h at step {macro}, stopping.")
                break

            # (d) 重建几何：使用最小 block 边长做保守统一侵蚀。
            # 平均边长在局部高温侵蚀很不均匀时会产生极小几何变化和求解奇异。
            geom_side = min(block_sides)
            new_blocks = self.eroded_blocks(self._blocks0, side0, geom_side)
            new_Renv = self.compute_envelope(new_blocks)
            try:
                # ★ geom_only=True：只重建几何+网格，不重建 S2S/EC/材料 ★
                # 与 zigzag_baseline.java 侵蚀循环一致，避免反复重建 S2S 导致服务器崩溃
                self._rebuild(new_blocks, side0, new_Renv, geom_only=True)
                r_now = self._solve_prepared(Vwork)
            except ServerDisconnectError:
                raise  # 断线不静默处理，直接向上抛出
            except Exception as e:
                if not self._is_server_alive():
                    raise ServerDisconnectError(self._safe_exception_text(e))
                failure = self._safe_exception_text(e)
                print(f"  WARN: rebuild failed step {macro}: {failure}")
                return self._annotate_voltage_result({
                    "Vwork_V": Vwork,
                    "initialTmax_K": r0_res["Tmax"],
                    "Tmin_K": r0_res["Tmin"],
                    "Tmean_K": r0_res["Tmean"],
                    "U_pct": r0_res["U_pct"],
                    "lifetimeH": time_s / 3600.0,
                    "initialP03sphere_W": r0_res["P03sphere"],
                    "initialPradSphere_W": r0_res["PradSphere"],
                    "lifeAvgP03sphere_W": float('nan'),
                    "lifeAvgPradSphere_W": float('nan'),
                    "lifeTotalP03sphere_J": p03s_int,
                    "selfViewLoss_pct": float('nan'),
                    "maxErosionTmax_K": max_erosion_tmax,
                    "failureReached": failed,
                    "erosionSteps": macro,
                    "status": "FAIL_EROSION_SOLVE",
                    "failure": failure,
                    "elapsed_sec": round(time.time() - t_start, 1),
                }, policy, objective, max_safe_v)

            if not r_now["solve_ok"]:
                failure = r_now.get("failure", "")
                print(f"  WARN: solve failed step {macro}: {failure}")
                return self._annotate_voltage_result({
                    "Vwork_V": Vwork,
                    "initialTmax_K": r0_res["Tmax"],
                    "Tmin_K": r0_res["Tmin"],
                    "Tmean_K": r0_res["Tmean"],
                    "U_pct": r0_res["U_pct"],
                    "lifetimeH": time_s / 3600.0,
                    "initialP03sphere_W": r0_res["P03sphere"],
                    "initialPradSphere_W": r0_res["PradSphere"],
                    "lifeAvgP03sphere_W": float('nan'),
                    "lifeAvgPradSphere_W": float('nan'),
                    "lifeTotalP03sphere_J": p03s_int,
                    "selfViewLoss_pct": float('nan'),
                    "maxErosionTmax_K": max_erosion_tmax,
                    "failureReached": failed,
                    "erosionSteps": macro,
                    "status": "FAIL_EROSION_SOLVE",
                    "failure": failure,
                    "elapsed_sec": round(time.time() - t_start, 1),
                }, policy, objective, max_safe_v)

            # (e) 梯形积分
            if r_now["solve_ok"] and r_now["Tmax"] > max_erosion_tmax:
                max_erosion_tmax = r_now["Tmax"]

            cur_p03   = r_now["P03steady"]  if r_now["solve_ok"] else prev_p03
            cur_prad  = r_now["PradSteady"] if r_now["solve_ok"] else prev_prad
            cur_p03s  = r_now["P03sphere"]  if r_now["solve_ok"] else prev_p03s
            cur_prads = r_now["PradSphere"] if r_now["solve_ok"] else prev_prads

            p03_int   += 0.5 * (prev_p03   + cur_p03)   * dt
            prad_int  += 0.5 * (prev_prad  + cur_prad)  * dt
            p03s_int  += 0.5 * (prev_p03s  + cur_p03s)  * dt
            prads_int += 0.5 * (prev_prads + cur_prads) * dt

            if r_now["solve_ok"] and r_now["Tmax"] >= self.temp_limit_K:
                elapsed = time.time() - t_start
                lifetime_h = time_s / 3600.0
                avg_p03s = (p03s_int / time_s) if time_s > 0 else float('nan')
                avg_prads = (prads_int / time_s) if time_s > 0 else float('nan')
                sv_loss = ((1.0 - p03s_int / p03_int) * 100.0
                           if (time_s > 0 and p03_int > 0) else float('nan'))
                return self._annotate_voltage_result({
                    "Vwork_V": Vwork,
                    "initialTmax_K": r0_res["Tmax"],
                    "Tmin_K": r0_res["Tmin"],
                    "Tmean_K": r0_res["Tmean"],
                    "U_pct": r0_res["U_pct"],
                    "lifetimeH": lifetime_h,
                    "initialP03sphere_W": r0_res["P03sphere"],
                    "initialPradSphere_W": r0_res["PradSphere"],
                    "lifeAvgP03sphere_W": avg_p03s,
                    "lifeAvgPradSphere_W": avg_prads,
                    "lifeTotalP03sphere_J": p03s_int,
                    "selfViewLoss_pct": sv_loss,
                    "maxErosionTmax_K": max_erosion_tmax,
                    "overtempStep": macro,
                    "overtempTimeH": lifetime_h,
                    "overtempTmax_K": r_now["Tmax"],
                    "failureReached": failed,
                    "erosionSteps": macro,
                    "status": "FAIL_OVERTEMP_DURING_EROSION",
                    "elapsed_sec": round(elapsed, 1),
                }, policy, objective, max_safe_v)

            prev_p03, prev_prad   = cur_p03, cur_prad
            prev_p03s, prev_prads = cur_p03s, cur_prads

            if r_now["solve_ok"] and "block_Tavg" in r_now:
                block_tavg = r_now["block_Tavg"]

            if macro % 5 == 0 or failed:
                print(f"  STEP={macro} t={time_s / 3600:.2f}h "
                      f"loss={max_loss:.4f}")

        # ---- Phase 3: 汇总结果 ----
        lifetime_h = time_s / 3600.0
        avg_p03s  = (p03s_int  / time_s) if time_s > 0 else float('nan')
        avg_prads = (prads_int / time_s) if time_s > 0 else float('nan')
        sv_loss = ((1.0 - p03s_int / p03_int) * 100.0
                   if (time_s > 0 and p03_int > 0) else float('nan'))

        elapsed = time.time() - t_start
        return self._annotate_voltage_result({
            "Vwork_V":              Vwork,
            "initialTmax_K":        r0_res["Tmax"],
            "Tmin_K":               r0_res["Tmin"],
            "Tmean_K":              r0_res["Tmean"],
            "U_pct":                r0_res["U_pct"],
            "lifetimeH":            lifetime_h,
            "initialP03sphere_W":   r0_res["P03sphere"],
            "initialPradSphere_W":  r0_res["PradSphere"],
            "lifeAvgP03sphere_W":   avg_p03s,
            "lifeAvgPradSphere_W":  avg_prads,
            "lifeTotalP03sphere_J": p03s_int,
            "selfViewLoss_pct":     sv_loss,
            "maxErosionTmax_K":     max_erosion_tmax,
            "failureReached":       failed,
            "erosionSteps":         macro,
            "status":               "OK",
            "elapsed_sec":          round(elapsed, 1),
        }, policy, objective, max_safe_v)

    # ================================================================
    #  清理
    # ================================================================

    def stop(self):
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass
            self.model = None
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass  # 服务器已崩溃时 disconnect 会抛 IllegalStateException，静默忽略
            self.client = None
        print("COMSOL disconnected.")


# ================================================================
#  独立测试（运行 baseline）
# ================================================================
if __name__ == "__main__":
    runner = COMSOLRunner()
    runner.start()
    try:
        print("Evaluating zigzag baseline: N_RUNS=8, L_RUN=104mm, z_first=0.8mm")
        result = runner.evaluate(N_RUNS=8, L_RUN_m=104e-3, z_first_m=0.8e-3)
        print("\n" + "=" * 60)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 60)
    finally:
        runner.stop()
