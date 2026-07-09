"""
comsol_runner.py — COMSOL 求解器（mph/JPype 方案）
===================================================
通过 mph 库连接 comsolmphserver，在 Python 中完整复现
cylinder_baseline.java 的建模、电压搜索、侵蚀循环逻辑。

优点：
  - 无需 comsolcompile / comsolbatch（避免内部类编译问题）
  - 所有 trial 共享同一个 COMSOL 服务进程（无 JVM 冷启动开销）
  - Python 侧直接读取求解结果，天然对接 Optuna

用法：
    runner = COMSOLRunner()
    runner.start()
    result = runner.evaluate([2.5e-3]*8)
    runner.stop()
"""

import math
import time
import mph
from jpype.types import JInt  # 避免 set(str, int) 与 set(str, boolean) 歧义


class COMSOLRunner:
    """通过 mph 库控制 COMSOL 服务器，评估钨棒配置。"""

    def __init__(self):
        self.client = None
        self.model = None
        self.j = None  # raw Java Model object

        # ---- 固定参数（与 cylinder_baseline.java 完全一致）----
        self.seg_count = 8
        self.L0 = 15e-3
        self.Lseg = self.L0 / self.seg_count
        self.temp_limit_K = 3000.0 + 273.15
        self.rho_mass = 19350.0
        self.vol_tol = 1.0e-4
        self.current_tol = 1.0e-9
        self.outer_sphere_margin = 1.05
        self.voltage_upper = 100.0
        self.voltage_floor = 1.0e-3
        self.voltage_tol = 0.05
        self.max_voltage_iters = 16
        self.voltage_policy = "max_safe"
        self.voltage_objective = "lifeTotalP03sphere_J"
        self.voltage_candidate_ratios = (
            1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60)
        self.Aev = 3.9e9
        self.Bev = 1.023e5
        self.failure_fraction = 0.20
        self.mesh_hauto_levels = (4, 5, 6, 7, 8)

        # 材料属性表达式
        self.rhoe_expr = (
            "max(1e-10[ohm*m], 5.5e-8[ohm*m]*(1+0.003836*(T-293.15[K])/1[K]"
            "+7.55e-7*((T-293.15[K])/1[K])^2))")
        self.k_expr = "max(75[W/(m*K)],175[W/(m*K)]-0.032[W/(m*K^2)]*(T-293.15[K]))"
        self.cp_expr = "min(195[J/(kg*K)],132[J/(kg*K)]+0.020[J/(kg*K^2)]*(T-293.15[K]))"

        # Planck f03 表达式（在 _build_expressions 中构建）
        self.q03_net_out_expr = None
        self.q_rad_net_out_expr = None
        self._build_expressions()

        # 运行时状态（每次 evaluate 时设置）
        self._r0 = None
        self._initial_radii = None
        self._fail_radii = None

    # ================================================================
    #  表达式构建
    # ================================================================

    def _build_expressions(self):
        """构建 Planck f03 黑体谱分数表达式（对应 cylinder_baseline.java）。"""
        x03T = "(c2bb/(lam03*T))"
        series_T_parts = []

        for n in range(1, 7):  # n = 1..6
            n2, n3, n4 = n * n, n * n * n, n * n * n * n
            term_T = (
                f"exp(-{n}*{x03T})*("
                f"{x03T}^3/{n}+3*{x03T}^2/{n2}"
                f"+6*{x03T}/{n3}+6/{n4})")
            series_T_parts.append(term_T)

        series_T = "+".join(series_T_parts)

        f03bb_T = f"min(1,max(0,(15/pi^4)*({series_T})))"

        # Effective radiation is scored against a 0 K black surface.
        self.q03_net_out_expr = (
            f"eps03*sigmaSB*(({f03bb_T})*T^4)")
        self.q_rad_net_out_expr = (
            f"sigmaSB*(epsRest*T^4+(eps03-epsRest)*(({f03bb_T})*T^4))")

    # ================================================================
    #  服务器管理
    # ================================================================

    def start(self):
        """启动 comsolmphserver 并连接（含暖机步骤）。"""
        print("Starting COMSOL server...")
        self.client = mph.start()
        print("COMSOL server connected. Running warm-up...")
        # 暖机：创建并移除一个空模型，触发 JVM/JPype 完整初始化
        try:
            dummy = self.client.create("warmup_dummy")
            dummy.java.component().create("comp1")
            self.client.remove(dummy)
            print("Warm-up done.")
        except Exception as e:
            print(f"Warm-up failed (non-fatal): {e}")

    # ================================================================
    #  几何计算
    # ================================================================

    def _compute_geom(self, radii_m):
        """从半径计算 r_max, R_env, A_env。"""
        r_max = max(radii_m)
        R_env = self.outer_sphere_margin * math.sqrt(
            (0.5 * self.L0) ** 2 + r_max ** 2)
        A_env = 4.0 * math.pi * R_env ** 2
        return r_max, R_env, A_env

    # ================================================================
    #  模型构建
    # ================================================================

    def _init_model(self, radii_m):
        """从空白创建完整 COMSOL 模型（对应 cylinder_baseline.java 全流程）。"""
        # 清理旧模型
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass

        self.model = self.client.create("tungsten_opt")
        j = self.model.java
        self.j = j

        r0 = self._r0
        _, R_env, A_env = self._compute_geom(radii_m)

        # ---- 组件 + 3D 几何 ----
        j.component().create("comp1")
        j.component("comp1").geom().create("geom1", 3)
        j.component("comp1").geom("geom1").lengthUnit("mm")

        # ---- 物理场 ----
        j.component("comp1").physics().create("ec", "ConductiveMedia", "geom1")
        j.component("comp1").physics().create("ht", "HeatTransfer", "geom1")

        # ---- 材料 ----
        j.component("comp1").material().create("mat1", "Common")
        j.component("comp1").material("mat1").label("Tungsten")
        j.component("comp1").material("mat1").selection().all()

        # ---- 网格容器 ----
        j.component("comp1").mesh().create("mesh1", "geom1")

        # ---- 稳态研究 ----
        j.study().create("std1")
        j.study("std1").create("stat", "Stationary")

        # ---- 全局参数 ----
        j.param().set("sigmaSB", "5.670374419e-8[W/(m^2*K^4)]",
                       "Stefan-Boltzmann constant")
        j.param().set("eps03", "0.35", "Emissivity 0-3 um band")
        j.param().set("epsRest", "0.15", "Emissivity outside 0-3 um band")
        j.param().set("rhoMassW", "19350[kg/m^3]", "Density of tungsten")
        j.param().set("Tamb", "293.15[K]", "Ambient temperature for S2S solve")
        j.param().set("Telectrode", "293.15[K]", "Copper electrode temperature")
        j.param().set("Vapp", f"{self.voltage_upper}[V]", "Applied DC voltage")
        j.param().set("lam03", "3[um]", "Upper wavelength bound")
        j.param().set("c2bb", "1.438776877e-2[m*K]",
                       "Second radiation constant")
        j.param().set("r0", f"{r0}[m]", "Reference radius (max of input)")
        j.param().set("L0", "15[mm]", "Reference length")
        j.param().set("Nseg", str(self.seg_count), "Segment count")
        j.param().set("Lseg", f"{self.Lseg}[m]", "Axial segment length")
        j.param().set("RenvInit", f"{R_env}[m]", "Enclosing sphere radius")
        j.param().set("AenvInit", f"{A_env}[m^2]", "Enclosing sphere area")
        for i in range(self.seg_count):
            j.param().set(f"r_seg{i + 1}", f"{r0}[m]",
                          f"Segment {i + 1} radius")

        # 设置实际半径 + 构建几何
        self._set_params(radii_m, R_env, A_env, self.voltage_upper)
        self._rebuild(radii_m)

        # ---- 多物理耦合：焦耳热 ----
        j.multiphysics().create("emh1",
                                "ElectromagneticHeatSource", "geom1", 3)
        j.multiphysics("emh1").selection().all()
        j.multiphysics("emh1").set("EMHeat_physics", "ec")
        j.multiphysics("emh1").set("Heat_physics", "ht")

        # 收敛辅助：3000K 暖启动 + 清旧 solver
        try:
            j.component("comp1").physics("ht").feature("init1").set(
                "Tinit", "3000[K]")
        except Exception:
            pass
        try:
            sol_tags = list(j.sol().tags())
            for st in sol_tags:
                try:
                    j.sol(st).clearSolution()
                except Exception:
                    pass
                try:
                    j.sol().remove(st)
                except Exception:
                    pass
        except Exception:
            pass

        # Phase 0: 1V 预热求解（触发 COMSOL 自动生成 S2S solver）
        self._set_params(radii_m, R_env, A_env, 1.0)
        j.study("std1").run()

        # ---- 创建数值算子 ----
        j.result().numerical().create("maxTS2S", "MaxVolume")
        j.result().numerical("maxTS2S").selection().all()
        j.result().numerical("maxTS2S").set("expr", ["T"])

        j.result().numerical().create("minTS2S", "MinVolume")
        j.result().numerical("minTS2S").selection().all()
        j.result().numerical("minTS2S").set("expr", ["T"])

        j.result().numerical().create("volS2S", "IntVolume")
        j.result().numerical("volS2S").selection().all()
        j.result().numerical("volS2S").set("expr", ["1"])

        j.result().numerical().create("TintVolS2S", "IntVolume")
        j.result().numerical("TintVolS2S").selection().all()
        j.result().numerical("TintVolS2S").set("expr", ["T"])

        j.result().numerical().create("IinS2S", "IntSurface")
        j.result().numerical("IinS2S").selection().named("selInS2S")
        j.result().numerical("IinS2S").set("expr",
                                           ["ec.Jx*nx+ec.Jy*ny+ec.Jz*nz"])

        j.result().numerical().create("AsurfS2S", "IntSurface")
        j.result().numerical("AsurfS2S").selection().named("selFreeS2S")
        j.result().numerical("AsurfS2S").set("expr", ["1"])

        j.result().numerical().create("P03emitS2S", "IntSurface")
        j.result().numerical("P03emitS2S").selection().named("selFreeS2S")
        j.result().numerical("P03emitS2S").set("expr",
                                               [self.q03_net_out_expr])

        j.result().numerical().create("PradEmitS2S", "IntSurface")
        j.result().numerical("PradEmitS2S").selection().named("selFreeS2S")
        j.result().numerical("PradEmitS2S").set("expr",
                                                [self.q_rad_net_out_expr])

        # [Fix-1] 每段侧面平均温度算子：TintSeg_{i+1}=∫T dA，AsegS2S_{i+1}=∫1 dA
        for i in range(self.seg_count):
            int_t_tag = f"TintSeg_{i + 1}"
            int_a_tag = f"AsegS2S_{i + 1}"
            sel_tag   = f"selSegLat_{i + 1}"
            try:
                j.result().numerical().remove(int_t_tag)
            except Exception:
                pass
            j.result().numerical().create(int_t_tag, "IntSurface")
            j.result().numerical(int_t_tag).selection().named(sel_tag)
            j.result().numerical(int_t_tag).set("expr", ["T"])
            try:
                j.result().numerical().remove(int_a_tag)
            except Exception:
                pass
            j.result().numerical().create(int_a_tag, "IntSurface")
            j.result().numerical(int_a_tag).selection().named(sel_tag)
            j.result().numerical(int_a_tag).set("expr", ["1"])

        # 健全性检查
        sanity_Tmax = float(
            j.result().numerical("maxTS2S").getReal()[0][0])
        sanity_I = abs(float(
            j.result().numerical("IinS2S").getReal()[0][0]))
        sanity_R = (1.0 / sanity_I) if sanity_I > 1e-20 else float('nan')
        if not math.isnan(sanity_R) and sanity_R > 1.0:
            raise RuntimeError(f"Sanity failed: R={sanity_R:.4f} ohm")
        if sanity_Tmax < 200.0:
            raise RuntimeError(
                f"S2S coupling failure: Tmax={sanity_Tmax:.1f}K")

        print(f"  Model init OK: sanity Tmax={sanity_Tmax:.1f}K, "
              f"I={sanity_I:.4f}A")

    # ================================================================
    #  参数设置
    # ================================================================

    def _set_params(self, radii_m, R_env, A_env, voltage):
        """设置段半径、包络球和电压参数。"""
        j = self.j
        for i in range(self.seg_count):
            j.param().set(f"r_seg{i + 1}", f"{radii_m[i]}[m]",
                          f"Segment {i + 1} radius")
        j.param().set("Vapp", f"{voltage}[V]", "Applied DC voltage")
        j.param().set("RenvInit", f"{R_env}[m]",
                       "Enclosing sphere radius")
        j.param().set("AenvInit", f"{A_env}[m^2]",
                       "Enclosing sphere area")

    def _remove_safe(self, container, tag):
        """安全删除 feature/selection（不存在则忽略）。"""
        try:
            container.remove(tag)
        except Exception:
            pass

    def _ensure_server_ready(self):
        """轻量心跳检查，避免 COMSOL server 断联后继续产出无效结果。"""
        if self.client is None:
            raise RuntimeError("COMSOL server is not initialized.")
        if self.model is None or self.j is None:
            return
        try:
            # 只读访问；server 断联、模型失效或 JVM 异常时这里会抛错。
            self.j.label()
        except Exception as exc:
            raise RuntimeError(f"COMSOL server/model heartbeat failed: {exc}") from exc

    def _clear_mesh_safe(self):
        """清理旧网格，失败时忽略（不同 COMSOL 版本 API 行为略有差异）。"""
        try:
            self.j.component("comp1").mesh("mesh1").clearMesh()
        except Exception:
            pass

    def _run_mesh_with_fallback(self, context="mesh"):
        """运行网格并按 hauto 逐级放宽，供初始建模和侵蚀更新共用。"""
        j = self.j
        try:
            j.component("comp1").mesh("mesh1").feature("ftet1")
        except Exception:
            j.component("comp1").mesh("mesh1").create("ftet1", "FreeTet")

        last_err = None
        for hauto in self.mesh_hauto_levels:
            try:
                self._clear_mesh_safe()
                # JInt 显式指定 Java int，避免 JPype 与 boolean 歧义
                j.component("comp1").mesh("mesh1").feature("size").set(
                    "hauto", JInt(hauto))
                j.component("comp1").mesh("mesh1").run()
                if hauto != self.mesh_hauto_levels[0]:
                    print(f"  NOTE: {context} mesh OK with hauto={hauto} (fallback)")
                return hauto
            except Exception as mesh_err:
                last_err = mesh_err
                next_levels = [h for h in self.mesh_hauto_levels if h > hauto]
                if next_levels:
                    print(f"  WARN: {context} mesh hauto={hauto} failed, "
                          f"retrying hauto={next_levels[0]}...")

        levels = ",".join(str(h) for h in self.mesh_hauto_levels)
        raise RuntimeError(
            f"{context} mesh failed at all levels (hauto={levels}): {last_err}"
        ) from last_err

    @staticmethod
    def _finite_number(value):
        """判断数值是否为有限浮点数。"""
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    # ================================================================
    #  几何 + 物理场构建
    # ================================================================

    def _rebuild(self, radii_m):
        """重建几何、边界条件、S2S 辐射、材料、网格。"""
        j = self.j
        N = self.seg_count

        # 清旧几何
        self._remove_safe(
            j.component("comp1").geom("geom1").feature(), "uniS2S")
        for i in range(N):
            self._remove_safe(
                j.component("comp1").geom("geom1").feature(),
                f"cS2S_{i + 1}")

        # 创建 8 段圆柱
        cyl_tags = []
        for i in range(N):
            tag = f"cS2S_{i + 1}"
            cyl_tags.append(tag)
            j.component("comp1").geom("geom1").create(tag, "Cylinder")
            j.component("comp1").geom("geom1").feature(tag).set(
                "r", f"r_seg{i + 1}")
            j.component("comp1").geom("geom1").feature(tag).set(
                "h", "Lseg")
            j.component("comp1").geom("geom1").feature(tag).set(
                "pos", ["0", "0", f"{float(i)}*Lseg"])

        # Union
        j.component("comp1").geom("geom1").create("uniS2S", "Union")
        j.component("comp1").geom("geom1").feature("uniS2S").selection(
            "input").set(cyl_tags)
        j.component("comp1").geom("geom1").feature("uniS2S").set(
            "intbnd", False)
        j.component("comp1").geom("geom1").run()

        # Box 选择
        self._remove_safe(j.component("comp1").selection(), "selInS2S")
        self._remove_safe(j.component("comp1").selection(), "selOutS2S")
        self._remove_safe(j.component("comp1").selection(), "selFreeS2S")

        j.component("comp1").selection().create("selInS2S", "Box")
        j.component("comp1").selection("selInS2S").geom("geom1", 2)
        j.component("comp1").selection("selInS2S").set("condition", "inside")
        j.component("comp1").selection("selInS2S").set("xmin", -10.0)
        j.component("comp1").selection("selInS2S").set("xmax", 10.0)
        j.component("comp1").selection("selInS2S").set("ymin", -10.0)
        j.component("comp1").selection("selInS2S").set("ymax", 10.0)
        j.component("comp1").selection("selInS2S").set("zmin", -1.0e-6)
        j.component("comp1").selection("selInS2S").set("zmax", 1.0e-6)

        j.component("comp1").selection().create("selOutS2S", "Box")
        j.component("comp1").selection("selOutS2S").geom("geom1", 2)
        j.component("comp1").selection("selOutS2S").set("condition", "inside")
        j.component("comp1").selection("selOutS2S").set("xmin", -10.0)
        j.component("comp1").selection("selOutS2S").set("xmax", 10.0)
        j.component("comp1").selection("selOutS2S").set("ymin", -10.0)
        j.component("comp1").selection("selOutS2S").set("ymax", 10.0)
        j.component("comp1").selection("selOutS2S").set("zmin", 14.999999)
        j.component("comp1").selection("selOutS2S").set("zmax", 15.000001)

        # Free radiation/evaporation surfaces: exclude the two electrode contact faces.
        j.component("comp1").selection().create("selFreeS2S", "Box")
        j.component("comp1").selection("selFreeS2S").geom("geom1", 2)
        j.component("comp1").selection("selFreeS2S").set("condition", "intersects")
        j.component("comp1").selection("selFreeS2S").set("xmin", -10.0)
        j.component("comp1").selection("selFreeS2S").set("xmax", 10.0)
        j.component("comp1").selection("selFreeS2S").set("ymin", -10.0)
        j.component("comp1").selection("selFreeS2S").set("ymax", 10.0)
        j.component("comp1").selection("selFreeS2S").set("zmin", 1.0e-6)
        j.component("comp1").selection("selFreeS2S").set("zmax", 14.999999)

        # [Fix-1] 每段侧面 Box 选择（用于精确段平均温度）
        # condition="intersects"：曲面节点只要有一个在 Box 内即被选中
        # delta=10% inset：排除端面（电极面）和段间过渡环面
        r_max_mm = max(radii_m) * 1e3
        Lseg_mm = self.Lseg * 1e3
        xy_safety = r_max_mm * 1.5
        delta = Lseg_mm * 0.1  # 10% inset，对应 cylinder_baseline.java Fix-1
        for i in range(N):
            sel_tag = f"selSegLat_{i + 1}"
            self._remove_safe(j.component("comp1").selection(), sel_tag)
            z_lo = i * Lseg_mm + delta
            z_hi = (i + 1) * Lseg_mm - delta
            j.component("comp1").selection().create(sel_tag, "Box")
            j.component("comp1").selection(sel_tag).geom("geom1", 2)
            j.component("comp1").selection(sel_tag).set("condition", "intersects")
            j.component("comp1").selection(sel_tag).set("xmin", -xy_safety)
            j.component("comp1").selection(sel_tag).set("xmax",  xy_safety)
            j.component("comp1").selection(sel_tag).set("ymin", -xy_safety)
            j.component("comp1").selection(sel_tag).set("ymax",  xy_safety)
            j.component("comp1").selection(sel_tag).set("zmin", z_lo)
            j.component("comp1").selection(sel_tag).set("zmax", z_hi)

        # EC 边界条件
        self._remove_safe(
            j.component("comp1").physics("ec").feature(), "potS2S")
        self._remove_safe(
            j.component("comp1").physics("ec").feature(), "gndS2S")
        j.component("comp1").physics("ec").create(
            "potS2S", "ElectricPotential", 2)
        j.component("comp1").physics("ec").feature("potS2S").selection(
            ).named("selInS2S")
        j.component("comp1").physics("ec").feature("potS2S").set(
            "V0", "Vapp")
        j.component("comp1").physics("ec").create("gndS2S", "Ground", 2)
        j.component("comp1").physics("ec").feature("gndS2S").selection(
            ).named("selOutS2S")

        # Electrode contact faces are held at copper room temperature.
        ht = j.component("comp1").physics("ht")
        self._remove_safe(ht.feature(), "tempInS2S")
        self._remove_safe(ht.feature(), "tempOutS2S")
        ht.create("tempInS2S", "TemperatureBoundary", 2)
        ht.feature("tempInS2S").selection().named("selInS2S")
        ht.feature("tempInS2S").set("T0", "Telectrode")
        ht.create("tempOutS2S", "TemperatureBoundary", 2)
        ht.feature("tempOutS2S").selection().named("selOutS2S")
        ht.feature("tempOutS2S").set("T0", "Telectrode")

        # S2S 面-面辐射
        self._setup_s2s()

        # 材料属性
        j.component("comp1").material("mat1").propertyGroup("def").set(
            "density", ["rhoMassW"])
        j.component("comp1").material("mat1").propertyGroup("def").set(
            "electricconductivity", [f"1/({self.rhoe_expr})"])
        j.component("comp1").material("mat1").propertyGroup("def").set(
            "thermalconductivity", [self.k_expr])
        j.component("comp1").material("mat1").propertyGroup("def").set(
            "heatcapacity", [self.cp_expr])

        # 网格：相邻段半径跳变大时可能产生相交边/移动域网格点失败，
        # 统一使用降级重试逻辑；侵蚀循环中的几何更新也复用同一逻辑。
        self._run_mesh_with_fallback("initial")

    def _setup_s2s(self):
        """设置 S2S 面-面辐射物理场（MultipleSpectralBands，与 cylinder_baseline.java 一致）。"""
        j = self.j
        eps_rad_multi = (
            "if(z<1e-9[m],0,"
            "if(z>L0-1e-9[m],0,"
            "if(comp1.rad.lambda<lam03,eps03,epsRest)))")

        self._remove_safe(j.component("comp1").physics(), "rad")
        self._remove_safe(j.multiphysics(), "htradLT")

        j.component("comp1").physics().create(
            "rad", "SurfaceToSurfaceRadiation", "geom1")
        j.component("comp1").physics("rad").prop(
            "RadiationSettings").set(
            "wavelengthDependenceOfSurfaceProperties", "MultipleSpectralBands")
        j.component("comp1").physics("rad").prop(
            "RadiationSettings").set("lambda_r", "lam03")

        j.component("comp1").physics("rad").create(
            "dsLT", "DiffuseSurface", 2)
        j.component("comp1").physics("rad").feature("dsLT").set(
            "defineSurfaceEmissivityOnEachSide", "0")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "epsilon_radMulti_mat", "userdef")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "epsilon_radMulti", eps_rad_multi)
        j.component("comp1").physics("rad").feature("dsLT").set(
            "spectralBandNameAmbientEmissivityMulti",
            [["[0, 3["], ["[3, +inf["]])
        j.component("comp1").physics("rad").feature("dsLT").set(
            "Tamb", "Tamb")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "Tambu", "Tamb")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "Tambd", "Tamb")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "ambientEmissivity", "userdef")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "epsilon_amb", "1")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "epsilon_ambu", "1")
        j.component("comp1").physics("rad").feature("dsLT").set(
            "epsilon_ambd", "1")
        j.component("comp1").physics("rad").feature("dsLT").selection().all()

        j.multiphysics().create(
            "htradLT",
            "HeatTransferWithSurfaceToSurfaceRadiation", "geom1", 2)
        j.multiphysics("htradLT").selection().all()

    # ================================================================
    #  求解器
    # ================================================================

    def _solve_prepared(self, radii_m, voltage):
        """在已建好的模型上设置参数并求解一次稳态。返回结果 dict。"""
        j = self.j
        r0 = self._r0
        _, R_env, A_env = self._compute_geom(radii_m)
        self._set_params(radii_m, R_env, A_env, voltage)

        result = {
            "solve_ok": False, "search_ok": False, "failure": "",
            "applied_V": voltage, "search_steps": 0,
            "Tmax": float('nan'), "Tmin": float('nan'),
            "Tmean": float('nan'), "U_pct": float('nan'),
            "I": float('nan'), "R": float('nan'),
            "Pelec": float('nan'), "P03steady": float('nan'),
            "PradSteady": float('nan'), "P03sphere": float('nan'),
            "PradSphere": float('nan'), "vol_err": float('nan'),
            "temp_ok": False, "volume_ok": False, "current_ok": False,
            "seg_Tavg": [0.0] * self.seg_count,
        }

        try:
            # 确保 IinS2S 选择正确
            try:
                j.result().numerical("IinS2S").selection().named("selInS2S")
            except Exception:
                pass

            j.study("std1").run()

            Tmax = float(
                j.result().numerical("maxTS2S").getReal()[0][0])
            try:
                Tmin = float(
                    j.result().numerical("minTS2S").getReal()[0][0])
            except Exception:
                Tmin = Tmax * 0.95
                print(f"  WARN: MinVolume failed, Tmin={Tmin:.1f}")

            V = float(
                j.result().numerical("volS2S").getReal()[0][0])
            TintVol = float(
                j.result().numerical("TintVolS2S").getReal()[0][0])
            Tmean = TintVol / V if V > 1e-20 else float('nan')
            U_pct = ((Tmax - Tmin) / Tmean * 100.0
                     if Tmean > 1e-20 else float('nan'))
            I = abs(float(
                j.result().numerical("IinS2S").getReal()[0][0]))
            P03steady = float(
                j.result().numerical("P03emitS2S").getReal()[0][0])
            PradSteady = float(
                j.result().numerical("PradEmitS2S").getReal()[0][0])

            # Effective outward radiation is integrated on free surfaces only.
            PradSphere = PradSteady
            P03sphere = P03steady

            # [Fix-1] 各段侧面平均温度：优先 IntSurface 算子，失败回退到抛物线
            seg_Tavg = []
            for i in range(self.seg_count):
                read_ok = False
                try:
                    Tint = float(j.result().numerical(
                        f"TintSeg_{i + 1}").getReal()[0][0])
                    Aseg = float(j.result().numerical(
                        f"AsegS2S_{i + 1}").getReal()[0][0])
                    if Aseg > 1e-20:
                        seg_Tavg.append(Tint / Aseg)
                        read_ok = True
                except Exception:
                    pass
                if not read_ok:
                    eta = ((i + 0.5) * self.Lseg) / self.L0
                    seg_Tavg.append(
                        Tmin + (Tmax - Tmin) * 4.0 * eta * (1.0 - eta))

            # 体积误差
            V0now = sum(
                math.pi * r ** 2 * self.Lseg for r in radii_m)
            V0ref = math.pi * r0 ** 2 * self.L0
            vol_err = abs(V - V0now) / V0ref

            finite_checks = {
                "Tmax": Tmax,
                "Tmin": Tmin,
                "Tmean": Tmean,
                "U_pct": U_pct,
                "volume": V,
                "current": I,
                "P03steady": P03steady,
                "PradSteady": PradSteady,
                "P03sphere": P03sphere,
                "PradSphere": PradSphere,
                "vol_err": vol_err,
            }
            invalid = [k for k, v in finite_checks.items()
                       if not self._finite_number(v)]
            invalid += [f"seg_Tavg[{i}]" for i, v in enumerate(seg_Tavg)
                        if not self._finite_number(v)]
            if invalid:
                raise RuntimeError(
                    "Invalid non-finite COMSOL result: " + ", ".join(invalid))

            result.update({
                "solve_ok": True,
                "Tmax": Tmax, "Tmin": Tmin,
                "Tmean": Tmean, "U_pct": U_pct,
                "I": I,
                "Pelec": voltage * I,
                "P03steady": P03steady, "PradSteady": PradSteady,
                "P03sphere": P03sphere, "PradSphere": PradSphere,
                "vol_err": vol_err,
                "temp_ok": Tmax < self.temp_limit_K,
                "volume_ok": vol_err <= self.vol_tol,
                "current_ok": I > self.current_tol,
                "R": (voltage / I) if I > self.current_tol else float('nan'),
                "seg_Tavg": seg_Tavg,
            })
        except Exception as e:
            result["failure"] = str(e)

        return result

    def _meets_constraint(self, r):
        """判断结果是否满足电压搜索约束。"""
        return (r["solve_ok"] and r["current_ok"]
                and r["volume_ok"] and r["temp_ok"])

    def _search_best_voltage(self, radii_m):
        """电压二分搜索，返回最优电压对应的结果 dict。"""
        steps = 0

        # 上界尝试
        high_res = self._solve_prepared(radii_m, self.voltage_upper)
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

        low_V = None
        low_res = None

        # 温度倍率猜测
        if (high_res["solve_ok"] and high_res["current_ok"]
                and high_res["volume_ok"]
                and not math.isnan(high_res["Tmax"])
                and high_res["Tmax"] > 0):
            guess_V = self.voltage_upper * math.sqrt(
                self.temp_limit_K / max(high_res["Tmax"], 1e-300))
            guess_V = max(self.voltage_floor,
                          min(0.98 * self.voltage_upper, guess_V))
            if guess_V < high_V - 1e-12:
                guess_res = self._solve_prepared(radii_m, guess_V)
                steps += 1
                if self._meets_constraint(guess_res):
                    low_V, low_res = guess_V, guess_res
                else:
                    high_V, high_res = guess_V, guess_res

        # 逐步减半寻找下界
        while low_res is None and high_V > self.voltage_floor + 1e-12:
            next_V = max(self.voltage_floor, 0.5 * high_V)
            if abs(next_V - high_V) <= 1e-12:
                break
            next_res = self._solve_prepared(radii_m, next_V)
            steps += 1
            if self._meets_constraint(next_res):
                low_V, low_res = next_V, next_res
            else:
                high_V, high_res = next_V, next_res

        if low_res is None:
            high_res["search_ok"] = False
            high_res["search_steps"] = steps
            return high_res

        # 二分精炼
        for _ in range(self.max_voltage_iters):
            if (high_V - low_V) <= self.voltage_tol:
                break
            mid_V = 0.5 * (low_V + high_V)
            mid_res = self._solve_prepared(radii_m, mid_V)
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

    def evaluate_voltage_candidates(self, radii_m, voltage_candidates=None,
                                    objective=None):
        """Run full lifecycle evaluations for candidate working voltages."""
        objective = objective or self.voltage_objective
        scan_start = time.time()

        print("  A4 voltage scan: evaluating max-safe voltage first...")
        first = self.evaluate(
            radii_m,
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
                radii_m,
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
    #  几何更新（侵蚀步用，比 rebuild 轻量）
    # ================================================================

    def _update_geometry(self, radii_m):
        """更新段参数 + 重建几何/网格（不重建物理场）。"""
        j = self.j
        _, R_env, A_env = self._compute_geom(radii_m)
        for i in range(self.seg_count):
            j.param().set(f"r_seg{i + 1}", f"{radii_m[i]}[m]",
                          f"Segment {i + 1} radius")
        j.param().set("RenvInit", f"{R_env}[m]",
                       "Enclosing sphere radius")
        j.param().set("AenvInit", f"{A_env}[m^2]",
                       "Enclosing sphere area")
        j.component("comp1").geom("geom1").run()
        self._run_mesh_with_fallback("erosion")

    def _solve_at_voltage(self, radii_m, voltage):
        """更新几何后求解（侵蚀循环内使用）。"""
        self._update_geometry(radii_m)
        return self._solve_prepared(radii_m, voltage)

    # ================================================================
    #  主评估入口
    # ================================================================

    def evaluate(self, radii_m, voltage_policy=None, voltage_candidates=None,
                 voltage_objective=None, voltage_override=None):
        """
        完整评估流程：建模 → 电压搜索 → 侵蚀循环。

        Args:
            radii_m: 长度 8 的 list，各段半径 (m)

        Returns:
            dict: 包含所有赛题指标，或 status != "OK" 表示失败
        """
        policy = voltage_policy or self.voltage_policy
        objective = voltage_objective or self.voltage_objective
        if policy in ("full_scan", "scan") and voltage_override is None:
            return self.evaluate_voltage_candidates(
                radii_m,
                voltage_candidates=voltage_candidates,
                objective=objective,
            )

        t_start = time.time()
        # 每个 trial 开始前做心跳检查，避免 server 断联后继续写入伪有效结果。
        self._ensure_server_ready()

        self._r0 = max(radii_m)
        self._initial_radii = list(radii_m)  # 保存每段初始半径
        self._fail_radii = [r * (1.0 - self.failure_fraction)
                            for r in radii_m]  # per-segment 失效阈值
        resolve_threshold = 0.02 * min(radii_m)

        # ---- Step 0: 建模 ----
        print("  Building model...")
        self._init_model(radii_m)

        # ---- Phase 1: 电压搜索 ----
        radii = list(radii_m)  # 可变副本
        if voltage_override is None:
            print("  Phase 1: voltage search...")
            r0_res = self._search_best_voltage(radii)
        else:
            print(f"  Phase 1: fixed voltage {voltage_override:.4f}V...")
            r0_res = self._solve_prepared(radii, voltage_override)
            r0_res["search_ok"] = self._meets_constraint(r0_res)
            r0_res["search_steps"] = 1
        Vwork = r0_res["applied_V"]
        max_safe_v = Vwork if (
            voltage_override is None and r0_res.get("search_ok", False)
        ) else None

        print(f"  PHASE1: Vwork={Vwork:.4f}V Tmax={r0_res['Tmax']:.1f}K "
              f"P03sph={r0_res['P03sphere']:.1f}W "
              f"steps={r0_res['search_steps']}")

        if not r0_res["search_ok"]:
            return self._annotate_voltage_result({
                "status": "FAIL_VOLTAGE_SEARCH",
                "failure": r0_res.get("failure", ""),
                "elapsed_sec": round(time.time() - t_start, 1),
            }, policy, objective, max_safe_v)

        # ---- Phase 2: 侵蚀循环 ----
        print("  Phase 2: erosion loop...")
        time_s = 0.0
        p03_integral = 0.0
        prad_integral = 0.0
        p03_sphere_integral = 0.0
        prad_sphere_integral = 0.0
        macro_step = 0
        failed = False
        max_macro_steps = 50

        prev_P03 = r0_res["P03steady"]
        prev_Prad = r0_res["PradSteady"]
        prev_P03sphere = r0_res["P03sphere"]
        prev_PradSphere = r0_res["PradSphere"]
        Tavg = r0_res["seg_Tavg"]
        max_erosion_tmax = r0_res["Tmax"]

        while macro_step < max_macro_steps and not failed:
            macro_step += 1

            # (a) 蒸发速率
            drdt = [0.0] * self.seg_count
            max_drdt = 0.0
            for i in range(self.seg_count):
                gamma = self.Aev * math.exp(-self.Bev / Tavg[i])
                drdt[i] = gamma / self.rho_mass
                max_drdt = max(max_drdt, drdt[i])

            if max_drdt < 1e-15:
                print("  Evaporation negligible. Infinite lifetime.")
                break

            # (b) 宏步时长
            dt_macro = float('inf')
            for i in range(self.seg_count):
                if drdt[i] > 1e-20:
                    dt_macro = min(dt_macro,
                                   resolve_threshold / drdt[i])
                    t_fail = (radii[i] - self._fail_radii[i]) / drdt[i]
                    if t_fail > 0:
                        dt_macro = min(dt_macro, t_fail)
            dt_macro = max(1.0, min(36000.0, dt_macro))

            # (c) 推进半径
            max_loss_frac = 0.0
            for i in range(self.seg_count):
                radii[i] -= drdt[i] * dt_macro
                radii[i] = max(1e-6, radii[i])
                loss_frac = ((self._initial_radii[i] - radii[i])
                             / self._initial_radii[i])
                max_loss_frac = max(max_loss_frac, loss_frac)
            time_s += dt_macro

            if max_loss_frac >= self.failure_fraction:
                failed = True

            # (d) 求解
            try:
                r_now = self._solve_at_voltage(radii, Vwork)
            except Exception as exc:
                print(f"  WARN: erosion solve failed step {macro_step}: {exc}")
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
                    "lifeTotalP03sphere_J": p03_sphere_integral,
                    "selfViewLoss_pct": float('nan'),
                    "maxErosionTmax_K": max_erosion_tmax,
                    "failureReached": failed,
                    "erosionSteps": macro_step,
                    "status": "FAIL_EROSION_SOLVE",
                    "failure": str(exc),
                    "elapsed_sec": round(time.time() - t_start, 1),
                }, policy, objective, max_safe_v)

            if not r_now["solve_ok"]:
                failure = r_now.get("failure", "")
                print(f"  WARN: solve failed step {macro_step}: {failure}")
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
                    "lifeTotalP03sphere_J": p03_sphere_integral,
                    "selfViewLoss_pct": float('nan'),
                    "maxErosionTmax_K": max_erosion_tmax,
                    "failureReached": failed,
                    "erosionSteps": macro_step,
                    "status": "FAIL_EROSION_SOLVE",
                    "failure": failure,
                    "elapsed_sec": round(time.time() - t_start, 1),
                }, policy, objective, max_safe_v)

            # (e) 梯形积分
            if r_now["Tmax"] > max_erosion_tmax:
                max_erosion_tmax = r_now["Tmax"]

            cur_P03 = (r_now["P03steady"]
                       if r_now["solve_ok"] else prev_P03)
            cur_Prad = (r_now["PradSteady"]
                        if r_now["solve_ok"] else prev_Prad)
            cur_P03sphere = (r_now["P03sphere"]
                             if r_now["solve_ok"] else prev_P03sphere)
            cur_PradSphere = (r_now["PradSphere"]
                              if r_now["solve_ok"] else prev_PradSphere)

            p03_integral += 0.5 * (prev_P03 + cur_P03) * dt_macro
            prad_integral += 0.5 * (prev_Prad + cur_Prad) * dt_macro
            p03_sphere_integral += 0.5 * (
                prev_P03sphere + cur_P03sphere) * dt_macro
            prad_sphere_integral += 0.5 * (
                prev_PradSphere + cur_PradSphere) * dt_macro

            if r_now["Tmax"] >= self.temp_limit_K:
                elapsed = time.time() - t_start
                lifetime_h = time_s / 3600.0
                avg_P03sphere = ((p03_sphere_integral / time_s)
                                 if time_s > 0 else float('nan'))
                avg_PradSphere = ((prad_sphere_integral / time_s)
                                  if time_s > 0 else float('nan'))
                self_view_loss = (
                    (1.0 - p03_sphere_integral / p03_integral) * 100.0
                    if (time_s > 0 and p03_integral > 0) else float('nan'))
                return self._annotate_voltage_result({
                    "Vwork_V": Vwork,
                    "initialTmax_K": r0_res["Tmax"],
                    "Tmin_K": r0_res["Tmin"],
                    "Tmean_K": r0_res["Tmean"],
                    "U_pct": r0_res["U_pct"],
                    "lifetimeH": lifetime_h,
                    "initialP03sphere_W": r0_res["P03sphere"],
                    "initialPradSphere_W": r0_res["PradSphere"],
                    "lifeAvgP03sphere_W": avg_P03sphere,
                    "lifeAvgPradSphere_W": avg_PradSphere,
                    "lifeTotalP03sphere_J": p03_sphere_integral,
                    "selfViewLoss_pct": self_view_loss,
                    "maxErosionTmax_K": max_erosion_tmax,
                    "overtempStep": macro_step,
                    "overtempTimeH": lifetime_h,
                    "overtempTmax_K": r_now["Tmax"],
                    "failureReached": failed,
                    "erosionSteps": macro_step,
                    "status": "FAIL_OVERTEMP_DURING_EROSION",
                    "elapsed_sec": round(elapsed, 1),
                }, policy, objective, max_safe_v)

            prev_P03 = cur_P03
            prev_Prad = cur_Prad
            prev_P03sphere = cur_P03sphere
            prev_PradSphere = cur_PradSphere
            Tavg = r_now["seg_Tavg"]

            if macro_step % 5 == 0 or failed:
                print(f"  STEP={macro_step} t={time_s / 3600:.2f}h "
                      f"loss={max_loss_frac:.4f}")

        # ---- Phase 3: 汇总结果 ----
        lifetime_h = time_s / 3600.0
        avg_P03sphere = ((p03_sphere_integral / time_s)
                         if time_s > 0 else float('nan'))
        avg_PradSphere = ((prad_sphere_integral / time_s)
                          if time_s > 0 else float('nan'))
        self_view_loss = (
            (1.0 - p03_sphere_integral / p03_integral) * 100.0
            if (time_s > 0 and p03_integral > 0) else float('nan'))

        elapsed = time.time() - t_start

        result = {
            "Vwork_V": Vwork,
            "initialTmax_K": r0_res["Tmax"],
            "Tmin_K": r0_res["Tmin"],
            "Tmean_K": r0_res["Tmean"],
            "U_pct": r0_res["U_pct"],
            "lifetimeH": lifetime_h,
            "initialP03sphere_W": r0_res["P03sphere"],
            "initialPradSphere_W": r0_res["PradSphere"],
            "lifeAvgP03sphere_W": avg_P03sphere,
            "lifeAvgPradSphere_W": avg_PradSphere,
            "lifeTotalP03sphere_J": p03_sphere_integral,
            "selfViewLoss_pct": self_view_loss,
            "maxErosionTmax_K": max_erosion_tmax,
            "failureReached": failed,
            "erosionSteps": macro_step,
            "status": "OK",
            "elapsed_sec": round(elapsed, 1),
        }

        required = [
            "Vwork_V", "initialTmax_K", "Tmin_K", "Tmean_K", "U_pct",
            "maxErosionTmax_K", "lifetimeH",
            "initialP03sphere_W", "initialPradSphere_W",
            "lifeAvgP03sphere_W", "lifeAvgPradSphere_W",
            "lifeTotalP03sphere_J",
            "selfViewLoss_pct", "erosionSteps",
        ]
        invalid = [k for k in required if not self._finite_number(result[k])]
        if invalid:
            result["status"] = "FAIL_INVALID_RESULT"
            result["failure"] = "Non-finite final metric(s): " + ", ".join(invalid)

        return self._annotate_voltage_result(
            result, policy, objective, max_safe_v)

    # ================================================================
    #  清理
    # ================================================================

    def stop(self):
        """断开并停止 COMSOL 服务器。"""
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass
            self.model = None
        if self.client is not None:
            self.client.disconnect()
            self.client = None
        print("COMSOL disconnected.")


# ================================================================
#  独立测试
# ================================================================

if __name__ == "__main__":
    runner = COMSOLRunner()
    runner.start()
    try:
        baseline = [2.5e-3] * 8
        print(f"Evaluating baseline: {[r * 1e3 for r in baseline]} mm")
        result = runner.evaluate(baseline)
        print("\n" + "=" * 60)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 60)
    finally:
        runner.stop()
