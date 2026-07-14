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


class ServerDisconnectError(RuntimeError):
    """COMSOL client lost its server and needs a fresh Python process."""


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
        self.reference_radius = 2.5e-3
        self.reference_volume = (
            math.pi * self.reference_radius ** 2 * self.L0)
        self.temp_limit_K = 3000.0 + 273.15
        self.rho_mass = 19350.0
        self.vol_tol = 1.0e-4
        self.current_tol = 1.0e-9
        self.outer_sphere_margin = 1.05
        self.voltage_upper = 100.0
        self.voltage_floor = 1.0e-3
        self.voltage_tol = 0.05
        self.max_voltage_iters = 16
        self.max_erosion_solve_retries = 2
        self.max_erosion_steps = 150
        self.max_lifetime_h = 1000.0
        self.max_erosion_step_s = 36000.0
        self.erosion_rel_tol = 1.0e-10
        self.erosion_time_tol_s = 1.0e-6
        self.voltage_policy = "rated_lifecycle_scan"
        self.voltage_objective = "lifeTotalP03escape_J"
        self.operating_point_version = "rated_lifecycle_energy_v1"
        self.metric_version = "radiation_escape_v2"
        self.physics_version = "thermal_s2s_d3_v1"
        self.geometry_version = "cylinder_segmented_erosion_v2"
        self.lifecycle_version = "lifecycle_v2"
        self.erosion_model = "local_sidewall_plus_shoulder_volume_balance"
        self.radiation_escape_method = "s2s_radiosity_famb"
        self.spectral_split_um = 3.0
        self.thermal_ambient_K = 293.15
        self.score_ambient_target_K = 0.0
        self.temperature_statistic_version = "temperature_domains_v1"
        self.temperature_primary_domain = "all_tungsten_volume"
        self.active_temperature_trim_m = 0.5e-3
        self.electrode_temperature_K = 293.15
        self.electrode_temperature_tolerance_K = 1.0
        self.electrode_boundary_mode = "fixed_temperature"
        self._active_electrode_boundary_mode = self.electrode_boundary_mode
        self.electrode_boundary_version = "electrode_thermal_v1"
        self.copper_thermal_conductivity_W_mK = 400.0
        self.electrode_boundary_approximation = (
            "circular_contact_half_space_spreading_resistance")
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

    def _segment_lateral_mask(self, index, radius):
        """Return a pointwise z mask and analytic active cylindrical area."""
        inset = 0.10 * self.Lseg
        z_lo = index * self.Lseg + inset
        z_hi = (index + 1) * self.Lseg - inset
        condition = f"z>{z_lo:.16g}[m]&&z<{z_hi:.16g}[m]"
        active_area = 2.0 * math.pi * radius * (z_hi - z_lo)
        return condition, active_area

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
        j.param().set("Tamb", f"{self.thermal_ambient_K}[K]",
                      "Ambient temperature for S2S solve")
        j.param().set(
            "Telectrode", f"{self.electrode_temperature_K}[K]",
            "Copper electrode reference temperature")
        j.param().set(
            "activeTrim", f"{self.active_temperature_trim_m}[m]",
            "Axial trim used only for active-region temperature diagnostics")
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

        active_condition = "z>activeTrim&&z<L0-activeTrim"
        j.result().numerical().create("maxTActiveS2S", "MaxVolume")
        j.result().numerical("maxTActiveS2S").selection().all()
        j.result().numerical("maxTActiveS2S").set(
            "expr", [f"if({active_condition},T,0[K])"])

        j.result().numerical().create("minTActiveS2S", "MinVolume")
        j.result().numerical("minTActiveS2S").selection().all()
        j.result().numerical("minTActiveS2S").set(
            "expr", [f"if({active_condition},T,1e9[K])"])

        j.result().numerical().create("volActiveS2S", "IntVolume")
        j.result().numerical("volActiveS2S").selection().all()
        j.result().numerical("volActiveS2S").set(
            "expr", [f"if({active_condition},1,0)"])

        j.result().numerical().create("TintActiveS2S", "IntVolume")
        j.result().numerical("TintActiveS2S").selection().all()
        j.result().numerical("TintActiveS2S").set(
            "expr", [f"if({active_condition},T,0[K])"])

        j.result().numerical().create("maxTFreeS2S", "MaxSurface")
        j.result().numerical("maxTFreeS2S").selection().named("selFreeS2S")
        j.result().numerical("maxTFreeS2S").set("expr", ["T"])

        j.result().numerical().create("minTFreeS2S", "MinSurface")
        j.result().numerical("minTFreeS2S").selection().named("selFreeS2S")
        j.result().numerical("minTFreeS2S").set("expr", ["T"])

        j.result().numerical().create("TintFreeS2S", "IntSurface")
        j.result().numerical("TintFreeS2S").selection().named("selFreeS2S")
        j.result().numerical("TintFreeS2S").set("expr", ["T"])

        j.result().numerical().create("IinS2S", "IntSurface")
        j.result().numerical("IinS2S").selection().named("selInS2S")
        j.result().numerical("IinS2S").set("expr",
                                           ["ec.Jx*nx+ec.Jy*ny+ec.Jz*nz"])

        j.result().numerical().create("AsurfS2S", "IntSurface")
        j.result().numerical("AsurfS2S").selection().named("selFreeS2S")
        j.result().numerical("AsurfS2S").set("expr", ["1"])

        # Gross emission uses COMSOL's exact spectral-band Planck integration.
        # Contact end faces have zero emissivity in the S2S interface.
        j.result().numerical().create("P03emitS2S", "IntSurface")
        j.result().numerical("P03emitS2S").selection().all()
        j.result().numerical("P03emitS2S").set(
            "expr", ["rad.epsilonu_band1*rad.ebu1"])

        j.result().numerical().create("PradEmitS2S", "IntSurface")
        j.result().numerical("PradEmitS2S").selection().all()
        j.result().numerical("PradEmitS2S").set(
            "expr", ["rad.epsilonu_band1*rad.ebu1+"
                     "rad.epsilonu_band2*rad.ebu2"])

        # Radiation reaching a black enclosing surface equals radiosity times
        # the ambient view factor, integrated over every S2S boundary.
        j.result().numerical().create("P03escapeS2S", "IntSurface")
        j.result().numerical("P03escapeS2S").selection().all()
        j.result().numerical("P03escapeS2S").set(
            "expr", ["rad.J_band1*rad.Famb1"])

        j.result().numerical().create("PradEscapeS2S", "IntSurface")
        j.result().numerical("PradEscapeS2S").selection().all()
        j.result().numerical("PradEscapeS2S").set(
            "expr", ["rad.J_band1*rad.Famb1+"
                     "rad.J_band2*rad.Famb2"])

        j.result().numerical().create("P03ambientS2S", "IntSurface")
        j.result().numerical("P03ambientS2S").selection().all()
        j.result().numerical("P03ambientS2S").set("expr", ["rad.Gamb1"])

        j.result().numerical().create("FambAreaS2S", "IntSurface")
        j.result().numerical("FambAreaS2S").selection().all()
        j.result().numerical("FambAreaS2S").set("expr", ["rad.Famb1"])

        j.result().numerical().create("AradS2S", "IntSurface")
        j.result().numerical("AradS2S").selection().all()
        j.result().numerical("AradS2S").set("expr", ["1"])

        # [Fix-1] 每段侧面平均温度算子：TintSeg_{i+1}=∫T dA，AsegS2S_{i+1}=∫1 dA
        for i in range(self.seg_count):
            int_t_tag = f"TintSeg_{i + 1}"
            int_a_tag = f"AsegS2S_{i + 1}"
            condition, _ = self._segment_lateral_mask(i, radii_m[i])
            try:
                j.result().numerical().remove(int_t_tag)
            except Exception:
                pass
            j.result().numerical().create(int_t_tag, "IntSurface")
            j.result().numerical(int_t_tag).selection().all()
            j.result().numerical(int_t_tag).set(
                "expr", [f"if({condition},T,0[K])"])
            try:
                j.result().numerical().remove(int_a_tag)
            except Exception:
                pass
            j.result().numerical().create(int_a_tag, "IntSurface")
            j.result().numerical(int_a_tag).selection().all()
            j.result().numerical(int_a_tag).set(
                "expr", [f"if({condition},1,0)"])

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
        j.param().set(
            "hCuIn", f"{self._copper_spreading_h(radii_m[0])}"
            "[W/(m^2*K)]", "Equivalent inlet copper spreading coefficient")
        j.param().set(
            "hCuOut", f"{self._copper_spreading_h(radii_m[-1])}"
            "[W/(m^2*K)]", "Equivalent outlet copper spreading coefficient")

    def _remove_safe(self, container, tag):
        """安全删除 feature/selection（不存在则忽略）。"""
        try:
            container.remove(tag)
        except Exception:
            pass

    def _ensure_server_ready(self):
        """轻量心跳检查，避免 COMSOL server 断联后继续产出无效结果。"""
        if self.client is None:
            raise ServerDisconnectError("COMSOL server is not initialized.")
        try:
            self.client.names()
        except Exception as exc:
            raise ServerDisconnectError(
                f"COMSOL server heartbeat failed: {exc}") from exc
        if self.model is None or self.j is None:
            return
        try:
            # 只读访问；server 断联、模型失效或 JVM 异常时这里会抛错。
            self.j.label()
        except Exception as exc:
            raise RuntimeError(f"COMSOL model heartbeat failed: {exc}") from exc

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

    @staticmethod
    def _temperature_uniformity(t_max, t_min, t_mean):
        """Return the official full-domain temperature nonuniformity."""
        values = (t_max, t_min, t_mean)
        if not all(math.isfinite(float(value)) for value in values):
            return float('nan')
        if float(t_mean) <= 1.0e-20 or float(t_max) < float(t_min):
            return float('nan')
        return ((float(t_max) - float(t_min))
                / float(t_mean) * 100.0)

    def _copper_spreading_h(self, contact_radius_m):
        """Equivalent Robin coefficient for a circular half-space contact."""
        radius = float(contact_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("electrode contact radius must be positive")
        return (4.0 * self.copper_thermal_conductivity_W_mK
                / (math.pi * radius))

    @staticmethod
    def _canonical_electrode_boundary_mode(mode):
        aliases = {
            "fixed": "fixed_temperature",
            "fixed_temperature": "fixed_temperature",
            "semi_infinite_copper": "semi_infinite_copper_spreading",
            "semi_infinite_copper_spreading": (
                "semi_infinite_copper_spreading"),
        }
        try:
            return aliases[str(mode)]
        except KeyError as exc:
            raise ValueError(
                "electrode_boundary_mode must be fixed_temperature or "
                "semi_infinite_copper_spreading") from exc

    def _configure_electrode_thermal_boundary(self):
        """Create the selected main or sensitivity electrode heat boundary."""
        ht = self.j.component("comp1").physics("ht")
        for tag in ("tempInS2S", "tempOutS2S",
                    "fluxInS2S", "fluxOutS2S"):
            self._remove_safe(ht.feature(), tag)

        mode = self._active_electrode_boundary_mode
        if mode == "fixed_temperature":
            for tag, selection in (
                    ("tempInS2S", "selInS2S"),
                    ("tempOutS2S", "selOutS2S")):
                ht.create(tag, "TemperatureBoundary", 2)
                ht.feature(tag).selection().named(selection)
                ht.feature(tag).set("T0", "Telectrode")
            return

        for tag, selection, coefficient in (
                ("fluxInS2S", "selInS2S", "hCuIn"),
                ("fluxOutS2S", "selOutS2S", "hCuOut")):
            ht.create(tag, "HeatFluxBoundary", 2)
            ht.feature(tag).selection().named(selection)
            ht.feature(tag).set(
                "HeatFluxType", "GeneralInwardHeatFlux")
            ht.feature(tag).set(
                "q0", f"{coefficient}*(Telectrode-T)")

    def _temperature_output_fields(self, steady_result):
        """Map one steady solve onto the versioned D3 output contract."""
        keys = (
            "TmaxAll_K", "TminAll_K", "TmeanAll_K", "UAll_pct",
            "TmaxActive_K", "TminActive_K", "TmeanActive_K",
            "UActive_pct", "activeVolumeFraction",
            "TmaxFreeSurface_K", "TminFreeSurface_K",
            "TmeanFreeSurface_K", "UFreeSurface_pct",
            "freeSurfaceArea_m2", "electrodeTemperatureUndershoot_K",
            "temperatureFallbackUsed",
        )
        return {key: steady_result[key] for key in keys}

    def _radiation_loss_metrics(self, gross, escape):
        """Return raw and reporting-safe self-view loss diagnostics."""
        if (not self._finite_number(gross) or not self._finite_number(escape)
                or float(gross) <= 0.0):
            nan = float('nan')
            return {
                "self_absorbed": nan,
                "loss_raw_pct": nan,
                "loss_pct": nan,
                "numerical_excess_pct": nan,
            }

        gross = float(gross)
        escape = float(escape)
        raw_pct = (1.0 - escape / gross) * 100.0
        return {
            "self_absorbed": max(0.0, gross - escape),
            "loss_raw_pct": raw_pct,
            "loss_pct": max(0.0, raw_pct),
            "numerical_excess_pct": max(0.0, -raw_pct),
        }

    @staticmethod
    def _feature_volume(radii_m, segment_length_m):
        """Return the volume of a piecewise-constant cylinder profile."""
        return sum(math.pi * radius ** 2 * segment_length_m
                   for radius in radii_m)

    @staticmethod
    def _shoulder_areas(radii_m):
        """Map each exposed inter-segment annulus to its larger segment.

        The two electrode contact faces are deliberately excluded. At an
        internal radius step, only the annulus outside the smaller neighbour
        is exposed and therefore contributes sublimation mass loss.
        """
        areas = [0.0] * len(radii_m)
        for index in range(len(radii_m) - 1):
            left = float(radii_m[index])
            right = float(radii_m[index + 1])
            if left > right:
                areas[index] += math.pi * (left ** 2 - right ** 2)
            elif right > left:
                areas[index + 1] += math.pi * (right ** 2 - left ** 2)
        return areas

    def _cylinder_erosion_rates(self, radii_m, temperatures_K):
        """Return radius recession rates including all exposed shoulders.

        Shoulder normal recession cannot be represented exactly by the eight
        fixed-length radius degrees of freedom. D2 therefore maps its volume
        loss conservatively onto the owning (larger) segment radius. The
        mapped radial volume loss exactly equals the sidewall plus shoulder
        surface-flux volume loss at the current state.
        """
        count = len(radii_m)
        if len(temperatures_K) != count:
            raise ValueError("radii and temperature arrays must have equal length")

        speeds = [
            self.Aev * math.exp(-self.Bev / float(temp)) / self.rho_mass
            for temp in temperatures_K
        ]
        volume_rates = [
            speeds[i] * 2.0 * math.pi * radii_m[i] * self.Lseg
            for i in range(count)
        ]
        shoulder_areas = [0.0] * count

        for index in range(count - 1):
            left = float(radii_m[index])
            right = float(radii_m[index + 1])
            if math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15):
                continue
            owner = index if left > right else index + 1
            area = math.pi * abs(left ** 2 - right ** 2)
            # The interface temperature is bracketed by its two adjacent
            # segment averages; use their mean in the exponential flux law.
            interface_temp = 0.5 * (
                float(temperatures_K[index])
                + float(temperatures_K[index + 1]))
            interface_speed = (
                self.Aev * math.exp(-self.Bev / interface_temp)
                / self.rho_mass)
            shoulder_areas[owner] += area
            volume_rates[owner] += interface_speed * area

        rates = []
        for index, radius in enumerate(radii_m):
            derivative = 2.0 * math.pi * float(radius) * self.Lseg
            rates.append(volume_rates[index] / derivative)
        return rates, shoulder_areas, volume_rates

    def _next_erosion_timestep(self, current, limits, rates,
                               resolution_delta, time_s):
        """Choose an exact, non-overshooting erosion interval."""
        candidates = [self.max_erosion_step_s]
        for value, limit, rate in zip(current, limits, rates):
            if rate <= 0.0:
                continue
            candidates.append(resolution_delta / rate)
            remaining = value - limit
            tolerance = self.erosion_rel_tol * max(abs(value), abs(limit), 1.0)
            if remaining > tolerance:
                candidates.append(remaining / rate)

        cap_remaining = self.max_lifetime_h * 3600.0 - time_s
        if cap_remaining <= self.erosion_time_tol_s:
            return 0.0
        candidates.append(cap_remaining)
        dt = min(candidates)
        return dt if dt > self.erosion_time_tol_s else 0.0

    def _advance_erosion_features(self, current, initial, limits, rates, dt):
        """Advance features and clamp floating-point noise at 20% failure."""
        updated = []
        losses = []
        failed_indices = []
        for index, (value, start, limit, rate) in enumerate(
                zip(current, initial, limits, rates)):
            candidate = max(limit, value - rate * dt)
            tolerance = self.erosion_rel_tol * max(abs(start), 1.0)
            if candidate <= limit + tolerance:
                candidate = limit
                failed_indices.append(index)
            updated.append(candidate)
            losses.append((start - candidate) / start)
        return updated, losses, failed_indices

    @staticmethod
    def _overtemperature_fraction(previous_K, current_K, limit_K):
        """Linear endpoint interpolation for the first overtemperature time."""
        if current_K < limit_K:
            return 1.0
        delta = current_K - previous_K
        if delta <= 0.0:
            return 0.0
        return min(1.0, max(0.0, (limit_K - previous_K) / delta))

    @staticmethod
    def _lifecycle_terminal_status(
            failed, cap_limited, step_limited, termination_reason):
        if failed:
            return "OK"
        if cap_limited:
            return "CENSORED_LIFETIME_CAP"
        if step_limited:
            return "CENSORED_STEP_LIMIT"
        if termination_reason == "negligible_erosion":
            return "CENSORED_NEGLIGIBLE_EROSION"
        return "CENSORED_UNRESOLVED"

    def _lifecycle_radiation_fields(
            self, initial, time_s, p03_gross_j, prad_gross_j,
            p03_escape_j, prad_escape_j):
        """Build explicit v2 radiation fields plus legacy compatibility aliases."""
        avg = lambda value: (value / time_s) if time_s > 0.0 else float('nan')
        loss = self._radiation_loss_metrics(p03_gross_j, p03_escape_j)
        return {
            "initialP03gross_W": initial["P03steady"],
            "initialPradGross_W": initial["PradSteady"],
            "initialP03escape_W": initial["P03sphere"],
            "initialPradEscape_W": initial["PradSphere"],
            "initialP03sphere_W": initial["P03sphere"],
            "initialPradSphere_W": initial["PradSphere"],
            "initialP03selfAbsorbed_W": initial["P03selfAbsorbed"],
            "initialSelfViewLossRaw_pct": initial["selfViewLossRaw_pct"],
            "initialSelfViewLoss_pct": initial["selfViewLoss_pct"],
            "initialRadiationNumericalExcess_pct":
                initial["radiationNumericalExcess_pct"],
            "initialP03ambient_W": initial["P03ambient"],
            "initialAmbient03ToEscape_pct": initial["ambient03ToEscape_pct"],
            "initialFambAreaAvg": initial["FambAreaAvg"],
            "lifeAvgP03gross_W": avg(p03_gross_j),
            "lifeAvgPradGross_W": avg(prad_gross_j),
            "lifeAvgP03escape_W": avg(p03_escape_j),
            "lifeAvgPradEscape_W": avg(prad_escape_j),
            "lifeAvgP03sphere_W": avg(p03_escape_j),
            "lifeAvgPradSphere_W": avg(prad_escape_j),
            "lifeTotalP03gross_J": p03_gross_j,
            "lifeTotalPradGross_J": prad_gross_j,
            "lifeTotalP03escape_J": p03_escape_j,
            "lifeTotalPradEscape_J": prad_escape_j,
            "lifeTotalP03sphere_J": p03_escape_j,
            "lifeTotalP03selfAbsorbed_J": loss["self_absorbed"],
            "selfViewLossRaw_pct": loss["loss_raw_pct"],
            "selfViewLoss_pct": loss["loss_pct"],
            "radiationNumericalExcess_pct": loss["numerical_excess_pct"],
        }

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

        # Fixed room-temperature copper is primary. The half-space spreading
        # resistance boundary is a D3 sensitivity case only.
        self._configure_electrode_thermal_boundary()

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
        # lambda_r is entered in micrometers by the COMSOL API. Passing the
        # unit-bearing parameter lam03 would multiply by [um] a second time.
        j.component("comp1").physics("rad").prop(
            "RadiationSettings").set("lambda_r", str(self.spectral_split_um))

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
        _, R_env, A_env = self._compute_geom(radii_m)
        self._set_params(radii_m, R_env, A_env, voltage)

        result = {
            "solve_ok": False, "search_ok": False, "failure": "",
            "applied_V": voltage, "search_steps": 0,
            "Tmax": float('nan'), "Tmin": float('nan'),
            "Tmean": float('nan'), "U_pct": float('nan'),
            "TmaxAll_K": float('nan'), "TminAll_K": float('nan'),
            "TmeanAll_K": float('nan'), "UAll_pct": float('nan'),
            "TmaxActive_K": float('nan'), "TminActive_K": float('nan'),
            "TmeanActive_K": float('nan'), "UActive_pct": float('nan'),
            "activeVolumeFraction": float('nan'),
            "TmaxFreeSurface_K": float('nan'),
            "TminFreeSurface_K": float('nan'),
            "TmeanFreeSurface_K": float('nan'),
            "UFreeSurface_pct": float('nan'),
            "freeSurfaceArea_m2": float('nan'),
            "electrodeTemperatureUndershoot_K": float('nan'),
            "temperatureFallbackUsed": False,
            "I": float('nan'), "R": float('nan'),
            "Pelec": float('nan'), "P03steady": float('nan'),
            "PradSteady": float('nan'), "P03sphere": float('nan'),
            "PradSphere": float('nan'), "P03gross": float('nan'),
            "PradGross": float('nan'), "P03escape": float('nan'),
            "PradEscape": float('nan'), "P03selfAbsorbed": float('nan'),
            "P03ambient": float('nan'), "ambient03ToEscape_pct": float('nan'),
            "FambAreaAvg": float('nan'), "selfViewLossRaw_pct": float('nan'),
            "selfViewLoss_pct": float('nan'),
            "radiationNumericalExcess_pct": float('nan'),
            "volume_m3": float('nan'),
            "expectedVolume_m3": float('nan'),
            "volumeLossFromInitial_pct": float('nan'),
            "geometryVolumeError_rel": float('nan'),
            "targetVolumeDeviation_rel": float('nan'),
            "vol_err": float('nan'),
            "temp_ok": False, "volume_ok": False, "current_ok": False,
            "seg_Tavg": [0.0] * self.seg_count,
            "segMaskAreaRatio": [0.0] * self.seg_count,
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
            Tmin = float(
                j.result().numerical("minTS2S").getReal()[0][0])

            V = float(
                j.result().numerical("volS2S").getReal()[0][0])
            TintVol = float(
                j.result().numerical("TintVolS2S").getReal()[0][0])
            Tmean = TintVol / V if V > 1e-20 else float('nan')
            U_pct = self._temperature_uniformity(Tmax, Tmin, Tmean)

            Tmax_active = float(j.result().numerical(
                "maxTActiveS2S").getReal()[0][0])
            Tmin_active = float(j.result().numerical(
                "minTActiveS2S").getReal()[0][0])
            V_active = float(j.result().numerical(
                "volActiveS2S").getReal()[0][0])
            Tint_active = float(j.result().numerical(
                "TintActiveS2S").getReal()[0][0])
            Tmean_active = (Tint_active / V_active
                            if V_active > 1e-20 else float('nan'))
            U_active = self._temperature_uniformity(
                Tmax_active, Tmin_active, Tmean_active)

            Tmax_free = float(j.result().numerical(
                "maxTFreeS2S").getReal()[0][0])
            Tmin_free = float(j.result().numerical(
                "minTFreeS2S").getReal()[0][0])
            A_free = float(j.result().numerical(
                "AsurfS2S").getReal()[0][0])
            Tint_free = float(j.result().numerical(
                "TintFreeS2S").getReal()[0][0])
            Tmean_free = (Tint_free / A_free
                          if A_free > 1e-20 else float('nan'))
            U_free = self._temperature_uniformity(
                Tmax_free, Tmin_free, Tmean_free)
            active_volume_fraction = (
                V_active / V if V > 1e-20 else float('nan'))
            electrode_undershoot = max(
                0.0, self.electrode_temperature_K - Tmin)
            if electrode_undershoot > self.electrode_temperature_tolerance_K:
                raise RuntimeError(
                    "temperature minimum is below the electrode reference "
                    f"by {electrode_undershoot:.6g} K")
            I = abs(float(
                j.result().numerical("IinS2S").getReal()[0][0]))
            P03steady = float(
                j.result().numerical("P03emitS2S").getReal()[0][0])
            PradSteady = float(
                j.result().numerical("PradEmitS2S").getReal()[0][0])
            P03sphere = float(
                j.result().numerical("P03escapeS2S").getReal()[0][0])
            PradSphere = float(
                j.result().numerical("PradEscapeS2S").getReal()[0][0])
            P03ambient = float(
                j.result().numerical("P03ambientS2S").getReal()[0][0])
            Famb_area = float(
                j.result().numerical("FambAreaS2S").getReal()[0][0])
            A_rad = float(
                j.result().numerical("AradS2S").getReal()[0][0])
            Famb_area_avg = (Famb_area / A_rad
                             if A_rad > 1e-20 else float('nan'))
            ambient_ratio = (P03ambient / P03sphere * 100.0
                             if P03sphere > 1e-20 else float('nan'))
            loss = self._radiation_loss_metrics(P03steady, P03sphere)

            # D2: every segment erosion rate must use its COMSOL surface value.
            seg_Tavg = []
            seg_mask_area_ratios = []
            for i in range(self.seg_count):
                try:
                    Tint = float(j.result().numerical(
                        f"TintSeg_{i + 1}").getReal()[0][0])
                    Aseg = float(j.result().numerical(
                        f"AsegS2S_{i + 1}").getReal()[0][0])
                    if Aseg > 1e-20:
                        _, expected_area = self._segment_lateral_mask(
                            i, radii_m[i])
                        value = Tint / Aseg
                        ratio = Aseg / expected_area
                        if (not self._finite_number(value)
                                or not self._finite_number(ratio)
                                or not 0.50 <= ratio <= 1.50):
                            raise RuntimeError(
                                f"invalid masked area ratio {ratio}")
                        seg_Tavg.append(value)
                        seg_mask_area_ratios.append(ratio)
                    else:
                        raise RuntimeError("non-positive segment area")
                except Exception as exc:
                    raise RuntimeError(
                        f"local surface integral failed for segment {i + 1}") from exc

            # 体积误差
            V0now = sum(
                math.pi * r ** 2 * self.Lseg for r in radii_m)
            V0ref = self.reference_volume
            geometry_vol_err = abs(V - V0now) / V0ref
            target_volume_deviation = abs(V0now - V0ref) / V0ref
            initial_volume = self._feature_volume(
                self._initial_radii, self.Lseg)
            volume_loss_pct = 100.0 * (1.0 - V0now / initial_volume)
            initial_state = all(
                math.isclose(value, initial, rel_tol=0.0, abs_tol=1.0e-14)
                for value, initial in zip(radii_m, self._initial_radii))

            finite_checks = {
                "Tmax": Tmax,
                "Tmin": Tmin,
                "Tmean": Tmean,
                "U_pct": U_pct,
                "TmaxActive_K": Tmax_active,
                "TminActive_K": Tmin_active,
                "TmeanActive_K": Tmean_active,
                "UActive_pct": U_active,
                "activeVolumeFraction": active_volume_fraction,
                "TmaxFreeSurface_K": Tmax_free,
                "TminFreeSurface_K": Tmin_free,
                "TmeanFreeSurface_K": Tmean_free,
                "UFreeSurface_pct": U_free,
                "freeSurfaceArea_m2": A_free,
                "electrodeTemperatureUndershoot_K": electrode_undershoot,
                "volume": V,
                "expectedVolume_m3": V0now,
                "volumeLossFromInitial_pct": volume_loss_pct,
                "geometryVolumeError_rel": geometry_vol_err,
                "targetVolumeDeviation_rel": target_volume_deviation,
                "current": I,
                "P03steady": P03steady,
                "PradSteady": PradSteady,
                "P03sphere": P03sphere,
                "PradSphere": PradSphere,
                "P03ambient": P03ambient,
                "ambient03ToEscape_pct": ambient_ratio,
                "FambAreaAvg": Famb_area_avg,
                "selfViewLossRaw_pct": loss["loss_raw_pct"],
                "selfViewLoss_pct": loss["loss_pct"],
                "radiationNumericalExcess_pct": loss["numerical_excess_pct"],
                "volume_m3": V,
                "expectedVolume_m3": V0now,
                "volumeLossFromInitial_pct": volume_loss_pct,
                "vol_err": geometry_vol_err,
            }
            invalid = [k for k, v in finite_checks.items()
                       if not self._finite_number(v)]
            invalid += [f"seg_Tavg[{i}]" for i, v in enumerate(seg_Tavg)
                        if not self._finite_number(v)]
            invalid += [f"segMaskAreaRatio[{i}]"
                        for i, value in enumerate(seg_mask_area_ratios)
                        if not self._finite_number(value)]
            if invalid:
                raise RuntimeError(
                    "Invalid non-finite COMSOL result: " + ", ".join(invalid))
            if not 0.0 < active_volume_fraction <= 1.0 + 1.0e-6:
                raise RuntimeError(
                    "Invalid active-region volume fraction: "
                    f"{active_volume_fraction}")
            if A_free <= 0.0:
                raise RuntimeError("Invalid non-positive free surface area")
            negative_powers = [
                name for name, value in (
                    ("P03steady", P03steady),
                    ("PradSteady", PradSteady),
                    ("P03sphere", P03sphere),
                    ("PradSphere", PradSphere),
                    ("P03ambient", P03ambient),
                ) if value < 0.0
            ]
            if negative_powers:
                raise RuntimeError(
                    "Invalid negative radiation result: "
                    + ", ".join(negative_powers))

            result.update({
                "solve_ok": True,
                "Tmax": Tmax, "Tmin": Tmin,
                "Tmean": Tmean, "U_pct": U_pct,
                "TmaxAll_K": Tmax, "TminAll_K": Tmin,
                "TmeanAll_K": Tmean, "UAll_pct": U_pct,
                "TmaxActive_K": Tmax_active,
                "TminActive_K": Tmin_active,
                "TmeanActive_K": Tmean_active,
                "UActive_pct": U_active,
                "activeVolumeFraction": active_volume_fraction,
                "TmaxFreeSurface_K": Tmax_free,
                "TminFreeSurface_K": Tmin_free,
                "TmeanFreeSurface_K": Tmean_free,
                "UFreeSurface_pct": U_free,
                "freeSurfaceArea_m2": A_free,
                "electrodeTemperatureUndershoot_K": electrode_undershoot,
                "temperatureFallbackUsed": False,
                "I": I,
                "Pelec": voltage * I,
                "P03steady": P03steady, "PradSteady": PradSteady,
                "P03sphere": P03sphere, "PradSphere": PradSphere,
                "P03gross": P03steady, "PradGross": PradSteady,
                "P03escape": P03sphere, "PradEscape": PradSphere,
                "P03selfAbsorbed": loss["self_absorbed"],
                "P03ambient": P03ambient,
                "ambient03ToEscape_pct": ambient_ratio,
                "FambAreaAvg": Famb_area_avg,
                "selfViewLossRaw_pct": loss["loss_raw_pct"],
                "selfViewLoss_pct": loss["loss_pct"],
                "radiationNumericalExcess_pct": loss["numerical_excess_pct"],
                "volume_m3": V,
                "expectedVolume_m3": V0now,
                "volumeLossFromInitial_pct": volume_loss_pct,
                "geometryVolumeError_rel": geometry_vol_err,
                "targetVolumeDeviation_rel": target_volume_deviation,
                "vol_err": geometry_vol_err,
                "temp_ok": Tmax < self.temp_limit_K,
                "volume_ok": (
                    geometry_vol_err <= self.vol_tol
                    and (not initial_state
                         or target_volume_deviation <= self.vol_tol)),
                "current_ok": I > self.current_tol,
                "R": (voltage / I) if I > self.current_tol else float('nan'),
                "seg_Tavg": seg_Tavg,
                "segMaskAreaRatio": seg_mask_area_ratios,
            })
        except Exception as e:
            try:
                self.client.names()
            except Exception:
                raise ServerDisconnectError(str(e)) from e
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
        """Return the scalar score used by the D3 rated-point selection."""
        value = result.get(objective, float('nan'))
        if self._finite_number(value):
            return float(value)

        if objective in ("lifeTotalP03sphere_J", "lifeTotalP03escape_J"):
            avg_key = ("lifeAvgP03escape_W"
                       if objective.endswith("escape_J")
                       else "lifeAvgP03sphere_W")
            avg = result.get(avg_key, float('nan'))
            life_h = result.get("lifetimeH", float('nan'))
            if self._finite_number(avg) and self._finite_number(life_h):
                return float(avg) * float(life_h) * 3600.0

        return float('nan')

    def _rated_voltage_candidate_eligible(self, result, objective):
        """Only exact, constraint-clean lifecycle results can be rated."""
        voltage = result.get("Vwork_V", float('nan'))
        max_temperature = result.get("maxErosionTmax_K", float('nan'))
        score = self._voltage_score(result, objective)
        return (
            result.get("status") == "OK"
            and result.get("failureReached") is True
            and result.get("lifetimeExact") is True
            and result.get("censored") is False
            and result.get("capLimited") is False
            and result.get("stepLimited") is False
            and self._finite_number(voltage)
            and 0.0 < float(voltage) <= self.voltage_upper
            and self._finite_number(max_temperature)
            and float(max_temperature) < self.temp_limit_K
            and self._finite_number(score)
        )

    def _rated_voltage_sort_key(self, result, objective):
        score = self._voltage_score(result, objective)
        uniformity = result.get("U_pct", float('inf'))
        lifetime_h = result.get("lifetimeH", float('-inf'))
        voltage = result.get("Vwork_V", float('inf'))
        uniformity = (float(uniformity)
                      if self._finite_number(uniformity) else float('inf'))
        lifetime_h = (float(lifetime_h)
                      if self._finite_number(lifetime_h) else float('-inf'))
        voltage = (float(voltage)
                   if self._finite_number(voltage) else float('inf'))
        return float(score), -uniformity, lifetime_h, -voltage

    def _voltage_scan_summary(self, results, objective):
        parts = []
        for item in results:
            voltage = item.get("Vwork_V", float('nan'))
            status = item.get("status", "UNKNOWN")
            score = self._voltage_score(item, objective)
            exact = self._rated_voltage_candidate_eligible(item, objective)
            v_txt = f"{voltage:.4g}V" if self._finite_number(voltage) else "nanV"
            s_txt = f"{score:.4g}" if self._finite_number(score) else "nan"
            parts.append(f"{v_txt}:{status}:exact={int(exact)}:{s_txt}")
        return "; ".join(parts)

    def _annotate_voltage_result(self, result, policy, objective,
                                 max_safe_v=None, candidate_count=1,
                                 scan_summary=""):
        result["voltagePolicy"] = policy
        result["voltageObjective"] = objective
        result["voltageCandidateCount"] = candidate_count
        result["operatingPointVersion"] = self.operating_point_version
        result.setdefault("ratedVoltageEligible", False)
        result.setdefault("ratedVoltageExactCandidateCount", 0)
        result.setdefault("ratedVoltageSelectionReason", "not_rated_scan")
        result["voltageCandidateRatios"] = ",".join(
            f"{ratio:g}" for ratio in self.voltage_candidate_ratios)
        result["metricVersion"] = self.metric_version
        result["physicsVersion"] = self.physics_version
        result["geometryVersion"] = self.geometry_version
        result["lifecycleVersion"] = self.lifecycle_version
        result["erosionModel"] = self.erosion_model
        result["failureFraction"] = self.failure_fraction
        result["maxErosionStep_s"] = self.max_erosion_step_s
        result["geometryVolumeTolerance_rel"] = self.vol_tol
        result["radiationEscapeMethod"] = self.radiation_escape_method
        result["spectralSplit_um"] = self.spectral_split_um
        result["thermalAmbient_K"] = self.thermal_ambient_K
        result["scoreAmbientTarget_K"] = self.score_ambient_target_K
        result["temperatureStatisticVersion"] = (
            self.temperature_statistic_version)
        result["temperaturePrimaryDomain"] = self.temperature_primary_domain
        result["activeTemperatureTrim_mm"] = (
            self.active_temperature_trim_m * 1.0e3)
        result["electrodeBoundaryMode"] = (
            self._active_electrode_boundary_mode)
        result["electrodeBoundaryVersion"] = self.electrode_boundary_version
        result["electrodeBoundaryApproximation"] = (
            self.electrode_boundary_approximation
            if self._active_electrode_boundary_mode
            == "semi_infinite_copper_spreading"
            else "none_fixed_temperature")
        result["electrodeTemperature_K"] = self.electrode_temperature_K
        result["copperThermalConductivity_W_mK"] = (
            self.copper_thermal_conductivity_W_mK)
        radii = self._initial_radii or [self.reference_radius] * self.seg_count
        result["electrodeContactRadiusIn_mm"] = radii[0] * 1.0e3
        result["electrodeContactRadiusOut_mm"] = radii[-1] * 1.0e3
        result["electrodeSpreadingHIn_W_m2K"] = (
            self._copper_spreading_h(radii[0]))
        result["electrodeSpreadingHOut_W_m2K"] = (
            self._copper_spreading_h(radii[-1]))
        if max_safe_v is not None:
            result["voltageMaxSafe_V"] = max_safe_v
        if scan_summary:
            result["voltageScanSummary"] = scan_summary
        return result

    def _select_voltage_scan_result(self, results, objective):
        eligible = [item for item in results
                    if self._rated_voltage_candidate_eligible(
                        item, objective)]
        if eligible:
            selected = dict(max(
                eligible,
                key=lambda item: self._rated_voltage_sort_key(
                    item, objective)))
            selected["ratedVoltageEligible"] = True
            selected["ratedVoltageExactCandidateCount"] = len(eligible)
            selected["ratedVoltageSelectionReason"] = (
                "maximum_exact_lifecycle_0_3um_escape_energy")
            selected["ratedVoltageSourceStatus"] = selected.get("status", "")
            return selected

        finite = [item for item in results
                  if self._finite_number(
                      self._voltage_score(item, objective))]
        if finite:
            diagnostic = max(
                finite,
                key=lambda item: self._rated_voltage_sort_key(
                    item, objective))
        elif results:
            diagnostic = results[0]
        else:
            diagnostic = {}
        selected = dict(diagnostic)
        selected["ratedVoltageSourceStatus"] = selected.get("status", "")
        selected["status"] = "FAIL_RATED_VOLTAGE_INCONCLUSIVE"
        selected["failure"] = (
            "No voltage candidate completed an exact, uncensored 20% "
            "failure lifecycle within all temperature constraints.")
        selected["ratedVoltageEligible"] = False
        selected["ratedVoltageExactCandidateCount"] = 0
        selected["ratedVoltageSelectionReason"] = (
            "no_exact_constraint_clean_lifecycle_candidate")
        selected["lifetimeExact"] = False
        return selected

    def evaluate_voltage_candidates(self, radii_m, voltage_candidates=None,
                                    objective=None,
                                    electrode_boundary_mode=None):
        """Run full lifecycle evaluations for candidate working voltages."""
        objective = objective or self.voltage_objective
        scan_start = time.time()
        boundary_mode = self._canonical_electrode_boundary_mode(
            electrode_boundary_mode or self.electrode_boundary_mode)

        print("  D3 rated scan: evaluating max-safe voltage first...")
        first = self.evaluate(
            radii_m,
            voltage_policy="max_safe",
            voltage_objective=objective,
            electrode_boundary_mode=boundary_mode,
        )
        if not self._finite_number(first.get("Vwork_V", float('nan'))):
            selected = self._select_voltage_scan_result([first], objective)
            selected["voltageScanElapsed_sec"] = round(
                time.time() - scan_start, 1)
            return self._annotate_voltage_result(
                selected, "rated_lifecycle_scan", objective, None, 1,
                self._voltage_scan_summary([first], objective))

        max_safe_v = float(first["Vwork_V"])
        candidates = self._build_voltage_candidates(
            max_safe_v, voltage_candidates)

        results = [first]
        for voltage in candidates:
            if abs(voltage - max_safe_v) <= self.voltage_tol:
                continue
            print(f"  D3 rated scan: evaluating {voltage:.4f}V...")
            result = self.evaluate(
                radii_m,
                voltage_policy="fixed",
                voltage_objective=objective,
                voltage_override=voltage,
                electrode_boundary_mode=boundary_mode,
            )
            results.append(result)

        selected = self._select_voltage_scan_result(results, objective)
        selected["voltageScanElapsed_sec"] = round(time.time() - scan_start, 1)
        return self._annotate_voltage_result(
            selected,
            "rated_lifecycle_scan",
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

    def _restart_at_radii(self, initial_radii, current_radii, voltage):
        """Rebuild the model on the live client and restore erosion state.

        mph only permits one client per Python process. Disconnecting and then
        calling mph.start() returns the same disconnected client, so a genuine
        server loss must be handled by the outer process-level resume worker.
        """
        try:
            self.client.names()
        except Exception as exc:
            raise ServerDisconnectError(
                "COMSOL server disconnected; process restart is required"
            ) from exc
        print("  Rebuilding COMSOL model and restoring current erosion radii...")
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass
        self.model = None
        self.j = None
        self._init_model(initial_radii)
        return self._solve_at_voltage(current_radii, voltage)

    # ================================================================
    #  主评估入口
    # ================================================================

    def evaluate(self, radii_m, voltage_policy=None, voltage_candidates=None,
                 voltage_objective=None, voltage_override=None,
                 electrode_boundary_mode=None):
        """
        完整评估流程：建模 → 电压搜索 → 侵蚀循环。

        Args:
            radii_m: 长度 8 的 list，各段半径 (m)

        Returns:
            dict: 包含所有赛题指标，或 status != "OK" 表示失败
        """
        policy = voltage_policy or self.voltage_policy
        if policy in ("full_scan", "scan"):
            policy = "rated_lifecycle_scan"
        if voltage_override is not None:
            policy = "fixed"
        objective = voltage_objective or self.voltage_objective
        self._active_electrode_boundary_mode = (
            self._canonical_electrode_boundary_mode(
                electrode_boundary_mode or self.electrode_boundary_mode))
        if policy == "rated_lifecycle_scan" and voltage_override is None:
            return self.evaluate_voltage_candidates(
                radii_m,
                voltage_candidates=voltage_candidates,
                objective=objective,
                electrode_boundary_mode=(
                    self._active_electrode_boundary_mode),
            )
        if policy not in ("max_safe", "fixed"):
            raise ValueError(
                "voltage_policy must be rated_lifecycle_scan, max_safe, or fixed")

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

        # ---- Phase 2: geometry/lifecycle v2 erosion loop ----
        print("  Phase 2: geometry/lifecycle v2 erosion loop...")
        time_s = 0.0
        p03_integral = 0.0
        prad_integral = 0.0
        p03_sphere_integral = 0.0
        prad_sphere_integral = 0.0
        macro_step = 0
        attempted_steps = 0
        failed = False
        cap_limited = False
        step_limited = False
        censored = False
        termination_reason = ""
        failure_feature = ""
        failure_index = ""
        max_loss_frac = 0.0

        initial_volume = self._feature_volume(
            self._initial_radii, self.Lseg)
        initial_shoulder_area = sum(
            self._shoulder_areas(self._initial_radii))
        max_shoulder_area = initial_shoulder_area

        prev_P03 = r0_res["P03steady"]
        prev_Prad = r0_res["PradSteady"]
        prev_P03sphere = r0_res["P03sphere"]
        prev_PradSphere = r0_res["PradSphere"]
        prev_Tmax = r0_res["Tmax"]
        Tavg = r0_res["seg_Tavg"]
        max_erosion_tmax = r0_res["Tmax"]
        erosion_retry_count = 0
        overtemp_fields = {}
        failure_text = ""

        def build_result(status):
            final_volume = self._feature_volume(radii, self.Lseg)
            final_shoulder_area = sum(self._shoulder_areas(radii))
            return {
                "Vwork_V": Vwork,
                "initialTmax_K": r0_res["Tmax"],
                "Tmin_K": r0_res["Tmin"],
                "Tmean_K": r0_res["Tmean"],
                "U_pct": r0_res["U_pct"],
                **self._temperature_output_fields(r0_res),
                "lifetimeH": time_s / 3600.0,
                **self._lifecycle_radiation_fields(
                    r0_res, time_s, p03_integral, prad_integral,
                    p03_sphere_integral, prad_sphere_integral),
                "maxErosionTmax_K": max_erosion_tmax,
                "failureReached": failed,
                "capLimited": cap_limited,
                "stepLimited": step_limited,
                "censored": censored,
                "lifetimeExact": failed and status == "OK",
                "terminationReason": termination_reason,
                "failureFeature": failure_feature,
                "failureIndex": failure_index,
                "maxFeatureLoss_pct": 100.0 * max_loss_frac,
                "erosionSteps": macro_step,
                "erosionAttemptedSteps": attempted_steps,
                "erosionSolveRetries": erosion_retry_count,
                "maxLifetimeCap_h": self.max_lifetime_h,
                "maxErosionSteps": self.max_erosion_steps,
                "initialCOMSOLVolume_m3": r0_res["volume_m3"],
                "initialExpectedVolume_m3": r0_res["expectedVolume_m3"],
                "initialGeometryVolumeError_rel": (
                    r0_res["geometryVolumeError_rel"]),
                "initialTargetVolumeDeviation_rel": (
                    r0_res["targetVolumeDeviation_rel"]),
                "initialSegmentMaskAreaRatioMin": min(
                    r0_res["segMaskAreaRatio"]),
                "initialSegmentMaskAreaRatioMax": max(
                    r0_res["segMaskAreaRatio"]),
                "initialVolume_m3": initial_volume,
                "finalVolume_m3": final_volume,
                "volumeLoss_pct": 100.0 * (
                    1.0 - final_volume / initial_volume),
                "initialShoulderArea_m2": initial_shoulder_area,
                "finalShoulderArea_m2": final_shoulder_area,
                "maxShoulderArea_m2": max_shoulder_area,
                **overtemp_fields,
                "status": status,
                "failure": failure_text,
                "elapsed_sec": round(time.time() - t_start, 1),
            }

        while macro_step < self.max_erosion_steps and not failed:
            drdt, _, _ = self._cylinder_erosion_rates(radii, Tavg)
            if max(drdt) < 1.0e-15:
                remaining = max(
                    0.0, self.max_lifetime_h * 3600.0 - time_s)
                p03_integral += prev_P03 * remaining
                prad_integral += prev_Prad * remaining
                p03_sphere_integral += prev_P03sphere * remaining
                prad_sphere_integral += prev_PradSphere * remaining
                time_s += remaining
                termination_reason = "negligible_erosion"
                censored = True
                print("  Evaporation negligible; integrated constant power "
                      "to the lifecycle cap and censored the lifetime.")
                break

            dt_macro = self._next_erosion_timestep(
                radii, self._fail_radii, drdt,
                resolve_threshold, time_s)
            if dt_macro <= 0.0:
                cap_limited = True
                censored = True
                termination_reason = "lifetime_cap"
                break

            candidate_radii, candidate_losses, failed_indices = (
                self._advance_erosion_features(
                    radii, self._initial_radii, self._fail_radii,
                    drdt, dt_macro))
            candidate_time_s = time_s + dt_macro
            candidate_max_loss = max(candidate_losses)
            attempted_steps += 1

            print(f"  SOLVING step={attempted_steps} "
                  f"t={candidate_time_s / 3600.0:.3f}h "
                  f"loss={candidate_max_loss:.6f}")
            r_now = None
            solve_failure = ""
            for attempt in range(self.max_erosion_solve_retries + 1):
                try:
                    if attempt == 0:
                        r_now = self._solve_at_voltage(candidate_radii, Vwork)
                    else:
                        erosion_retry_count += 1
                        print(f"  RETRY step {attempted_steps}: {attempt}/"
                              f"{self.max_erosion_solve_retries}")
                        r_now = self._restart_at_radii(
                            self._initial_radii, candidate_radii, Vwork)
                    if r_now.get("solve_ok", False):
                        break
                    solve_failure = r_now.get(
                        "failure", "unknown solve failure")
                except ServerDisconnectError:
                    raise
                except Exception as exc:
                    try:
                        solve_failure = str(exc)
                    except Exception:
                        solve_failure = exc.__class__.__name__
                    r_now = None
                if attempt < self.max_erosion_solve_retries:
                    print(f"  WARN step {attempted_steps} attempt "
                          f"{attempt + 1} failed: {solve_failure}")

            if r_now is None or not r_now.get("solve_ok", False):
                # The candidate interval is not committed until its endpoint
                # has solved. Report only the last verified state and energy.
                censored = True
                termination_reason = "erosion_solve_failure"
                failure_text = solve_failure
                print(f"  WARN: solve failed step {attempted_steps}: "
                      f"{solve_failure}")
                result = build_result("FAIL_EROSION_SOLVE")
                return self._annotate_voltage_result(
                    result, policy, objective, max_safe_v)

            print(f"  SOLVED step={attempted_steps} "
                  f"Tmax={r_now['Tmax']:.1f}K "
                  f"P03escape={r_now['P03sphere']:.2f}W")
            max_erosion_tmax = max(max_erosion_tmax, r_now["Tmax"])

            cur_P03 = r_now["P03steady"]
            cur_Prad = r_now["PradSteady"]
            cur_P03sphere = r_now["P03sphere"]
            cur_PradSphere = r_now["PradSphere"]

            if r_now["Tmax"] >= self.temp_limit_K:
                fraction = self._overtemperature_fraction(
                    prev_Tmax, r_now["Tmax"], self.temp_limit_K)
                valid_dt = fraction * dt_macro

                def at_crossing(previous, current):
                    return previous + fraction * (current - previous)

                cross_P03 = at_crossing(prev_P03, cur_P03)
                cross_Prad = at_crossing(prev_Prad, cur_Prad)
                cross_P03sphere = at_crossing(
                    prev_P03sphere, cur_P03sphere)
                cross_PradSphere = at_crossing(
                    prev_PradSphere, cur_PradSphere)
                p03_integral += 0.5 * (prev_P03 + cross_P03) * valid_dt
                prad_integral += 0.5 * (prev_Prad + cross_Prad) * valid_dt
                p03_sphere_integral += 0.5 * (
                    prev_P03sphere + cross_P03sphere) * valid_dt
                prad_sphere_integral += 0.5 * (
                    prev_PradSphere + cross_PradSphere) * valid_dt

                radii, crossing_losses, crossing_failed = (
                    self._advance_erosion_features(
                        radii, self._initial_radii, self._fail_radii,
                        drdt, valid_dt))
                time_s += valid_dt
                macro_step += 1
                max_loss_frac = max(crossing_losses)
                failed = bool(crossing_failed)
                if crossing_failed:
                    failure_feature = "segment_radius"
                    failure_index = crossing_failed[0] + 1
                censored = not failed
                termination_reason = "overtemperature"
                overtemp_fields = {
                    "overtempStep": macro_step,
                    "overtempTimeH": time_s / 3600.0,
                    "overtempTmax_K": r_now["Tmax"],
                    "overtempInterpolationFraction": fraction,
                    "overtempBracketEndTimeH": candidate_time_s / 3600.0,
                }
                result = build_result("FAIL_OVERTEMP_DURING_EROSION")
                return self._annotate_voltage_result(
                    result, policy, objective, max_safe_v)

            # Commit the verified endpoint, then integrate the exact interval.
            p03_integral += 0.5 * (prev_P03 + cur_P03) * dt_macro
            prad_integral += 0.5 * (prev_Prad + cur_Prad) * dt_macro
            p03_sphere_integral += 0.5 * (
                prev_P03sphere + cur_P03sphere) * dt_macro
            prad_sphere_integral += 0.5 * (
                prev_PradSphere + cur_PradSphere) * dt_macro
            radii = candidate_radii
            time_s = candidate_time_s
            macro_step += 1
            max_loss_frac = candidate_max_loss
            max_shoulder_area = max(
                max_shoulder_area, sum(self._shoulder_areas(radii)))

            prev_P03 = cur_P03
            prev_Prad = cur_Prad
            prev_P03sphere = cur_P03sphere
            prev_PradSphere = cur_PradSphere
            prev_Tmax = r_now["Tmax"]
            Tavg = r_now["seg_Tavg"]

            if failed_indices:
                failed = True
                termination_reason = "feature_loss_20pct"
                failure_feature = "segment_radius"
                failure_index = failed_indices[0] + 1
            elif time_s >= (self.max_lifetime_h * 3600.0
                            - self.erosion_time_tol_s):
                cap_limited = True
                censored = True
                termination_reason = "lifetime_cap"

            if macro_step % 5 == 0 or failed or cap_limited:
                print(f"  STEP={macro_step} t={time_s / 3600:.2f}h "
                      f"loss={max_loss_frac:.6f}")
            if cap_limited:
                break

        if not termination_reason:
            step_limited = not failed
            censored = step_limited
            termination_reason = (
                "step_limit" if step_limited else "feature_loss_20pct")

        status = self._lifecycle_terminal_status(
            failed, cap_limited, step_limited, termination_reason)

        result = build_result(status)
        required = [
            "Vwork_V", "initialTmax_K", "Tmin_K", "Tmean_K", "U_pct",
            "TmaxAll_K", "TminAll_K", "TmeanAll_K", "UAll_pct",
            "TmaxActive_K", "TminActive_K", "TmeanActive_K",
            "UActive_pct", "activeVolumeFraction",
            "TmaxFreeSurface_K", "TminFreeSurface_K",
            "TmeanFreeSurface_K", "UFreeSurface_pct",
            "maxErosionTmax_K", "lifetimeH",
            "initialP03sphere_W", "initialPradSphere_W",
            "lifeAvgP03sphere_W", "lifeAvgPradSphere_W",
            "lifeTotalP03sphere_J", "initialP03gross_W",
            "initialP03escape_W", "lifeAvgP03gross_W",
            "lifeAvgP03escape_W", "lifeTotalP03gross_J",
            "lifeTotalP03escape_J", "selfViewLossRaw_pct",
            "selfViewLoss_pct", "erosionSteps",
        ]
        invalid = [key for key in required
                   if not self._finite_number(result.get(key))]
        if invalid:
            result["status"] = "FAIL_INVALID_RESULT"
            result["failure"] = (
                "Non-finite final metric(s): " + ", ".join(invalid))
            result["censored"] = True
            result["lifetimeExact"] = False

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
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client = None
        self.j = None
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
