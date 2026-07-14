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

退蚀模型（geometry/lifecycle v2）：
  - 每个方截面路径段保留独立边长；较大截面的 connector cube 连接转角
  - 侧壁、转角肩面、两端 stub 侧壁及内肩面统一换算为体积损失
  - 接触电极的两个外端面不辐射、不升华
  - 任意 block 边长或 stub 半径损失达到 20% 时失效

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
        self.vol_tol = 1.0e-4
        self.current_tol = 1e-9
        self.sphere_margin = 1.05
        self.voltage_upper = 100.0
        self.voltage_floor = 1e-3
        self.voltage_tol = 0.05
        self.max_voltage_iters = 20
        self.voltage_policy = "max_safe"
        self.voltage_objective = "lifeTotalP03sphere_J"
        self.metric_version = "radiation_escape_v2"
        self.physics_version = "thermal_s2s_v2"
        self.geometry_version = "zigzag_local_erosion_v2"
        self.lifecycle_version = "lifecycle_v2"
        self.erosion_model = "local_blocks_turn_caps_and_terminal_stubs"
        self.turn_connector_rule = "max_adjacent_side_cube_split_external_faces"
        self.geometry_side_quantum_fraction = 0.01
        self.radiation_escape_method = "s2s_radiosity_famb"
        self.spectral_split_um = 3.0
        self.thermal_ambient_K = 293.15
        self.score_ambient_target_K = 0.0
        self.voltage_candidate_ratios = (
            1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60)
        self.Aev = 3.9e9
        self.Bev = 1.023e5
        self.failure_fraction = 0.20
        self.MAX_BLOCK_SLOTS = 64
        self.max_erosion_steps = 50
        self.max_erosion_solve_retries = 2
        self.max_lifetime_h = 200.0   # 侵蚀循环最大仿真时长，避免低蒸发率构型无限运行
        self.max_erosion_step_s = 36000.0
        self.erosion_rel_tol = 1.0e-10
        self.erosion_time_tol_s = 1.0e-6

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
        self._current_blocks = None
        self._path_points = None
        self._segment_endpoints = None
        self._turn_flags = None
        self._block_lengths = None
        self._current_block_sides = None
        self._current_stub_radii = None
        self._last_block_mask_area_ratios = []
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
        """Build equal-side path segments; turns are separate connector cubes."""
        count = sum(
            1 for p0, p1 in zip(pts[:-1], pts[1:])
            if abs(p1[0] - p0[0]) + abs(p1[1] - p0[1]) > 1.0e-12)
        return COMSOLRunner.build_blocks_with_sides(pts, [side] * count)

    @staticmethod
    def build_blocks_with_sides(pts, sides):
        """Build one exact centreline-length cuboid per path segment.

        A turn is completed by :meth:`build_turn_blocks`. Keeping segment
        cuboids at their exact centreline length avoids the non-manifold,
        micron-wide ledges created when unequal segments both extend through
        the same corner.
        """
        segments = []
        for p0, p1 in zip(pts[:-1], pts[1:]):
            dx, dz = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1.0e-12 and abs(dz) < 1.0e-12:
                continue
            segments.append((p0, p1, dx, dz))
        if len(sides) != len(segments):
            raise ValueError("one block side is required per non-zero path segment")

        blocks = []
        for index, ((p0, p1, dx, dz), side_value) in enumerate(
                zip(segments, sides)):
            side = float(side_value)
            if side <= 0.0:
                raise ValueError("block sides must be positive")
            half = 0.5 * side
            tag = f"blk_{index + 1}"
            if abs(dx) > 1.0e-12:
                x_lo, x_hi = sorted((p0[0], p1[0]))
                blocks.append((tag, x_lo, -half, p0[1] - half,
                               x_hi - x_lo, side, side))
            else:
                z_lo, z_hi = sorted((p0[1], p1[1]))
                blocks.append((tag, p0[0] - half, -half, z_lo,
                               side, side, z_hi - z_lo))
        return blocks

    @staticmethod
    def build_turn_blocks(pts, sides):
        """Build robust 90-degree connector cubes owned by the larger side.

        The max-side cube gives each incoming segment a finite-volume overlap
        and turns a local side mismatch into a conventional shoulder. COMSOL
        can mesh this topology reliably, unlike intersecting unequal cuboids
        whose face terminates on another cuboid's edge.
        """
        path_points = []
        segments = []
        for p0, p1 in zip(pts[:-1], pts[1:]):
            dx, dz = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1.0e-12 and abs(dz) < 1.0e-12:
                continue
            if not path_points:
                path_points.append(p0)
            path_points.append(p1)
            segments.append((dx, dz))
        if len(sides) != len(segments):
            raise ValueError("one block side is required per non-zero path segment")

        turns = []
        for index in range(len(segments) - 1):
            dx0, dz0 = segments[index]
            dx1, dz1 = segments[index + 1]
            if (abs(dx0) > 1.0e-12) == (abs(dx1) > 1.0e-12):
                continue
            side = max(float(sides[index]), float(sides[index + 1]))
            half = 0.5 * side
            x, z = path_points[index + 1]
            turns.append((f"turn_{index + 1}", x - half, -half, z - half,
                          side, side, side))
        return turns

    @staticmethod
    def path_segment_lengths(pts):
        """Return centerline lengths for non-zero Manhattan segments."""
        lengths = []
        for p0, p1 in zip(pts[:-1], pts[1:]):
            length = abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
            if length > 1.0e-12:
                lengths.append(length)
        return lengths

    @staticmethod
    def path_segments(pts):
        """Return endpoint pairs for non-zero Manhattan path segments."""
        return [
            (p0, p1) for p0, p1 in zip(pts[:-1], pts[1:])
            if (abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
                > 1.0e-12)
        ]

    @staticmethod
    def turn_flags(segment_endpoints):
        """Mark adjacent path segments joined by a 90-degree connector."""
        flags = []
        for (p0, p1), (q0, q1) in zip(
                segment_endpoints[:-1], segment_endpoints[1:]):
            first_horizontal = abs(p1[0] - p0[0]) > 1.0e-12
            second_horizontal = abs(q1[0] - q0[0]) > 1.0e-12
            flags.append(first_horizontal != second_horizontal)
        return flags

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

    def compute_envelope(self, blocks, stub_radii=None, block_sides=None):
        """计算包含所有 blocks + terminal stubs 的外接球半径。"""
        max_dist = 0.0
        envelope_blocks = list(blocks)
        if block_sides is not None:
            if self._path_points is None:
                raise RuntimeError("path points are not initialized")
            envelope_blocks.extend(
                self.build_turn_blocks(self._path_points, block_sides))
        for _, x0, y0, z0, sx, sy, sz in envelope_blocks:
            for x in (x0, x0 + sx):
                for y in (y0, y0 + sy):
                    for z in (z0, z0 + sz):
                        max_dist = max(max_dist, math.sqrt(
                            x ** 2 + y ** 2 + (z - 0.5 * self.L0) ** 2))
        radii = list(stub_radii or (self.R0, self.R0))
        for radius, z_values in zip(
                radii,
                ((0.0, self.STUB_LEN),
                 (self.L0 - self.STUB_LEN, self.L0))):
            for zz in z_values:
                max_dist = max(max_dist, math.sqrt(
                    radius ** 2 + (zz - 0.5 * self.L0) ** 2))
        return self.sphere_margin * max_dist

    # ================================================================
    #  侵蚀后几何
    # ================================================================

    def eroded_blocks(self, block_sides):
        """Rebuild every block from its own current side length."""
        if self._path_points is None:
            raise RuntimeError("path points are not initialized")
        return self.build_blocks_with_sides(self._path_points, block_sides)

    def _project_block_sides_to_geometry(self, exact_sides):
        """Project exact erosion state onto mesh-resolvable local geometry.

        Lifecycle state and the 20% failure check remain continuous. Only the
        COMSOL geometry is quantized, to prevent sub-micron shoulder faces from
        falling below the mesh/S2S resolution. D6 owns convergence of this
        numerical quantum.
        """
        if self._init_side is None or self._init_side <= 0.0:
            raise RuntimeError("initial block side is not initialized")
        quantum = self.geometry_side_quantum_fraction * self._init_side
        lower = (1.0 - self.failure_fraction) * self._init_side
        projected = []
        for side in exact_sides:
            loss = max(0.0, self._init_side - float(side))
            levels = math.floor(loss / quantum + 0.5 + 1.0e-12)
            represented = self._init_side - levels * quantum
            projected.append(min(self._init_side, max(lower, represented)))
        return projected

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
    def _turn_cap_areas(block_sides):
        """Map each connector shoulder to its larger adjacent block."""
        areas = [0.0] * len(block_sides)
        for index in range(len(block_sides) - 1):
            left = float(block_sides[index])
            right = float(block_sides[index + 1])
            if left > right:
                areas[index] += left ** 2 - right ** 2
            elif right > left:
                areas[index + 1] += right ** 2 - left ** 2
        return areas

    @staticmethod
    def _exposed_segment_lengths(block_sides, path_lengths,
                                 turn_flags=None):
        """Return straight lengths outside all max-side connector cubes."""
        sides = [float(value) for value in block_sides]
        lengths = [float(value) for value in path_lengths]
        if len(sides) != len(lengths):
            raise ValueError("block sides and path lengths must align")
        if not sides:
            return []
        flags = (list(turn_flags) if turn_flags is not None
                 else [True] * (len(sides) - 1))
        if len(flags) != len(sides) - 1:
            raise ValueError("turn flags must align with adjacent blocks")

        exposed = list(lengths)
        for index, is_turn in enumerate(flags):
            if not is_turn:
                continue
            overlap = 0.5 * max(sides[index], sides[index + 1])
            exposed[index] -= overlap
            exposed[index + 1] -= overlap
        for index, length in enumerate(exposed):
            if length <= 1.0e-12:
                raise ValueError(
                    f"connector cubes consume block {index + 1} "
                    f"(exposed length={length:.6e} m)")
        return exposed

    @staticmethod
    def _turn_surface_areas(block_sides, turn_flags=None):
        """Return connector exterior and shoulder areas for every turn."""
        sides = [float(value) for value in block_sides]
        flags = (list(turn_flags) if turn_flags is not None
                 else [True] * max(0, len(sides) - 1))
        if len(flags) != max(0, len(sides) - 1):
            raise ValueError("turn flags must align with adjacent blocks")
        areas = []
        for index, is_turn in enumerate(flags):
            if not is_turn:
                areas.append((0.0, 0.0))
                continue
            left = sides[index]
            right = sides[index + 1]
            large = max(left, right)
            connector_exterior = 4.0 * large ** 2
            shoulder = ((large ** 2 - left ** 2)
                        + (large ** 2 - right ** 2))
            areas.append((connector_exterior, shoulder))
        return areas

    @staticmethod
    def _turn_volume_corrections(block_sides):
        """Extra union volume introduced by max-side connector cubes."""
        corrections = []
        for left, right in zip(block_sides[:-1], block_sides[1:]):
            left = float(left)
            right = float(right)
            large = max(left, right)
            corrections.append(
                large ** 3 - 0.5 * large * (left ** 2 + right ** 2))
        return corrections

    @staticmethod
    def _block_volume_derivatives(block_sides, path_lengths):
        """Return dV/d(side_i) for segments plus max-side turn connectors."""
        derivatives = [
            2.0 * float(side) * float(length)
            for side, length in zip(block_sides, path_lengths)]
        for index in range(len(block_sides) - 1):
            left = float(block_sides[index])
            right = float(block_sides[index + 1])
            if math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15):
                continue
            if left > right:
                derivatives[index] += 1.5 * left ** 2 - 0.5 * right ** 2
                derivatives[index + 1] -= left * right
            else:
                derivatives[index] -= left * right
                derivatives[index + 1] += 1.5 * right ** 2 - 0.5 * left ** 2
        return derivatives

    @staticmethod
    def _circle_square_overlap_area(radius, side):
        """Exact overlap of a centred circle and axis-aligned square."""
        radius = max(0.0, float(radius))
        side = max(0.0, float(side))
        half = 0.5 * side
        if radius <= half:
            return math.pi * radius ** 2
        if radius >= math.sqrt(2.0) * half:
            return side ** 2

        crossing = math.sqrt(max(0.0, radius ** 2 - half ** 2))

        def primitive(x):
            root = math.sqrt(max(0.0, radius ** 2 - x ** 2))
            return 0.5 * (
                x * root + radius ** 2 * math.asin(x / radius))

        quadrant = half * crossing + primitive(half) - primitive(crossing)
        return 4.0 * quadrant

    def _block_erosion_rates(self, block_sides, temperatures_K,
                             path_lengths=None, lateral_areas=None,
                             surface_sides=None):
        """Return local recession from all exposed straight and turn faces."""
        lengths = list(path_lengths or self._block_lengths or [])
        if not (len(block_sides) == len(temperatures_K) == len(lengths)):
            raise ValueError(
                "block sides, temperatures and path lengths must align")
        represented_sides = list(surface_sides or block_sides)
        if len(represented_sides) != len(block_sides):
            raise ValueError("represented block sides must align")
        turn_flags = (
            self._turn_flags
            if self._turn_flags is not None
            and len(self._turn_flags) == len(block_sides) - 1
            else [True] * max(0, len(block_sides) - 1))

        speeds = [
            self.Aev * math.exp(-self.Bev / float(temp)) / self.rho_mass
            for temp in temperatures_K
        ]
        if lateral_areas is None:
            exposed_lengths = self._exposed_segment_lengths(
                represented_sides, lengths, turn_flags)
            areas = [
                4.0 * represented_sides[i] * exposed_lengths[i]
                for i in range(len(block_sides))]
        else:
            areas = list(lateral_areas)
            if len(areas) != len(block_sides):
                raise ValueError("block lateral areas must align")
        volume_rates = [
            speeds[i] * areas[i] for i in range(len(block_sides))]
        cap_areas = [0.0] * len(block_sides)

        turn_areas = self._turn_surface_areas(
            represented_sides, turn_flags)
        for index, (connector_area, shoulder_area) in enumerate(turn_areas):
            if connector_area <= 0.0:
                continue
            left = float(represented_sides[index])
            right = float(represented_sides[index + 1])
            interface_temp = 0.5 * (
                float(temperatures_K[index])
                + float(temperatures_K[index + 1]))
            interface_speed = (
                self.Aev * math.exp(-self.Bev / interface_temp)
                / self.rho_mass)
            # Two connector exterior faces are associated with each branch.
            # This split preserves uniform recession for equal neighbouring
            # sides; the dimensional shoulder belongs to the larger branch.
            half_connector_area = 0.5 * connector_area
            volume_rates[index] += interface_speed * half_connector_area
            volume_rates[index + 1] += (
                interface_speed * half_connector_area)
            if shoulder_area > 0.0:
                owner = index if left > right else index + 1
                cap_areas[owner] += shoulder_area
                volume_rates[owner] += interface_speed * shoulder_area

        derivatives = self._block_volume_derivatives(block_sides, lengths)
        rates = [
            volume_rate / derivative
            for volume_rate, derivative in zip(volume_rates, derivatives)]
        return rates, cap_areas, volume_rates

    def _stub_erosion_rates(self, stub_radii, stub_temperatures_K,
                            end_block_sides, end_block_temperatures_K,
                            lateral_areas=None):
        """Map stub sidewall and exposed inner shoulder loss to stub radii."""
        if not (len(stub_radii) == len(stub_temperatures_K)
                == len(end_block_sides) == len(end_block_temperatures_K)
                == 2):
            raise ValueError("two terminal states are required")

        rates = []
        shoulder_areas = []
        volume_rates = []
        if lateral_areas is not None and len(lateral_areas) != 2:
            raise ValueError("two stub lateral areas are required")
        for terminal_index, (radius, stub_temp, block_side, block_temp) in enumerate(zip(
                stub_radii, stub_temperatures_K,
                end_block_sides, end_block_temperatures_K)):
            radius = float(radius)
            lateral_area = (
                float(lateral_areas[terminal_index])
                if lateral_areas is not None
                else 2.0 * math.pi * radius * self.STUB_LEN)
            overlap = self._circle_square_overlap_area(radius, block_side)
            shoulder_area = max(0.0, math.pi * radius ** 2 - overlap)
            side_speed = (
                self.Aev * math.exp(-self.Bev / float(stub_temp))
                / self.rho_mass)
            interface_temp = 0.5 * (float(stub_temp) + float(block_temp))
            shoulder_speed = (
                self.Aev * math.exp(-self.Bev / interface_temp)
                / self.rho_mass)
            volume_rate = (
                side_speed * lateral_area
                + shoulder_speed * shoulder_area)
            derivative = 2.0 * math.pi * radius * self.STUB_LEN
            rates.append(volume_rate / derivative)
            shoulder_areas.append(shoulder_area)
            volume_rates.append(volume_rate)
        return rates, shoulder_areas, volume_rates

    @staticmethod
    def _erosion_state_volume(block_sides, path_lengths,
                              stub_radii, stub_length):
        """Volume represented by the lifecycle degrees of freedom."""
        block_volume = sum(
            side ** 2 * length
            for side, length in zip(block_sides, path_lengths))
        turn_volume = sum(
            COMSOLRunner._turn_volume_corrections(block_sides))
        stub_volume = sum(
            math.pi * radius ** 2 * stub_length for radius in stub_radii)
        return block_volume + turn_volume + stub_volume

    def _next_erosion_timestep(self, current, limits, rates,
                               resolution_delta, time_s):
        """Choose an exact interval bounded by resolution, failure and cap."""
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
        """Advance local dimensions and clamp exactly at their failure limits."""
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

    def _is_server_alive(self):
        """Quick check: returns False if COMSOL server is disconnected."""
        if self.client is None:
            return False
        try:
            self.client.names()
            return True
        except Exception:
            return False

    def _restart_at_geometry(self, N_RUNS, L_RUN_m, z_first_m,
                             blocks, block_sides, stub_radii,
                             R_env, voltage):
        """Rebuild on the live client and restore exact erosion geometry.

        A disconnected mph client cannot be restarted inside the same Python
        process. In that case the process-level resume worker must retry the
        same trial in a fresh process.
        """
        if not self._is_server_alive():
            raise ServerDisconnectError(
                "COMSOL server disconnected; process restart is required")
        print("  Rebuilding COMSOL model and restoring erosion geometry...")
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass
        self.model = None
        self.j = None
        self._init_model(
            N_RUNS, L_RUN_m, z_first_m,
            geometry_state=(blocks, block_sides, stub_radii, R_env))
        return self._solve_prepared(voltage)

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

    def _block_exposed_interval(self, index, block_sides):
        """Return axis and interval outside the adjacent connector cubes."""
        if self._segment_endpoints is None or self._turn_flags is None:
            raise RuntimeError("path segments are not initialized")
        if len(block_sides) != len(self._segment_endpoints):
            raise ValueError("block sides and path segments must align")
        if not 0 <= index < len(block_sides):
            raise IndexError("block index is out of range")

        p0, p1 = self._segment_endpoints[index]
        dx, dz = p1[0] - p0[0], p1[1] - p0[1]
        if abs(dx) > 1.0e-12:
            axis = "x"
            coordinate = 0
        elif abs(dz) > 1.0e-12:
            axis = "z"
            coordinate = 1
        else:
            raise ValueError(f"block {index + 1} has zero path length")

        start_trim = 0.0
        if index > 0 and self._turn_flags[index - 1]:
            start_trim = 0.5 * max(
                float(block_sides[index - 1]), float(block_sides[index]))
        end_trim = 0.0
        if index < len(block_sides) - 1 and self._turn_flags[index]:
            end_trim = 0.5 * max(
                float(block_sides[index]), float(block_sides[index + 1]))

        start = float(p0[coordinate])
        end = float(p1[coordinate])
        direction = 1.0 if end > start else -1.0
        exposed_start = start + direction * start_trim
        exposed_end = end - direction * end_trim
        exposed_length = direction * (exposed_end - exposed_start)
        if exposed_length <= 1.0e-12:
            raise ValueError(
                f"connector cubes consume block {index + 1} "
                f"(exposed length={exposed_length:.6e} m)")
        return (axis, min(exposed_start, exposed_end),
                max(exposed_start, exposed_end), exposed_length)

    def _block_lateral_mask(self, index, block, block_sides):
        """Return an exposed-straight point mask and its analytic area."""
        side = float(block_sides[index])
        _, x0, y0, z0, sx, sy, sz = block
        path_axis, exposed_low, exposed_high, exposed_length = (
            self._block_exposed_interval(index, block_sides))
        inset = min(max(1.0e-12, 0.02 * exposed_length),
                    0.20 * exposed_length)
        active_low = exposed_low + inset
        active_high = exposed_high - inset
        active_length = active_high - active_low
        pad = max(1.0e-12, 1.0e-6 * side)

        def gt(name, value):
            return f"{name}>{value:.16g}[m]"

        def lt(name, value):
            return f"{name}<{value:.16g}[m]"

        conditions = [
            gt("y", y0 - pad), lt("y", y0 + sy + pad),
        ]
        if path_axis == "x":
            conditions.extend((
                gt("x", active_low), lt("x", active_high),
                gt("z", z0 - pad), lt("z", z0 + sz + pad),
            ))
        else:
            conditions.extend((
                gt("x", x0 - pad), lt("x", x0 + sx + pad),
                gt("z", active_low), lt("z", active_high),
            ))
        return "&&".join(conditions), 4.0 * side * active_length

    def _update_block_surface_operators(self, blocks, block_sides):
        """Update per-block exposed-surface integration masks."""
        if len(blocks) != len(block_sides):
            raise ValueError("block selection and side arrays must align")
        for i, block in enumerate(blocks):
            condition, _ = self._block_lateral_mask(
                i, block, block_sides)
            try:
                self.j.result().numerical(f"TintBlk_{i + 1}").set(
                    "expr", [f"if({condition},T,0[K])"])
                self.j.result().numerical(f"AblkLat_{i + 1}").set(
                    "expr", [f"if({condition},1,0)"])
            except Exception:
                pass

    def _init_model(self, N_RUNS, L_RUN_m, z_first_m,
                    geometry_state=None):
        """从空白创建完整 COMSOL 模型（含 1V 预热求解）。"""
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass

        side, blocks, _ = self.compute_side_and_blocks(
            N_RUNS, L_RUN_m, z_first_m)
        z_last = self.L0 - z_first_m
        self._path_points = self.build_full_path(
            N_RUNS, L_RUN_m, z_first_m, z_last,
            self.STUB_LEN, self.L0)
        self._segment_endpoints = self.path_segments(self._path_points)
        self._turn_flags = self.turn_flags(self._segment_endpoints)
        self._block_lengths = self.path_segment_lengths(self._path_points)
        self._blocks0 = blocks
        self._init_side = side
        self._n_blocks = len(blocks)
        initial_sides = [side] * self._n_blocks
        initial_stub_radii = [self.R0, self.R0]
        if geometry_state is None:
            build_blocks = blocks
            build_sides = initial_sides
            build_stub_radii = initial_stub_radii
            R_env = self.compute_envelope(
                build_blocks, build_stub_radii, build_sides)
        else:
            build_blocks, build_sides, build_stub_radii, R_env = (
                geometry_state)
            build_blocks = list(build_blocks)
            build_sides = list(build_sides)
            build_stub_radii = list(build_stub_radii)
            if len(build_blocks) != self._n_blocks:
                raise ValueError("restart geometry block count changed")
        self._current_blocks = list(build_blocks)
        self._current_block_sides = list(build_sides)
        self._current_stub_radii = list(build_stub_radii)

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
        j.param().set("Tamb", f"{self.thermal_ambient_K}[K]",
                      "Ambient temperature for S2S solve")
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
        self._rebuild(
            build_blocks, build_sides, build_stub_radii, R_env)

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
        j.result().numerical("P03emitZZ").selection().all()
        j.result().numerical("P03emitZZ").set(
            "expr", ["rad.epsilonu_band1*rad.ebu1"])

        j.result().numerical().create("PradEmitZZ", "IntSurface")
        j.result().numerical("PradEmitZZ").selection().all()
        j.result().numerical("PradEmitZZ").set(
            "expr", ["rad.epsilonu_band1*rad.ebu1+"
                     "rad.epsilonu_band2*rad.ebu2"])

        j.result().numerical().create("P03escapeZZ", "IntSurface")
        j.result().numerical("P03escapeZZ").selection().all()
        j.result().numerical("P03escapeZZ").set(
            "expr", ["rad.J_band1*rad.Famb1"])

        j.result().numerical().create("PradEscapeZZ", "IntSurface")
        j.result().numerical("PradEscapeZZ").selection().all()
        j.result().numerical("PradEscapeZZ").set(
            "expr", ["rad.J_band1*rad.Famb1+"
                     "rad.J_band2*rad.Famb2"])

        j.result().numerical().create("P03ambientZZ", "IntSurface")
        j.result().numerical("P03ambientZZ").selection().all()
        j.result().numerical("P03ambientZZ").set("expr", ["rad.Gamb1"])

        j.result().numerical().create("FambAreaZZ", "IntSurface")
        j.result().numerical("FambAreaZZ").selection().all()
        j.result().numerical("FambAreaZZ").set("expr", ["rad.Famb1"])

        j.result().numerical().create("AradZZ", "IntSurface")
        j.result().numerical("AradZZ").selection().all()
        j.result().numerical("AradZZ").set("expr", ["1"])

        for i in range(self._n_blocks):
            int_t_tag = f"TintBlk_{i + 1}"
            int_a_tag = f"AblkLat_{i + 1}"
            try:
                j.result().numerical().remove(int_t_tag)
            except Exception:
                pass
            j.result().numerical().create(int_t_tag, "IntSurface")
            j.result().numerical(int_t_tag).selection().all()
            condition, _ = self._block_lateral_mask(
                i, self._current_blocks[i], self._current_block_sides)
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

        for suffix, selection in (
                ("In", "selStubInLatZZ"),
                ("Out", "selStubOutLatZZ")):
            temp_tag = f"TintStub{suffix}ZZ"
            area_tag = f"AStub{suffix}LatZZ"
            j.result().numerical().create(temp_tag, "IntSurface")
            j.result().numerical(temp_tag).selection().named(selection)
            j.result().numerical(temp_tag).set("expr", ["T"])
            j.result().numerical().create(area_tag, "IntSurface")
            j.result().numerical(area_tag).selection().named(selection)
            j.result().numerical(area_tag).set("expr", ["1"])

        # 健全性检查
        san_T = float(j.result().numerical("maxTZZ").getReal()[0][0])
        san_I = abs(float(j.result().numerical("IinZZ").getReal()[0][0]))
        if san_T < 200.0:
            raise RuntimeError(f"S2S coupling failure: Tmax={san_T:.1f}K")
        print(f"  Model init OK: sanity Tmax={san_T:.1f}K I={san_I:.4f}A")

    # ================================================================
    #  几何 + 物理场构建
    # ================================================================

    def _rebuild(self, blocks, block_sides, stub_radii,
                 R_env, geom_only=False):
        """重建几何+网格，可选重建物理场。

        geom_only=True：只重建几何+网格（侵蚀循环专用），保留 S2S/EC/材料不动。
        geom_only=False：完整重建（初始建模专用）。

        Box 选择（selInZZ/selOutZZ）是坐标驱动的，几何重建后自动刷新，
        无需在 geom_only 模式下重建。S2S 视角因子由 study.run() 自动重算。
        """
        j = self.j
        geom = j.component("comp1").geom("geom1")
        block_sides = list(block_sides)
        stub_radii = list(stub_radii)
        if len(block_sides) != len(blocks):
            raise ValueError("block geometry and side arrays must align")
        if len(stub_radii) != 2:
            raise ValueError("two terminal stub radii are required")

        # 清旧几何
        self._remove_safe(geom.feature(), "uniZZ")
        self._remove_safe(geom.feature(), "term_in")
        self._remove_safe(geom.feature(), "term_out")
        for i in range(self.MAX_BLOCK_SLOTS):
            self._remove_safe(geom.feature(), f"blk_{i + 1}")
            self._remove_safe(geom.feature(), f"turn_{i + 1}")

        # 创建 blocks
        tags = []
        for tag, x0, y0, z0, sx, sy, sz in blocks:
            tags.append(tag)
            geom.create(tag, "Block")
            geom.feature(tag).set("size",
                                  [f"{sx}[m]", f"{sy}[m]", f"{sz}[m]"])
            geom.feature(tag).set("pos",
                                  [f"{x0}[m]", f"{y0}[m]", f"{z0}[m]"])

        turn_blocks = self.build_turn_blocks(
            self._path_points, block_sides)
        for tag, x0, y0, z0, sx, sy, sz in turn_blocks:
            tags.append(tag)
            geom.create(tag, "Block")
            geom.feature(tag).set("size",
                                  [f"{sx}[m]", f"{sy}[m]", f"{sz}[m]"])
            geom.feature(tag).set("pos",
                                  [f"{x0}[m]", f"{y0}[m]", f"{z0}[m]"])

        # Terminal stubs（圆柱）
        geom.create("term_in", "Cylinder")
        geom.feature("term_in").set("r", f"{stub_radii[0]}[m]")
        geom.feature("term_in").set("h", f"{self.STUB_LEN}[m]")
        geom.feature("term_in").set("pos", ["0[m]", "0[m]", "0[m]"])

        geom.create("term_out", "Cylinder")
        geom.feature("term_out").set("r", f"{stub_radii[1]}[m]")
        geom.feature("term_out").set("h", f"{self.STUB_LEN}[m]")
        geom.feature("term_out").set("pos",
            ["0[m]", "0[m]", f"{self.L0 - self.STUB_LEN}[m]"])

        # Union
        geom.create("uniZZ", "Union")
        geom.feature("uniZZ").selection("input").set(tags + ["term_in", "term_out"])
        geom.feature("uniZZ").set("intbnd", False)
        geom.run()
        self._current_blocks = list(blocks)
        self._current_block_sides = list(block_sides)
        self._current_stub_radii = list(stub_radii)
        self._update_block_surface_operators(blocks, block_sides)

        if not geom_only:
            # Box selections — 电极面（坐标驱动，geom_only 时无需重建）
            self._remove_safe(j.component("comp1").selection(), "selInZZ")
            self._remove_safe(j.component("comp1").selection(), "selOutZZ")
            self._remove_safe(j.component("comp1").selection(), "selFreeZZ")
            self._remove_safe(
                j.component("comp1").selection(), "selStubInLatZZ")
            self._remove_safe(
                j.component("comp1").selection(), "selStubOutLatZZ")

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

            # Stub lateral faces drive their own local recession. Insets in z
            # exclude both the protected electrode contacts and inner caps.
            stub_pad_mm = (self.R0 * 1e3) + 0.1
            stub_inset_mm = 0.10 * self.STUB_LEN * 1e3
            for tag, zmin, zmax in (
                    ("selStubInLatZZ", stub_inset_mm,
                     self.STUB_LEN * 1e3 - stub_inset_mm),
                    ("selStubOutLatZZ",
                     (self.L0 - self.STUB_LEN) * 1e3 + stub_inset_mm,
                     self.L0 * 1e3 - stub_inset_mm)):
                j.component("comp1").selection().create(tag, "Box")
                j.component("comp1").selection(tag).geom("geom1", 2)
                j.component("comp1").selection(tag).set(
                    "condition", "intersects")
                j.component("comp1").selection(tag).set("xmin", -stub_pad_mm)
                j.component("comp1").selection(tag).set("xmax", stub_pad_mm)
                j.component("comp1").selection(tag).set("ymin", -stub_pad_mm)
                j.component("comp1").selection(tag).set("ymax", stub_pad_mm)
                j.component("comp1").selection(tag).set("zmin", zmin)
                j.component("comp1").selection(tag).set("zmax", zmax)

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
            "lambda_r", str(self.spectral_split_um))

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

    def _read_block_surface_states(self, Tmin, Tmax):
        """Read masked local temperatures and return analytic sidewall areas."""
        blocks = self._current_blocks or self._blocks0 or []
        block_Tavg = []
        block_areas = []
        area_ratios = []
        for i in range(self._n_blocks):
            read_ok = False
            block = blocks[i] if i < len(blocks) else self._blocks0[i]
            side = self._current_block_sides[i]
            _, expected_mask_area = self._block_lateral_mask(
                i, block, self._current_block_sides)
            _, _, _, exposed_length = self._block_exposed_interval(
                i, self._current_block_sides)
            try:
                Tint = float(self.j.result().numerical(
                    f"TintBlk_{i + 1}").getReal()[0][0])
                masked_area = float(self.j.result().numerical(
                    f"AblkLat_{i + 1}").getReal()[0][0])
                if masked_area > 1.0e-20:
                    value = Tint / masked_area
                    ratio = masked_area / expected_mask_area
                    if (self._finite_number(value)
                            and self._finite_number(ratio)
                            and 0.50 <= ratio <= 1.50):
                        block_Tavg.append(value)
                        block_areas.append(4.0 * side * exposed_length)
                        area_ratios.append(ratio)
                        read_ok = True
            except Exception:
                pass

            if not read_ok:
                raise RuntimeError(
                    f"masked local surface integral failed for block {i + 1}")

        self._last_block_mask_area_ratios = area_ratios
        return block_Tavg, block_areas

    def _read_stub_surface_states(self, Tmin, Tmax):
        """Read lateral temperature and actual area for terminal stubs."""
        values = []
        areas = []
        for index, suffix in enumerate(("In", "Out")):
            radius = self._current_stub_radii[index]
            try:
                integral = float(self.j.result().numerical(
                    f"TintStub{suffix}ZZ").getReal()[0][0])
                area = float(self.j.result().numerical(
                    f"AStub{suffix}LatZZ").getReal()[0][0])
                value = integral / area if area > 1.0e-20 else float('nan')
                if not self._finite_number(value):
                    raise RuntimeError("non-finite stub temperature")
            except Exception as exc:
                raise RuntimeError(
                    f"local stub surface integral failed for {suffix}") from exc
            values.append(value)
            areas.append(2.0 * math.pi * radius * self.STUB_LEN)
        return values, areas

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
            "block_Tavg": [0.0] * self._n_blocks,
            "block_A_lat": [0.0] * self._n_blocks,
            "blockMaskAreaRatio": [0.0] * self._n_blocks,
            "stub_Tavg": [0.0, 0.0],
            "stub_A_lat": [0.0, 0.0],
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
            P03sphere = float(
                j.result().numerical("P03escapeZZ").getReal()[0][0])
            PradSphere = float(
                j.result().numerical("PradEscapeZZ").getReal()[0][0])
            P03ambient = float(
                j.result().numerical("P03ambientZZ").getReal()[0][0])
            Famb_area = float(
                j.result().numerical("FambAreaZZ").getReal()[0][0])
            A_rad = float(j.result().numerical("AradZZ").getReal()[0][0])
            Famb_area_avg = (Famb_area / A_rad
                             if A_rad > 1e-20 else float('nan'))
            ambient_ratio = (P03ambient / P03sphere * 100.0
                             if P03sphere > 1e-20 else float('nan'))
            loss = self._radiation_loss_metrics(P03, P03sphere)

            # Per-block temperatures drive erosion. Prefer COMSOL
            # lateral-surface integrals; fallback is local.
            block_Tavg, block_A_lat = self._read_block_surface_states(
                Tmin, Tmax)
            block_mask_area_ratios = list(
                self._last_block_mask_area_ratios)
            stub_Tavg, stub_A_lat = self._read_stub_surface_states(
                Tmin, Tmax)

            expected_volume = self._erosion_state_volume(
                self._current_block_sides, self._block_lengths,
                self._current_stub_radii, self.STUB_LEN)
            geometry_vol_err = abs(V - expected_volume) / self.V0
            target_volume_deviation = (
                abs(expected_volume - self.V0) / self.V0)
            volume_loss_pct = 100.0 * (1.0 - expected_volume / self.V0)
            initial_state = (
                all(math.isclose(side, self._init_side, rel_tol=0.0,
                                 abs_tol=1.0e-14)
                    for side in self._current_block_sides)
                and all(math.isclose(radius, self.R0, rel_tol=0.0,
                                     abs_tol=1.0e-14)
                        for radius in self._current_stub_radii))

            finite_checks = {
                "Tmax": Tmax,
                "Tmin": Tmin,
                "Tmean": Tmean,
                "U_pct": U_pct,
                "volume": V,
                "expectedVolume_m3": expected_volume,
                "volumeLossFromInitial_pct": volume_loss_pct,
                "geometryVolumeError_rel": geometry_vol_err,
                "targetVolumeDeviation_rel": target_volume_deviation,
                "current": I,
                "P03steady": P03,
                "PradSteady": Prad,
                "P03sphere": P03sphere,
                "PradSphere": PradSphere,
                "P03ambient": P03ambient,
                "ambient03ToEscape_pct": ambient_ratio,
                "FambAreaAvg": Famb_area_avg,
                "selfViewLossRaw_pct": loss["loss_raw_pct"],
                "selfViewLoss_pct": loss["loss_pct"],
                "radiationNumericalExcess_pct": loss["numerical_excess_pct"],
                "volume_m3": V,
                "expectedVolume_m3": expected_volume,
                "volumeLossFromInitial_pct": volume_loss_pct,
                "vol_err": geometry_vol_err,
            }
            invalid = [key for key, value in finite_checks.items()
                       if not self._finite_number(value)]
            invalid += [f"block_Tavg[{i}]"
                        for i, value in enumerate(block_Tavg)
                        if not self._finite_number(value)]
            invalid += [f"stub_Tavg[{i}]"
                        for i, value in enumerate(stub_Tavg)
                        if not self._finite_number(value)]
            invalid += [f"block_A_lat[{i}]"
                        for i, value in enumerate(block_A_lat)
                        if not self._finite_number(value) or value <= 0.0]
            invalid += [f"blockMaskAreaRatio[{i}]"
                        for i, value in enumerate(block_mask_area_ratios)
                        if not self._finite_number(value)]
            invalid += [f"stub_A_lat[{i}]"
                        for i, value in enumerate(stub_A_lat)
                        if not self._finite_number(value) or value <= 0.0]
            if invalid:
                raise RuntimeError(
                    "Invalid non-finite COMSOL result: " + ", ".join(invalid))
            negative_powers = [
                name for name, value in (
                    ("P03steady", P03),
                    ("PradSteady", Prad),
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
                "I": I,
                "R": (voltage / I) if I > self.current_tol else float('nan'),
                "Pelec": voltage * I,
                "P03steady": P03, "PradSteady": Prad,
                "P03sphere": P03sphere, "PradSphere": PradSphere,
                "P03gross": P03, "PradGross": Prad,
                "P03escape": P03sphere, "PradEscape": PradSphere,
                "P03selfAbsorbed": loss["self_absorbed"],
                "P03ambient": P03ambient,
                "ambient03ToEscape_pct": ambient_ratio,
                "FambAreaAvg": Famb_area_avg,
                "selfViewLossRaw_pct": loss["loss_raw_pct"],
                "selfViewLoss_pct": loss["loss_pct"],
                "radiationNumericalExcess_pct": loss["numerical_excess_pct"],
                "volume_m3": V,
                "expectedVolume_m3": expected_volume,
                "volumeLossFromInitial_pct": volume_loss_pct,
                "geometryVolumeError_rel": geometry_vol_err,
                "targetVolumeDeviation_rel": target_volume_deviation,
                "vol_err": geometry_vol_err,
                "temp_ok":    Tmax < self.temp_limit_K,
                "volume_ok": (
                    geometry_vol_err <= self.vol_tol
                    and (not initial_state
                         or target_volume_deviation <= self.vol_tol)),
                "current_ok": I > self.current_tol,
                "block_Tavg": block_Tavg,
                "block_A_lat": block_A_lat,
                "blockMaskAreaRatio": block_mask_area_ratios,
                "stub_Tavg": stub_Tavg,
                "stub_A_lat": stub_A_lat,
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
        result["metricVersion"] = self.metric_version
        result["physicsVersion"] = self.physics_version
        result["geometryVersion"] = self.geometry_version
        result["lifecycleVersion"] = self.lifecycle_version
        result["erosionModel"] = self.erosion_model
        result["turnConnectorRule"] = self.turn_connector_rule
        result["geometrySideQuantum_pct"] = (
            100.0 * self.geometry_side_quantum_fraction)
        result["failureFraction"] = self.failure_fraction
        result["maxErosionStep_s"] = self.max_erosion_step_s
        result["geometryVolumeTolerance_rel"] = self.vol_tol
        result["radiationEscapeMethod"] = self.radiation_escape_method
        result["spectralSplit_um"] = self.spectral_split_um
        result["thermalAmbient_K"] = self.thermal_ambient_K
        result["scoreAmbientTarget_K"] = self.score_ambient_target_K
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

        # ---- Phase 2: geometry/lifecycle v2 erosion loop ----
        print("  Phase 2: geometry/lifecycle v2 erosion loop...")
        time_s = 0.0
        p03_int, prad_int = 0.0, 0.0
        p03s_int, prads_int = 0.0, 0.0
        macro = 0
        attempted_steps = 0
        failed = False
        cap_limited = False
        step_limited = False
        censored = False
        termination_reason = ""
        failure_feature = ""
        failure_index = ""
        max_loss = 0.0

        side0 = self._init_side
        block_sides = [side0] * n_blocks
        geometry_block_sides = [side0] * n_blocks
        stub_radii = [self.R0, self.R0]
        initial_features = block_sides + stub_radii
        failure_limits = [
            value * (1.0 - self.failure_fraction)
            for value in initial_features]
        resolution_delta = 0.01 * min(initial_features)

        initial_volume = self._erosion_state_volume(
            block_sides, self._block_lengths,
            stub_radii, self.STUB_LEN)
        initial_turn_area = sum(self._turn_cap_areas(block_sides))

        def stub_shoulder_area_sum(current_sides, current_stub_radii):
            return sum(
                max(0.0, math.pi * radius ** 2
                    - self._circle_square_overlap_area(radius, end_side))
                for radius, end_side in zip(
                    current_stub_radii,
                    (current_sides[0], current_sides[-1])))

        initial_stub_shoulder_area = stub_shoulder_area_sum(
            block_sides, stub_radii)
        max_turn_area = initial_turn_area
        max_stub_shoulder_area = initial_stub_shoulder_area
        max_geometry_projection_error = 0.0

        prev_p03 = r0_res["P03steady"]
        prev_prad = r0_res["PradSteady"]
        prev_p03s = r0_res["P03sphere"]
        prev_prads = r0_res["PradSphere"]
        prev_Tmax = r0_res["Tmax"]
        block_tavg = r0_res["block_Tavg"]
        block_A_lat = r0_res["block_A_lat"]
        stub_tavg = r0_res["stub_Tavg"]
        stub_A_lat = r0_res["stub_A_lat"]
        max_erosion_tmax = r0_res["Tmax"]
        erosion_retry_count = 0
        overtemp_fields = {}
        failure_text = ""

        def assign_failure(indices):
            nonlocal failure_feature, failure_index
            if not indices:
                return
            index = indices[0]
            if index < n_blocks:
                failure_feature = "block_side"
                failure_index = index + 1
            else:
                failure_feature = "stub_radius"
                failure_index = "in" if index == n_blocks else "out"

        def build_result(status):
            final_volume = self._erosion_state_volume(
                block_sides, self._block_lengths,
                stub_radii, self.STUB_LEN)
            final_turn_area = sum(self._turn_cap_areas(block_sides))
            final_stub_shoulder_area = stub_shoulder_area_sum(
                block_sides, stub_radii)
            side_spread = max(block_sides) - min(block_sides)
            geometry_side_spread = (
                max(geometry_block_sides) - min(geometry_block_sides))
            geometry_volume = self._erosion_state_volume(
                geometry_block_sides, self._block_lengths,
                stub_radii, self.STUB_LEN)
            return {
                "Vwork_V": Vwork,
                "initialTmax_K": r0_res["Tmax"],
                "Tmin_K": r0_res["Tmin"],
                "Tmean_K": r0_res["Tmean"],
                "U_pct": r0_res["U_pct"],
                "lifetimeH": time_s / 3600.0,
                **self._lifecycle_radiation_fields(
                    r0_res, time_s, p03_int, prad_int,
                    p03s_int, prads_int),
                "maxErosionTmax_K": max_erosion_tmax,
                "failureReached": failed,
                "capLimited": cap_limited,
                "stepLimited": step_limited,
                "censored": censored,
                "lifetimeExact": failed and status == "OK",
                "terminationReason": termination_reason,
                "failureFeature": failure_feature,
                "failureIndex": failure_index,
                "maxFeatureLoss_pct": 100.0 * max_loss,
                "erosionSteps": macro,
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
                "initialBlockMaskAreaRatioMin": min(
                    r0_res["blockMaskAreaRatio"]),
                "initialBlockMaskAreaRatioMax": max(
                    r0_res["blockMaskAreaRatio"]),
                "initialErosionStateVolume_m3": initial_volume,
                "finalErosionStateVolume_m3": final_volume,
                "erosionStateVolumeLoss_pct": 100.0 * (
                    1.0 - final_volume / initial_volume),
                "finalGeometryStateVolume_m3": geometry_volume,
                "maxGeometrySideProjectionError_pct": (
                    100.0 * max_geometry_projection_error / side0),
                "initialTurnCapArea_m2": initial_turn_area,
                "finalTurnCapArea_m2": final_turn_area,
                "maxTurnCapArea_m2": max_turn_area,
                "initialStubShoulderArea_m2": initial_stub_shoulder_area,
                "finalStubShoulderArea_m2": final_stub_shoulder_area,
                "maxStubShoulderArea_m2": max_stub_shoulder_area,
                "finalMinBlockSide_mm": min(block_sides) * 1.0e3,
                "finalMaxBlockSide_mm": max(block_sides) * 1.0e3,
                "finalBlockSideSpread_mm": side_spread * 1.0e3,
                "finalGeometryMinBlockSide_mm": (
                    min(geometry_block_sides) * 1.0e3),
                "finalGeometryMaxBlockSide_mm": (
                    max(geometry_block_sides) * 1.0e3),
                "finalGeometryBlockSideSpread_mm": (
                    geometry_side_spread * 1.0e3),
                "finalStubInRadius_mm": stub_radii[0] * 1.0e3,
                "finalStubOutRadius_mm": stub_radii[1] * 1.0e3,
                **overtemp_fields,
                "status": status,
                "failure": failure_text,
                "elapsed_sec": round(time.time() - t_start, 1),
            }

        while macro < self.max_erosion_steps and not failed:
            block_rates, _, _ = self._block_erosion_rates(
                block_sides, block_tavg, self._block_lengths,
                block_A_lat, geometry_block_sides)
            stub_rates, _, _ = self._stub_erosion_rates(
                stub_radii, stub_tavg,
                (geometry_block_sides[0], geometry_block_sides[-1]),
                (block_tavg[0], block_tavg[-1]),
                stub_A_lat)
            feature_rates = block_rates + stub_rates
            current_features = block_sides + stub_radii

            if max(feature_rates) < 1.0e-15:
                remaining = max(
                    0.0, self.max_lifetime_h * 3600.0 - time_s)
                p03_int += prev_p03 * remaining
                prad_int += prev_prad * remaining
                p03s_int += prev_p03s * remaining
                prads_int += prev_prads * remaining
                time_s += remaining
                termination_reason = "negligible_erosion"
                censored = True
                print("  Evaporation negligible; integrated constant power "
                      "to the lifecycle cap and censored the lifetime.")
                break

            dt = self._next_erosion_timestep(
                current_features, failure_limits, feature_rates,
                resolution_delta, time_s)
            if dt <= 0.0:
                cap_limited = True
                censored = True
                termination_reason = "lifetime_cap"
                break

            candidate_features, candidate_losses, failed_indices = (
                self._advance_erosion_features(
                    current_features, initial_features, failure_limits,
                    feature_rates, dt))
            candidate_sides = candidate_features[:n_blocks]
            candidate_stub_radii = candidate_features[n_blocks:]
            candidate_time_s = time_s + dt
            candidate_max_loss = max(candidate_losses)
            candidate_geometry_sides = (
                self._project_block_sides_to_geometry(candidate_sides))
            candidate_projection_error = max(
                abs(exact - represented)
                for exact, represented in zip(
                    candidate_sides, candidate_geometry_sides))
            new_blocks = self.eroded_blocks(candidate_geometry_sides)
            new_Renv = self.compute_envelope(
                new_blocks, candidate_stub_radii, candidate_geometry_sides)
            attempted_steps += 1

            print(f"  SOLVING step={attempted_steps} "
                  f"t={candidate_time_s / 3600.0:.3f}h "
                  f"loss={candidate_max_loss:.6f} "
                  f"sideSpread={(max(candidate_sides) - min(candidate_sides)) * 1e3:.6f}mm")
            r_now = None
            solve_failure = ""
            for attempt in range(self.max_erosion_solve_retries + 1):
                try:
                    if attempt == 0:
                        self._rebuild(
                            new_blocks, candidate_geometry_sides,
                            candidate_stub_radii, new_Renv,
                            geom_only=True)
                        r_now = self._solve_prepared(Vwork)
                    else:
                        erosion_retry_count += 1
                        print(f"  RETRY step {attempted_steps}: {attempt}/"
                              f"{self.max_erosion_solve_retries}")
                        r_now = self._restart_at_geometry(
                            N_RUNS, L_RUN_m, z_first_m,
                            new_blocks, candidate_geometry_sides,
                            candidate_stub_radii, new_Renv, Vwork)
                    if r_now.get("solve_ok", False):
                        break
                    solve_failure = r_now.get(
                        "failure", "unknown solve failure")
                except ServerDisconnectError:
                    raise
                except Exception as exc:
                    solve_failure = self._safe_exception_text(exc)
                    r_now = None
                if attempt < self.max_erosion_solve_retries:
                    print(f"  WARN step {attempted_steps} attempt "
                          f"{attempt + 1} failed: {solve_failure}")

            if r_now is None or not r_now.get("solve_ok", False):
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

            cur_p03 = r_now["P03steady"]
            cur_prad = r_now["PradSteady"]
            cur_p03s = r_now["P03sphere"]
            cur_prads = r_now["PradSphere"]

            if r_now["Tmax"] >= self.temp_limit_K:
                fraction = self._overtemperature_fraction(
                    prev_Tmax, r_now["Tmax"], self.temp_limit_K)
                valid_dt = fraction * dt

                def at_crossing(previous, current):
                    return previous + fraction * (current - previous)

                cross_p03 = at_crossing(prev_p03, cur_p03)
                cross_prad = at_crossing(prev_prad, cur_prad)
                cross_p03s = at_crossing(prev_p03s, cur_p03s)
                cross_prads = at_crossing(prev_prads, cur_prads)
                p03_int += 0.5 * (prev_p03 + cross_p03) * valid_dt
                prad_int += 0.5 * (prev_prad + cross_prad) * valid_dt
                p03s_int += 0.5 * (prev_p03s + cross_p03s) * valid_dt
                prads_int += 0.5 * (prev_prads + cross_prads) * valid_dt

                crossing_features, crossing_losses, crossing_failed = (
                    self._advance_erosion_features(
                        current_features, initial_features, failure_limits,
                        feature_rates, valid_dt))
                block_sides = crossing_features[:n_blocks]
                geometry_block_sides = (
                    self._project_block_sides_to_geometry(block_sides))
                stub_radii = crossing_features[n_blocks:]
                max_geometry_projection_error = max(
                    max_geometry_projection_error,
                    max(abs(exact - represented)
                        for exact, represented in zip(
                            block_sides, geometry_block_sides)))
                time_s += valid_dt
                macro += 1
                max_loss = max(crossing_losses)
                failed = bool(crossing_failed)
                assign_failure(crossing_failed)
                censored = not failed
                termination_reason = "overtemperature"
                overtemp_fields = {
                    "overtempStep": macro,
                    "overtempTimeH": time_s / 3600.0,
                    "overtempTmax_K": r_now["Tmax"],
                    "overtempInterpolationFraction": fraction,
                    "overtempBracketEndTimeH": candidate_time_s / 3600.0,
                }
                result = build_result("FAIL_OVERTEMP_DURING_EROSION")
                return self._annotate_voltage_result(
                    result, policy, objective, max_safe_v)

            # Commit only a solved endpoint, then integrate its exact interval.
            p03_int += 0.5 * (prev_p03 + cur_p03) * dt
            prad_int += 0.5 * (prev_prad + cur_prad) * dt
            p03s_int += 0.5 * (prev_p03s + cur_p03s) * dt
            prads_int += 0.5 * (prev_prads + cur_prads) * dt
            block_sides = list(candidate_sides)
            geometry_block_sides = list(candidate_geometry_sides)
            stub_radii = list(candidate_stub_radii)
            max_geometry_projection_error = max(
                max_geometry_projection_error, candidate_projection_error)
            time_s = candidate_time_s
            macro += 1
            max_loss = candidate_max_loss
            max_turn_area = max(
                max_turn_area, sum(self._turn_cap_areas(block_sides)))
            max_stub_shoulder_area = max(
                max_stub_shoulder_area,
                stub_shoulder_area_sum(block_sides, stub_radii))

            prev_p03, prev_prad = cur_p03, cur_prad
            prev_p03s, prev_prads = cur_p03s, cur_prads
            prev_Tmax = r_now["Tmax"]
            block_tavg = r_now["block_Tavg"]
            block_A_lat = r_now["block_A_lat"]
            stub_tavg = r_now["stub_Tavg"]
            stub_A_lat = r_now["stub_A_lat"]

            if failed_indices:
                failed = True
                termination_reason = "feature_loss_20pct"
                assign_failure(failed_indices)
            elif time_s >= (self.max_lifetime_h * 3600.0
                            - self.erosion_time_tol_s):
                cap_limited = True
                censored = True
                termination_reason = "lifetime_cap"

            if macro % 5 == 0 or failed or cap_limited:
                print(f"  STEP={macro} t={time_s / 3600:.2f}h "
                      f"loss={max_loss:.6f}")
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
            "maxErosionTmax_K", "lifetimeH", "initialP03gross_W",
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
        self.j = None
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
