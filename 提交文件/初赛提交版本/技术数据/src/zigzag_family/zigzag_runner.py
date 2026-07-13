import math
import time
import mph
from jpype.types import JInt

# Expected failures are separated from COMSOL disconnects so Optuna can resume safely.
class MeshError(RuntimeError):
    pass

class ServerDisconnectError(RuntimeError):
    pass

# COMSOL model runner for one zigzag-family candidate.
class COMSOLRunner:

    def __init__(self):
        self.client = None
        self.model = None
        self.j = None
        self.R0 = 0.0025
        self.L0 = 0.015
        self.V0 = math.pi * self.R0 ** 2 * self.L0
        self.STUB_LEN = 0.0005
        self.temp_limit_K = 3273.15
        self.rho_mass = 19350.0
        self.vol_tol = 0.03
        self.current_tol = 1e-09
        self.sphere_margin = 1.05
        self.voltage_upper = 100.0
        self.voltage_floor = 0.001
        self.voltage_tol = 0.05
        self.max_voltage_iters = 20
        self.Aev = 3900000000.0
        self.Bev = 102300.0
        self.failure_fraction = 0.2
        self.MAX_BLOCK_SLOTS = 64
        self.max_erosion_steps = 50
        self.max_lifetime_h = 200.0
        self.rhoe_expr = 'max(1e-10[ohm*m], 5.5e-8[ohm*m]*(1+0.003836*(T-293.15[K])/1[K]+7.55e-7*((T-293.15[K])/1[K])^2))'
        self.k_expr = 'max(75[W/(m*K)],175[W/(m*K)]-0.032[W/(m*K^2)]*(T-293.15[K]))'
        self.cp_expr = 'min(195[J/(kg*K)],132[J/(kg*K)]+0.020[J/(kg*K^2)]*(T-293.15[K]))'
        self.q03_expr = None
        self.qrad_expr = None
        self._build_expressions()
        self._blocks0 = None
        self._init_side = None
        self._n_blocks = None

    # Radiation band expressions used by the COMSOL surface integrals.
    def _build_expressions(self):
        x03T = '(c2bb/(lam03*T))'
        x03Tamb = '(c2bb/(lam03*Tamb))'
        parts_T, parts_Tamb = ([], [])
        for n in range(1, 7):
            n2, n3, n4 = (n * n, n ** 3, n ** 4)
            parts_T.append(f'exp(-{n}*{x03T})*({x03T}^3/{n}+3*{x03T}^2/{n2}+6*{x03T}/{n3}+6/{n4})')
            parts_Tamb.append(f'exp(-{n}*{x03Tamb})*({x03Tamb}^3/{n}+3*{x03Tamb}^2/{n2}+6*{x03Tamb}/{n3}+6/{n4})')
        series_T = '+'.join(parts_T)
        series_Tamb = '+'.join(parts_Tamb)
        f03T = f'min(1,max(0,(15/pi^4)*({series_T})))'
        f03Tamb = f'min(1,max(0,(15/pi^4)*({series_Tamb})))'
        self.q03_expr = f'eps03*sigmaSB*(({f03T})*T^4-({f03Tamb})*Tamb^4)'
        self.qrad_expr = f'sigmaSB*(epsRest*(T^4-Tamb^4)+(eps03-epsRest)*(({f03T})*T^4-({f03Tamb})*Tamb^4))'

    # Server lifecycle.
    def start(self):
        print('Starting COMSOL server...')
        self.client = mph.start()
        print('COMSOL server connected. Running warm-up...')
        try:
            dummy = self.client.create('warmup_dummy')
            dummy.java.component().create('comp1')
            self.client.remove(dummy)
            print('Warm-up done.')
        except Exception as e:
            print(f'Warm-up failed (non-fatal): {self._safe_exception_text(e)}')

    @staticmethod
    # Centerline construction for the planar zigzag path.
    def build_full_path(N_RUNS, L_RUN, z_first, z_last, stub_len, L0):
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
        return sum((abs(pts[i + 1][0] - pts[i][0]) + abs(pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)))

    @staticmethod
    def build_blocks(pts, side):
        blocks = []
        half = 0.5 * side
        for i in range(len(pts) - 1):
            p0, p1 = (pts[i], pts[i + 1])
            dx, dz = (p1[0] - p0[0], p1[1] - p0[1])
            if abs(dx) < 1e-12 and abs(dz) < 1e-12:
                continue
            ext_s = 0.0 if i == 0 else half
            ext_e = 0.0 if i == len(pts) - 2 else half
            tag = f'blk_{len(blocks) + 1}'
            if abs(dx) > 1e-12:
                d = 1.0 if dx > 0 else -1.0
                xa = p0[0] - d * ext_s
                xb = p1[0] + d * ext_e
                x_lo, x_hi = (min(xa, xb), max(xa, xb))
                blocks.append((tag, x_lo, -half, p0[1] - half, x_hi - x_lo, side, side))
            else:
                d = 1.0 if dz > 0 else -1.0
                za = p0[1] - d * ext_s
                zb = p1[1] + d * ext_e
                z_lo, z_hi = (min(za, zb), max(za, zb))
                blocks.append((tag, p0[0] - half, -half, z_lo, side, side, z_hi - z_lo))
        return blocks

    # Side length is derived from volume conservation.
    def compute_side_and_blocks(self, N_RUNS, L_RUN_m, z_first_m):
        z_last = self.L0 - z_first_m
        pts = self.build_full_path(N_RUNS, L_RUN_m, z_first_m, z_last, self.STUB_LEN, self.L0)
        plen = self.path_length(pts)
        fixed_vol = 2 * math.pi * self.R0 ** 2 * self.STUB_LEN
        flex_vol = self.V0 - fixed_vol
        side = math.sqrt(flex_vol / max(plen, 1e-300))
        blocks = self.build_blocks(pts, side)
        return (side, blocks, plen)

    # Outer sphere used by the power accounting.
    def compute_envelope(self, blocks):
        max_dist = 0.0
        for _, x0, y0, z0, sx, sy, sz in blocks:
            for x in (x0, x0 + sx):
                for y in (y0, y0 + sy):
                    for z in (z0, z0 + sz):
                        max_dist = max(max_dist, math.sqrt(x ** 2 + y ** 2 + (z - 0.5 * self.L0) ** 2))
        for rx in (-self.R0, self.R0):
            for ry in (-self.R0, self.R0):
                for zz in (0, self.STUB_LEN, self.L0 - self.STUB_LEN, self.L0):
                    max_dist = max(max_dist, math.sqrt(rx ** 2 + ry ** 2 + (zz - 0.5 * self.L0) ** 2))
        return self.sphere_margin * max_dist

    @staticmethod
    # Convert uniform side erosion into updated block geometry.
    def eroded_blocks(blocks0, init_side, geom_side):
        s0 = init_side
        s_new = geom_side
        shrink = (s0 - s_new) * 0.5
        new_blocks = []
        for idx, (_, x0, y0, z0, sx, sy, sz) in enumerate(blocks0):
            tag = f'blk_{idx + 1}'
            is_horiz = abs(sz - s0) < 1e-06 * s0
            if is_horiz:
                new_blocks.append((tag, x0, y0 + shrink, z0 + shrink, sx, s_new, s_new))
            else:
                new_blocks.append((tag, x0 + shrink, y0 + shrink, z0, s_new, s_new, sz))
        return new_blocks

    # Small wrappers keep repeated COMSOL cleanup calls local.
    def _remove_safe(self, container, tag):
        try:
            container.remove(tag)
        except Exception:
            pass

    @staticmethod
    def _safe_exception_text(exc):
        try:
            return str(exc)
        except Exception:
            return exc.__class__.__name__

    def _is_server_alive(self):
        if self.client is None:
            return False
        try:
            self.client.names()
            return True
        except Exception:
            return False

    def _clear_solutions(self, remove=False):
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

    # Create the COMSOL model skeleton and the first geometry.
    def _init_model(self, N_RUNS, L_RUN_m, z_first_m):
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass
        side, blocks, _ = self.compute_side_and_blocks(N_RUNS, L_RUN_m, z_first_m)
        self._blocks0 = blocks
        self._init_side = side
        self._n_blocks = len(blocks)
        R_env = self.compute_envelope(blocks)
        self.model = self.client.create('zigzag_opt')
        j = self.model.java
        self.j = j
        j.component().create('comp1')
        j.component('comp1').geom().create('geom1', 3)
        j.component('comp1').geom('geom1').lengthUnit('mm')
        j.component('comp1').physics().create('ec', 'ConductiveMedia', 'geom1')
        j.component('comp1').physics().create('ht', 'HeatTransfer', 'geom1')
        j.component('comp1').material().create('mat1', 'Common')
        j.component('comp1').material('mat1').label('Tungsten')
        j.component('comp1').material('mat1').selection().all()
        j.component('comp1').mesh().create('mesh1', 'geom1')
        j.study().create('std1')
        j.study('std1').create('stat', 'Stationary')
        j.param().set('sigmaSB', '5.670374419e-8[W/(m^2*K^4)]', 'Stefan-Boltzmann constant')
        j.param().set('eps03', '0.35', 'Emissivity 0-3um')
        j.param().set('epsRest', '0.15', 'Emissivity >3um')
        j.param().set('rhoMassW', '19350[kg/m^3]', 'Tungsten density')
        j.param().set('Tamb', '293.15[K]', 'Ambient temperature')
        j.param().set('Vapp', '1[V]', 'Applied DC voltage')
        j.param().set('lam03', '3[um]', 'Upper wavelength bound')
        j.param().set('c2bb', '1.438776877e-2[m*K]', 'Second radiation constant')
        j.param().set('r0', f'{self.R0}[m]', 'Reference radius')
        j.param().set('L0', '15[mm]', 'Reference length')
        j.param().set('Aev', '3.9e9[kg/(m^2*s)]', 'Evaporation prefactor (SI)')
        j.param().set('Bev', '1.023e5[K]', 'Evaporation temperature coefficient')
        j.param().set('RenvZZ', f'{R_env}[m]', 'Enclosing sphere radius')
        j.param().set('AenvZZ', f'{4 * math.pi * R_env ** 2}[m^2]', 'Enclosing sphere area')
        self._rebuild(blocks, side, R_env)
        j.multiphysics().create('emh1', 'ElectromagneticHeatSource', 'geom1', 3)
        j.multiphysics('emh1').selection().all()
        j.multiphysics('emh1').set('EMHeat_physics', 'ec')
        j.multiphysics('emh1').set('Heat_physics', 'ht')
        try:
            j.component('comp1').physics('ht').feature('init1').set('Tinit', '1500[K]')
        except Exception:
            pass
        self._clear_solutions(remove=True)
        last_phase0_error = None
        for warmup_voltage in (1.0, 10.0, 100.0):
            j.param().set('Vapp', f'{warmup_voltage}[V]')
            self._clear_solutions(remove=True)
            try:
                j.study('std1').run()
                if warmup_voltage != 1.0:
                    print(f'  NOTE: Phase-0 warm-up succeeded at {warmup_voltage:g}V')
                break
            except Exception as exc:
                last_phase0_error = exc
        else:
            raise RuntimeError('Phase-0 warm-up failed at 1V/10V/100V: ' + self._safe_exception_text(last_phase0_error))
        j.result().numerical().create('maxTZZ', 'MaxVolume')
        j.result().numerical('maxTZZ').selection().all()
        j.result().numerical('maxTZZ').set('expr', ['T'])
        j.result().numerical().create('minTZZ', 'MinVolume')
        j.result().numerical('minTZZ').selection().all()
        j.result().numerical('minTZZ').set('expr', ['T'])
        j.result().numerical().create('volZZ', 'IntVolume')
        j.result().numerical('volZZ').selection().all()
        j.result().numerical('volZZ').set('expr', ['1'])
        j.result().numerical().create('IinZZ', 'IntSurface')
        j.result().numerical('IinZZ').selection().named('selInZZ')
        j.result().numerical('IinZZ').set('expr', ['ec.Jx*nx+ec.Jy*ny+ec.Jz*nz'])
        j.result().numerical().create('AsurfZZ', 'IntSurface')
        j.result().numerical('AsurfZZ').selection().all()
        j.result().numerical('AsurfZZ').set('expr', ['1'])
        j.result().numerical().create('P03emitZZ', 'IntSurface')
        j.result().numerical('P03emitZZ').selection().all()
        j.result().numerical('P03emitZZ').set('expr', [self.q03_expr])
        j.result().numerical().create('PradEmitZZ', 'IntSurface')
        j.result().numerical('PradEmitZZ').selection().all()
        j.result().numerical('PradEmitZZ').set('expr', [self.qrad_expr])
        san_T = float(j.result().numerical('maxTZZ').getReal()[0][0])
        san_I = abs(float(j.result().numerical('IinZZ').getReal()[0][0]))
        if san_T < 200.0:
            raise RuntimeError(f'S2S coupling failure: Tmax={san_T:.1f}K')
        print(f'  Model init OK: sanity Tmax={san_T:.1f}K I={san_I:.4f}A')

    # Rebuild current block geometry after erosion.
    def _rebuild(self, blocks, side, R_env, geom_only=False):
        j = self.j
        geom = j.component('comp1').geom('geom1')
        self._remove_safe(geom.feature(), 'uniZZ')
        self._remove_safe(geom.feature(), 'term_in')
        self._remove_safe(geom.feature(), 'term_out')
        for i in range(self.MAX_BLOCK_SLOTS):
            self._remove_safe(geom.feature(), f'blk_{i + 1}')
        tags = []
        for tag, x0, y0, z0, sx, sy, sz in blocks:
            tags.append(tag)
            geom.create(tag, 'Block')
            geom.feature(tag).set('size', [f'{sx}[m]', f'{sy}[m]', f'{sz}[m]'])
            geom.feature(tag).set('pos', [f'{x0}[m]', f'{y0}[m]', f'{z0}[m]'])
        geom.create('term_in', 'Cylinder')
        geom.feature('term_in').set('r', f'{self.R0}[m]')
        geom.feature('term_in').set('h', f'{self.STUB_LEN}[m]')
        geom.feature('term_in').set('pos', ['0[m]', '0[m]', '0[m]'])
        geom.create('term_out', 'Cylinder')
        geom.feature('term_out').set('r', f'{self.R0}[m]')
        geom.feature('term_out').set('h', f'{self.STUB_LEN}[m]')
        geom.feature('term_out').set('pos', ['0[m]', '0[m]', f"{self.L0 - self.STUB_LEN}[m]"])
        geom.create('uniZZ', 'Union')
        geom.feature('uniZZ').selection('input').set(tags + ['term_in', 'term_out'])
        geom.feature('uniZZ').set('intbnd', False)
        geom.run()
        if not geom_only:
            self._remove_safe(j.component('comp1').selection(), 'selInZZ')
            self._remove_safe(j.component('comp1').selection(), 'selOutZZ')
            j.component('comp1').selection().create('selInZZ', 'Box')
            j.component('comp1').selection('selInZZ').geom('geom1', 2)
            j.component('comp1').selection('selInZZ').set('condition', 'inside')
            j.component('comp1').selection('selInZZ').set('xmin', -10.0)
            j.component('comp1').selection('selInZZ').set('xmax', 10.0)
            j.component('comp1').selection('selInZZ').set('ymin', -10.0)
            j.component('comp1').selection('selInZZ').set('ymax', 10.0)
            j.component('comp1').selection('selInZZ').set('zmin', -1e-06)
            j.component('comp1').selection('selInZZ').set('zmax', 1e-06)
            j.component('comp1').selection().create('selOutZZ', 'Box')
            j.component('comp1').selection('selOutZZ').geom('geom1', 2)
            j.component('comp1').selection('selOutZZ').set('condition', 'inside')
            j.component('comp1').selection('selOutZZ').set('xmin', -10.0)
            j.component('comp1').selection('selOutZZ').set('xmax', 10.0)
            j.component('comp1').selection('selOutZZ').set('ymin', -10.0)
            j.component('comp1').selection('selOutZZ').set('ymax', 10.0)
            j.component('comp1').selection('selOutZZ').set('zmin', 14.999999)
            j.component('comp1').selection('selOutZZ').set('zmax', 15.000001)
            ec = j.component('comp1').physics('ec')
            self._remove_safe(ec.feature(), 'potZZ')
            self._remove_safe(ec.feature(), 'gndZZ')
            ec.create('potZZ', 'ElectricPotential', 2)
            ec.feature('potZZ').selection().named('selInZZ')
            ec.feature('potZZ').set('V0', 'Vapp')
            ec.create('gndZZ', 'Ground', 2)
            ec.feature('gndZZ').selection().named('selOutZZ')
            self._setup_s2s()
            mp = j.component('comp1').material('mat1').propertyGroup('def')
            mp.set('density', ['rhoMassW'])
            mp.set('electricconductivity', [f'1/({self.rhoe_expr})'])
            mp.set('thermalconductivity', [self.k_expr])
            mp.set('heatcapacity', [self.cp_expr])
        try:
            j.component('comp1').mesh('mesh1').feature('ftet1')
        except Exception:
            j.component('comp1').mesh('mesh1').create('ftet1', 'FreeTet')
        for hauto in [5, 6, 7, 8, 9]:
            try:
                j.component('comp1').mesh('mesh1').feature('size').set('hauto', JInt(hauto))
                j.component('comp1').mesh('mesh1').run()
                if hauto > 5:
                    print(f'  NOTE: mesh OK with hauto={hauto} (fallback)')
                break
            except Exception as mesh_err:
                if hauto < 9:
                    print(f'  WARN: mesh hauto={hauto} failed, retrying hauto={hauto + 1}...')
                else:
                    raise MeshError(f'Mesh failed at all levels (hauto=5..9): {mesh_err}') from mesh_err
        j.param().set('RenvZZ', f'{R_env}[m]')
        j.param().set('AenvZZ', f'{4 * math.pi * R_env ** 2}[m^2]')

    # Multiple-spectral-band S2S radiation settings.
    def _setup_s2s(self):
        j = self.j
        self._remove_safe(j.component('comp1').physics(), 'rad')
        self._remove_safe(j.multiphysics(), 'htradZZ')
        j.component('comp1').physics().create('rad', 'SurfaceToSurfaceRadiation', 'geom1')
        j.component('comp1').physics('rad').prop('RadiationSettings').set('wavelengthDependenceOfSurfaceProperties', 'MultipleSpectralBands')
        j.component('comp1').physics('rad').prop('RadiationSettings').set('lambda_r', 'lam03')
        j.component('comp1').physics('rad').create('dsZZ', 'DiffuseSurface', 2)
        ds = j.component('comp1').physics('rad').feature('dsZZ')
        ds.selection().all()
        ds.set('defineSurfaceEmissivityOnEachSide', '0')
        ds.set('epsilon_radMulti_mat', 'userdef')
        ds.set('epsilon_radMulti', 'if(comp1.rad.lambda<lam03,eps03,epsRest)')
        ds.set('spectralBandNameAmbientEmissivityMulti', [['[0, 3['], ['[3, +inf[']])
        ds.set('Tamb', 'Tamb')
        ds.set('Tambu', 'Tamb')
        ds.set('Tambd', 'Tamb')
        ds.set('ambientEmissivity', 'userdef')
        ds.set('epsilon_amb', '1')
        ds.set('epsilon_ambu', '1')
        ds.set('epsilon_ambd', '1')
        j.multiphysics().create('htradZZ', 'HeatTransferWithSurfaceToSurfaceRadiation', 'geom1', 2)
        j.multiphysics('htradZZ').selection().all()

    # Solve current geometry at one voltage and collect scalar outputs.
    def _solve_prepared(self, voltage):
        j = self.j
        j.param().set('Vapp', f'{voltage}[V]')
        result = {'solve_ok': False, 'applied_V': voltage, 'Tmax': float('nan'), 'Tmin': float('nan'), 'I': float('nan'), 'P03steady': float('nan'), 'PradSteady': float('nan'), 'P03sphere': float('nan'), 'PradSphere': float('nan'), 'vol_err': float('nan'), 'temp_ok': False, 'volume_ok': False, 'current_ok': False, 'block_Tavg': [0.0] * self._n_blocks}
        try:
            self._clear_solutions(remove=False)
            try:
                j.result().numerical('IinZZ').selection().named('selInZZ')
            except Exception:
                pass
            j.study('std1').run()
            Tmax = float(j.result().numerical('maxTZZ').getReal()[0][0])
            try:
                Tmin = float(j.result().numerical('minTZZ').getReal()[0][0])
            except Exception:
                Tmin = Tmax * 0.95
            V = float(j.result().numerical('volZZ').getReal()[0][0])
            I = abs(float(j.result().numerical('IinZZ').getReal()[0][0]))
            P03 = float(j.result().numerical('P03emitZZ').getReal()[0][0])
            Prad = float(j.result().numerical('PradEmitZZ').getReal()[0][0])
            PradSphere = voltage * I
            P03sphere = PradSphere * P03 / Prad if Prad > 1e-10 else 0.0
            block_Tavg = []
            for _, x0, y0, z0, sx, sy, sz in self._blocks0:
                zc = z0 + 0.5 * sz
                eta = zc / self.L0
                block_Tavg.append(Tmin + (Tmax - Tmin) * 4.0 * eta * (1.0 - eta))
            vol_err = abs(V - self.V0) / self.V0
            result.update({'solve_ok': True, 'Tmax': Tmax, 'Tmin': Tmin, 'I': I, 'R': voltage / I if I > self.current_tol else float('nan'), 'Pelec': voltage * I, 'P03steady': P03, 'PradSteady': Prad, 'P03sphere': P03sphere, 'PradSphere': PradSphere, 'vol_err': vol_err, 'temp_ok': Tmax < self.temp_limit_K, 'volume_ok': vol_err <= self.vol_tol, 'current_ok': I > self.current_tol, 'block_Tavg': block_Tavg})
        except Exception as e:
            self._clear_solutions(remove=False)
            if not self._is_server_alive():
                raise ServerDisconnectError(self._safe_exception_text(e))
            result['failure'] = self._safe_exception_text(e)
            print(f"  WARN solve failed: {result['failure']}")
        return result

    # Voltage search keeps each candidate below the temperature limit.
    def _meets_constraint(self, r):
        return r['solve_ok'] and r['current_ok'] and r['volume_ok'] and r['temp_ok']

    def _search_best_voltage(self):
        steps = 0
        high_res = self._solve_prepared(self.voltage_upper)
        high_V = self.voltage_upper
        steps += 1
        if self._meets_constraint(high_res):
            high_res['search_ok'] = True
            high_res['search_steps'] = steps
            return high_res
        if high_res['solve_ok'] and (not high_res['current_ok'] or not high_res['volume_ok']):
            high_res['search_ok'] = False
            high_res['search_steps'] = steps
            return high_res
        low_V, low_res = (None, None)
        if high_res['solve_ok'] and high_res['current_ok'] and high_res['volume_ok'] and (not math.isnan(high_res['Tmax'])) and (high_res['Tmax'] > 0):
            guess_V = self.voltage_upper * math.sqrt(self.temp_limit_K / max(high_res['Tmax'], 1e-300))
            guess_V = max(self.voltage_floor, min(0.98 * self.voltage_upper, guess_V))
            if guess_V < high_V - 1e-12:
                guess_res = self._solve_prepared(guess_V)
                steps += 1
                if self._meets_constraint(guess_res):
                    low_V, low_res = (guess_V, guess_res)
                else:
                    high_V, high_res = (guess_V, guess_res)
        while low_res is None and high_V > self.voltage_floor + 1e-12:
            next_V = max(self.voltage_floor, 0.5 * high_V)
            if abs(next_V - high_V) <= 1e-12:
                break
            next_res = self._solve_prepared(next_V)
            steps += 1
            if self._meets_constraint(next_res):
                low_V, low_res = (next_V, next_res)
            else:
                high_V, high_res = (next_V, next_res)
        if low_res is None:
            high_res['search_ok'] = False
            high_res['search_steps'] = steps
            return high_res
        for _ in range(self.max_voltage_iters):
            if high_V - low_V <= self.voltage_tol:
                break
            mid_V = 0.5 * (low_V + high_V)
            mid_res = self._solve_prepared(mid_V)
            steps += 1
            if self._meets_constraint(mid_res):
                low_V, low_res = (mid_V, mid_res)
            else:
                high_V, high_res = (mid_V, mid_res)
        low_res['search_ok'] = True
        low_res['search_steps'] = steps
        return low_res

    # Full lifecycle evaluation: voltage search, erosion, and time integration.
    def evaluate(self, N_RUNS, L_RUN_m, z_first_m):
        t_start = time.time()
        side, blocks, plen = self.compute_side_and_blocks(N_RUNS, L_RUN_m, z_first_m)
        if side < 0.0001:
            return {'status': 'FAIL_SIDE_TOO_SMALL', 'elapsed_sec': round(time.time() - t_start, 1)}
        if side > 0.0012:
            return {'status': 'FAIL_SIDE_TOO_LARGE', 'elapsed_sec': round(time.time() - t_start, 1)}
        z_last = self.L0 - z_first_m
        z_step = (z_last - z_first_m) / max(N_RUNS - 1, 1)
        if z_step < side * 1.2:
            return {'status': 'FAIL_Z_OVERLAP', 'elapsed_sec': round(time.time() - t_start, 1)}
        n_blocks = len(blocks)
        resolve_thr = 0.02 * side
        print(f'  Building model: N={N_RUNS} L={L_RUN_m * 1000.0:.1f}mm zf={z_first_m * 1000.0:.2f}mm side={side * 1000.0:.4f}mm nblk={n_blocks}')
        try:
            self._init_model(N_RUNS, L_RUN_m, z_first_m)
        except MeshError as e:
            return {'status': 'FAIL_MESH: ' + self._safe_exception_text(e), 'elapsed_sec': round(time.time() - t_start, 1)}
        except ServerDisconnectError:
            raise
        except Exception as e:
            if not self._is_server_alive():
                raise ServerDisconnectError(self._safe_exception_text(e))
            return {'status': 'FAIL_INIT: ' + self._safe_exception_text(e), 'elapsed_sec': round(time.time() - t_start, 1)}
        print('  Phase 1: voltage search...')
        r0_res = self._search_best_voltage()
        Vwork = r0_res['applied_V']
        print(f"  PHASE1: Vwork={Vwork:.4f}V Tmax={r0_res['Tmax']:.1f}K P03sph={r0_res['P03sphere']:.1f}W steps={r0_res['search_steps']}")
        if not r0_res.get('search_ok', False):
            return {'status': 'FAIL_VOLTAGE_SEARCH', 'elapsed_sec': round(time.time() - t_start, 1)}
        print('  Phase 2: erosion loop...')
        time_s = 0.0
        p03_int, prad_int = (0.0, 0.0)
        p03s_int, prads_int = (0.0, 0.0)
        macro = 0
        failed = False
        side0 = self._init_side
        block_sides = [side0] * n_blocks
        side_min = side0 * (1.0 - self.failure_fraction)
        prev_p03 = r0_res['P03steady']
        prev_prad = r0_res['PradSteady']
        prev_p03s = r0_res['P03sphere']
        prev_prads = r0_res['PradSphere']
        block_tavg = r0_res['block_Tavg']
        while macro < self.max_erosion_steps and (not failed):
            macro += 1
            dsdt = [2.0 * self.Aev * math.exp(-self.Bev / block_tavg[i]) / self.rho_mass for i in range(n_blocks)]
            max_dsdt = max(dsdt)
            if max_dsdt < 1e-15:
                print('  Evaporation negligible. Infinite lifetime.')
                break
            dt = float('inf')
            for i in range(n_blocks):
                if dsdt[i] > 1e-20:
                    dt = min(dt, resolve_thr / dsdt[i])
                    t_fail = (block_sides[i] - side_min) / dsdt[i]
                    if t_fail > 0:
                        dt = min(dt, t_fail)
            dt = max(1.0, min(36000.0, dt))
            max_loss = 0.0
            for i in range(n_blocks):
                block_sides[i] = max(1e-06, block_sides[i] - dsdt[i] * dt)
                loss = (side0 - block_sides[i]) / side0
                max_loss = max(max_loss, loss)
            time_s += dt
            if max_loss >= self.failure_fraction:
                failed = True
            if not failed and time_s / 3600.0 >= self.max_lifetime_h:
                print(f'  Lifetime cap {self.max_lifetime_h:.0f}h at step {macro}, stopping.')
                break
            geom_side = sum(block_sides) / n_blocks
            new_blocks = self.eroded_blocks(self._blocks0, side0, geom_side)
            new_Renv = self.compute_envelope(new_blocks)
            try:
                self._rebuild(new_blocks, side0, new_Renv, geom_only=True)
                r_now = self._solve_prepared(Vwork)
            except ServerDisconnectError:
                raise
            except Exception as e:
                if not self._is_server_alive():
                    raise ServerDisconnectError(self._safe_exception_text(e))
                print(f'  WARN: rebuild failed step {macro}: {self._safe_exception_text(e)}')
                r_now = {'solve_ok': False, 'P03steady': prev_p03, 'PradSteady': prev_prad, 'P03sphere': prev_p03s, 'PradSphere': prev_prads, 'Tmax': 0.0, 'Tmin': 0.0}
                failed = True
            if not r_now['solve_ok'] and (not failed):
                print(f'  WARN: solve failed step {macro}')
                failed = True
            cur_p03 = r_now['P03steady'] if r_now['solve_ok'] else prev_p03
            cur_prad = r_now['PradSteady'] if r_now['solve_ok'] else prev_prad
            cur_p03s = r_now['P03sphere'] if r_now['solve_ok'] else prev_p03s
            cur_prads = r_now['PradSphere'] if r_now['solve_ok'] else prev_prads
            p03_int += 0.5 * (prev_p03 + cur_p03) * dt
            prad_int += 0.5 * (prev_prad + cur_prad) * dt
            p03s_int += 0.5 * (prev_p03s + cur_p03s) * dt
            prads_int += 0.5 * (prev_prads + cur_prads) * dt
            prev_p03, prev_prad = (cur_p03, cur_prad)
            prev_p03s, prev_prads = (cur_p03s, cur_prads)
            if r_now['solve_ok'] and 'block_Tavg' in r_now:
                block_tavg = r_now['block_Tavg']
            if macro % 5 == 0 or failed:
                print(f'  STEP={macro} t={time_s / 3600:.2f}h loss={max_loss:.4f}')
        lifetime_h = time_s / 3600.0
        avg_p03s = p03s_int / time_s if time_s > 0 else float('nan')
        avg_prads = prads_int / time_s if time_s > 0 else float('nan')
        sv_loss = (1.0 - p03s_int / p03_int) * 100.0 if time_s > 0 and p03_int > 0 else float('nan')
        elapsed = time.time() - t_start
        return {'Vwork_V': Vwork, 'initialTmax_K': r0_res['Tmax'], 'lifetimeH': lifetime_h, 'initialP03sphere_W': r0_res['P03sphere'], 'initialPradSphere_W': r0_res['PradSphere'], 'lifeAvgP03sphere_W': avg_p03s, 'lifeAvgPradSphere_W': avg_prads, 'selfViewLoss_pct': sv_loss, 'failureReached': failed, 'erosionSteps': macro, 'status': 'OK', 'elapsed_sec': round(elapsed, 1)}

    # Release COMSOL resources.
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
                pass
            self.client = None
        print('COMSOL disconnected.')
if __name__ == '__main__':
    runner = COMSOLRunner()
    runner.start()
    try:
        print('Evaluating zigzag baseline: N_RUNS=8, L_RUN=104mm, z_first=0.8mm')
        result = runner.evaluate(N_RUNS=8, L_RUN_m=0.104, z_first_m=0.0008)
        print('\n' + '=' * 60)
        for k, v in result.items():
            print(f'  {k}: {v}')
        print('=' * 60)
    finally:
        runner.stop()
