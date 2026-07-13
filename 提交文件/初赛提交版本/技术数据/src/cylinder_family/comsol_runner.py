import math
import time
import mph
from jpype.types import JInt

# COMSOL model runner for one cylinder-family candidate.
class COMSOLRunner:

    def __init__(self):
        self.client = None
        self.model = None
        self.j = None
        self.seg_count = 8
        self.L0 = 0.015
        self.Lseg = self.L0 / self.seg_count
        self.temp_limit_K = 3000.0 + 273.15
        self.rho_mass = 19350.0
        self.vol_tol = 0.0001
        self.current_tol = 1e-09
        self.outer_sphere_margin = 1.05
        self.voltage_upper = 100.0
        self.voltage_floor = 0.001
        self.voltage_tol = 0.05
        self.max_voltage_iters = 16
        self.Aev = 3900000000.0
        self.Bev = 102300.0
        self.failure_fraction = 0.2
        self.mesh_hauto_levels = (4, 5, 6, 7, 8)
        self.rhoe_expr = 'max(1e-10[ohm*m], 5.5e-8[ohm*m]*(1+0.003836*(T-293.15[K])/1[K]+7.55e-7*((T-293.15[K])/1[K])^2))'
        self.k_expr = 'max(75[W/(m*K)],175[W/(m*K)]-0.032[W/(m*K^2)]*(T-293.15[K]))'
        self.cp_expr = 'min(195[J/(kg*K)],132[J/(kg*K)]+0.020[J/(kg*K^2)]*(T-293.15[K]))'
        self.q03_net_out_expr = None
        self.q_rad_net_out_expr = None
        self._build_expressions()
        self._r0 = None
        self._initial_radii = None
        self._fail_radii = None

    # Radiation band expressions used by the COMSOL surface integrals.
    def _build_expressions(self):
        x03T = '(c2bb/(lam03*T))'
        x03Tamb = '(c2bb/(lam03*Tamb))'
        series_T_parts = []
        series_Tamb_parts = []
        for n in range(1, 7):
            n2, n3, n4 = (n * n, n * n * n, n * n * n * n)
            term_T = f'exp(-{n}*{x03T})*({x03T}^3/{n}+3*{x03T}^2/{n2}+6*{x03T}/{n3}+6/{n4})'
            term_Tamb = f'exp(-{n}*{x03Tamb})*({x03Tamb}^3/{n}+3*{x03Tamb}^2/{n2}+6*{x03Tamb}/{n3}+6/{n4})'
            series_T_parts.append(term_T)
            series_Tamb_parts.append(term_Tamb)
        series_T = '+'.join(series_T_parts)
        series_Tamb = '+'.join(series_Tamb_parts)
        f03bb_T = f'min(1,max(0,(15/pi^4)*({series_T})))'
        f03bb_Tamb = f'min(1,max(0,(15/pi^4)*({series_Tamb})))'
        self.q03_net_out_expr = f'eps03*sigmaSB*(({f03bb_T})*T^4-({f03bb_Tamb})*Tamb^4)'
        self.q_rad_net_out_expr = f'sigmaSB*(epsRest*(T^4-Tamb^4)+(eps03-epsRest)*(({f03bb_T})*T^4-({f03bb_Tamb})*Tamb^4))'

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
            print(f'Warm-up failed (non-fatal): {e}')

    # Geometry envelope used by the outer-sphere power accounting.
    def _compute_geom(self, radii_m):
        r_max = max(radii_m)
        R_env = self.outer_sphere_margin * math.sqrt((0.5 * self.L0) ** 2 + r_max ** 2)
        A_env = 4.0 * math.pi * R_env ** 2
        return (r_max, R_env, A_env)

    # Create the COMSOL model skeleton and the first geometry.
    def _init_model(self, radii_m):
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass
        self.model = self.client.create('tungsten_opt')
        j = self.model.java
        self.j = j
        r0 = self._r0
        _, R_env, A_env = self._compute_geom(radii_m)
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
        j.param().set('eps03', '0.35', 'Emissivity 0-3 um band')
        j.param().set('epsRest', '0.15', 'Emissivity outside 0-3 um band')
        j.param().set('rhoMassW', '19350[kg/m^3]', 'Density of tungsten')
        j.param().set('Tamb', '293.15[K]', 'Ambient temperature')
        j.param().set('Vapp', f'{self.voltage_upper}[V]', 'Applied DC voltage')
        j.param().set('lam03', '3[um]', 'Upper wavelength bound')
        j.param().set('c2bb', '1.438776877e-2[m*K]', 'Second radiation constant')
        j.param().set('r0', f'{r0}[m]', 'Reference radius (max of input)')
        j.param().set('L0', '15[mm]', 'Reference length')
        j.param().set('Nseg', str(self.seg_count), 'Segment count')
        j.param().set('Lseg', f'{self.Lseg}[m]', 'Axial segment length')
        j.param().set('RenvInit', f'{R_env}[m]', 'Enclosing sphere radius')
        j.param().set('AenvInit', f'{A_env}[m^2]', 'Enclosing sphere area')
        for i in range(self.seg_count):
            j.param().set(f'r_seg{i + 1}', f'{r0}[m]', f'Segment {i + 1} radius')
        self._set_params(radii_m, R_env, A_env, self.voltage_upper)
        self._rebuild(radii_m)
        j.multiphysics().create('emh1', 'ElectromagneticHeatSource', 'geom1', 3)
        j.multiphysics('emh1').selection().all()
        j.multiphysics('emh1').set('EMHeat_physics', 'ec')
        j.multiphysics('emh1').set('Heat_physics', 'ht')
        try:
            j.component('comp1').physics('ht').feature('init1').set('Tinit', '3000[K]')
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
        self._set_params(radii_m, R_env, A_env, 1.0)
        j.study('std1').run()
        j.result().numerical().create('maxTS2S', 'MaxVolume')
        j.result().numerical('maxTS2S').selection().all()
        j.result().numerical('maxTS2S').set('expr', ['T'])
        j.result().numerical().create('minTS2S', 'MinVolume')
        j.result().numerical('minTS2S').selection().all()
        j.result().numerical('minTS2S').set('expr', ['T'])
        j.result().numerical().create('volS2S', 'IntVolume')
        j.result().numerical('volS2S').selection().all()
        j.result().numerical('volS2S').set('expr', ['1'])
        j.result().numerical().create('IinS2S', 'IntSurface')
        j.result().numerical('IinS2S').selection().named('selInS2S')
        j.result().numerical('IinS2S').set('expr', ['ec.Jx*nx+ec.Jy*ny+ec.Jz*nz'])
        j.result().numerical().create('AsurfS2S', 'IntSurface')
        j.result().numerical('AsurfS2S').selection().all()
        j.result().numerical('AsurfS2S').set('expr', ['1'])
        j.result().numerical().create('P03emitS2S', 'IntSurface')
        j.result().numerical('P03emitS2S').selection().all()
        j.result().numerical('P03emitS2S').set('expr', [self.q03_net_out_expr])
        j.result().numerical().create('PradEmitS2S', 'IntSurface')
        j.result().numerical('PradEmitS2S').selection().all()
        j.result().numerical('PradEmitS2S').set('expr', [self.q_rad_net_out_expr])
        for i in range(self.seg_count):
            int_t_tag = f'TintSeg_{i + 1}'
            int_a_tag = f'AsegS2S_{i + 1}'
            sel_tag = f'selSegLat_{i + 1}'
            try:
                j.result().numerical().remove(int_t_tag)
            except Exception:
                pass
            j.result().numerical().create(int_t_tag, 'IntSurface')
            j.result().numerical(int_t_tag).selection().named(sel_tag)
            j.result().numerical(int_t_tag).set('expr', ['T'])
            try:
                j.result().numerical().remove(int_a_tag)
            except Exception:
                pass
            j.result().numerical().create(int_a_tag, 'IntSurface')
            j.result().numerical(int_a_tag).selection().named(sel_tag)
            j.result().numerical(int_a_tag).set('expr', ['1'])
        sanity_Tmax = float(j.result().numerical('maxTS2S').getReal()[0][0])
        sanity_I = abs(float(j.result().numerical('IinS2S').getReal()[0][0]))
        sanity_R = 1.0 / sanity_I if sanity_I > 1e-20 else float('nan')
        if not math.isnan(sanity_R) and sanity_R > 1.0:
            raise RuntimeError(f'Sanity failed: R={sanity_R:.4f} ohm')
        if sanity_Tmax < 200.0:
            raise RuntimeError(f'S2S coupling failure: Tmax={sanity_Tmax:.1f}K')
        print(f'  Model init OK: sanity Tmax={sanity_Tmax:.1f}K, I={sanity_I:.4f}A')

    # Update scalar parameters before each solve.
    def _set_params(self, radii_m, R_env, A_env, voltage):
        j = self.j
        for i in range(self.seg_count):
            j.param().set(f'r_seg{i + 1}', f'{radii_m[i]}[m]', f'Segment {i + 1} radius')
        j.param().set('Vapp', f'{voltage}[V]', 'Applied DC voltage')
        j.param().set('RenvInit', f'{R_env}[m]', 'Enclosing sphere radius')
        j.param().set('AenvInit', f'{A_env}[m^2]', 'Enclosing sphere area')

    # Small wrappers keep repeated COMSOL cleanup calls local.
    def _remove_safe(self, container, tag):
        try:
            container.remove(tag)
        except Exception:
            pass

    def _ensure_server_ready(self):
        if self.client is None:
            raise RuntimeError('COMSOL server is not initialized.')
        if self.model is None or self.j is None:
            return
        try:
            self.j.label()
        except Exception as exc:
            raise RuntimeError(f'COMSOL server/model heartbeat failed: {exc}') from exc

    def _clear_mesh_safe(self):
        try:
            self.j.component('comp1').mesh('mesh1').clearMesh()
        except Exception:
            pass

    def _run_mesh_with_fallback(self, context='mesh'):
        j = self.j
        try:
            j.component('comp1').mesh('mesh1').feature('ftet1')
        except Exception:
            j.component('comp1').mesh('mesh1').create('ftet1', 'FreeTet')
        last_err = None
        for hauto in self.mesh_hauto_levels:
            try:
                self._clear_mesh_safe()
                j.component('comp1').mesh('mesh1').feature('size').set('hauto', JInt(hauto))
                j.component('comp1').mesh('mesh1').run()
                if hauto != self.mesh_hauto_levels[0]:
                    print(f'  NOTE: {context} mesh OK with hauto={hauto} (fallback)')
                return hauto
            except Exception as mesh_err:
                last_err = mesh_err
                next_levels = [h for h in self.mesh_hauto_levels if h > hauto]
                if next_levels:
                    print(f'  WARN: {context} mesh hauto={hauto} failed, retrying hauto={next_levels[0]}...')
        levels = ','.join((str(h) for h in self.mesh_hauto_levels))
        raise RuntimeError(f'{context} mesh failed at all levels (hauto={levels}): {last_err}') from last_err

    @staticmethod
    def _finite_number(value):
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    # Rebuild current segmented geometry after radius changes.
    def _rebuild(self, radii_m):
        j = self.j
        N = self.seg_count
        self._remove_safe(j.component('comp1').geom('geom1').feature(), 'uniS2S')
        for i in range(N):
            self._remove_safe(j.component('comp1').geom('geom1').feature(), f'cS2S_{i + 1}')
        cyl_tags = []
        for i in range(N):
            tag = f'cS2S_{i + 1}'
            cyl_tags.append(tag)
            j.component('comp1').geom('geom1').create(tag, 'Cylinder')
            j.component('comp1').geom('geom1').feature(tag).set('r', f'r_seg{i + 1}')
            j.component('comp1').geom('geom1').feature(tag).set('h', 'Lseg')
            j.component('comp1').geom('geom1').feature(tag).set('pos', ['0', '0', f"{float(i)}*Lseg"])
        j.component('comp1').geom('geom1').create('uniS2S', 'Union')
        j.component('comp1').geom('geom1').feature('uniS2S').selection('input').set(cyl_tags)
        j.component('comp1').geom('geom1').feature('uniS2S').set('intbnd', False)
        j.component('comp1').geom('geom1').run()
        self._remove_safe(j.component('comp1').selection(), 'selInS2S')
        self._remove_safe(j.component('comp1').selection(), 'selOutS2S')
        j.component('comp1').selection().create('selInS2S', 'Box')
        j.component('comp1').selection('selInS2S').geom('geom1', 2)
        j.component('comp1').selection('selInS2S').set('condition', 'inside')
        j.component('comp1').selection('selInS2S').set('xmin', -10.0)
        j.component('comp1').selection('selInS2S').set('xmax', 10.0)
        j.component('comp1').selection('selInS2S').set('ymin', -10.0)
        j.component('comp1').selection('selInS2S').set('ymax', 10.0)
        j.component('comp1').selection('selInS2S').set('zmin', -1e-06)
        j.component('comp1').selection('selInS2S').set('zmax', 1e-06)
        j.component('comp1').selection().create('selOutS2S', 'Box')
        j.component('comp1').selection('selOutS2S').geom('geom1', 2)
        j.component('comp1').selection('selOutS2S').set('condition', 'inside')
        j.component('comp1').selection('selOutS2S').set('xmin', -10.0)
        j.component('comp1').selection('selOutS2S').set('xmax', 10.0)
        j.component('comp1').selection('selOutS2S').set('ymin', -10.0)
        j.component('comp1').selection('selOutS2S').set('ymax', 10.0)
        j.component('comp1').selection('selOutS2S').set('zmin', 14.999999)
        j.component('comp1').selection('selOutS2S').set('zmax', 15.000001)
        r_max_mm = max(radii_m) * 1000.0
        Lseg_mm = self.Lseg * 1000.0
        xy_safety = r_max_mm * 1.5
        delta = Lseg_mm * 0.1
        for i in range(N):
            sel_tag = f'selSegLat_{i + 1}'
            self._remove_safe(j.component('comp1').selection(), sel_tag)
            z_lo = i * Lseg_mm + delta
            z_hi = (i + 1) * Lseg_mm - delta
            j.component('comp1').selection().create(sel_tag, 'Box')
            j.component('comp1').selection(sel_tag).geom('geom1', 2)
            j.component('comp1').selection(sel_tag).set('condition', 'intersects')
            j.component('comp1').selection(sel_tag).set('xmin', -xy_safety)
            j.component('comp1').selection(sel_tag).set('xmax', xy_safety)
            j.component('comp1').selection(sel_tag).set('ymin', -xy_safety)
            j.component('comp1').selection(sel_tag).set('ymax', xy_safety)
            j.component('comp1').selection(sel_tag).set('zmin', z_lo)
            j.component('comp1').selection(sel_tag).set('zmax', z_hi)
        self._remove_safe(j.component('comp1').physics('ec').feature(), 'potS2S')
        self._remove_safe(j.component('comp1').physics('ec').feature(), 'gndS2S')
        j.component('comp1').physics('ec').create('potS2S', 'ElectricPotential', 2)
        j.component('comp1').physics('ec').feature('potS2S').selection().named('selInS2S')
        j.component('comp1').physics('ec').feature('potS2S').set('V0', 'Vapp')
        j.component('comp1').physics('ec').create('gndS2S', 'Ground', 2)
        j.component('comp1').physics('ec').feature('gndS2S').selection().named('selOutS2S')
        self._setup_s2s()
        j.component('comp1').material('mat1').propertyGroup('def').set('density', ['rhoMassW'])
        j.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', [f'1/({self.rhoe_expr})'])
        j.component('comp1').material('mat1').propertyGroup('def').set('thermalconductivity', [self.k_expr])
        j.component('comp1').material('mat1').propertyGroup('def').set('heatcapacity', [self.cp_expr])
        self._run_mesh_with_fallback('initial')

    # Multiple-spectral-band S2S radiation settings.
    def _setup_s2s(self):
        j = self.j
        eps_rad_multi = 'if(comp1.rad.lambda<lam03,eps03,epsRest)'
        self._remove_safe(j.component('comp1').physics(), 'rad')
        self._remove_safe(j.multiphysics(), 'htradLT')
        j.component('comp1').physics().create('rad', 'SurfaceToSurfaceRadiation', 'geom1')
        j.component('comp1').physics('rad').prop('RadiationSettings').set('wavelengthDependenceOfSurfaceProperties', 'MultipleSpectralBands')
        j.component('comp1').physics('rad').prop('RadiationSettings').set('lambda_r', 'lam03')
        j.component('comp1').physics('rad').create('dsLT', 'DiffuseSurface', 2)
        j.component('comp1').physics('rad').feature('dsLT').selection().all()
        j.component('comp1').physics('rad').feature('dsLT').set('defineSurfaceEmissivityOnEachSide', '0')
        j.component('comp1').physics('rad').feature('dsLT').set('epsilon_radMulti_mat', 'userdef')
        j.component('comp1').physics('rad').feature('dsLT').set('epsilon_radMulti', eps_rad_multi)
        j.component('comp1').physics('rad').feature('dsLT').set('spectralBandNameAmbientEmissivityMulti', [['[0, 3['], ['[3, +inf[']])
        j.component('comp1').physics('rad').feature('dsLT').set('Tamb', 'Tamb')
        j.component('comp1').physics('rad').feature('dsLT').set('Tambu', 'Tamb')
        j.component('comp1').physics('rad').feature('dsLT').set('Tambd', 'Tamb')
        j.component('comp1').physics('rad').feature('dsLT').set('ambientEmissivity', 'userdef')
        j.component('comp1').physics('rad').feature('dsLT').set('epsilon_amb', '1')
        j.component('comp1').physics('rad').feature('dsLT').set('epsilon_ambu', '1')
        j.component('comp1').physics('rad').feature('dsLT').set('epsilon_ambd', '1')
        j.multiphysics().create('htradLT', 'HeatTransferWithSurfaceToSurfaceRadiation', 'geom1', 2)
        j.multiphysics('htradLT').selection().all()

    # Solve current geometry at one voltage and collect scalar outputs.
    def _solve_prepared(self, radii_m, voltage):
        j = self.j
        r0 = self._r0
        _, R_env, A_env = self._compute_geom(radii_m)
        self._set_params(radii_m, R_env, A_env, voltage)
        result = {'solve_ok': False, 'search_ok': False, 'failure': '', 'applied_V': voltage, 'search_steps': 0, 'Tmax': float('nan'), 'I': float('nan'), 'R': float('nan'), 'Pelec': float('nan'), 'P03steady': float('nan'), 'PradSteady': float('nan'), 'P03sphere': float('nan'), 'PradSphere': float('nan'), 'vol_err': float('nan'), 'temp_ok': False, 'volume_ok': False, 'current_ok': False, 'seg_Tavg': [0.0] * self.seg_count}
        try:
            try:
                j.result().numerical('IinS2S').selection().named('selInS2S')
            except Exception:
                pass
            j.study('std1').run()
            Tmax = float(j.result().numerical('maxTS2S').getReal()[0][0])
            try:
                Tmin = float(j.result().numerical('minTS2S').getReal()[0][0])
            except Exception:
                Tmin = Tmax * 0.95
                print(f'  WARN: MinVolume failed, Tmin={Tmin:.1f}')
            V = float(j.result().numerical('volS2S').getReal()[0][0])
            I = abs(float(j.result().numerical('IinS2S').getReal()[0][0]))
            P03steady = float(j.result().numerical('P03emitS2S').getReal()[0][0])
            PradSteady = float(j.result().numerical('PradEmitS2S').getReal()[0][0])
            PradSphere = voltage * I
            P03sphere = PradSphere * P03steady / PradSteady if PradSteady > 1e-10 else 0.0
            seg_Tavg = []
            for i in range(self.seg_count):
                read_ok = False
                try:
                    Tint = float(j.result().numerical(f'TintSeg_{i + 1}').getReal()[0][0])
                    Aseg = float(j.result().numerical(f'AsegS2S_{i + 1}').getReal()[0][0])
                    if Aseg > 1e-20:
                        seg_Tavg.append(Tint / Aseg)
                        read_ok = True
                except Exception:
                    pass
                if not read_ok:
                    eta = (i + 0.5) * self.Lseg / self.L0
                    seg_Tavg.append(Tmin + (Tmax - Tmin) * 4.0 * eta * (1.0 - eta))
            V0now = sum((math.pi * r ** 2 * self.Lseg for r in radii_m))
            V0ref = math.pi * r0 ** 2 * self.L0
            vol_err = abs(V - V0now) / V0ref
            finite_checks = {'Tmax': Tmax, 'volume': V, 'current': I, 'P03steady': P03steady, 'PradSteady': PradSteady, 'P03sphere': P03sphere, 'PradSphere': PradSphere, 'vol_err': vol_err}
            invalid = [k for k, v in finite_checks.items() if not self._finite_number(v)]
            invalid += [f'seg_Tavg[{i}]' for i, v in enumerate(seg_Tavg) if not self._finite_number(v)]
            if invalid:
                raise RuntimeError('Invalid non-finite COMSOL result: ' + ', '.join(invalid))
            result.update({'solve_ok': True, 'Tmax': Tmax, 'I': I, 'Pelec': voltage * I, 'P03steady': P03steady, 'PradSteady': PradSteady, 'P03sphere': P03sphere, 'PradSphere': PradSphere, 'vol_err': vol_err, 'temp_ok': Tmax < self.temp_limit_K, 'volume_ok': vol_err <= self.vol_tol, 'current_ok': I > self.current_tol, 'R': voltage / I if I > self.current_tol else float('nan'), 'seg_Tavg': seg_Tavg})
        except Exception as e:
            result['failure'] = str(e)
        return result

    # Voltage search keeps each candidate below the temperature limit.
    def _meets_constraint(self, r):
        return r['solve_ok'] and r['current_ok'] and r['volume_ok'] and r['temp_ok']

    def _search_best_voltage(self, radii_m):
        steps = 0
        high_res = self._solve_prepared(radii_m, self.voltage_upper)
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
        low_V = None
        low_res = None
        if high_res['solve_ok'] and high_res['current_ok'] and high_res['volume_ok'] and (not math.isnan(high_res['Tmax'])) and (high_res['Tmax'] > 0):
            guess_V = self.voltage_upper * math.sqrt(self.temp_limit_K / max(high_res['Tmax'], 1e-300))
            guess_V = max(self.voltage_floor, min(0.98 * self.voltage_upper, guess_V))
            if guess_V < high_V - 1e-12:
                guess_res = self._solve_prepared(radii_m, guess_V)
                steps += 1
                if self._meets_constraint(guess_res):
                    low_V, low_res = (guess_V, guess_res)
                else:
                    high_V, high_res = (guess_V, guess_res)
        while low_res is None and high_V > self.voltage_floor + 1e-12:
            next_V = max(self.voltage_floor, 0.5 * high_V)
            if abs(next_V - high_V) <= 1e-12:
                break
            next_res = self._solve_prepared(radii_m, next_V)
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
            mid_res = self._solve_prepared(radii_m, mid_V)
            steps += 1
            if self._meets_constraint(mid_res):
                low_V, low_res = (mid_V, mid_res)
            else:
                high_V, high_res = (mid_V, mid_res)
        low_res['search_ok'] = True
        low_res['search_steps'] = steps
        return low_res

    # Geometry-only update used inside the erosion loop.
    def _update_geometry(self, radii_m):
        j = self.j
        _, R_env, A_env = self._compute_geom(radii_m)
        for i in range(self.seg_count):
            j.param().set(f'r_seg{i + 1}', f'{radii_m[i]}[m]', f'Segment {i + 1} radius')
        j.param().set('RenvInit', f'{R_env}[m]', 'Enclosing sphere radius')
        j.param().set('AenvInit', f'{A_env}[m^2]', 'Enclosing sphere area')
        j.component('comp1').geom('geom1').run()
        self._run_mesh_with_fallback('erosion')

    def _solve_at_voltage(self, radii_m, voltage):
        self._update_geometry(radii_m)
        return self._solve_prepared(radii_m, voltage)

    # Full lifecycle evaluation: voltage search, erosion, and time integration.
    def evaluate(self, radii_m):
        t_start = time.time()
        self._ensure_server_ready()
        self._r0 = max(radii_m)
        self._initial_radii = list(radii_m)
        self._fail_radii = [r * (1.0 - self.failure_fraction) for r in radii_m]
        resolve_threshold = 0.02 * min(radii_m)
        print('  Building model...')
        self._init_model(radii_m)
        print('  Phase 1: voltage search...')
        radii = list(radii_m)
        r0_res = self._search_best_voltage(radii)
        Vwork = r0_res['applied_V']
        print(f"  PHASE1: Vwork={Vwork:.4f}V Tmax={r0_res['Tmax']:.1f}K P03sph={r0_res['P03sphere']:.1f}W steps={r0_res['search_steps']}")
        if not r0_res['search_ok']:
            return {'status': 'FAIL_VOLTAGE_SEARCH', 'failure': r0_res.get('failure', ''), 'elapsed_sec': round(time.time() - t_start, 1)}
        print('  Phase 2: erosion loop...')
        time_s = 0.0
        p03_integral = 0.0
        prad_integral = 0.0
        p03_sphere_integral = 0.0
        prad_sphere_integral = 0.0
        macro_step = 0
        failed = False
        max_macro_steps = 50
        prev_P03 = r0_res['P03steady']
        prev_Prad = r0_res['PradSteady']
        prev_P03sphere = r0_res['P03sphere']
        prev_PradSphere = r0_res['PradSphere']
        Tavg = r0_res['seg_Tavg']
        while macro_step < max_macro_steps and (not failed):
            macro_step += 1
            drdt = [0.0] * self.seg_count
            max_drdt = 0.0
            for i in range(self.seg_count):
                gamma = self.Aev * math.exp(-self.Bev / Tavg[i])
                drdt[i] = gamma / self.rho_mass
                max_drdt = max(max_drdt, drdt[i])
            if max_drdt < 1e-15:
                print('  Evaporation negligible. Infinite lifetime.')
                break
            dt_macro = float('inf')
            for i in range(self.seg_count):
                if drdt[i] > 1e-20:
                    dt_macro = min(dt_macro, resolve_threshold / drdt[i])
                    t_fail = (radii[i] - self._fail_radii[i]) / drdt[i]
                    if t_fail > 0:
                        dt_macro = min(dt_macro, t_fail)
            dt_macro = max(1.0, min(36000.0, dt_macro))
            max_loss_frac = 0.0
            for i in range(self.seg_count):
                radii[i] -= drdt[i] * dt_macro
                radii[i] = max(1e-06, radii[i])
                loss_frac = (self._initial_radii[i] - radii[i]) / self._initial_radii[i]
                max_loss_frac = max(max_loss_frac, loss_frac)
            time_s += dt_macro
            if max_loss_frac >= self.failure_fraction:
                failed = True
            try:
                r_now = self._solve_at_voltage(radii, Vwork)
            except Exception as exc:
                print(f'  WARN: erosion solve failed step {macro_step}: {exc}')
                return {'status': 'FAIL_EROSION_SOLVE', 'failure': str(exc), 'elapsed_sec': round(time.time() - t_start, 1)}
            if not r_now['solve_ok']:
                failure = r_now.get('failure', '')
                print(f'  WARN: solve failed step {macro_step}: {failure}')
                return {'status': 'FAIL_EROSION_SOLVE', 'failure': failure, 'elapsed_sec': round(time.time() - t_start, 1)}
            cur_P03 = r_now['P03steady'] if r_now['solve_ok'] else prev_P03
            cur_Prad = r_now['PradSteady'] if r_now['solve_ok'] else prev_Prad
            cur_P03sphere = r_now['P03sphere'] if r_now['solve_ok'] else prev_P03sphere
            cur_PradSphere = r_now['PradSphere'] if r_now['solve_ok'] else prev_PradSphere
            p03_integral += 0.5 * (prev_P03 + cur_P03) * dt_macro
            prad_integral += 0.5 * (prev_Prad + cur_Prad) * dt_macro
            p03_sphere_integral += 0.5 * (prev_P03sphere + cur_P03sphere) * dt_macro
            prad_sphere_integral += 0.5 * (prev_PradSphere + cur_PradSphere) * dt_macro
            prev_P03 = cur_P03
            prev_Prad = cur_Prad
            prev_P03sphere = cur_P03sphere
            prev_PradSphere = cur_PradSphere
            Tavg = r_now['seg_Tavg']
            if macro_step % 5 == 0 or failed:
                print(f'  STEP={macro_step} t={time_s / 3600:.2f}h loss={max_loss_frac:.4f}')
        lifetime_h = time_s / 3600.0
        avg_P03sphere = p03_sphere_integral / time_s if time_s > 0 else float('nan')
        avg_PradSphere = prad_sphere_integral / time_s if time_s > 0 else float('nan')
        self_view_loss = (1.0 - p03_sphere_integral / p03_integral) * 100.0 if time_s > 0 and p03_integral > 0 else float('nan')
        elapsed = time.time() - t_start
        result = {'Vwork_V': Vwork, 'initialTmax_K': r0_res['Tmax'], 'lifetimeH': lifetime_h, 'initialP03sphere_W': r0_res['P03sphere'], 'initialPradSphere_W': r0_res['PradSphere'], 'lifeAvgP03sphere_W': avg_P03sphere, 'lifeAvgPradSphere_W': avg_PradSphere, 'selfViewLoss_pct': self_view_loss, 'failureReached': failed, 'erosionSteps': macro_step, 'status': 'OK', 'elapsed_sec': round(elapsed, 1)}
        required = ['Vwork_V', 'initialTmax_K', 'lifetimeH', 'initialP03sphere_W', 'initialPradSphere_W', 'lifeAvgP03sphere_W', 'lifeAvgPradSphere_W', 'selfViewLoss_pct', 'erosionSteps']
        invalid = [k for k in required if not self._finite_number(result[k])]
        if invalid:
            result['status'] = 'FAIL_INVALID_RESULT'
            result['failure'] = 'Non-finite final metric(s): ' + ', '.join(invalid)
        return result

    # Release COMSOL resources.
    def stop(self):
        if self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                pass
            self.model = None
        if self.client is not None:
            self.client.disconnect()
            self.client = None
        print('COMSOL disconnected.')
if __name__ == '__main__':
    runner = COMSOLRunner()
    runner.start()
    try:
        baseline = [0.0025] * 8
        print(f'Evaluating baseline: {[r * 1000.0 for r in baseline]} mm')
        result = runner.evaluate(baseline)
        print('\n' + '=' * 60)
        for k, v in result.items():
            print(f'  {k}: {v}')
        print('=' * 60)
    finally:
        runner.stop()
