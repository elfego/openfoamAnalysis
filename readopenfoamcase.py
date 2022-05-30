from os.path import join, exists
from os import makedirs
from numpy import arange, array, r_, pi, cbrt, sum, save, load, sqrt, dot, zeros_like, savez_compressed
from numpy.linalg import norm, det
import re
import openfoamparser as Ofpp
from misctools import calc_val_weighted, calc_3rd_inv

#

#

#


class readOFcase:
    def __init__(self, case_dir=None):
        if case_dir is not None:
            self.setup(case_dir)

    def setup(self, case_dir):
        self.case_dir = case_dir
        self.system_dir = join(case_dir, 'system')
        self.constant_dir = join(case_dir, 'constant')
        self.times = self.list_time_dirs()
        self.existing = list(map(lambda f: exists(join(case_dir, f)),
                                 self.times))
        self.app_name = self.get_application()
        phases = self.get_phases()
        self.surface_tension = self.get_surface_tension()
        self.diffusivity = self.get_diffusivity()

        self.transport_properties = {p: self.get_properties(p)
                                     for p in phases}
        self.g = self.get_gravity()
        self.out_dir = join(case_dir, 'postProcessing')

        self.mesh_loaded = False

    def load_mesh(self):
        self.mesh_loaded = True
        self.C = self.load_field('C', '0.orig')
        self.V = self.load_field('V', '0.orig')

    def load_field(self, field_name, path):
        ifile = join(self.case_dir, path, field_name)
        if exists(ifile):
            field = Ofpp.parse_internal_field(ifile)
        else:
            field = None
            raise FileExistsError
        return field

    def load_post_field(self, field_name, path):
        ifile = join(self.out_dir, path, field_name)
        if exists(ifile):
            field = load(ifile)
        else:
            field = None
            raise FileExistsError
        return field

    def set_nozzle_radius(self, nozzle_radius):
        self.Rnozzle = nozzle_radius
        return None

    def list_time_dirs(self):
        with open(join(self.system_dir, 'controlDict'), 'r') as handler:
            for ln in handler.readlines():
                ln2 = ln.split(' ')
                if ln2[0] == 'startTime':
                    startTime = float(ln2[1][:-2])
                if ln2[0] == 'endTime':
                    endTime = float(ln2[1][:-2])
                if ln2[0] == 'writeInterval':
                    writeInterval = float(ln2[1][:-2])

        frames = arange(startTime, endTime, writeInterval) + writeInterval
        t_dirs = [f'{t:g}' for t in frames]

        return t_dirs

    def get_application(self):
        with open(join(self.system_dir, 'controlDict'), 'r') as handler:
            for ln in handler.readlines():
                ln2 = ln.split(' ')
                if ln2[0] == 'application':
                    return ln2[1][:-2]

    def get_phases(self):
        with open(join(self.constant_dir, 'transportProperties'), 'r') as handler:
            for ln in handler.readlines():
                ln2 = ln.split(' ')
                if ln2[0] == 'phases':
                    return ' '.join(ln2[1:]).split('(')[1].split(')')[0].split(' ')

    def get_surface_tension(self):
        with open(join(self.constant_dir, 'transportProperties'), 'r') as handler:
            for ln in handler.readlines():
                ln2 = ln.split(' ')

                if ln2[0] == 'sigma12':
                    result = float(' '.join(ln2[1:])[:-2])
                if ln2[0] == 'sigma13':
                    if result != float(' '.join(ln2[1:])[:-2]):
                        print('Surface tensions are unequal!')
                    return result

    def get_properties(self, phase):
        with open(join(self.constant_dir, 'transportProperties'), 'r') as handler:
            lns = ''.join(handler.readlines())
            tmp = lns.split('phases')[1]

            tmp, = re.findall(phase + '\s*\{[^\}]*\}', tmp)

            transport_model = re.findall('transportModel\s*\w*', tmp)[0].split(' ')[-1]
            viscosity = float(re.findall('nu\s*[^\;]*;', tmp)[0].split(' ')[-1][:-1])
            density = float(re.findall('rho\s*[^\;]*;', tmp)[0].split(' ')[-1][:-1])

        return {'transportModel': transport_model,
                'viscosity': viscosity,
                'density': density}

    def get_diffusivity(self):
        with open(join(self.constant_dir, 'transportProperties'), 'r') as handler:
            lns = ''.join(handler.readlines())
            D = float(re.findall('D23\s*[^\;]*;', lns)[0].split(' ')[-1][:-1])
            return D

    def get_gravity(self):
        with open(join(self.constant_dir, 'g'), 'r') as handler:
            lns = ''.join(handler.readlines())
            tmp, = re.findall('value\s*\([^\(]*\)', lns)
            tmp, = re.findall('\([^\)]*\)', tmp)
            return array(list(map(float, tmp[1:-1].split(' '))))

    def calc_dimensinless_numbers(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        nu1 = self.transport_properties['pregel']['viscosity']
        nu2 = self.transport_properties['crosslinker']['viscosity']

        rho1 = self.transport_properties['pregel']['density']
        rho2 = self.transport_properties['crosslinker']['density']

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        alpha3 = self.load_field('alpha.air', t_dir)
        U      = self.load_field('U', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        dv1 = alpha1 * self.V
        dv2 = alpha2 * self.V

        V1 = sum(dv1)
        save(join(o_dir,      'V.pregel.npy'), r_[V1, 2. * cbrt(0.75 * V1 / pi)])
        V2 = sum(dv2)
        save(join(o_dir, 'V.crosslinker.npy'), r_[V2, 2. * cbrt(0.75 * V2 / pi)])

        X1 = calc_val_weighted(self.C, dv1,
                               normalised=True,
                               fsave=join(o_dir, 'X.pregel.npy'))
        X2 = calc_val_weighted(self.C, dv2,
                               normalised=True,
                               fsave=join(o_dir, 'X.crosslinker.npy'))

        U1 = calc_val_weighted(U, dv1,
                               normalised=True,
                               fsave=join(o_dir, 'U.pregel.npy'))
        U2 = calc_val_weighted(U, dv2,
                               normalised=True,
                               fsave=join(o_dir, 'U.crosslinker.npy'))

        Ur = norm(U2 - U1)

        Ucm = (V1 * U1 + V2 * U2) / (V1 + V2)
        save(join(o_dir, 'Ucm.npy'), Ucm)

        b1 = norm(X2 - X1)
        b2 = dot(U2 - U1, X2 - X1) / Ur
        save(join(o_dir, 'distance.npy'), r_[b1, b2])
        b = sqrt(b1 ** 2 - b2 ** 2)

        B = 0.5 * b / self.Rnozzle
        save(join(o_dir, 'impact_param.npy'), r_[b, B])

        We_collision = (rho1 + rho2) * self.Rnozzle * Ur * Ur / self.surface_tension
        Re_collision = 4 * self.Rnozzle * Ur / (nu1 + nu2)
        save(join(o_dir, 'We_Re.npy'), r_[We_collision, Re_collision])
        return None

    def calc_mixture_measures(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        gradAlpha1 = self.load_field('grad(alpha.pregel)', t_dir)
        gradAlpha2 = self.load_field('grad(alpha.crosslinker)', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        dv1 = alpha1 * self.V
        dv2 = alpha2 * self.V

        magGradAlpha1 = norm(gradAlpha1, axis=1)
        magGradAlpha2 = norm(gradAlpha2, axis=1)

        A_c = dot(magGradAlpha2, dv1) + dot(magGradAlpha1, dv2)
        save(join(o_dir, 'surfaceDensityFunction.npy'), A_c)
        V_m = 4 * dot(alpha1, dv2)
        save(join(o_dir, 'mixtureVolume'), V_m)
        E_mu = self.diffusivity * (dot(magGradAlpha2**2, dv1) +
                                   dot(magGradAlpha1**2, dv2))
        save(join(o_dir, 'scalarDissipationRate.npy'), E_mu)

        return None

    def calc_classification(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        Q = self.load_field('Q', t_dir)
        R = self.load_post_field('3rd_invariant.npy', time)
        n = 2 * (R > zeros_like(R)) + (4 * Q ** 3 + 27 * R ** 2 > zeros_like(Q))
        savez_compressed(join(o_dir, 'classification.npz'), n)

    def calc_R(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        gradU = self.load_field('grad(U)', t_dir)
        R = array(list(map(calc_3rd_inv, gradU)))
        save(join(o_dir, '3rd_invariant.npy'), R)

    def forAllTimes(self, func, *args, **kwargs):
        for e, t in zip(self.existing, self.times):
            if not e:
                print('Skipping', t, '...')
                continue
            print(t, ' : ', func.__name__, '...')
            func(t, *args, **kwargs)

    def calc_visc_dissipation(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        enstrophy = self.load_field('enstrophy', t_dir)
        Q = self.load_field('Q', t_dir)
        eps = 2.0 *(enstrophy - 2 * Q)
        save(join(o_dir, 'visc_dissipation.npy'), eps)

    def calc_eigprojection(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        alpha1 = self.load_field('alpha.crosslinker', time)
        gradAlpha1 = self.load_field('grad(alpha.crosslinker)', time)

        alpha2 = self.load_field('alpha.pregel', time)
        gradAlpha2 = self.load_field('grad(alpha.pregel)', time)

        dsigma = (alpha1 * gradAlpha2 - alpha2 * gradAlpha1) * self.V

        evecs = self.load_post_field('eigenvectorsGradU.npy', time)



#


rc = readOFcase('/home/vvmv9/workspace/OFSims/mix_L1/ur-6')
rc.set_nozzle_radius(2.5e-4)
# rc.calc_dimensinless_numbers('0.001')
# rc.calc_mixture_measures('0.001')
# rc.calc_visc_dissipation('0.001')
# rc.forAllTimes(rc.calc_R)
# rc.forAllTimes(rc.calc_classification)
# rc.forAllTimes(rc.calc_visc_dissipation)
# rc.forAllTimes(rc.calc_mixture_measures)
# rc.forAllTimes(rc.calc_dimensinless_numbers)

