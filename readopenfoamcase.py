from os.path import join, exists, makedirs
from numpy import arange, array, r_, pi, cbrt, sum, save, sqrt, dot
from numpy.linalg import norm
import re
import openfoamparser as Ofpp
from misctools import calc_val_weighted

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

        self.C = self.load_field('C', '0.orig')
        self.V = self.load_field('V', '0.orig')

    def load_field(self, field_name, path):
        ifile = join(self.case_dir, path, field_name)
        if exists(ifile):
            field = Ofpp.parse_boundary_field(ifile)
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

    def compute_dimensinless_numbers(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        nu1 = self.transport_properties['alpha.pregel']['viscosity']
        nu2 = self.transport_properties['alpha.crosslinker']['viscosity']

        rho1 = self.transport_properties['alpha.pregel']['density']
        rho2 = self.transport_properties['alpha.crosslinker']['density']

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        alpha3 = self.load_field('alpha.air', t_dir)
        U      = self.load_field('U', t_dir)

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

#


rc = readOFcase('/DataB/mix_L2/base')
print(rc.g)
