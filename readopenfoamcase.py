from os.path import join, exists
from os import makedirs
from re import findall
from numpy import (arange, array, r_, sum, save, savez_compressed,
                   load, sqrt, dot, zeros, zeros_like)
from numpy.linalg import norm
from openfoamparser import parse_internal_field
from misctools import (calc_val_weighted, calc_3rd_inv, dSigma,
                       local_eigensystem)


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
            field = parse_internal_field(ifile)
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

            tmp, = findall(phase + '\s*\{[^\}]*\}', tmp)

            transport_model = findall('transportModel\s*\w*', tmp)[0].split(' ')[-1]
            viscosity = float(findall('nu\s*[^\;]*;', tmp)[0].split(' ')[-1][:-1])
            density = float(findall('rho\s*[^\;]*;', tmp)[0].split(' ')[-1][:-1])

        return {'transportModel': transport_model,
                'viscosity': viscosity,
                'density': density}

    def get_diffusivity(self):
        with open(join(self.constant_dir, 'transportProperties'), 'r') as handler:
            lns = ''.join(handler.readlines())
            D = float(findall('D23\s*[^\;]*;', lns)[0].split(' ')[-1][:-1])
            return D

    def get_gravity(self):
        with open(join(self.constant_dir, 'g'), 'r') as handler:
            lns = ''.join(handler.readlines())
            tmp, = findall('value\s*\([^\(]*\)', lns)
            tmp, = findall('\([^\)]*\)', tmp)
            return array(list(map(float, tmp[1:-1].split(' '))))

    def forAllTimes(self, func, *args, **kwargs):
        for e, t in zip(self.existing, self.times):
            if not e:
                print('Skipping', t, '...')
                continue
            print(t, ' : ', func.__name__, '...')
            func(t, *args, **kwargs)

    def calc_droplet_volumes(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'V.pregel.npy')) and
            exists(join(o_dir, 'V.crosslinker.npy'))):
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        save(join(o_dir,      'V.pregel.npy'), dot(alpha1, self.V))
        save(join(o_dir, 'V.crosslinker.npy'), dot(alpha2, self.V))

    def calc_Xcm(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'X.pregel.npy')) and
            exists(join(o_dir, 'X.crosslinker.npy'))):
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        dv1 = alpha1 * self.V
        dv2 = alpha2 * self.V

        calc_val_weighted(self.C, dv1,
                          normalised=True,
                          fsave=join(o_dir, 'X.pregel.npy'))
        calc_val_weighted(self.C, dv2,
                          normalised=True,
                          fsave=join(o_dir, 'X.crosslinker.npy'))

    def calc_Ucm(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'U.pregel.npy')) and
            exists(join(o_dir, 'U.crosslinker.npy'))):
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        U = self.load_field('U', t_dir)
        V1 = self.load_post_field('V.pregel.npy', time)
        V2 = self.load_post_field('V.crosslinker.npy', time)

        if not self.mesh_loaded:
            self.load_mesh()

        dv1 = alpha1 * self.V
        dv2 = alpha2 * self.V

        U1 = calc_val_weighted(U, dv1,
                               normalised=True,
                               fsave=join(o_dir, 'U.pregel.npy'))
        U2 = calc_val_weighted(U, dv2,
                               normalised=True,
                               fsave=join(o_dir, 'U.crosslinker.npy'))
        Ucm = (V1 * U1 + V2 * U2) / (V1 + V2)
        save(join(o_dir, 'Ucm.npy'), Ucm)

    def calc_impact_parameter(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'impact_param.npy')):
            return None

        X1 = self.load_post_field('X.pregel.npy', time)
        X2 = self.load_post_field('X.crosslinker.npy', time)

        U1 = self.load_post_field('U.pregel.npy', time)
        U2 = self.load_post_field('U.crosslinker.npy', time)

        Ur = norm(U2 - U1)
        b1 = norm(X2 - X1)
        b2 = dot(U2 - U1, X2 - X1) / Ur
        b = sqrt(b1 ** 2 - b2 ** 2)

        B = 0.5 * b / self.Rnozzle
        save(join(o_dir, 'impact_param.npy'), r_[b, B])

    def calc_Reynolds(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'Re_collision.npy')):
            return None

        U1 = self.load_post_field('U.pregel.npy', time)
        U2 = self.load_post_field('U.crosslinker.npy', time)

        Ur = norm(U2 - U1)
        nu1 = self.transport_properties['pregel']['viscosity']
        nu2 = self.transport_properties['crosslinker']['viscosity']

        Re_collision = 4 * self.Rnozzle * Ur / (nu1 + nu2)
        save(join(o_dir, 'Re_collision.npy'), Re_collision)

    def calc_Weber(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'We_collision.npy')):
            return None

        U1 = self.load_post_field('U.pregel.npy', time)
        U2 = self.load_post_field('U.crosslinker.npy', time)

        Ur = norm(U2 - U1)
        rho1 = self.transport_properties['pregel']['density']
        rho2 = self.transport_properties['crosslinker']['density']

        We_collision = (rho1 + rho2) * self.Rnozzle * \
            Ur * Ur / self.surface_tension
        save(join(o_dir, 'We_collision.npy'), We_collision)

    def calc_dSigma(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'dSigma.npy')):
            return None

        alpha1 = self.load_field('alpha.crosslinker', time)
        alpha2 = self.load_field('alpha.pregel', time)
        gradAlpha1 = self.load_field('grad(alpha.crosslinker)', time)
        gradAlpha2 = self.load_field('grad(alpha.pregel)', time)

        if not self.mesh_loaded:
            self.load_mesh()

        dS = dSigma(alpha1, alpha2, gradAlpha1, gradAlpha2, self.V)
        save(join(o_dir, 'dSigma.npy'), dS)

    def calc_contact_area(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'contact_surface_area.npy')):
            return None

        dS = self.load_post_field('dSigma.npy', time)
        S = sum(norm(dS, axis=1))
        save(join(o_dir, 'contact_surface_area.npy'), S)

    def calc_volume_mixture(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'mixtureVolume.npy')):
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        dChi = 4.0 * alpha1 * alpha2 * self.V
        save(join(o_dir, 'mixtureVolume.npy'), sum(dChi))

    def calc_dissipation_rate(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'scalarDissipationRate.npy')):
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        magGradAlpha1 = norm(self.load_field('grad(alpha.pregel)', t_dir),
                             axis=1)
        magGradAlpha2 = norm(self.load_field('grad(alpha.crosslinker)', t_dir),
                             axis=1)

        if not self.mesh_loaded:
            self.load_mesh()

        dv1 = alpha1 * self.V
        dv2 = alpha2 * self.V

        E_mu = self.diffusivity * (dot(magGradAlpha2**2, dv1) +
                                   dot(magGradAlpha1**2, dv2))
        save(join(o_dir, 'scalarDissipationRate.npy'), E_mu)
        return None

    def calc_R(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, '3rd_invariant.npy')):
            return None

        gradU = self.load_field('grad(U)', t_dir)
        R = array(list(map(calc_3rd_inv, gradU)))
        save(join(o_dir, '3rd_invariant.npy'), R)
        return None

    def calc_classification(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'classification.npz')):
            return None

        Q = self.load_field('Q', t_dir)
        R = self.load_post_field('3rd_invariant.npy', time)
        n = 2 * (R > zeros_like(R)) +\
            (4 * Q ** 3 + 27 * R ** 2 > zeros_like(Q))
        savez_compressed(join(o_dir, 'classification.npz'), n)
        return None

    def calc_visc_dissipation_density(self, time):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'visc_dissipation_density.npy')):
            return None

        enstrophy = self.load_field('enstrophy', t_dir)
        Q = self.load_field('Q', t_dir)
        eps = 2.0 * (enstrophy - 2.0 * Q)
        save(join(o_dir, 'visc_dissipation_density.npy'), eps)
        return None

    def calc_eigensystem(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'eigenvector_1.npy')) and
            exists(join(o_dir, 'eigenvector_2.npy')) and
            exists(join(o_dir, 'eigenvector_3.npy')) and
            exists(join(o_dir, 'eigenvalues.npy'))):
            return None

        gradU = self.load_field('grad(U)', time)
        N = len(gradU)
        W = zeros((N, 12))
        for i in range(N):
            W[i] = local_eigensystem(gradU[i])

        save(join(o_dir, 'eigenvector_1.npy'), W[:,  0:3])
        save(join(o_dir, 'eigenvector_2.npy'), W[:,  3:6])
        save(join(o_dir, 'eigenvector_3.npy'), W[:,  6:9])
        save(join(o_dir,   'eigenvalues.npy'), W[:, 9:12])
        return None

    def calc_eigprojection(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'eigvec_1_projection.npy')) and
            exists(join(o_dir, 'eigvec_3_projection.npy'))):
            return None

        dS = self.load_post_field('dSigma.npy', time)
        E1 = self.load_post_field('eigenvector_1.npy', time)
        E3 = self.load_post_field('eigenvector_3.npy', time)

        S = sum(norm(dS, axis=1))
        save(join(o_dir, 'contact_surface_area.npy'), S)
        pE1 = sum(abs(sum(E1 * dS, axis=1)))
        pE3 = sum(abs(sum(E3 * dS, axis=1)))

        save(join(o_dir, 'eigvec_1_projection.npy'), pE1)
        save(join(o_dir, 'eigvec_3_projection.npy'), pE3)
        return pE1, pE3

    def calc_topology_contact_surface(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'surface_area_topology.npy')):
            return None

        C = self.load_post_field('classification.npz', time)
        dS = norm(self.load_post_field('dSigma.npy', time), axis=1)

        Cs = array([dot(C == i, dS) for i in range(4)])
        save(join(o_dir, 'surface_area_topology.npy'), Cs)
        return None
