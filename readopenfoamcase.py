from os.path import join, exists
from os import makedirs
from re import findall
from numpy import (arange, array, r_, sum, save, savez_compressed,
                   load, sqrt, dot, zeros, zeros_like, ones_like,
                   vstack)
from numpy.linalg import norm
from openfoamparser import parse_internal_field
from misctools import (calc_val_weighted, calc_2nd_inv, calc_3rd_inv,
                       dSigma, local_eigensystem, get_vorticity)


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

    def forAllTimes(self, func, *args, interval=None, **kwargs):
        ran = zip(self.existing, self.times)
        if interval is not None:
            interval[1] = min(len(self.times), interval[1])
            ran = zip(self.existing[interval[0]:interval[1]],
                      self.times[interval[0]:interval[1]])
        for e, t in ran:
            if not e:
                print('Skipping', t, '...')
                continue
            print(t, ' : ', func.__name__, '...')
            func(t, *args, **kwargs)

    def calc_droplet_volumes(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'V.pregel.npy')) and
            exists(join(o_dir, 'V.crosslinker.npy')) and
            not overwrite):
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        save(join(o_dir,      'V.pregel.npy'), dot(alpha1, self.V))
        save(join(o_dir, 'V.crosslinker.npy'), dot(alpha2, self.V))

    def calc_Xcm(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'X.pregel.npy')) and
            exists(join(o_dir, 'X.crosslinker.npy')) and
            not overwrite):
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

    def calc_Ucm(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'U.pregel.npy')) and
            exists(join(o_dir, 'U.crosslinker.npy')) and
            exists(join(o_dir, 'Ucm.npy')) and
            not overwrite):
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

    def calc_impact_parameter(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'impact_param.npy')) and not overwrite:
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

    def calc_Reynolds(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'Re_collision.npy')) and not overwrite:
            return None

        U1 = self.load_post_field('U.pregel.npy', time)
        U2 = self.load_post_field('U.crosslinker.npy', time)

        Ur = norm(U2 - U1)
        nu1 = self.transport_properties['pregel']['viscosity']
        nu2 = self.transport_properties['crosslinker']['viscosity']

        Re_collision = 4 * self.Rnozzle * Ur / (nu1 + nu2)
        save(join(o_dir, 'Re_collision.npy'), Re_collision)

    def calc_Weber(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'We_collision.npy')) and not overwrite:
            return None

        U1 = self.load_post_field('U.pregel.npy', time)
        U2 = self.load_post_field('U.crosslinker.npy', time)

        Ur = norm(U2 - U1)
        rho1 = self.transport_properties['pregel']['density']
        rho2 = self.transport_properties['crosslinker']['density']

        We_collision = (rho1 + rho2) * self.Rnozzle * \
            Ur * Ur / self.surface_tension
        save(join(o_dir, 'We_collision.npy'), We_collision)

    def calc_vorticity(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'vorticity.npy')) and not overwrite:
            return None

        gradU = self.load_field('grad(U)', t_dir)
        W = vstack(list(map(get_vorticity, gradU)))
        save(join(o_dir, 'vorticity.npy'), W)
        return None

    def calc_enstrophy(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'enstrophy.npy')) and not overwrite:
            return None

        W = self.load_post_field('vorticity.npy', time)
        xi = 0.5 * norm(W, axis=1)**2
        save(join(o_dir, 'enstrophy.npy'), xi)
        return None

    def calc_Q(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'Q.npy')) and not overwrite:
            return None

        gradU = self.load_field('grad(U)', t_dir)
        Q = array(list(map(calc_2nd_inv, gradU)))
        save(join(o_dir, 'Q.npy'), Q)
        return None

    def calc_R(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'R.npy')) and not overwrite:
            return None

        gradU = self.load_field('grad(U)', t_dir)
        R = array(list(map(calc_3rd_inv, gradU)))
        save(join(o_dir, 'R.npy'), R)
        return None

    def calc_dSigma(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'dSigma.npy')) and not overwrite:
            return None

        alpha1 = self.load_field('alpha.crosslinker', time)
        alpha2 = self.load_field('alpha.pregel', time)
        gradAlpha1 = self.load_field('grad(alpha.crosslinker)', time)
        gradAlpha2 = self.load_field('grad(alpha.pregel)', time)

        if not self.mesh_loaded:
            self.load_mesh()

        dS = dSigma(alpha1, alpha2, gradAlpha1, gradAlpha2, self.V)
        save(join(o_dir, 'dSigma.npy'), dS)

    def calc_contact_area(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'contact_surface_area.npy')) and not overwrite:
            return None

        dS = self.load_post_field('dSigma.npy', time)
        S = sum(norm(dS, axis=1))
        save(join(o_dir, 'contact_surface_area.npy'), S)

    def calc_volume_mixture(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'mixtureVolume.npy')) and not overwrite:
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        save(join(o_dir, 'mixtureVolume.npy'),
             4.0 * sum(alpha1 * alpha2 * self.V))
        return None

    def calc_dissipation_rate(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'scalarDissipationRate.npy')) and not overwrite:
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

    def calc_classification(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'classification.npz')) and not overwrite:
            return None

        Q = self.load_post_field('Q.npy', time)
        R = self.load_post_field('R.npy', time)
        n = 2 * (R > zeros_like(R)) +\
            (4 * Q ** 3 + 27 * R ** 2 > zeros_like(Q))
        savez_compressed(join(o_dir, 'classification.npz'), n)
        return None

    def calc_visc_dissipation_density(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'visc_dissipation_density.npy')) and not overwrite:
            return None

        enstrophy = self.load_post_field('enstrophy.npy', time)
        Q = self.load_post_field('Q.npy', time)
        save(join(o_dir, 'visc_dissipation_density.npy'),
             2.0 * (enstrophy - 2.0 * Q))
        return None

    def calc_visc_dissipation(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'visc_dissipation.npy')) and not overwrite:
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        rho1 = self.transport_properties['pregel']['density']
        nu1 = self.transport_properties['pregel']['viscosity']

        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        rho2 = self.transport_properties['crosslinker']['density']
        nu2 = self.transport_properties['crosslinker']['viscosity']

        if not self.mesh_loaded:
            self.load_mesh()

        eps = self.load_post_field('visc_dissipation_density.npy', time)
        save(join(o_dir, 'visc_dissipation.npy'),
             dot(eps * self.V, rho1 * nu1 * alpha1 + rho2 * nu2 * alpha2))
        return None

    def calc_eigensystem(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'eigenvector_1.npy')) and
            exists(join(o_dir, 'eigenvector_3.npy')) and
            exists(join(o_dir, 'eigenvalues.npy')) and
            not overwrite):
            return None

        gradU = self.load_field('grad(U)', time)
        N = len(gradU)
        W = zeros((N, 12))
        for i in range(N):
            W[i] = local_eigensystem(gradU[i])

        save(join(o_dir, 'eigenvector_1.npy'), W[:,  0:3])
        save(join(o_dir, 'eigenvector_3.npy'), W[:,  6:9])
        save(join(o_dir,   'eigenvalues.npy'), W[:, 9:12])
        return None

    def calc_eigprojection(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'eigvec_1_projection.npy')) and
            exists(join(o_dir, 'eigvec_3_projection.npy')) and
            not overwrite):
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

    def calc_topology_contact_surface(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'surface_area_topology.npy')) and not overwrite:
            return None

        C = self.load_post_field('classification.npz', time)['arr_0']
        dS = norm(self.load_post_field('dSigma.npy', time), axis=1)

        save(join(o_dir, 'surface_area_topology.npy'),
             array([dot(C == i, dS) for i in range(4)]))
        return None

    def calc_topology_diffusive(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'diffusive_topology.npy')) and not overwrite:
            return None

        C = self.load_post_field('classification.npz', time)['arr_0']
        dV = self.load_post_field('scalarDissipationRate.npy', time)

        save(join(o_dir, 'diffusive_topology.npy'),
             array([dot(C == i, dV) for i in range(4)]))
        return None

    def calc_topology_mixture_volume(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'mixing_topology.npy')) and not overwrite:
            return None

        C = self.load_post_field('classification.npz', time)['arr_0']
        dV = self.load_post_field('mixtureVolume.npy', time)

        save(join(o_dir, 'mixing_topology.npy'),
             array([dot(C == i, dV) for i in range(4)]))
        return None

    def calc_vortprojection(self, time, overwrite=False):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'w_dot_n.npy')) and not overwrite:
            return None

        W = self.load_post_field('vorticity.npy', time)
        dS = self.load_post_field('dSigma.npy', time)
        magW = norm(W, axis=1)
        save(join(o_dir, 'w_dot_n.npy'),
             sum(abs(sum(W * dS, axis=1) / magW)))
        return None

    def calc_surface_energy(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'surface_energy.npy')) and not overwrite:
            return None

        gradAlpha1 = self.load_field('grad(alpha.pregel)', t_dir)
        gradAlpha2 = self.load_field('grad(alpha.crosslinker)', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()
        w = self.C[:, 3] >= 0.0002 * ones_like(self.C[:, 3])

        save(join(o_dir, 'surface_energy.npy'),
             self.surface_tension * dot(self.V * w, norm(gradAlpha1 + gradAlpha2, axis=1)))
        return None

    def calc_kinetic_energy(self, time, overwrite=False):
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'kinetic_energy.npy')) and not overwrite:
            return None

        U = self.load_field('U', t_dir)

        alpha1 = self.load_field('alpha.pregel', t_dir)
        rho1 = self.transport_properties['pregel']['density']

        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        rho2 = self.transport_properties['crosslinker']['density']

        if not self.mesh_loaded:
            self.load_mesh()

        save(join(o_dir, 'kinetic_energy.npy'),
             0.5 * dot(rho1 * alpha1 + rho2 * alpha2, self.V * norm(U, axis=1)**2))
        return None

    def calc_rot_energy(self, time, overwrite=False):
        return None

    def cleanup():
        return None

    def measureAll(self, time, overwrite=False, cleanup=False):
        self.calc_vorticity(time, overwrite=overwrite)
        self.calc_enstrophy(time, overwrite=overwrite)
        self.calc_Q(time, overwrite=overwrite)
        self.calc_R(time, overwrite=overwrite)
        self.calc_droplet_volumes(time, overwrite=overwrite)
        self.calc_Xcm(time, overwrite=overwrite)
        self.calc_Ucm(time, overwrite=overwrite)
        self.calc_impact_parameter(time, overwrite=overwrite)
        self.calc_Reynolds(time, overwrite=overwrite)
        self.calc_Weber(time, overwrite=overwrite)
        self.calc_dSigma(time, overwrite=overwrite)
        self.calc_contact_area(time, overwrite=overwrite)
        self.calc_volume_mixture(time, overwrite=overwrite)
        self.calc_dissipation_rate(time, overwrite=overwrite)
        self.calc_classification(time, overwrite=overwrite)
        self.calc_visc_dissipation_density(time, overwrite=overwrite)
        self.calc_visc_dissipation(time, overwrite=overwrite)
        self.calc_eigensystem(time, overwrite=overwrite)
        self.calc_eigprojection(time, overwrite=overwrite)
        self.calc_topology_contact_surface(time, overwrite=overwrite)
        self.calc_topology_diffusive(time, overwrite=overwrite)
        self.calc_topology_mixture_volume(time, overwrite=overwrite)
        self.calc_vortprojection(time, overwrite=overwrite)
        self.calc_surface_energy(time, overwrite=overwrite)
        self.calc_kinetic_energy(time, overwrite=overwrite)
        self.calc_rot_energy(time, overwrite=overwrite)
        self.cleanup()
        return None



