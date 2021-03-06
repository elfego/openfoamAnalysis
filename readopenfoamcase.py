from os.path import join, exists
from os import makedirs, remove
from re import findall
from sys import stderr
from numpy import (arange, array, r_, sum, save, savez_compressed,
                   load, sqrt, dot, zeros, zeros_like, ones_like,
                   vstack, cross, histogram, histogram2d, logspace,
                   linspace, log10, meshgrid)
from numpy.linalg import norm
from openfoamparser import parse_internal_field
from misctools import (calc_val_weighted, dSigma, get_vorticity,
                       local_eigensystem, prod, calc_2nd_inv, calc_3rd_inv)
from nanoalgebra import (asymmVec, magSq, Qinv, Rinv, symmTraceless,
                         eigvecsh, eigvecs)
import time as tm


def normalise(v):
    N = len(v)
    L = norm(v, axis=1)
    w = zeros_like(v)
    for idx in range(N):
        if L[idx] > 1e-13:
            w[idx] = v[idx] / L[idx]
    return w


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
        self.old_style = False

    def set_oldstyle(self):
        self.old_style = True

    def load_mesh(self):
        self.mesh_loaded = True
        self.R = self.load_field('C', '0.orig')
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
            field = load(ifile, allow_pickle=True)
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
            try:
                func(t, *args, **kwargs)
            except Exception as error:
                print("Case: ", self.case_dir, file=stderr)
                print("   Something went wrong at time:", t, file=stderr)
                print(error, file=stderr)

    def calc_droplet_volumes(self, time, overwrite=False):
        print('\tCalculating droplet volumes V1 and V2...')
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
        print('\tCalculating centre of mass...')
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

        calc_val_weighted(self.R, dv1,
                          normalised=True,
                          fsave=join(o_dir, 'X.pregel.npy'))
        calc_val_weighted(self.R, dv2,
                          normalised=True,
                          fsave=join(o_dir, 'X.crosslinker.npy'))

    def calc_Ucm(self, time, overwrite=False):
        print('\tCalculating velocity of the centre of mass...')
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
        print('\tCalculating impact parameter...')
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
        print('\tCalculating Reynolds number...')
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
        print('\tCalculating Weber number...')
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

    def calc_gradU_deriv(self, time, overwrite=False):
        print('\tCalculating gradU derived fields...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'vorticity.npy')) and
            exists(join(o_dir, 'enstrophy.npy')) and
            exists(join(o_dir, 'Q.npy')) and
            exists(join(o_dir, 'R.npy')) and
            exists(join(o_dir, 'visc_dissipation_density.npy')) and
            exists(join(o_dir, 'eigenvector_1.npy')) and
            exists(join(o_dir, 'eigenvector_2.npy')) and
            exists(join(o_dir, 'eigenvector_3.npy')) and
            exists(join(o_dir, 'eigenvalues.npy')) and
            not overwrite):
            return None

        A = self.load_field('grad(U)', t_dir)

        N = len(A)

        omega = array(list(map(asymmVec, A)))
        save(join(o_dir, 'vorticity.npy'), omega)
        print('\t\tsaving vorticity.npy')

        xi = array(list(map(magSq, omega)))
        save(join(o_dir, 'enstrophy.npy'), xi)
        print('\t\tsaving enstrophy.npy')
        del omega

        Q = array(list(map(Qinv, A)))
        save(join(o_dir, 'Q.npy'), Q)
        print('\t\tsaving Q.npy')

        R = array(list(map(Rinv, A)))
        save(join(o_dir, 'R.npy'), R)
        print('\t\tsaving R.npy')
        del R

        save(join(o_dir, 'visc_dissipation_density.npy'),
             2.0 * (xi - 2.0 * Q))
        print('\t\tsaving visc_dissipation_density.npy')
        del xi
        del Q

        W = zeros((N, 12))
        for i in range(N):
            # W[i] = eigvecsh(symmTraceless(A[i]))
            W[i] = eigvecs(A[i])
        save(join(o_dir, 'eigenvector_1.npy'), W[:,  0:3])
        save(join(o_dir, 'eigenvector_2.npy'), W[:,  3:6])
        save(join(o_dir, 'eigenvector_3.npy'), W[:,  6:9])
        save(join(o_dir,   'eigenvalues.npy'), W[:, 9:12])
        print('\t\tsaving eigenvector_#.npy')
        del W

        return None

    def calc_vorticity(self, time, overwrite=False):
        print('\tCalculating viscosity...')
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
        print('\tCalculating enstrophy...')
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
        print('\tCalculating second invariant (Q)...')
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
        print('\tCalculating third invariant (R)...')
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
        print('\tCalculating surface density vector...')
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
        print('\tCalculating contact area...')

        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'contact_surface_area.npy')) and not overwrite:
            return None

        dS = self.load_post_field('dSigma.npy', time)
        S = sum(norm(dS, axis=1))
        save(join(o_dir, 'contact_surface_area.npy'), S)

    def calc_volume_mixture(self, time, overwrite=False):
        print('\tCalculating volume of the mixture...')

        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'mixtureVolume.npy')) and not overwrite:
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        save(join(o_dir, 'mixtureVolume.npy'), 4 * sum(alpha1 * alpha2 * self.V))
        return None

    def calc_segregation(self, time, overwrite=False):
        print('\tCalculating segregation')

        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'segregation.npy')) and not overwrite:
            return None

        if not self.mesh_loaded:
            self.load_mesh()

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        V1 = self.load_post_field('V.pregel.npy', time)
        V2 = self.load_post_field('V.crosslinker.npy', time)

        seg = ((V2 / V1) * dot(alpha1**2, self.V)
             + (V1 / V2) * dot(alpha2**2, self.V)
             - 2.0 * dot(alpha1 * alpha2, self.V)) / (V1 + V2)

        save(join(o_dir, 'segregation.npy'), seg)
        return None

    def calc_dissipation_density(self, time, overwrite=False):
        print('\tCalculating dissipation density...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'scalar_dissipation_density.npy')) and not overwrite:
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        magGradAlpha1 = norm(self.load_field('grad(alpha.pregel)', t_dir),
                             axis=1)

        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        magGradAlpha2 = norm(self.load_field('grad(alpha.crosslinker)', t_dir),
                             axis=1)

        if not self.mesh_loaded:
            self.load_mesh()

        dv1 = alpha1 * self.V
        dv2 = alpha2 * self.V

        save(join(o_dir, 'scalar_dissipation_density.npy'),
             (dv1 * magGradAlpha2**2 + dv2 * magGradAlpha1**2))
        return None

    def calc_dissipation_rate(self, time, overwrite=False):
        print('\tCalculating dissipation rate...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'scalar_dissipation_rate.npy')) and not overwrite:
            return None
        eps_D = self.load_post_field('scalar_dissipation_density.npy', time)
        save(join(o_dir, 'scalar_dissipation_rate.npy'), self.diffusivity * sum(eps_D))
        return None

    def calc_classification(self, time, overwrite=False):
        print('\tCalculating topology classification...')

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
        print('\tCalculating viscous dissipation density...')

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
        print('\tCalculating viscous dissipation...')

        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'visc_dissipation.npy')) and not overwrite:
            return None

        alpha1 = self.load_field('alpha.pregel', t_dir)
        eta1 = self.transport_properties['pregel']['viscosity'] \
            * self.transport_properties['pregel']['density']

        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        eta2 = self.transport_properties['crosslinker']['viscosity'] \
            * self.transport_properties['crosslinker']['density']

        if not self.mesh_loaded:
            self.load_mesh()

        eps = self.load_post_field('visc_dissipation_density.npy', time)
        save(join(o_dir, 'visc_dissipation.npy'),
             dot(eps * self.V, eta1 * alpha1 + eta2 * alpha2))
        return None

    def calc_eigensystem(self, time, overwrite=False):
        print('\tCalculating eigensystem...')
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'eigenvector_1.npy')) and
            exists(join(o_dir, 'eigenvector_2.npy')) and
            exists(join(o_dir, 'eigenvector_3.npy')) and
            exists(join(o_dir, 'eigenvalues.npy')) and
            not overwrite):
            return None

        gradU = self.load_field('grad(U)', time)
        N = len(gradU)
        W = zeros((N, 12))

        # for i in range(N):
        #     W[i] = local_eigensystem(gradU[i])

        for i in range(N):
            W[i] = eigvecs(gradU[i])

        save(join(o_dir, 'eigenvector_1.npy'), W[:,  0:3])
        save(join(o_dir, 'eigenvector_2.npy'), W[:,  3:6])
        save(join(o_dir, 'eigenvector_3.npy'), W[:,  6:9])
        save(join(o_dir,   'eigenvalues.npy'), W[:, 9:12])
        print('\t\tsaving eigenvector_#.npy')
        del W

        return None

    def calc_eigprojection(self, time, overwrite=False):
        print('\tCalculating projection against eigenvectors...')
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'eigvec_1_projection.npy')) and
            exists(join(o_dir, 'eigvec_2_projection.npy')) and
            exists(join(o_dir, 'eigvec_3_projection.npy')) and
            not overwrite):
            return None

        dS = self.load_post_field('dSigma.npy', time)
        E1 = self.load_post_field('eigenvector_1.npy', time)
        E2 = self.load_post_field('eigenvector_2.npy', time)
        E3 = self.load_post_field('eigenvector_3.npy', time)

        S = sum(norm(dS, axis=1))
        save(join(o_dir, 'contact_surface_area.npy'), S)
        pE1 = sum(abs(sum(E1 * dS, axis=1)))
        pE2 = sum(abs(sum(E2 * dS, axis=1)))
        pE3 = sum(abs(sum(E3 * dS, axis=1)))

        save(join(o_dir, 'eigvec_1_projection.npy'), pE1)
        save(join(o_dir, 'eigvec_2_projection.npy'), pE2)
        save(join(o_dir, 'eigvec_3_projection.npy'), pE3)
        return pE1, pE2, pE3

    def calc_topology_contact_surface(self, time, overwrite=False):
        print('\tCalculating topology classification at the surface...')
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'topology_surface_area.npy')) and not overwrite:
            return None

        TC = self.load_post_field('classification.npz', time)['arr_0']
        dS = norm(self.load_post_field('dSigma.npy', time), axis=1)

        save(join(o_dir, 'topology_surface_area.npy'),
             array([dot(TC == i, dS) for i in range(4)]))
        return None

    def calc_topology_diffusive(self, time, overwrite=False):
        print('\tCalculating topology classification in the diffusive region...')
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'topology_diffusive.npy')) and not overwrite:
            return None

        TC = self.load_post_field('classification.npz', time)['arr_0']
        dV = self.load_post_field('scalar_dissipation_density.npy', time)

        save(join(o_dir, 'topology_diffusive.npy'),
             self.diffusivity * array([dot(TC == i, dV) for i in range(4)]))
        return None

    def calc_topology_mixture_volume(self, time, overwrite=False):
        print('\tCalculating topology classification at the mixing volume...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'topology_mixing.npy')) and not overwrite:
            return None

        TC = self.load_post_field('classification.npz', time)['arr_0']

        alpha1 = self.load_field('alpha.pregel', t_dir)
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()

        dV = 4 * alpha1 * alpha2 * self.V

        save(join(o_dir, 'topology_mixing.npy'),
             array([dot(TC == i, dV) for i in range(4)]))
        return None

    def calc_topology_viscous(self, time, overwrite=False):
        print('\tCalculating topology classification in viscous dissipation region...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'topology_viscous.npy')) and not overwrite:
            return None

        if not self.mesh_loaded:
            self.load_mesh()

        TC = self.load_post_field('classification.npz', time)['arr_0']
        eps = self.load_post_field('visc_dissipation_density.npy', time)

        alpha1 = self.load_field('alpha.pregel', t_dir)
        eta1 = self.transport_properties['pregel']['viscosity'] \
            * self.transport_properties['pregel']['density']

        alpha2 = self.load_field('alpha.crosslinker', t_dir)
        eta2 = self.transport_properties['crosslinker']['viscosity'] \
            * self.transport_properties['crosslinker']['density']
        dEps = (eta1 * alpha1 + eta2 * alpha2) * eps * self.V

        save(join(o_dir, 'topology_viscous.npy'),
             array([dot(TC == i, dEps) for i in range(4)]))
        return None


    def calc_vortprojection(self, time, overwrite=False):
        print('\tCalculating projection against vorticity...')
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
        print('\tCalculating surface energy...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'surface_energy.npy')) and not overwrite:
            return None

        gradAlpha1 = self.load_field('grad(alpha.pregel)', t_dir)
        gradAlpha2 = self.load_field('grad(alpha.crosslinker)', t_dir)

        if not self.mesh_loaded:
            self.load_mesh()
        w = self.R[:, 2] >= 0.0002 * ones_like(self.R[:, 2])

        save(join(o_dir, 'surface_energy.npy'),
             self.surface_tension * dot(self.V * w, norm(gradAlpha1 + gradAlpha2, axis=1)))
        return None

    def calc_kinetic_energy(self, time, overwrite=False):
        r"""
        Calculates the total kinetic energy at a paritcular time.
        It is the discretised volume integral

        \[
            K = \frac{1}{2} \int_\Omega \rho |\bm{u}|^2 dV,
        \]

        where $\rho$ and $\bm{u}$ are the density and velocity fields.

        Params:
        ======

        time (str):
        The time of the snapshot (as written by OpenFOAM).

        overwrite (bool, optional):
        Whereas to recalculate and dump the resulting value.

        Returns:
        =======
        None, but produces the file `kinetic_energy.npy` which is a single
        number.

        """
        print('\tCalculating kinetic energy...')
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

    def calc_angular_momentum(self, time, overwrite=False):
        print('\tCalculating angular_momentum...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'angular_momentum.npy')) and not overwrite:
            return None

        if not self.mesh_loaded:
            self.load_mesh()

        U = self.load_field('U', t_dir)

        rho1 = self.transport_properties['pregel']['density']
        alpha1 = self.load_field('alpha.pregel', t_dir)

        rho2 = self.transport_properties['crosslinker']['density']
        alpha2 = self.load_field('alpha.crosslinker', t_dir)

        M = (rho1 * alpha1 + rho2 * alpha2) * self.V
        RcrossU = cross(self.R, U, axis=1)
        L = sum(prod(M, RcrossU), axis=0)

        save(join(o_dir, 'angular_momentum.npy'), L)
        return None

    def calc_QR_histograms(self, time, overwrite=False):
        print('\tCalculating QR joint histograms...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'QR_hist_bins.npy')) and
            exists(join(o_dir, 'QR_histogram_g.npy')) and
            exists(join(o_dir, 'QR_histogram_l.npy')) and
            not overwrite):
            return None

        if not self.mesh_loaded:
            self.load_mesh()

        bins = 128
        extent = [[-1e12, 1e12],
                  [-3e18 / 2 ** (2 / 3), 3e18 / 2 ** (2 / 3)]]

        alpha = self.load_field('alpha.pregel', t_dir) \
            + self.load_field('alpha.crosslinker', t_dir)
        Q = self.load_post_field('Q.npy', time)
        R = self.load_post_field('R.npy', time)

        tol = 1e-2
        idx_gas = alpha < ones_like(alpha) * tol
        idx_liq = alpha > ones_like(alpha) * (1.0 - tol)

        P_gas, X, Y = histogram2d(R[idx_gas], Q[idx_gas], weights=self.V[idx_gas],
                                  bins=bins, range=extent)

        P_liq, X, Y = histogram2d(R[idx_liq], Q[idx_liq], weights=self.V[idx_liq],
                                  bins=bins, range=extent)

        save(join(o_dir, 'QR_hist_bins.npy'), [X, Y])
        save(join(o_dir, 'QR_histogram_g.npy'), P_gas)
        save(join(o_dir, 'QR_histogram_l.npy'), P_liq)

        return None

    def calc_enstrophy_histogram(self, time, overwrite=False):
        print('\tCalculating enstrophy histogram...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if exists(join(o_dir, 'enstrophy_histogram.npy')) and not overwrite:
            return None

        bins = 128
        bins_ = logspace(0, 10, bins)
        sbins = sqrt(bins_[:-1] * bins_[1:])
        db = bins_[1:] - bins_[:-1]


        if not self.mesh_loaded:
            self.load_mesh()

        alpha = self.load_field('alpha.pregel', t_dir) \
            + self.load_field('alpha.crosslinker', t_dir)
        xi = self.load_post_field('enstrophy.npy', time)

        tol = 1e-2
        idx_gas = (alpha < ones_like(alpha) * tol) \
            * (sqrt(self.R[:, 0]**2 + self.R[:, 1]**2) < 3.5e-3 * ones_like(alpha))
        idx_gas = idx_gas.astype(bool)
        idx_liq = alpha > ones_like(alpha) * (1.0 - tol)

        pdf_gas, _ = histogram(xi[idx_gas], bins=bins_, weights=self.V[idx_gas])
        pdf_liq, _ = histogram(xi[idx_liq], bins=bins_, weights=self.V[idx_liq])

        save(join(o_dir, 'enstrophy_histogram.npy'),
             vstack((sbins, pdf_gas / db, pdf_liq / db)))
        return None

    def calc_topo_dissip_histogram(self, time, overwrite=False, bins=16):
        print('\tCalculating topology-dissipation histogram...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'topology_dissip_histogram_0.npy')) and
            exists(join(o_dir, 'topology_dissip_histogram_1.npy')) and
            exists(join(o_dir, 'topology_dissip_histogram_2.npy')) and
            exists(join(o_dir, 'topology_dissip_histogram_3.npy')) and
            not overwrite):
            return None

        if not self.mesh_loaded:
            self.load_mesh()

        dS = norm(self.load_post_field('dSigma.npy', time), axis=1)
        dS /= sum(dS)
        TC = self.load_post_field('classification.npz', time)['arr_0']
        eps_D = self.load_post_field('scalar_dissipation_density.npy', time)
        eps_D *= self.diffusivity / self.V 
       
        W = linspace(0, 192.0, bins + 1)
        Bins = 0.5 * (W[1:] + W[:-1])
        dB = W[1] - W[0]

        for i in range(4):
            fltr = (TC == i * ones_like(TC, dtype=int))
            Hist, _ = histogram(eps_D[fltr], bins=W, weights=dS[fltr] / dB)
            save(join(o_dir, f'topology_dissip_histogram_{i}.npy'),
                 [Hist, Bins]) 
        Hist, _ = histogram(eps_D, bins=W, weights=dS / dB)
        save(join(o_dir, 'dissipation_histogram.npy'),
             [Hist, Bins])
        return None

    def calc_enstrophy_diff_dissip_histogram(self, time, overwrite=False, bins=16):
        print('\tCalculating enstrophy-dissipation histogram...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)
        
        if (exists(join(o_dir, 'enstrophy_dissip_histogram.npy')) and
            not overwrite):
            return None

        if not self.mesh_loaded:
            self.load_mesh()

        xi = self.load_post_field('enstrophy.npy', time)
        eps_D = self.load_post_field('scalar_dissipation_density.npy', time)
        eps_D *= self.diffusivity / self.V

        bins_lxi = linspace(-13, 11, bins + 1)
        bins_eps = linspace(0, 192.0, bins + 1)

        H, X, Y = histogram2d(log10(xi), eps_D, density=True,
                              bins=[bins_lxi, bins_eps])
        X = 0.5 * (X[1:] + X[:-1])
        Y = 0.5 * (Y[1:] + Y[:-1])
        XX, YY = meshgrid(X, Y)

        save(join(o_dir, 'enstrophy_dissip_histogram.npy'),
             [H, XX, YY])
        return None

    def calc_eigenvec_eps_histograms(self, time, overwrite=False, bins=64):
        print('\tCalculating eigenvector proj. dissipation histograms...')
        t_dir = join(self.case_dir, time)
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)

        if (exists(join(o_dir, 'n-dot-e_eps_hist2d.npy')) and
            exists(join(o_dir, 'avgs_ndote_eps.npy')) and
            not overwrite):
            return None

        if not self.mesh_loaded:
            self.load_mesh()
        
        u0 = 1.2
        tau = 2 * self.Rnozzle / u0

        eps_D = self.load_post_field('scalar_dissipation_density.npy', time)
        eps_D = self.diffusivity * abs(eps_D) / self.V

        dS = self.load_post_field('dSigma.npy', time)
        ds = norm(dS, axis=1)
        ds /= sum(ds)

        bins_e = linspace(0, 1, bins + 1)
        bins_d = linspace(0, 0.1 / tau, bins + 1)
        X, Y = meshgrid(bins_e, bins_d)
        aveD = sum(eps_D * ds)

        avE = [[]] * 3
        H = [[]] * 3

        for i in range(3):
            E = self.load_post_field(f'eigenvector_{i+1}.npy', time)
            pE = abs(sum(E * normalise(dS), axis=1))
            avE[i] = sum(pE * ds)
            H[i], _, _ = histogram2d(pE, eps_D, bins=[bins_e, bins_d],
                                     weights=ds, density=True)
            H[i] = H[i].transpose()

        save(join(o_dir, 'n-dot-e_eps_hist2d.npy'), [*H, X, Y])
        save(join(o_dir, 'avgs_ndote_eps.npy'), [*avE, aveD])
        return None

    def clean(self, time):
        print('\tCleaning up...')
        o_dir = join(self.out_dir, time)

        file_list = [
            'vorticity.npy',
            'enstrophy.npy',
            'Q.npy',
            'R.npy',
            'dSigma.npy',
            'eigenvalues.npy',
            'eigenvector_1.npy',
            'eigenvector_2.npy',
            'eigenvector_3.npy',
            'visc_dissipation_density.npy',
            'classification.npz',
            'scalar_dissipation_density.npy'
        ]
        for fl in file_list:
            try:
                remove(join(o_dir, fl))
                print(f'\t\tRemoved {fl}')
            except FileNotFoundError:
                print(f'\t\t{fl} missing')
        return None

    def check_done(self, time):
        o_dir = join(self.out_dir, time)
        makedirs(o_dir, exist_ok=True)
        return exists(join(o_dir, 'end.lck'))

    def measureAll(self, time, overwrite=False, clean=False):
        print('Time: ', time)
        o_dir = join(self.out_dir, time)

        if overwrite or not self.check_done(time):
            clock_times = []
            run_funcs = []

            clock_times.append(tm.time())

            self.calc_droplet_volumes(time, overwrite=overwrite)
            run_funcs.append('calc_droplet_volumes')
            clock_times.append(tm.time())

            self.calc_Xcm(time, overwrite=overwrite)
            run_funcs.append('calc_Xcm')
            clock_times.append(tm.time())

            self.calc_Ucm(time, overwrite=overwrite)
            run_funcs.append('calc_Ucm')
            clock_times.append(tm.time())

            self.calc_impact_parameter(time, overwrite=overwrite)
            run_funcs.append('calc_impact_parameter')
            clock_times.append(tm.time())

            self.calc_Reynolds(time, overwrite=overwrite)
            run_funcs.append('calc_Reynolds')
            clock_times.append(tm.time())

            self.calc_Weber(time, overwrite=overwrite)
            run_funcs.append('calc_Weber')
            clock_times.append(tm.time())

            self.calc_dSigma(time, overwrite=overwrite)
            run_funcs.append('calc_dSigma')
            clock_times.append(tm.time())

            if self.old_style:
                self.calc_vorticity(time, overwrite=overwrite)
                run_funcs.append('calc_vorticity')
                clock_times.append(tm.time())

                self.calc_enstrophy(time, overwrite=overwrite)
                run_funcs.append('calc_enstrophy')
                clock_times.append(tm.time())

                self.calc_Q(time, overwrite=overwrite)
                run_funcs.append('calc_Q')
                clock_times.append(tm.time())

                self.calc_R(time, overwrite=overwrite)
                run_funcs.append('calc_R')
                clock_times.append(tm.time())

                self.calc_visc_dissipation_density(time, overwrite=overwrite)
                run_funcs.append('calc_visc_dissipation_density')
                clock_times.append(tm.time())

                self.calc_eigensystem(time, overwrite=overwrite)
                run_funcs.append('calc_eigensystem')
                clock_times.append(tm.time())

            else:
                self.calc_gradU_deriv(time, overwrite=overwrite)
                run_funcs.append('calc_gradU_deriv')
                clock_times.append(tm.time())

            self.calc_contact_area(time, overwrite=overwrite)
            run_funcs.append('calc_contact_area')
            clock_times.append(tm.time())

            self.calc_volume_mixture(time, overwrite=overwrite)
            run_funcs.append('calc_volume_mixture')
            clock_times.append(tm.time())

            self.calc_segregation(time, overwrite=overwrite)
            run_funcs.append('calc_segregation')
            clock_times.append(tm.time())

            self.calc_dissipation_density(time, overwrite=overwrite)
            run_funcs.append('calc_dissipation_density')
            clock_times.append(tm.time())

            self.calc_dissipation_rate(time, overwrite=overwrite)
            run_funcs.append('calc_dissipation_rate')
            clock_times.append(tm.time())

            self.calc_classification(time, overwrite=overwrite)
            run_funcs.append('calc_classification')
            clock_times.append(tm.time())

            self.calc_visc_dissipation(time, overwrite=overwrite)
            run_funcs.append('calc_visc_dissipation')
            clock_times.append(tm.time())

            self.calc_eigprojection(time, overwrite=overwrite)
            run_funcs.append('calc_eigprojection')
            clock_times.append(tm.time())

            self.calc_topology_contact_surface(time, overwrite=overwrite)
            run_funcs.append('calc_topology_contact_surface')
            clock_times.append(tm.time())

            self.calc_topology_diffusive(time, overwrite=overwrite)
            run_funcs.append('calc_topology_diffusive')
            clock_times.append(tm.time())

            self.calc_topology_mixture_volume(time, overwrite=overwrite)
            run_funcs.append('calc_topology_mixture_volume')
            clock_times.append(tm.time())

            self.calc_topology_viscous(time, overwrite=overwrite)
            run_funcs.append('calc_topology_viscous')
            clock_times.append(tm.time())

            self.calc_vortprojection(time, overwrite=overwrite)
            run_funcs.append('calc_vortprojection')
            clock_times.append(tm.time())

            self.calc_surface_energy(time, overwrite=overwrite)
            run_funcs.append('calc_surface_energy')
            clock_times.append(tm.time())

            self.calc_kinetic_energy(time, overwrite=overwrite)
            run_funcs.append('calc_kinetic_energy')
            clock_times.append(tm.time())

            self.calc_angular_momentum(time, overwrite=overwrite)
            run_funcs.append('calc_angular_momentum')
            clock_times.append(tm.time())

            self.calc_QR_histograms(time, overwrite=overwrite)
            run_funcs.append('calc_QR_histograms')
            clock_times.append(tm.time())

            self.calc_enstrophy_histogram(time, overwrite=overwrite)
            run_funcs.append('calc_enstrophy_histogram')
            clock_times.append(tm.time())

            self.calc_eigenvec_eps_histograms(time, overwrite=overwrite)
            run_funcs.append('calc_eigenvec_eps_histograms')
            clock_times.append(tm.time())

            print('\tWriting down the `end.lck` file...')
            with open(join(o_dir, 'end.lck'), 'wt') as ofile:
                ofile.write(f'Starting at: {clock_times[0]}\n\n')
                for i in range(len(run_funcs)):
                    ofile.write(f'{run_funcs[i]:s}: {clock_times[i + 1] - clock_times[i]}\n')
                ofile.write(f'Ended at: {clock_times[-1]}\n')

        if clean:
            self.clean(time)

        print('Done', time)
        return None
