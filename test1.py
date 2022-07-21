import sys
import numpy as np
from readopenfoamcase import readOFcase
import matplotlib.pylab as plt


vv = [r'\alpha', r'\beta', r'\gamma']

plt.ion()


def normalise(v):
    N = len(v)
    L = np.linalg.norm(v, axis=1)
    w = np.zeros_like(v)
    for idx in range(N):
        if L[idx] > 1e-13:
            w[idx] = v[idx] / L[idx]
    return w


case = sys.argv[1]
rc = readOFcase(case)
R_nozzle = 2.5e-4
u0 = 1.2
tau = 2 * R_nozzle / u0
rc.set_nozzle_radius(R_nozzle)

time = rc.times[int(sys.argv[2])]
print(time)

# alpha3 = rc.load_field('alpha.air', time)

if not rc.mesh_loaded:
    rc.load_mesh()

rc.calc_dissipation_density(time)
eps_D = rc.diffusivity * abs(rc.load_post_field('scalar_dissipation_density.npy', time)) / rc.V
rc.calc_dSigma(time)
dS = rc.load_post_field('dSigma.npy', time)

ds = np.linalg.norm(dS, axis=1)
ds /= np.sum(ds)

bins = 64
bins_e = np.linspace(0, 1, bins + 1)
dx = bins_e[1] - bins_e[0]
bins_d = np.linspace(0, 0.1 / tau, bins + 1)
dy = bins_d[1] - bins_d[0]
X, Y = np.meshgrid(bins_e, bins_d)
X_, Y_ = np.meshgrid(0.5 * (bins_e[1:] + bins_e[:-1]),
                     0.5 * (bins_d[1:] + bins_d[:-1]))

rc.calc_eigensystem(time)
E1 = rc.load_post_field('eigenvector_1.npy', time)
E2 = rc.load_post_field('eigenvector_2.npy', time)
E3 = rc.load_post_field('eigenvector_3.npy', time)

pE1 = abs(np.sum(E1 * normalise(dS), axis=1))
pE2 = abs(np.sum(E2 * normalise(dS), axis=1))
pE3 = abs(np.sum(E3 * normalise(dS), axis=1))

fig = plt.figure(1, clear=True, figsize=[9.6, 3.2])
ax = fig.subplots(1, 3, sharey=True)

H1, _, _ = np.histogram2d(pE1, eps_D, bins=[bins_e, bins_d],
                          weights=ds, density=True)
H1 = H1.transpose()
ax[0].pcolormesh(X, Y * tau, H1, cmap='Reds')
# ax[0].plot(np.sum(H1 * X_) * dx * dy,
           # np.sum(H1 * Y_ * tau) * dx * dy,
           # 'ko')

H2, _, _ = np.histogram2d(pE2, eps_D, bins=[bins_e, bins_d],
                          weights=ds, density=True)
H2 = H2.transpose()
ax[1].pcolormesh(X, Y * tau, H2, cmap='Greens')
# ax[1].plot(np.sum(H2 * X_) * dx * dy,
           # np.sum(H2 * Y_ * tau) * dx * dy,
           # 'ko')

H3, Bx, By = np.histogram2d(pE3, eps_D, bins=[bins_e, bins_d],
                            weights=ds, density=True)
H3 = H3.transpose()
ax[2].pcolormesh(X, Y * tau, H3, cmap='Blues')
# ax[2].plot(np.sum(H3 * X_) * dx * dy,
           # np.sum(H3 * Y_ * tau) * dx * dy,
           # 'ko')

for i, a in enumerate(ax):
    a.set_xlabel(r'$|\hat{\mathbf{n}} \cdot \hat{\mathbf{e}}_' + vv[i] + r'|$')
    a.set_xlim((0, 1))
    a.set_xticks([0, 0.5, 1])
ax[0].set_ylabel(r'$\dot{\epsilon}_D \, \cdot R_{\mathrm{n}} / U_{\mathrm{n}}$')

fig.tight_layout()

# plt.ylabel(r'acos$|\hat{\mathbf{n}} \cdot \hat{\mathbf{e}}_\gamma|$ (deg)')
