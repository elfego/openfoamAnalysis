import sys
import numpy as np
from readopenfoamcase import readOFcase
import matplotlib.pylab as plt


vv = [r'\alpha', r'\beta', r'\gamma']
Cm = ['Reds', 'Greens', 'Blues']
C = ['r', 'g', 'b']


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

avE = [[], [], []]
H = [[], [], []]

rc.calc_eigenvec_eps_histograms(time)

H[0], H[1], H[2], X, Y = rc.load_post_field('n-dot-e_eps_hist2d.npy', time)
avE[0], avE[1], avE[2], aveD = rc.load_post_field('avgs_ndote_eps.npy', time)

Cm = ['Reds', 'Greens', 'Blues']
C = ['r', 'g', 'b']

fig = plt.figure(1, clear=True, figsize=[9.6, 3.2])
ax = fig.subplots(1, 3, sharey=True)

for i in range(3):
    ax[i].pcolormesh(X, Y * tau, H[i], cmap=Cm[i])
    ax[i].scatter(avE[i], aveD * tau, color=C[i], s=60,
                  marker='o', facecolor='none')

for i, a in enumerate(ax):
    a.set_xlabel(r'$|\hat{\mathbf{n}} \cdot \hat{\mathbf{e}}_' + vv[i] + r'|$')
    a.set_xlim((0, 1))
    a.set_xticks([0, 0.5, 1])
ax[0].set_ylabel(r'$\dot{\epsilon}_D \, \cdot R_{\mathrm{n}} / U_{\mathrm{n}}$')

fig.tight_layout()

