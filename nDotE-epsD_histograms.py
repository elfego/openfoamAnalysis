import sys
import numpy as np
from readopenfoamcase import readOFcase
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, PowerNorm


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
u0 = 1
tau = 2 * R_nozzle / u0
rc.set_nozzle_radius(R_nozzle)

time = rc.times[int(sys.argv[2])]
print(time)

avE = [[], [], []]
H = [[], [], []]

# rc.calc_eigenvec_eps_histograms(time, maxEd=0.3, bins=2)
# rc.calc_eigenvec_eps_histograms(time, maxEd=0.3, bins=128, overwrite=True)
rc.calc_eigenvec_eps_histograms(time, maxEd=0.3, bins=128)

H[0], H[1], H[2], X, Y = rc.load_post_field('n-dot-e_eps_hist2d.npy', time)
avE[0], avE[1], avE[2], aveD = rc.load_post_field('avgs_ndote_eps.npy', time)

Cm = ['Reds', 'Greens', 'Blues']
C = ['r', 'g', 'b']

fig = plt.figure(1, clear=True, figsize=np.r_[9.6, 3.2] * 0.6)
ax = fig.subplots(1, 3, sharey=True)

# vmin = min([np.nanmin(H[i][H[i] != 0]) for i in range(3)])
# print(vmin)
# vmax = max([np.nanmax(H[i]) for i in range(3)])
# print(vmax)
# vmin = 1e-6
# vmax = 1e-1
# vmin = None
# vmax = None
vmax = 0.03
vmin = 0

for i in range(3):
    c = ax[i].pcolormesh(X, Y * tau, H[i], cmap=Cm[i],
                         norm=PowerNorm(0.5, vmax=vmax))
    if i == 2:
        cb = fig.colorbar(c, ax=ax[i], ticks=[0, 0.01, 0.02, 0.03], label='PDF')
    else:
        cb = fig.colorbar(c, ax=ax[i], ticks=[])
    # ax[i].scatter(avE[i], aveD * tau, color=C[i], s=60,
                  # marker='o', facecolor='none')

for i, a in enumerate(ax):
    a.set_xlabel(r'$|\hat{\mathbf{n}} \cdot \hat{\mathbf{e}}_' + vv[i] + r'|$')
    a.set_xlim((0, 1))
    a.set_xticks([0, 0.5, 1])
    a.set_yticks([0, 0.1, 0.2, 0.3])
ax[0].set_ylabel(r'$\dot{\epsilon}_D \, \cdot R_{\mathrm{nzl}} / U_{\mathrm{nzl}}$')

fig.tight_layout()

tag = case.split('/')[-1]
N = H[0].shape[0]

if N < 10:
    fig.savefig('/tmp/frame.pdf')
else:
    fig.savefig(f'/tmp/fig_4_{tag}.png')
