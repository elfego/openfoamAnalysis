import sys
from os.path import join
from os import makedirs
import numpy as np
from matplotlib import use; use('Agg')
import matplotlib.pylab as plt
from readopenfoamcase import readOFcase


R_nozzle = 2.5e-04
U_nozzle = 1.2
Tau = 2 * R_nozzle / U_nozzle
case = sys.argv[1]

rc = readOFcase(case)
rc.set_nozzle_radius(R_nozzle)

for time in rc.times[20:]:
    print(f'Time {time}')
    
    rc.calc_enstrophy_diff_dissip_histogram(time, bins=32, overwrite=True)
    H, X, Y = rc.load_post_field('enstrophy_dissip_histogram.npy', time)
    
    fig = plt.figure(1, clear=True)
    ax = fig.subplots(1, 1)
    
    xmin, xmax = np.nanmin(10**X), np.nanmax(10**X)
    
    pcm = ax.pcolormesh(10**X * Tau**2, Y * Tau,
                        np.log(H / Tau**3), cmap='Greys')
    fig.colorbar(pcm, label=r'Probability density, log$\,P$')
    ax.set_xscale('log')
    ax.set_xlim(10 ** (-18), 10 ** 2)
    
    ax.set_xlabel(r'Enstrophy, $\xi \cdot D_{nozzle}^2 / U_{nozzle}^2$')
    ax.set_ylabel(r'Diffusive dissipation density, $\dot{\epsilon}_D \cdot D_{nozzle} / U_{nozzle}$')
    
    fig.tight_layout()
    
    ofile = join(rc.case_dir, 'pics', time)
    makedirs(ofile, exist_ok=True)
    ofile = join(ofile, 'enstrophy-dissip_histogram.png')
    print(f'\t\tSaving {ofile}...')
    fig.savefig(ofile)
