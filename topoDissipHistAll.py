import sys
from os.path import join
from os import makedirs
import numpy as np
from matplotlib import use; use('Agg')
import matplotlib.pylab as plt

from readopenfoamcase import readOFcase
from scipy.interpolate import interp1d
from scipy.integrate import trapz

diffusivity = 2e-07
R_nozzle = 2.5e-04
U_nozzle = 1.2
Tau = 2 * R_nozzle / U_nozzle
colours = [3, 2, 0, 1]

case = sys.argv[1]

rc = readOFcase(case)
rc.set_nozzle_radius(2.5e-04)

for time in rc.times[20:]:
    print(f"Time {time}") 
    rc.calc_topo_dissip_histogram(time, bins=16)
    
    TDH = []
    for i in range(4):
        TDH.append(rc.load_post_field(f'topology_dissip_histogram_{i}.npy',
                   time))
    
    fig = plt.figure(1, clear=True)
    ax = fig.subplots(1, 1)
    ax.set_xlabel(r'Diffusive dissipation density, $\dot{\epsilon}_D \; (U_{nozzle} / D_{nozzle})$')
    ax.set_ylabel(r'Probability density')
    
    
    for i, l in enumerate([3, 2, 0, 1]):
        Hist, Bins = TDH[l]
        avEps_D = trapz(Hist * Bins, Bins)
        Hist /= Tau
        Bins *= Tau
    
        x = np.linspace(Bins[0], Bins[-1], 512 + 1)
        HistF = interp1d(Bins, Hist, kind='nearest',
                fill_value='extrapolate')
        ax.plot(x, HistF(x), '-', c=f'C{i}')
    
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0, top=80)
    fig.tight_layout()
    ofile = join(rc.case_dir, 'pics', time)
    makedirs(ofile, exist_ok=True)
    ofile = join(ofile, 'topology_dissip_histogram.png')
    print(f'\t\tSaving {ofile}...')
    fig.savefig(ofile)

