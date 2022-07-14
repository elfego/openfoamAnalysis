#!/usr/bin/env python

from os import makedirs
from os.path import join, exists
import sys
import pandas as pd
import numpy as np


def list_time_dirs(case_dir):
    sys_dir = join(case_dir, 'system')
    with open(join(sys_dir, 'controlDict'), 'r') as handler:
        for ln in handler.readlines():
            ln2 = ln.split(' ')
            if ln2[0] == 'startTime':
                startTime = float(ln2[1][:-2])
            if ln2[0] == 'endTime':
                endTime = float(ln2[1][:-2])
            if ln2[0] == 'writeInterval':
                writeInterval = float(ln2[1][:-2])

    frames = np.arange(startTime, endTime, writeInterval) + writeInterval
    t_dirs = [f'{t:g}' for t in frames]

    return t_dirs


case_dir = sys.argv[1]

times = list_time_dirs(case_dir)

preDatabase = {
    'time': [],
    'impact parameter': [],
    'mixture volume': [],
    'segregation': [],
    'mixing intensity': [],
    'scalar dissipation rate': [],
    'visc dissipation': [],
    'contact surface area': [],
    'velocity centre of mass': [],
    'velocity pregel': [],
    'velocity crosslinker': [],
    'Ur': [],
    'volume crosslinker': [],
    'volume pregel': [],
    'We': [],
    'Re': [],
    'topology surf area 0': [],
    'topology surf area 1': [],
    'topology surf area 2': [],
    'topology surf area 3': [],
    'topology diffusive 0': [],
    'topology diffusive 1': [],
    'topology diffusive 2': [],
    'topology diffusive 3': [],
    'topology mix vol 0': [],
    'topology mix vol 1': [],
    'topology mix vol 2': [],
    'topology mix vol 3': [],
    'topology viscous 0': [],
    'topology viscous 1': [],
    'topology viscous 2': [],
    'topology viscous 3': [],
    'eigvec 1 proj': [],
    'eigvec 2 proj': [],
    'eigvec 3 proj': [],
    'vort proj': [],
    'surface energy': [],
    'kinetic energy': [],
    'angular momentum': []
}

print("Iterating over the results.")

for time in times:
    print(time)
    read_dir = join(case_dir, 'postProcessing', time)
    if not exists(read_dir):
        print('  ---->  Skipping')
        continue

    preDatabase['time'].append(float(time))

    tmp = np.load(join(read_dir, 'impact_param.npy'))
    preDatabase['impact parameter'].append(tmp[1])

    Sc = np.load(join(read_dir, 'contact_surface_area.npy'))
    preDatabase['contact surface area'].append(Sc)

    tmp = np.load(join(read_dir, 'Ucm.npy'))
    preDatabase['velocity centre of mass'].append(np.linalg.norm(tmp))

    U1 = np.load(join(read_dir, 'U.pregel.npy'))
    U2 = np.load(join(read_dir, 'U.crosslinker.npy'))
    preDatabase['velocity pregel'].append(np.linalg.norm(U1))
    preDatabase['velocity crosslinker'].append(np.linalg.norm(U2))
    preDatabase['Ur'].append(np.linalg.norm(U2 - U1))

    V1 = np.load(join(read_dir, 'V.crosslinker.npy'))
    V2 = np.load(join(read_dir, 'V.pregel.npy'))
    preDatabase['volume crosslinker'].append(V1)
    preDatabase['volume pregel'].append(V2)

    tmp = np.load(join(read_dir, 'Re_collision.npy'))
    preDatabase['Re'].append(tmp)

    tmp = np.load(join(read_dir, 'We_collision.npy'))
    preDatabase['We'].append(tmp)

    Vm = np.load(join(read_dir, 'mixtureVolume.npy'))
    preDatabase['mixture volume'].append(Vm / (V1 + V2))
    preDatabase['mixing intensity'].append(1 - np.sqrt(1 - Vm / (V1 + V2)))

    tmp = np.load(join(read_dir, 'segregation.npy'))
    preDatabase['segregation'].append(tmp)

    eps_D = np.load(join(read_dir, 'scalar_dissipation_rate.npy'))
    preDatabase['scalar dissipation rate'].append(eps_D)

    eps_mu = np.load(join(read_dir, 'visc_dissipation.npy'))
    preDatabase['visc dissipation'].append(eps_mu)

    tmp = np.load(join(read_dir, 'topology_surface_area.npy'))
    for i in range(4):
        k = f'topology surf area {i:d}'
        preDatabase[k].append(tmp[i] / Sc)

    tmp = np.load(join(read_dir, 'topology_diffusive.npy'))
    for i in range(4):
        k = f'topology diffusive {i:d}'
        preDatabase[k].append(tmp[i] / eps_D)

    tmp = np.load(join(read_dir, 'topology_mixing.npy'))
    for i in range(4):
        k = f'topology mix vol {i:d}'
        preDatabase[k].append(tmp[i] / Vm)

    tmp = np.load(join(read_dir, 'topology_viscous.npy'))
    for i in range(4):
        k = f'topology viscous {i:d}'
        preDatabase[k].append(tmp[i] / eps_mu)

    for i in range(1, 4):
        tmp = np.load(join(read_dir, f'eigvec_{i:d}_projection.npy'))
        preDatabase[f'eigvec {i:d} proj'].append(tmp / Sc)

    tmp = np.load(join(read_dir, 'w_dot_n.npy'))
    preDatabase['vort proj'].append(tmp / Sc)

    tmp = np.load(join(read_dir, 'surface_energy.npy'))
    preDatabase['surface energy'].append(tmp)

    tmp = np.load(join(read_dir, 'kinetic_energy.npy'))
    preDatabase['kinetic energy'].append(tmp)

    tmp = np.load(join(read_dir, 'angular_momentum.npy'))
    preDatabase['angular momentum'].append(np.linalg.norm(tmp))

print("Done extracting the values. Building database.")
df = pd.DataFrame(preDatabase)

savefile = join(case_dir, 'measurements.pkl')
print("Saving database as", savefile)

df.to_csv(savefile[:-3] + 'csv')
df.to_pickle(savefile)

