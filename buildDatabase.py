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
    'mixing intensity': [],
    'scalar dissipation rate': [],
    'visc dissipation': [],
    'contact surface area': [],
    'velocity centre of mass': [],
    'velocity pregel':, [],
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
    'most expansive proj': [],
    'most compressive proj': [],
    'vort proj': [],
    'surface energy': [],
    'kinetic energy': []
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
    U2 = np.load(join(read_dir, 'U.crosslinker'))
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

    tmp = np.load(join(read_dir, 'scalarDissipationRate.npy'))
    preDatabase['scalar dissipation rate'].append(tmp)

    tmp = np.load(join(read_dir, 'visc_dissipation.npy'))
    preDatabase['visc dissipation'].append(tmp)

    tmp = np.load(join(read_dir, 'surface_area_topology.npy'))
    preDatabase['topology surf area 0'].append(tmp[0] / Sc)
    preDatabase['topology surf area 1'].append(tmp[1] / Sc)
    preDatabase['topology surf area 2'].append(tmp[2] / Sc)
    preDatabase['topology surf area 3'].append(tmp[3] / Sc)

    tmp = np.load(join(read_dir, 'eigvec_1_projection.npy'))
    preDatabase['most expansive proj'].append(tmp / Sc)

    tmp = np.load(join(read_dir, 'eigvec_3_projection.npy'))
    preDatabase['most compressive proj'].append(tmp / Sc)

    tmp = np.load(join(read_dir, 'w_dot_n.npy'))
    preDatabase['vort proj'].append(tmp / Sc)

    tmp = np.load(join(read_dir, 'surface_energy.npy'))
    preDatabase['surface energy'].append(tmp)

    tmp = np.load(join(read_dir, 'kinetic_energy.npy'))
    preDatabase['kinetic energy'].append(tmp)

print("Done extracting the values. Building database.")
df = pd.DataFrame(preDatabase)

print("Saving database as",
      join(case_dir, 'postProcessing', 'measurements'))

df.to_csv(   join(case_dir, 'postProcessing', 'measurements.csv'))
df.to_pickle(join(case_dir, 'postProcessing', 'measurements.pkl'))

