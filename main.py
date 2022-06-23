#!/bin/env python

import sys
from readopenfoamcase import readOFcase


def main():
    case = sys.argv[1]
    interval = None
    time = None
    clean = False
    ow = False

    for i in range(2, len(sys.argv)):
        if '--range' == sys.argv[i] and len(sys.argv) >= i + 1:
            interval = list(map(int, sys.argv[i + 1].split(':')))
        if '--time' == sys.argv[i] and len(sys.argv) >= i + 1:
            time = sys.argv[i + 1]
        if '--clean' == sys.argv[i]:
            clean = True
        if '--overwrite' == sys.argv[i]:
            ow = True

    rc = readOFcase(case)
    rc.set_nozzle_radius(2.5e-4)

    if time is None:
        rc.forAllTimes(rc.measureAll, interval=interval, clean=clean, overwrite=ow)
    else:
        rc.measureAll(time, overwrite=ow, clean=clean)

    # rc.forAllTimes(rc.calc_vorticity, interval=interval)
    # rc.forAllTimes(rc.calc_enstrophy, interval=interval)
    # rc.forAllTimes(rc.calc_Q, interval=interval)
    # rc.forAllTimes(rc.calc_droplet_volumes, interval=interval) #
    # rc.forAllTimes(rc.calc_Xcm, interval=interval)
    # rc.forAllTimes(rc.calc_Ucm, interval=interval) #
    # rc.forAllTimes(rc.calc_impact_parameter, interval=interval) #
    # rc.forAllTimes(rc.calc_Reynolds, interval=interval) #
    # rc.forAllTimes(rc.calc_Weber, interval=interval) #
    # rc.forAllTimes(rc.calc_dSigma, interval=interval)
    # rc.forAllTimes(rc.calc_contact_area, interval=interval) #
    # rc.forAllTimes(rc.calc_volume_mixture, interval=interval) #
    # rc.forAllTimes(rc.calc_dissipation_rate, interval=interval) #
    # rc.forAllTimes(rc.calc_R, interval=interval)
    # rc.forAllTimes(rc.calc_classification, interval=interval)
    # rc.forAllTimes(rc.calc_visc_dissipation_density, interval=interval)
    # rc.forAllTimes(rc.calc_visc_dissipation, interval=interval) #
    # rc.forAllTimes(rc.calc_eigensystem, interval=interval)
    # rc.forAllTimes(rc.calc_eigprojection, interval=interval) #
    # rc.forAllTimes(rc.calc_topology_contact_surface, interval=interval) #
    # rc.forAllTimes(rc.calc_vortprojection, interval=interval) #
    # rc.forAllTimes(rc.calc_surface_energy, interval=interval) #
    # rc.forAllTimes(rc.calc_kinetic_energy, interval=interval) #


if __name__ == '__main__':
    main()

