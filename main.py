#!/bin/env python

import sys
from readopenfoamcase import readOFcase


def main():
    rc = readOFcase(sys.argv[1])
    rc.set_nozzle_radius(2.5e-4)

    rc.forAllTimes(rc.calc_droplet_volumes)
    rc.forAllTimes(rc.calc_Xcm)
    rc.forAllTimes(rc.calc_Ucm)
    rc.forAllTimes(rc.calc_impact_parameter)
    rc.forAllTimes(rc.calc_Reynolds)
    rc.forAllTimes(rc.calc_Weber)
    rc.forAllTimes(rc.calc_dSigma)
    rc.forAllTimes(rc.calc_contact_area)
    rc.forAllTimes(rc.calc_volume_mixture)
    rc.forAllTimes(rc.calc_dissipation_rate)
    rc.forAllTimes(rc.calc_R)
    rc.forAllTimes(rc.calc_classification)
    rc.forAllTimes(rc.calc_visc_dissipation_density)
    rc.forAllTimes(rc.calc_eigensystem)
    rc.forAllTimes(rc.calc_eigprojection)
    rc.forAllTimes(rc.calc_topology_contact_surface)


if __name__ == '__main__':
    main()

