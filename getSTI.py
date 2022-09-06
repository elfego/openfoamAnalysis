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

    def _sti(time, overwrite=False, clean=False):

        if rc.check_done(time) and not overwrite:
            return None

        rc.calc_dSigma(time, overwrite=overwrite)
        rc.calc_scalar_turbulence_interaction_density(time,
                                                      overwrite=overwrite)
        rc.calc_scalar_turbulence_interaction(time,
                                              overwrite=overwrite)

    if time is None:
        rc.forAllTimes(_sti,
                       interval=interval,
                       clean=clean,
                       overwrite=ow)
    else:
        _sti(time, overwrite=ow, clean=clean)

    if clean:
        rc.clean(time)

    return None


if __name__ == '__main__':
    main()
