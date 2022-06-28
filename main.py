#!/bin/env python

import sys
from readopenfoamcase import readOFcase


def main():
    case = sys.argv[1]
    interval = None
    time = None
    clean = False
    ow = False
    oldstyle = False

    for i in range(2, len(sys.argv)):
        if '--range' == sys.argv[i] and len(sys.argv) >= i + 1:
            interval = list(map(int, sys.argv[i + 1].split(':')))
        if '--time' == sys.argv[i] and len(sys.argv) >= i + 1:
            time = sys.argv[i + 1]
        if '--clean' == sys.argv[i]:
            clean = True
        if '--overwrite' == sys.argv[i]:
            ow = True
        if '--old' == sys.argv[i]:
            oldstyle = True

    rc = readOFcase(case)
    rc.set_nozzle_radius(2.5e-4)
    if oldstyle:
        rc.set_oldstyle()

    if time is None:
        rc.forAllTimes(rc.measureAll, interval=interval,
                       clean=clean, overwrite=ow)
    else:
        rc.measureAll(time, overwrite=ow, clean=clean)


if __name__ == '__main__':
    main()

