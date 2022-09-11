#!/bin/env python

import sys
from readopenfoamcase import readOFcase
from parseargs import parseargs


def main():
    ops = parseargs(sys.argv)
    # case = sys.argv[1]
    # interval = None
    # time = None
    # clean = False
    # ow = False
    # oldstyle = False

    # for i in range(2, len(sys.argv)):
    #     if '--range' == sys.argv[i] and len(sys.argv) >= i + 1:
    #         interval = list(map(int, sys.argv[i + 1].split(':')))
    #     if '--time' == sys.argv[i] and len(sys.argv) >= i + 1:
    #         time = sys.argv[i + 1]
    #     if '--clean' == sys.argv[i]:
    #         clean = True
    #     if '--overwrite' == sys.argv[i]:
    #         ow = True
    #     if '--old' == sys.argv[i]:
    #         oldstyle = True

    rc = readOFcase(ops['case'])
    rc.set_nozzle_radius(2.5e-4)

    if ops['time'] is None:
        rc.forAllTimes(rc.measureAll, interval=ops['interval'],
                       clean=ops['clean'],
                       overwrite=ops['overwrite'])
    else:
        rc.measureAll(ops['time'],
                      overwrite=ops['overwrite'],
                      clean=ops['clean'])


if __name__ == '__main__':
    main()

