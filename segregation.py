import sys
from readopenfoamcase import readOFcase
from parseargs import parseargs


def main():
    ops = parseargs(sys.argv)
    R_nozzle = 2.5e-04
    rc = readOFcase(ops['case'])
    rc.set_nozzle_radius(R_nozzle)

    if ops['time'] is None:
        rc.forAllTimes(rc.calc_segregation,
                       interval=ops['interval'],
                       clean=ops['clean'],
                       overwrite=ops['overwrite'])
    else:
        rc.calc_segregation(ops['time'],
                            overwrite=ops['overwrite'],
                            clean=ops['clean'])

    # rc.forAllTimes(rc.calc_segregation)


if __name__ == '__main__':
    main()

