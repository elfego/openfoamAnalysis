import sys
from readopenfoamcase import readOFcase

R_nozzle = 2.5e-04
case = sys.argv[1]
rc = readOFcase(case)
rc.set_nozzle_radius(R_nozzle)

rc.forAllTimes(rc.calc_segregation)
