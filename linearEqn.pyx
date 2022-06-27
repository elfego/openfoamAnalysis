import cython


def evalLinearPolynomial(double x, double a, double b) -> double:
    return a * x + b


cdef double _linearEqnRoots(double a, double b):
    return -b / a


def linearEqnRoots(a, b):
    return _linearEqnRoots(a, b)

