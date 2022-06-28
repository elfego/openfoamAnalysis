import cython
from libc.math cimport sqrt, fabs, FP_NAN


def evalQuadraticPolynomial(double x, double a, double b, double c) -> double:
    return a * x**2 + b * x + c


cdef (double, double) _quadraticEqnRoots(double a, double b, double c):
    cdef double discr = b * b - 4.0 * a * c
    cdef double x_[2]

    if discr > 0:
        x_[0] = -0.5 * (b - sqrt(discr)) / a
        x_[1] = -0.5 * (b + sqrt(discr)) / a
    elif fabs(discr) < 1e-15:
        x_[0] = -0.5 * b / a
        x_[1] = x_[0]
    else:
        x_[0] = FP_NAN
        x_[1] = FP_NAN

    return (x_[0], x_[1])


def quadraticEqnRoots(a, b, c):
    return _quadraticEqnRoots(a, b, c)
