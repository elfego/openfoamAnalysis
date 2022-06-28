
cimport cython
from libc.math cimport sqrt, fabs, FP_NAN
from cubicEqn cimport _cubicEqnRoots
from numpy import array


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _dot(double[:] u, double[:] v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double, double) _cross(double[:] u, double[:] v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _tr(double[:] A):
    return A[0] + A[4] + A[8]


def tr(double[:] A):
    return _tr(A)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _det(double[:] A):
    return (
        A[0] * A[4] * A[8] +
        A[3] * A[7] * A[2] +
        A[6] * A[1] * A[5] -
        A[2] * A[4] * A[6] -
        A[5] * A[7] * A[0] -
        A[8] * A[1] * A[3]
    )


def det(double[:] A):
    return _det(A)


cdef double _Pinv(double[:] A):
    return -_tr(A)


def Pinv(double[:] A) -> double:
    return _Pinv(A)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _Qinv(double[:] A):
    return (
        A[0] * A[4] - A[1] * A[3] +
        A[0] * A[8] - A[2] * A[6] +
        A[4] * A[8] - A[5] * A[7]
    )


def Qinv(double[:] A) -> double:
    return _Qinv(A)


cdef double _Rinv(double[:] A):
    return -_det(A)


def Rinv(double[:] A):
    return _Rinv(A)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _mag(double[:] U):
    return sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2])


def mag(double[:] v):
    return _mag(v)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _magSq(double[:] U):
    return U[0] * U[0] + U[1] * U[1] + U[2] * U[2]


def magSq(double[:] v):
    return _magSq(v)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _Hat(double[:] U):
    cdef double magU = _mag(U)
    U[0] /= magU
    U[1] /= magU
    U[2] /= magU


@cython.boundscheck(False)
@cython.wraparound(False)
def symmTraceless(double[:] A):
    return array([
        A[0],
        0.5 * (A[1] + A[3]),
        0.5 * (A[2] + A[6]),
        A[4],
        0.5 * (A[5] + A[7])
    ])


@cython.boundscheck(False)
@cython.wraparound(False)
def asymmVec(double[:] A):
    return array([
        A[7] - A[5],
        A[2] - A[6],
        A[3] - A[1]
    ])


cdef double _Qinvh(double[:] A):
    return -1.0 * (
        A[0] * A[0] +
        A[1] * A[1] +
        A[2] * A[2] +
        A[3] * A[3] +
        A[4] * A[4] +
        A[0] * A[3]
    )


def Qinvh(double[:] A) -> double:
    return _Qinvh(A)


cdef double _Rinvh(double[:] A):
    return (
        + A[0] * A[0] * A[3]
        - A[0] * A[1] * A[1]
        + A[0] * A[3] * A[3]
        + A[0] * A[4] * A[4]
        - A[1] * A[1] * A[3]
        + A[2] * A[2] * A[3]
        - A[1] * A[2] * A[4] * 2
    )


def Rinvh(double[:] A) -> double:
    return _Rinvh(A)

