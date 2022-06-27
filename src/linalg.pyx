
cimport cython
from libc.math cimport sqrt
from cubicEqn cimport _cubicEqnRoots
from numpy import array


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _tr(double[:] A):
    assert len(A) == 9
    return A[0] + A[4] + A[8]


def tr(double[:] A):
    return _tr(A)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _det(double[:] A):
    assert len(A) == 9
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _Qinv(double[:] A):
    assert len(A) == 9
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


cdef (double, double, double) invariants(double[:] A):
    return (_Pinv(A), _Qinv(A), _Rinv(A))


cdef double _mag(double[:] v):
    assert len(v) == 3
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def mag(double[:] v):
    return _mag(v)


cdef (double, double, double) _eigvals(double[:] A):
    return _cubicEqnRoots(1.0, _Pinv(A), _Qinv(A), _Rinv(A))


def eigvals(double[:] A):
    return _eigvals(A)


@cython.boundscheck(False)
@cython.wraparound(False)
def symmTraceless(double[:] A):
    assert len(A) == 9
    return array([
        A[0],
        0.5 * (A[1] + A[3]),
        0.5 * (A[2] + A[6]),
        A[4],
        0.5 * (A[5] + A[7])
    ])


cdef double _Qinvh(double[:] A):
    assert len(A) == 5
    return -(
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
    assert len(A) == 5
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


cdef (double, double, double) _eigvals(double[:] A):
    return _cubicEqnRoots(1.0, _Pinv(A), _Qinv(A), _Rinv(A))


def eigvalsh(double[:] A):
    return _eigvalsh(A)

# cdef (double, double, double) _eigvector(double[:] A, double w):
#     return (1.0, 0.0, 0.0)

