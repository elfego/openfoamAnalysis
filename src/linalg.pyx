
cimport cython
from libc.math cimport sqrt, fabs, FP_NAN
from cubicEqn cimport _cubicEqnRoots
from numpy import array, sort, eye


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double, double) _cross(double[:] u, double[:] v):
    assert len(u) == 3
    assert len(v) == 3
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _Mcross(double[:] W):
    assert len(W) == 12
    W[6] = W[1] * W[5] - W[2] * W[4]
    W[7] = W[2] * W[3] - W[0] * W[5]
    W[8] = W[0] * W[4] - W[1] * W[3]
    pass


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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _mag(double[:] U):
    assert len(U) == 3
    return sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2])


def mag(double[:] v):
    return _mag(v)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _unit(double[:] U):
    assert len(U) == 3
    cdef double magU = _mag(U)
    U[0] /= magU
    U[1] /= magU
    U[2] /= magU


cdef (double, double, double) _eigvals(double[:] A):
    cdef double w0, w1, w2, tmp
    w0, w1, w2 = _cubicEqnRoots(1.0, _Pinv(A), _Qinv(A), _Rinv(A))
    if w0 < w1:
        tmp = w0
        w0 = w1
        w1 = tmp
    if w1 < w2:
        tmp = w1
        w1 = w2
        w2 = tmp
    if w0 < w1:
        tmp = w0
        w0 = w1
        w1 = tmp
    return (w0, w1, w2)


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


cdef (double, double, double) _eigvalsh(double[:] A):
    cdef double w0, w1, w2, tmp
    w0, w1, w2 = _cubicEqnRoots(1.0, 0.0, _Qinvh(A), _Rinvh(A))
    if w0 < w1:
        tmp = w0
        w0 = w1
        w1 = tmp
    if w1 < w2:
        tmp = w1
        w1 = w2
        w2 = tmp
    if w0 < w1:
        tmp = w0
        w0 = w1
        w1 = tmp
    return (w0, w1, w2)


def eigvalsh(double[:] A):
    return _eigvalsh(A)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double, double) _eigvec(double[:] T, double w, double[:] v1, double[:] v2):
    assert len(T) == 9
    assert len(v1) == 3

    cdef double A[9]
    cdef double ev[3]
    cdef SMALL = 1e-12

    # Construct the linear system for this eigenvalue
    A[0] = T[0] - w
    A[1] = T[1]
    A[2] = T[2]

    A[3] = T[3]
    A[4] = T[4] - w
    A[5] = T[5]

    A[6] = T[6]
    A[7] = T[7]
    A[8] = T[8] - w

    # Determinants of the 2x2 sub-matrices used to find the eigenvectors
    cdef double sd0, sd1, sd2
    cdef double aSd0, aSd1, aSd2

    sd0 = A[4] * A[8] - A[5] * A[7]
    sd1 = A[8] * A[0] - A[6] * A[2]
    sd2 = A[0] * A[4] - A[1] * A[3]
    aSd0 = fabs(sd0)
    aSd1 = fabs(sd1)
    aSd2 = fabs(sd2)

    # Evaluate the eigenvector using the largest sub-determinant
    if aSd0 >= aSd1 and aSd0 >= aSd2 and aSd0 > SMALL:
        ev[0] = 1.0
        ev[1] = (A[5] * A[6] - A[8] * A[3]) / sd0
        ev[2] = (A[7] * A[3] - A[4] * A[6]) / sd0
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)

    elif aSd1 >= aSd2 and aSd1 > SMALL:
        ev[0] = (A[2] * A[7] - A[8] * A[1]) / sd1
        ev[1] = 1.0
        ev[2] = (A[6] * A[1] - A[0] * A[7]) / sd1
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)

    elif aSd2 > SMALL:
        ev[0] = (A[1] * A[5] - A[4] * A[2]) / sd2
        ev[1] = (A[3] * A[2] - A[0] * A[5]) / sd2
        ev[2] = 1.0
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)

    # Sub-determinants for a repeated eigenvalue
    sd0 = A[4] * v1[2] - A[5] * v1[1]
    sd1 = A[8] * v1[0] - A[6] * v1[2]
    sd2 = A[0] * v1[1] - A[1] * v1[0]
    aSd0 = fabs(sd0);
    aSd1 = fabs(sd1);
    aSd2 = fabs(sd2);

    # Evaluate the eigenvector using the largest sub-determinant
    if aSd0 >= aSd1 and aSd0 >= aSd2 and aSd0 > SMALL:
        ev[0] = 1.0
        ev[1] = (A[5] * v1[0] - v1[2] * A[3]) / sd0
        ev[2] = (v1[1] * A[3] - A[4] * v1[0]) / sd0
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)

    elif aSd1 >= aSd2 and aSd1 > SMALL:
        ev[0] = (v1[2] * A[7] - A[8] * v1[1]) / sd1
        ev[1] = 1.0
        ev[2] = (A[6] * v1[1] - v1[0] * A[7]) / sd1
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)

    elif aSd2 > SMALL:
        ev[0] = (A[1] * v1[2] - v1[1] * A[2]) / sd2
        ev[1] = (v1[0] * A[2] - A[0] * v1[2]) / sd2
        ev[2] = 1.0
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)


    # Triple eigenvalue
    return _cross(v1, v2);


@cython.boundscheck(False)
@cython.wraparound(False)
def eigvec(double[:] A):
    cdef double W[12]
    cdef double v0[3], v1[3], v2[3]

    v0[0], v0[1], v0[2] = 1, 0, 0
    v1[0], v1[1], v1[2] = 0, 1, 0
    v2[0], v2[1], v2[2] = 0, 0, 1

    W[9], W[10], W[11] = _eigvals(A)

    W[0], W[1], W[2] = _eigvec(A, W[ 9], v1, v2)
    v0[0], v0[1], v0[2] = W[0], W[1], W[2]
    W[3], W[4], W[5] = _eigvec(A, W[10], v2, v0)
    v1[0], v1[1], v1[2] = W[3], W[4], W[5]
    W[6], W[7], W[8] = _eigvec(A, W[11], v0, v1)

    return W













@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double, double) _eigvech(double[:] T, double w):
    assert len(T) == 9

    cdef double A[9]
    cdef double ev[3]
    cdef SMALL = 1e-12

    # Construct the linear system for this eigenvalue
    A[0] = T[0] - w
    A[1] = T[1]
    A[2] = T[2]

    A[3] = T[3]
    A[4] = T[4] - w
    A[5] = T[5]

    A[6] = T[6]
    A[7] = T[7]
    A[8] = T[8] - w

    # Determinants of the 2x2 sub-matrices used to find the eigenvectors
    cdef double sd0, sd1, sd2
    cdef double aSd0, aSd1, aSd2

    sd0 = A[4] * A[8] - A[5] * A[7]
    sd1 = A[8] * A[0] - A[6] * A[2]
    sd2 = A[0] * A[4] - A[1] * A[3]
    aSd0 = fabs(sd0)
    aSd1 = fabs(sd1)
    aSd2 = fabs(sd2)

    # Evaluate the eigenvector using the largest sub-determinant
    if aSd0 >= aSd1 and aSd0 >= aSd2 and aSd0 > SMALL:
        ev[0] = 1.0
        ev[1] = (A[5] * A[6] - A[8] * A[3]) / sd0
        ev[2] = (A[7] * A[3] - A[4] * A[6]) / sd0
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)

    elif aSd1 >= aSd2 and aSd1 > SMALL:
        ev[0] = (A[2] * A[7] - A[8] * A[1]) / sd1
        ev[1] = 1.0
        ev[2] = (A[6] * A[1] - A[0] * A[7]) / sd1
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)

    elif aSd2 > SMALL:
        ev[0] = (A[1] * A[5] - A[4] * A[2]) / sd2
        ev[1] = (A[3] * A[2] - A[0] * A[5]) / sd2
        ev[2] = 1.0
        mEv = _mag(ev)
        return (ev[0] / mEv, ev[1] / mEv, ev[2] / mEv)
    pass


@cython.boundscheck(False)
@cython.wraparound(False)
def eigvech(double[:] A):
    cdef double W[12]

    W[9], W[10], W[11] = _eigvals(A)

    W[0], W[1], W[2] = _eigvech(A, W[ 9])
    W[3], W[4], W[5] = _eigvech(A, W[10])
    _Mcross(W)

    return W

