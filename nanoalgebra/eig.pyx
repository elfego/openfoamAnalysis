
cimport cython
from libc.math cimport fabs
from cubicEqn cimport _cubicEqnRoots
from linalg cimport _cross, _mag
from invars cimport _Pinv, _Qinv, _Rinv


@cython.boundscheck(False)
@cython.wraparound(False)
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
cdef (double, double, double) _eigvecs(double[:] T, double w, double[:] v1, double[:] v2):
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
def eigvecs(double[:] A):
    cdef double W[12]
    cdef double v0[3]
    cdef double v1[3]
    cdef double v2[3]

    v0[0], v0[1], v0[2] = 1, 0, 0
    v1[0], v1[1], v1[2] = 0, 1, 0
    v2[0], v2[1], v2[2] = 0, 0, 1

    W[9], W[10], W[11] = _eigvals(A)

    W[0], W[1], W[2] = _eigvecs(A, W[ 9], v1, v2)
    v0[0], v0[1], v0[2] = W[0], W[1], W[2]
    W[3], W[4], W[5] = _eigvecs(A, W[10], v2, v0)
    v1[0], v1[1], v1[2] = W[3], W[4], W[5]
    W[6], W[7], W[8] = _eigvecs(A, W[11], v0, v1)

    return W
