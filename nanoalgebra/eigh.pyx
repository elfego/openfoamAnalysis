
cimport cython
from libc.math cimport sqrt, fabs, FP_NAN
from cubicEqn cimport _cubicEqnRoots
from linalg cimport _mag, _symmTraceless
from invarsh cimport _Qinvh, _Rinvh
from eig cimport _eigvals


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _m_cross(double[:] W):
    W[3] = W[1] * W[8] - W[2] * W[7]
    W[4] = W[2] * W[6] - W[0] * W[8]
    W[5] = W[0] * W[7] - W[1] * W[6]
    pass


@cython.boundscheck(False)
@cython.wraparound(False)
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
cdef _eigvecsh(double[:] T, double w):
    cdef double A[9]
    cdef double ev[3]
    cdef SMALL = 1e-12

    # Construct the linear system for this eigenvalue
    A[0] = T[0] - w
    A[1] = T[1]
    A[2] = T[2]

    A[3] = T[1]
    A[4] = T[3] - w
    A[5] = T[4]

    A[6] = T[2]
    A[7] = T[4]
    A[8] = -1.0 * (T[0] + T[3] + w)

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
def eigvecsh(double[:] A):
    cdef double W[12]
    W[9], W[10], W[11] = _eigvalsh(A)
    W[0], W[1], W[2] = _eigvecsh(A, W[ 9])
    W[6], W[7], W[8] = _eigvecsh(A, W[11])
    _m_cross(W)

    return W

