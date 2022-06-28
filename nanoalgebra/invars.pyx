cimport cython
from linalg cimport _tr, _det


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _Pinv(double[:] A):
    return -_tr(A)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _Qinv(double[:] A):
    return (
        A[0] * A[4] - A[1] * A[3] +
        A[0] * A[8] - A[2] * A[6] +
        A[4] * A[8] - A[5] * A[7]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _Rinv(double[:] A):
    return -_det(A)


@cython.boundscheck(False)
@cython.wraparound(False)
def Pinv(double[:] A) -> double:
    return _Pinv(A)


@cython.boundscheck(False)
@cython.wraparound(False)
def Qinv(double[:] A) -> double:
    return _Qinv(A)


@cython.boundscheck(False)
@cython.wraparound(False)
def Rinv(double[:] A) -> double:
    return _Rinv(A)
