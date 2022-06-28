cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _Qinvh(double[:] A):
    return -1.0 * (
        A[0] * A[0] +
        A[1] * A[1] +
        A[2] * A[2] +
        A[3] * A[3] +
        A[4] * A[4] +
        A[0] * A[3]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
def Qinvh(double[:] A) -> double:
    return _Qinvh(A)


@cython.boundscheck(False)
@cython.wraparound(False)
def Rinvh(double[:] A) -> double:
    return _Rinvh(A)

