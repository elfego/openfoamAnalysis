from numpy import sum, dot, save, nditer, zeros_like, hstack, array, trace
from numpy.linalg import det, eigh, norm
import numba_scipy
from numba import jit, prange, float64


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def calc_1st_inv(A_flat):
    return trace(A_flat.reshape((3, 3)))


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def calc_2nd_inv(A_flat):
    A = A_flat.reshape((3, 3))
    return -0.5 * trace(A @ A)


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def calc_3rd_inv(A_flat):
    return -det(A_flat.reshape((3, 3)))


def calc_val_weighted(X, dV, normalised=False, fsave=None):
    result = dot(dV, X)
    if normalised:
        result /= sum(dV)
    if fsave is not None:
        save(fsave, result)
    return result


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def get_vorticity(gradU):
    A = gradU.reshape(3, 3)
    return array([
        A[2, 1] - A[1, 2],
        A[0, 2] - A[2, 0],
        A[1, 0] - A[0, 1]
    ])


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def enstrophy(A):
    return 0.5 * ((A[2, 1] - A[1, 2])**2 +
                  (A[0, 2] - A[2, 0])**2 +
                  (A[1, 0] - A[0, 1])**2)


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def normalise(v):
    n = norm(v)
    if n < 1.0e-12:
        return v / n
    else:
        return v


def dSigma(alpha1, alpha2, gradAlpha1, gradAlpha2, V, out=None):
    with nditer([alpha1, alpha2, gradAlpha1.T, gradAlpha2.T, V, out],
                flags=['external_loop'] * 5 + ['buffered'],
                op_flags=[['readonly']] * 5 + [['writeonly', 'allocate']]) as it:
        for a1, a2, ga1, ga2, v, y in it:
            y[...] = (a1 * ga2 - a2 * ga1) * v
        return it.operands[5].T


def local_eigensystem(gradU):
    A = gradU.reshape(3, 3)
    w, v = eigh(0.5 * (A + A.T))
    idx = w.argsort()[::-1]

    return hstack((normalise(v[:, idx[0]]),
                   normalise(v[:, idx[1]]),
                   normalise(v[:, idx[2]]),
                   w[idx]))
