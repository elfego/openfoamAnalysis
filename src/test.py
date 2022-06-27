from cubicEqn import cubicEqnRoots, evalCubicPolynomial
import linalg
import eig
import eigh
import numpy as np
from numba import jit


def compare_vecs(v1, v2):
    v1dv2 = abs(np.dot(v1, v2))
    v1_v2 = np.linalg.norm(v1) / np.linalg.norm(v2)
    return abs(v1dv2 / v1_v2 - 1)


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def normalise(v):
    n = np.linalg.norm(v)
    if n < 1.0e-12:
        return v / n
    else:
        return v


def local_eigensystem(gradU):
    A = gradU.reshape(3, 3)
    w, v = np.linalg.eigh(0.5 * (A + A.T))
    idx = w.argsort()[::-1]

    return np.hstack((normalise(v[:, idx[0]]),
                      normalise(v[:, idx[1]]),
                      normalise(v[:, idx[2]]),
                      w[idx]))


# if __name__ == '__main__':
A = 8 * np.random.rand(3, 3) - 4
print(np.trace(A) == linalg.tr(A.flatten()))
A -= np.trace(A) / 3
B = A.flatten()
As = 0.5 * (A + A.T)
Bs = As.flatten()


Cs = linalg.symmTraceless(B)
print('Cs =\n', Cs)
print('Bs =\n', Bs[[0, 1, 2, 4, 5]])


print("P")
p = np.trace(As)
print(p)
print(linalg.tr(Bs))

print("Q")
q = 0.5 * (np.trace(As)**2 - np.trace(As @ As))
print(q)
print(linalg.Qinv(Bs))
print(linalg.Qinvh(Cs))


print("R")
r = -np.linalg.det(As)
print(r)
print(linalg.Rinv(Bs))
print(linalg.Rinvh(Cs))


w1 = np.sort(np.linalg.eigvalsh(As))[::-1]
w2 = eig.eigvals(Bs)
w3 = eigh.eigvalsh(Cs)
print("eigenvalues")
print(w1)
print(w2)
print(w3)

print("eigenvectors")
W1 = local_eigensystem(As)
W2 = np.array(eig.eigvecs(Bs))
W3 = np.array(eigh.eigvecsh(Cs))
print(W1)
print(W2)
print(W3)

print("checking...")
for i in range(3):
    print("  eigenvalue", W1[9 + i], W3[9 + i])
    print("  eigenequation")
    print("  ", np.dot(As, W1[3*i:3*(i + 1)]), W1[9 + i] * W1[3*i:3*(i + 1)])
    print("  ", np.dot(As, W2[3*i:3*(i + 1)]), W2[9 + i] * W2[3*i:3*(i + 1)])
    print("  ", np.dot(As, W3[3*i:3*(i + 1)]), W3[9 + i] * W3[3*i:3*(i + 1)])


print("Error")
for i in range(3):
    err2 = compare_vecs(W1[3*i: 3*(i + 1)], W3[3*i: 3*(i + 1)])
    err3 = compare_vecs(W1[3*i: 3*(i + 1)], W3[3*i: 3*(i + 1)])
    print("  ", err2, err3)


