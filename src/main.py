from cubicEqn import cubicEqnRoots, evalCubicPolynomial
import linalg
import numpy as np


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


w1 = np.linalg.eigvalsh(As)
w2 = list(linalg.eigvals(Bs))
print(np.linalg.norm(np.sort(w1) - np.sort(w2)) < 1e-10)


# print(linalg.det(A.flatten()) == np.linalg.det(A))

# coeffs = (1.0, p, q, r)

# # a = 1.0
# # b = 0.0
# # c = -4.2
# # d = 2.0
# # coeffs = (a, b, c, d)

# roots = cubicEqnRoots(*coeffs)

# for x_ in roots:
#     print(f"Root x = {x_} gives f(x) = {evalCubicPolynomial(x_, *coeffs)}")
