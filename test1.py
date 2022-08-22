import sys
import numpy as np
from readopenfoamcase import readOFcase
import matplotlib.pylab as plt
import numba as nb
from skimage.measure import find_contours
from scipy.interpolate import interp1d

plt.ion()


@nb.njit
def truncate(x, a=0, b=1):
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x


@nb.njit
def normalise(v):
    L = np.sqrt(np.sum(v * v))
    if L < 1e-13:
        return v
    else:
        return v / L


case = sys.argv[1]
rc = readOFcase(case)
R_nozzle = 2.5e-4
rc.set_nozzle_radius(R_nozzle)

time = rc.times[int(sys.argv[2])]
print(time)

if not rc.mesh_loaded:
    rc.load_mesh()

alpha1 = rc.load_field('alpha.pregel', time)
alpha1 = np.where(alpha1 > 0, alpha1, 0)
alpha2 = rc.load_field('alpha.crosslinker', time)
alpha2 = np.where(alpha2 > 0, alpha2, 0)



X1 = rc.load_post_field('X.pregel.npy', time)
X2 = rc.load_post_field('X.crosslinker.npy', time)

vm = 4 * alpha1 * alpha2 * rc.V
Vm = np.sum(vm)

X_col = np.r_[np.dot(rc.R[:, 0], vm) / Vm,
              np.dot(rc.R[:, 1], vm) / Vm,
              np.dot(rc.R[:, 2], vm) / Vm]


U1 = rc.load_post_field('U.pregel.npy', time)
U2 = rc.load_post_field('U.crosslinker.npy', time)

n = normalise(U2 - U1)
 
t1 = np.r_[0, -1, 0]
t1 = normalise(t1 - np.dot(n, t1) * n)
t2 = np.cross(t1, n)

fig = plt.figure(1, clear=True, figsize=[3.1, 4.8])
ax = fig.subplots(1, 1)

bins = 512
thres = 1.2e-1

P1, bx, by = np.histogram2d(rc.R.dot(t1) - X_col.dot(t1),
                            rc.R.dot(t2) - X_col.dot(t2),
                            bins=[bins, bins],
                            weights=alpha1 * rc.V,
                            density=True)
P1 = P1.transpose()
X, Y = np.meshgrid(bx, by)
# ax.pcolormesh(X, Y, np.tanh(P1 * 1e-6), cmap='Greens', alpha=0.3)
X_, Y_ = np.meshgrid(0.5 * (bx[1:] + bx[:-1]), 0.5 * (by[1:] + by[:-1]))
ax.contour(X_, Y_, P1, [np.max(P1) * thres], colors='#408256')

P2, bx, by = np.histogram2d(rc.R.dot(t1) - X_col.dot(t1),
                            rc.R.dot(t2) - X_col.dot(t2),
                            bins=[bins, bins],
                            weights=alpha2 * rc.V,
                            density=True)
P2 = P2.transpose()
X, Y = np.meshgrid(bx, by)
# ax.pcolormesh(X, Y, np.tanh(P2 * 1e-6), cmap='Purples', alpha=0.3)
X_, Y_ = np.meshgrid(0.5 * (bx[1:] + bx[:-1]), 0.5 * (by[1:] + by[:-1]))
ax.contour(X_, Y_, P2, [np.max(P2) * thres], colors='#8c58a6')

xx1 = np.r_[t1.dot(X1 - X_col), t2.dot(X1 - X_col)]
xx2 = np.r_[t1.dot(X2 - X_col), t2.dot(X2 - X_col)]
ax.scatter(*xx1, marker='o',
           color='seagreen', s=60)
ax.plot([xx2[0], xx1[0]], [xx2[1], xx1[1]], 'k')
ax.scatter(*xx2, marker='o',
           color='darkorchid', s=60)

P3 = np.sqrt(P1 * P2)
# ax.pcolormesh(X, Y, np.tanh(P3 * 1e-6), cmap='Greys', alpha=0.3)
contours = ax.contour(X_, Y_, P3, [np.max(P3) * thres * 1.6], colors='k', linestyles='--', linewidths=1)
m = xx1 - xx2
m /= np.linalg.norm(m)
R = np.r_[m[0], m[1], -m[1], m[0]].reshape((2, 2))
CC = contours.collections[0].get_paths()[0].vertices
cc = CC.dot(R.T)
cc_ = np.roll(cc, -cc[:, 1].argmin(), axis=0)
iM = cc_[:, 1].argmax()
f1 = interp1d(cc_[:iM + 1, 1][::-1], cc_[:iM + 1, 0][::-1], fill_value='extrapolate')
f2 = interp1d(cc_[iM:, 1], cc_[iM:, 0], fill_value='extrapolate')
yy = np.linspace(np.min(cc[:, 1]), np.max(cc[:, 1]), 1024)
w = abs(f2(yy) - f1(yy))


# contours = find_contours(X_, Y_, P3, [np.max(P3) * thres * 1.6])
# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=1, linestyle='--', color='k')

ax.set_xlim(-0.00066, 0.00066)
ax.set_ylim(-0.0008, 0.0022)
ax.set_aspect('equal')


fig.tight_layout()


# print("Contact point")
# print(X_col)

# print("Normal vector")
# print(n)

# print("Droplet centres")
# print(X1)
# print(X2)

# print("Plane corners")
# y = 1.6e-3
# z = 1.6e-3
# x = -(n[1] * y + n[2] * z) / n[0]
# C0 = np.r_[x + X_col[0], y + X_col[1], z + X_col[2]]
# print(C0)

# y = -1.6e-3
# z = 1.6e-3
# x = -(n[1] * y + n[2] * z) / n[0]
# C1 = np.r_[x + X_col[0], y + X_col[1], z + X_col[2]]
# print(C1)

# y = 1.6e-3
# z = -1.6e-3
# x = -(n[1] * y + n[2] * z) / n[0]
# C2 = np.r_[x + X_col[0], y + X_col[1], z + X_col[2]]
# print(C2)

# tmp = np.cross(C2 - C0, C1 - C0)
# tmp /= np.linalg.norm(tmp)

# print("Vector plotting")
# y = 1e-3
# z = 1e-3
# x = -(n[1] * y + n[2] * z) / n[0]
# C3 = np.r_[x + X_col[0], y + X_col[1], z + X_col[2]]
# print(C3)
# print(C3 + 2e-4 * n)



