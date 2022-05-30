from numpy import sum, dot, save
from numpy.linalg import det


def calc_3rd_inv(A_flat):
    return -det(A_flat.reshape((3, 3)))


def calc_val_weighted(X, dV, normalised=False, fsave=None):
    result = dot(dV, X)
    if normalised:
        result /= sum(dV)
    if fsave is not None:
        save(fsave, result)
    return result


class OFphase:
    def __init__(self, name, transport_model, viscosity, density):
        self.name = name
        self.transport_model = transport_model
        self.viscosity = viscosity
        self.density = density
