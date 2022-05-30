from numpy import sum, dot, save


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
