import cython
from libc.math cimport cos, sin, atan2, cbrt, sqrt, fma, fabs, FP_NAN
from linearEqn cimport _linearEqnRoots
from quadraticEqn cimport _quadraticEqnRoots


def evalCubicPolynomial(double x, double a, double b, double c, double d) -> double:
    return a * x**3 + b * x**2 + c * x + d


cdef (double, double, double) _cubicEqnRoots(double a, double b, double c, double d):

    # (JLM:p. 2246) [p = a * c - b * b / 3]
    cdef double w = a * c
    cdef double p = -(fma(-a, c, w) + fma(b, b / 3.0, -w))
    cdef double q = b * b * b * 2.0 / 27.0 - b * c * a / 3.0 + d * a * a
    cdef double numDiscr = p*p*p/27.0 + q*q/4.0
    cdef double discr =  numDiscr if fabs(numDiscr) > 1.0e-15 else 0.0

    # Determine the number and types of the roots
    threeReal = discr < 0
    oneRealTwoComplex = discr > 0
    twoReal = fabs(p) > sqrt(1.0e-15) and not (threeReal or oneRealTwoComplex)

    cdef double sqrt3 = sqrt(3.0)
    cdef double x = 0.0
    cdef double x_[3]

    cdef double wCbRe, wCbIm, wAbs, wArg, wRe, wIm

    if threeReal:
        wCbRe = -q / 2.0
        wCbIm = sqrt(-discr)
        wAbs = cbrt(sqrt(wCbRe * wCbRe + wCbIm * wCbIm))
        wArg = atan2(wCbIm, wCbRe) / 3.0
        wRe = wAbs * cos(wArg)
        wIm = wAbs * sin(wArg)

        if b > 0.0:
            x = -wRe - fabs(wIm) * sqrt3 - b / 3.0
        else:
            x = 2.0*wRe - b/3.0

    elif oneRealTwoComplex:
    #     const scalar wCb = -q/2 - sign(q)*sqrt(discr);
    #     const scalar w = cbrt(wCb);
    #     const scalar t = w - p/(3*w);

    #     if (p + t*b < 0):
    #         x = t - b/3;
    #     else:
    #         const scalar xRe = -t/2 - b/3;
    #         const scalar xIm = sqrt3/2*(w + p/3/w);

    #         return
    #             Roots<3>
    #             (
    #                 Roots<1>(roots::real, -a*d/(xRe*xRe + xIm*xIm)),
    #                 Roots<2>
    #                 (
    #                     Roots<1>(roots::complex, xRe),
    #                     Roots<1>(roots::complex, xIm)
    #                 )
    #             );
        return (FP_NAN, FP_NAN, FP_NAN)
    elif twoReal:
        if q * b > 0:
            x = -2.0 * cbrt(0.5 * q) - b / 3.0
        else:
            x = cbrt(0.5 * q) - b / 3.0
            x_[0] = _linearEqnRoots(-a, x)
            x_[2] = _linearEqnRoots(x * x, a * d)
            return (x_[0], x_[0], x_[2])
    # (oneReal)
    else:
        x_[0] = _linearEqnRoots(a, b / 3.0)
        return [x_[0], x_[0], x_[0]]

    x_[0] = _linearEqnRoots(-a, x)
    x_[1], x_[2] = _quadraticEqnRoots(-x * x, c * x + a * d, d * x)

    return (x_[0], x_[1], x_[2])


def cubicEqnRoots(a, b, c, d):
    return _cubicEqnRoots(a, b, c, d)

