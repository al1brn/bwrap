#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Interpolation functions

Rewrite the Blender interpolation functions in order to be able to use it on vectorized coordinates

The parameters can be either a float or an array of floats

Created on Jul 2020
"""

__author__     = "Alain Bernard"
__copyright__  = "Copyright 2020, Alain Bernard"
__credits__    = ["Alain Bernard"]

__license__    = "GPL"
__version__    = "1.0"
__maintainer__ = "Alain Bernard"
__email__      = "wrapanime@ligloo.net"
__status__     = "Production"


import numpy as np
from math import pi

from .commons import base_error_title

error_title = base_error_title % "interpolation.%s"



# =============================================================================================================================
# Useful constants

zero  = 1e-6
twopi = pi*2
hlfpi = pi/2

# =============================================================================================================================
# Easings canonic functions
# Parameter is betwwen 0 and 1. Return from 0 to 1

def f_constant(t):
    y = np.ones_like(t)
    try:
        y[np.where(t<1)[0]] = 1.
    except:
        return 0. if t < 1 else 1.
    return y

def f_linear(t):
    return t

def f_bezier(t):
    return f_linear(t)

def f_sine(t):
    return 1 - np.cos(t * hlfpi)

def f_quadratic(t):
    return t*t

def f_cubic(t):
    return t*t*t

def f_quartic(t):
    t2 = t*t
    return t2*t

def f_quintic(t):
    t2 = t*t
    t3 = t2*t
    return t3*t2

def f_exponential(t):
    return 1. - np.power(2, -10 * t)

def f_circular(t):
    return 1 - np.sqrt(1 - t*t)

def f_back(t, factor):
    return t*t*((factor + 1)*t - factor)

def f_elastic(t):
    amplitude    = 0.
    period       = 0.

    if period == 0:
        period = .3

    if (amplitude == 0) or (amplitude < 1.):
        amplitude = 1.
        s         = period/4
    else:
        s = period / twopi * np.sin(1/amplitude)

    t -= 1

    return -amplitude * np.power(2, 10*t) * np.sin((t-s)*twopi / period)


def f_bounce(t, a, xi, di, ci):
    """Compute bounces based on xi, di and ci params

    Let's name n the number of bounces after the half initial one.
    Each bounces is half of the previouse one. Let's note q the length
    of bounce 0 (the half one).
    The total length is L = q + q/2 + q/4 + ... -q/2
    Hence: L/q = (1-q/2^(n+1))/(1-1/2) - 1/2 = 3/2 - 1/2^n
    We want L = 1, hence 1/q = 3/2 - 1/2^n

    Let's note: d = q/2

    The equation of the initial parabola (half one) is: y = a(d^2 - x^2)
    At x= 0, y = 1, hence: a = 4/q^2

    Each xi is given by: xi = q(3/2-1/2^i)

    The parameters are the following:
        - a  : float -> used to compute the parabola a*(t^2 - ??)
        - xi : [0, q/2, q, ... 3/2 - 1/2^i ... 1]
        - di : [q/2, q/2,  ... xi+1 - xi]
        - ci : [0.,  3/2q, ... xi + di/2]

    These parameters are computed at initialization time

    NOTE that the ease in falls from right to left !
    The parameters must be initialized in consequence :-)
"""
    # Number of bounces

    n = len(di)

    # Duplicaton of the t for each of the intervals starting abscissa
    # NOTE: 1-t because the computation is made on falling from 1 to 0
    # but the default ease in is time reversed from 0 to 1

    ats = np.full((n, len(t)), 1-t).transpose()

    # Distances to the parabola centers
    y = ats - ci

    # Parabola equation
    # a and di are supposed to be correctly initialized :-)
    y = a*y*y + di

    # Return the max values (normally, only one positive value per line)

    return np.max(y, axis=1)


# =============================================================================================================================
# Interpolation

class Interpolation():
    """Interpolation function from an interval towards another interval.

    Interpolated values are computed depending upon an interpolation code.

    Parameters
    ----------
    x0: float
        User min x value
    x1: float
        User max x value
    y0: float
        User min y value
    y1: float
        User max y value
    interpolation: str
        A valid code for interpolation in
    mode: str
        A valid code for easing mode in [AUTO', 'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT'
    """

    RESOL  = 0.01

    EASINGS = ['AUTO', 'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT']

    INTERPS = {
        'CONSTANT'  : {'func': f_constant,    'auto': 'EASE_IN', 'tangents': [0, 0]},
        'LINEAR'    : {'func': f_linear,      'auto': 'EASE_IN',  'tangents': [1, 1]},
        'BEZIER'    : {'func': f_bezier,      'auto': 'EASE_IN',  'tangents': [0, 0]},
        'SINE'      : {'func': f_sine,        'auto': 'EASE_IN',  'tangents': [0, 1]},
        'QUAD'      : {'func': f_quadratic,   'auto': 'EASE_IN',  'tangents': [0, 1]},
        'CUBIC'     : {'func': f_cubic,       'auto': 'EASE_IN',  'tangents': [0, 1]},
        'QUART'     : {'func': f_quartic,     'auto': 'EASE_IN',  'tangents': [0, 1]},
        'QUINT'     : {'func': f_quintic,     'auto': 'EASE_IN',  'tangents': [0, 1]},
        'EXPO'      : {'func': f_exponential, 'auto': 'EASE_IN',  'tangents': [10*np.log(2), 0]},
        'CIRC'      : {'func': f_circular,    'auto': 'EASE_IN',  'tangents': [0, 0]},
        'BACK'      : {'func': f_back,        'auto': 'EASE_IN',  'tangents': [0, 0]},
        'BOUNCE'    : {'func': f_bounce,      'auto': 'EASE_OUT', 'tangents': [0, 0]},
        'ELASTIC'   : {'func': f_elastic,     'auto': 'EASE_OUT', 'tangents': [0, 0]},
        }

    def __init__(self, interpolation='BEZIER', easing='AUTO', input=(0., 1.), output=(0., 1.)):
    #x0=0., x1=1., y0=0., y1=1.):

        x0 = input[0]
        x1 = max(x0+self.RESOL, input[1])
        y0 = output[0]
        y1 = output[1]

        Dx3 = (x1-x0)/3
        self._bpoints   = np.array(((x0, y0), (x0+Dx3, y0), (x1-Dx3, y1), (x1, y1)))

        # The control points can be managed by the curve
        self.curve        = None
        self.curve_index  = None

        # Specific Parameters
        self.amplitude = 0.
        self.back      = 0.
        self.period    = 0.
        self.factor    = 1.70158

        # Interpolation
        self._interpolation = ""
        self.interpolation  = interpolation

        # Easing
        self._auto     = self.INTERPS[interpolation]['auto']
        self.easing    = easing

    # ---------------------------------------------------------------------------
    # Bezier points

    @property
    def bpoints(self):
        if self.curve is None:
            return self._bpoints
        else:
            return self.curve.bpoints[self.curve_index: self.curve_index + 4]

    @bpoints.setter
    def bpoints(self, value):
        if self.curve is None:
            self._bpoints = value
        else:
            self.curve.bpoints[self.curve_index:self.curve_index + 4] = value

    def capture_bpoints(self, curve, index):
        curve.bpoints[index:index + 4] = self._bpoints
        self.curve       = curve
        self.curve_index = index

    # ---------------------------------------------------------------------------
    # Amplitudes

    @property
    def x0(self):
        return self.bpoints[0, 0]

    @x0.setter
    def x0(self, value):
        dx = value - self.x0
        self.bpoints[:, 0] += dx

    @property
    def y0(self):
        return self.bpoints[0, 1]

    @y0.setter
    def y0(self, value):
        dy = value - self.y0
        self.bpoints[:, 1] += dy

    @property
    def x1(self):
        return self.bpoints[3, 0]

    @x1.setter
    def x1(self, value):
        self.delta = value - self.x0

    @property
    def y1(self):
        return self.bpoints[3, 1]

    @y1.setter
    def y1(self, value):
        dy = value - self.y1
        self.bpoints[2:, 1] += dy

    @property
    def delta(self):
        return self.bpoints[3, 0] - self.bpoints[0, 0]

    @delta.setter
    def delta(self, value):
        v = max(self.RESOL, value)
        a = (self.bpoints[1:, 0] - self.bpoints[0, 0]) * v / self.delta
        self.bpoints[1:, 0] = a

    @property
    def y_delta(self):
        return self.bpoints[3, 1] - self.bpoints[0, 1]

    # ---------------------------------------------------------------------------
    # Bezier points

    @property
    def P0(self):
        return self.bpoints[0]

    @property
    def P1(self):
        return self.bpoints[1]

    @property
    def P2(self):
        return self.bpoints[2]

    @property
    def P3(self):
        return self.bpoints[3]

    @P0.setter
    def P0(self, value):
        self.x0 = value[0]
        self.y0 = value[1]

    @P1.setter
    def P1(self, value):
        self.bpoints[1] = (max(self.x0, value[0]), value[1])

    @P2.setter
    def P2(self, value):
        self.bpoints[2] = (min(self.x1, value[0]), value[1])

    @P3.setter
    def P3(self, value):
        self.x1 = value[0]
        self.y1 = value[1]

    # ---------------------------------------------------------------------------
    # A user friendly representation

    def __repr__(self):
        easing = f"{self.easing}"
        if easing == 'AUTO':
            easing += f" {self.easing_mode}"
        return f"Interpolation({self.interpolation}) [{self.x0:.2f} {self.x1:.2f}] -> [{self.y0:.2f} {self.y1:.2f} {easing}]"

    # ---------------------------------------------------------------------------
    # Initialize from two Blender KeyFrame points

    @classmethod
    def FromKFPoints(cls, kf0, kf1):
        interp = Interpolation(
            kf0.co.x, kf1.co.x, kf0.co.y, kf1.co.y,
            interpolation=kf0.interpolation, easing=kf0.easing)
        interp.P1 = kf0.handle_right
        interp.P2 = kf1.handle_left

        interp.amplitude = kf0.amplitude
        interp.back      = kf0.back
        interp.period    = kf0.period
        interp.comp_bounces()
        return interp

    # ---------------------------------------------------------------------------
    # Initializers

    @classmethod
    def Constant(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('CONSTANT', easing, input=input, output=output)

    @classmethod
    def Linear(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('LINEAR', easing, input=input, output=output)

    @classmethod
    def Bezier(cls, easing='AUTO', input=(0., 1.), output=(0., 1.), P1=(1/3, 0.), P2=(2/3, 1.)):
        interp = Interpolation('BEZIER', easing, input=input, output=output)
        interp.P1 = (interp.x0 + P1[0]*(interp.x1-interp.x0), interp.y0 + P1[1]*(interp.y1-interp.y0))
        interp.P2 = (interp.x0 + P2[0]*(interp.x1-interp.x0), interp.y0 + P2[1]*(interp.y1-interp.y0))
        return interp

    @classmethod
    def Sine(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('SINE', easing, input=input, output=output)

    @classmethod
    def Quadratic(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('QUAD', easing, input=input, output=output)

    @classmethod
    def Cubic(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('CUBIC', easing, input=input, output=output)

    @classmethod
    def Quartic(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('QUART', easing, input=input, output=output)

    @classmethod
    def Quintic(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('QUINT', easing, input=input, output=output)

    @classmethod
    def Exponential(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('EXPO', easing, input=input, output=output)

    @classmethod
    def Circular(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('CIRC', easing, input=input, output=output)

    @classmethod
    def Back(cls, easing='AUTO', factor=1.70158, input=(0., 1.), output=(0., 1.)):
        interp = Interpolation('BACK', easing, input=input, output=output)
        interp.factor = factor
        return interp

    @classmethod
    def Bounce(cls, easing='AUTO', n=3, input=(0., 1.), output=(0., 1.)):
        interp = Interpolation('BOUNCE', easing, input=input, output=output)
        interp.comp_bounces(n)
        return interp

    @classmethod
    def Elastic(cls, easing='AUTO', input=(0., 1.), output=(0., 1.)):
        return Interpolation('ELASTIC', easing, input=input, output=output)

    # ---------------------------------------------------------------------------
    # Interpolation property

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value):
        if not value in self.INTERPS.keys():
            raise RuntimeError(
                error_title % "interpolation" +
                f"Interpolation initialization error: invalid interpolation {value}." +
                f"Valid codes are {self.INTERPS.keys()}"
                )

        self._interpolation = value
        self._canonic  = self.INTERPS[value]['func']
        self._auto     = self.INTERPS[value]['auto']
        self._tangents = self.INTERPS[value]['tangents']

        self.comp_bounces()

    # ---------------------------------------------------------------------------
    # Easing property

    @property
    def easing(self):
        return self._easing

    @easing.setter
    def easing(self, value):
        if not value in self.EASINGS:
            raise RuntimeError(
                error_title % "easing" +
                f"Easing initialization error: invalid easing mode {value}. Valid modes are {self.EASINGS}"
                )

        self._easing = value

    # ---------------------------------------------------------------------------
    # Easing mode

    @property
    def easing_mode(self):
        return self._auto if self._easing == 'AUTO' else self._easing

    # ---------------------------------------------------------------------------
    # Canonic computation
    # Can be overriden for custom easings

    def canonic(self, t):
        if self.interpolation == 'BOUNCE':
            return f_bounce(t, self.a, self.xi, self.di, self.ci)
        elif self.interpolation == 'BACK':
            return f_back(t, self.factor)
        else:
            return self._canonic(t)


    # ---------------------------------------------------------------------------
    # Initialization specific to bounces

    def comp_bounces(self, n=3):

        r = 2 # Default

        # All but the first half one

        n      = min(7, max(0, n))

        qinv   = 1.5 - 1/r**n
        q      = 1/qinv
        xi     = np.array([q*(1.5 - 1/r**i) for i in range(n+1)])
        xi[-1] = 1
        di     = xi[1:] - xi[:-1]
        a      = 4*qinv*qinv

        self.a  = -a
        self.xi = np.insert(xi[:-1], 0, 0)                # 0, q/2, q, ...
        self.di = np.insert(a * di * di / 4, 0, 1)  # Parabola equation : a*x*x + di
        self.ci = np.insert(xi[:-1] + di/2, 0, 0)

    # ---------------------------------------------------------------------------
    # Bezier computation
    # Call only after ensuring x is within the [x0, x1] interval

    def bezier(self, x):

        xs = np.array(x)
        single = len(xs.shape) == 0
        if single:
            xs = np.array([xs])
        count = len(xs)

        t0 = np.zeros((count, 2), np.float)
        t1 = np.ones( (count, 2), np.float)

        P0 = np.resize(self.P0, (count, 2))
        P1 = np.resize(self.P1, (count, 2))
        P2 = np.resize(self.P2, (count, 2))
        P3 = np.resize(self.P3, (count, 2))

        P  = np.empty((count, 2), np.float)
        ix = np.arange(count)

        # Start with t close to the location of x in the interval

        alpha = (xs - P0[:, 0])/(P3[:, 0] - P0[:, 0])
        t     = t1 * np.resize(alpha, (2, count)).transpose()

        for i in range(15):

            #print(f"step {i}: {len(ix)} --> {ix}")

            # Compute the current bezier point for t
            # t0 and t1 have the length of ix

            t2   = t*t
            t3   = t2*t

            umt  = 1-t
            umt2 = umt*umt
            umt3 = umt2*umt

            P[ix] = umt3*P0[ix] + 3*umt2*t*P1[ix] + 3*umt*t2*P2[ix] +  t3*P3[ix]
            #print(f"t: {t[:, 0]}, x: {P[ix, 0]}, gap: {P[ix, 0]-xs[ix]}")

            # Dichotomy step

            zeros  = np.where(np.abs(P[ix, 0] - xs[ix]) < 1e-4)[0]
            new_ix = np.delete(ix, zeros)

            # Done for all the t values
            if len(new_ix) == 0:
                break

            # Update t0 and t1

            imin = np.where(P[ix, 0] < xs[ix])[0]
            imax = np.delete(np.arange(len(ix)), imin)

            t0[imin] = t[imin]
            t1[imax] = t[imax]

            # reduce the size of the arrays

            nzs = np.delete(np.arange(len(ix)), zeros)
            t0 = t0[nzs]
            t1 = t1[nzs]
            t  = (t0 + t1)/2

            ix = new_ix

        if single:
            return P[0, 1]
        else:
            return P[:, 1]

    # ---------------------------------------------------------------------------
    # x factors: position of x in [0 - 1]

    def xfactors(self, x):

        xs = np.array(x)
        if len(xs.shape) == 0:
            return min(1., max(0., (x-self.x0)/(self.x1-self.x0)))

        return np.minimum(1., np.maximum(0., (xs-self.x0)/(self.x1-self.x0)))

    # ---------------------------------------------------------------------------
    # y factors: position of f(x) in [0 - 1]

    def yfactors(self, x):

        y = self(x)

        ys = np.array(y)
        if len(ys.shape) == 0:
            return min(1., max(0., (y-self.y0)/(self.y1-self.y0)))

        return np.minimum(1., np.maximum(0., (ys-self.y0)/(self.y1-self.y0)))

    # ---------------------------------------------------------------------------
    # Interpolation

    def __call__(self, x):

        # Work only with arrays
        xs = np.array(x)

        # A single value
        single = len(xs.shape) == 0
        if single:
            xs = np.array([xs])

        # Normalized the abscissa between 0 and 1
        ts = (xs - self.x0)/self.delta

        single = len(ts.shape) == 0
        if single:
            ts = np.array([ts])

        # Points outside the interval
        i_inf = np.where(ts <= 0)[0]
        i_sup = np.where(ts >= 1)[0]

        # Abscissas exist outside
        idx     = np.arange(len(ts))
        outside = (len(i_inf) + len(i_sup)) > 0
        if outside:
            ys = np.empty_like(ts)

            ys[i_inf] = self.y0
            ys[i_sup] = self.y1

            idx = np.delete(idx, np.concatenate((i_inf, i_sup)))

        t = ts[idx]

        # Compute on the required abscissa
        if self.interpolation == 'BEZIER':

            vals = self.bezier(xs[idx])

        else:
            mode = self.easing_mode

            if mode == 'EASE_IN':
                vals = self.y0 + self.y_delta*self.canonic(t)

            elif mode == 'EASE_OUT':
                vals = self.y0 + self.y_delta*(1-self.canonic(1-t))

            else:
                t *= 2

                if len(t.shape) > 0:
                    y = np.empty_like(t)

                    inf = np.where(t<=1)[0]
                    sup = np.delete(np.arange(len(t)), inf)

                    y[inf] = self.canonic(t[inf])/2
                    y[sup] = 1 - self.canonic(2-t[sup])/2

                    vals = self.y0 + self.y_delta*y

                else:
                    if t <= 1:
                        vals = self.y0 + self.y_delta*self.canonic(t)/2
                    else:
                        vals = self.y0 + self.y_delta*(1 - self.canonic(2-t))/2

        # The results
        if outside:
            ys[idx] = vals
        else:
            ys = vals

        # Single result
        if single:
            return ys[0]
        else:
            return ys

    # ---------------------------------------------------------------------------
    # Integral
    # The values y is interpretated as the speed relative to x
    # The integral method returns the value integrated at a given value x

    def integral(self, x=None):

        if x is None:
            x = self.x1

        xs = np.array(x)
        single = len(xs.shape) == 0

        # Array of values

        if not single:
            r = np.zeros(len(xs), np.float)
            for i in range(len(xs)):
                r[i] = self.integral(xs[i])
            return r

        # Below initial value of the interval

        if x <= self.x0:
            return 0.

        # Full integration

        xamp = self.x1 - self.x0
        count = max(20, int(xamp + 1))
        dx = xamp / count
        vs = self.x0 + np.arange(count)*dx
        ys = self(vs)

        # After x1: full integration on the interval

        if x >= self.x1:
            return np.sum(ys)*dx

        # x is in the middle

        idx = len(np.where(vs <= x)[0])-1
        y = np.sum(ys[np.arange(idx)])*dx

        y += (x-vs[idx]) * ys[idx]

        return y


    # ---------------------------------------------------------------------------
    # Tangents

    @property
    def left_tangent(self):
        if self.interpolation == 'BEZIER':
            dx = self.bpoints[1, 0] - self.bpoints[0, 0]
            dy = self.bpoints[1, 1] - self.bpoints[0, 1]
            if abs(dx) < zero:
                return 0.
            else:
                return dy/dx
        else:
            tg = self.y_delta/self.delta

            mode = self.easing_mode

            if mode == 'EASE_IN':
                return tg*self._tangents[0]
            elif mode == 'EASE_OUT':
                return tg*(1-self._tangents[0])
            if mode == 'EASE_IN_OUT':
                return tg*self._tangents[0]/2

        return 0.

    @property
    def right_tangent(self):
        if self.interpolation == 'BEZIER':
            dx = self.bpoints[3, 0] - self.bpoints[2, 0]
            dy = self.bpoints[3, 1] - self.bpoints[2, 1]
            if abs(dx) < zero:
                return 0.
            else:
                return dy/dx
        else:
            tg = self.y_delta/self.delta

            mode = self.easing_mode

            if mode == 'EASE_IN':
                return tg*self._tangents[1]
            elif mode == 'EASE_OUT':
                return tg*(1-self._tangents[1])
            if mode == 'EASE_IN_OUT':
                return tg*self._tangents[1]/2

        return 0.

    # ---------------------------------------------------------------------------
    # _plot for development

    def _plot(self, count=100, margin=0., fcomp=None, integ=False):
        
        import matplotlib.pyplot as plt

        x0 = self.x0
        x1 = self.x1
        amp = x1-x0

        x0 -= margin*amp
        x1 += margin*amp
        dx = (x1-x0)/(count-1)

        xs = np.arange(x0, x1+dx, dx, dtype=np.float)

        fig, ax = plt.subplots()

        def splot(mode):
            mmode = self.easing
            self.easing = mode
            ys = self(xs)
            self.easing = mmode

            ax.plot(xs, ys)

        splot('EASE_OUT')

        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])

        if integ:
            ax.plot(xs, [self.integral(x) for x in xs])

        ax.set(xlabel='x', ylabel='easing',
               title=f"{self}")
        ax.grid()

        fig.savefig("test.png")
        plt.show()


# =============================================================================================================================
# A curve

class Interpolations():
    """A fcurve Blender compatible.

    The Fcurve is a series of successive interpolations. Each interpolation
    occupies an interval

    Parameters
    ----------
    bpoints: array(n, 3, 2) of float
        The bpoints of the fcurve
    params:
        Parameters
    funcs
    modes
    """

    def __init__(self, extrapolation='CONSTANT'):
        self.interpolations = []
        self.extrapolation  = extrapolation
        self.bpoints        = None

    def __repr__(self):
        s = ""
        for interp in self.interpolations:
            s += f"{interp.x0:.2f} '{interp.interpolation}' "
        s = "[" + s + f"{self.x1:.2f}]"

        return f"Interpolations({len(self)})\n{s}"

    # ---------------------------------------------------------------------------
    # Initialize from a Blender fcurve

    @classmethod
    def FromFCurve(cls, fcurve):

        wfc = cls()
        wfc.extrapolation = fcurve.extrapolation

        for i in range(len(fcurve.keyframe_points)-1):
            kf0 = fcurve.keyframe_points[i]
            kf1 = fcurve.keyframe_points[i+1]
            wfc.append(Interpolation.FromKFPoints(kf0, kf1))

        wfc.capture_bpoints()

        return wfc

    # ---------------------------------------------------------------------------
    # To fcurve
    # Not a true fcurve but an array of keyframe like

    @property
    def keyframe_points(self):

        class Kf():
            pass

        kfs = []
        for i in range(len(self)+1):
            itp = self[min(i, len(self)-1)]
            kf = Kf()

            index = self.interp_index(i)
            kf.handle_left   = np.array(self.bpoints[index-1])
            kf.co            = np.array(self.bpoints[index])
            kf.handle_right  = np.array(self.bpoints[index+1])

            kf.interpolation = itp.interpolation
            kf.amplitude     = itp.amplitude
            kf.back          = itp.back
            kf.easing        = itp.easing
            kf.period        = itp.period

            kfs.append(kf)

        return kfs


    # ---------------------------------------------------------------------------
    # As an array of interpolations

    def __len__(self):
        return len(self.interpolations)

    def __getitem__(self, index):
        return self.interpolations[index]

    def __setitem__(self, index, value):
        self.interpolations[index] = value

    # ---------------------------------------------------------------------------
    # Capture the control of the bezier points from the interpolations

    def capture_bpoints(self):
        if self.bpoints is not None:
            return

        self.bpoints = np.zeros( ((len(self)+1)*3, 2), np.float)
        for i in range(len(self)):
            self.interpolations[i].capture_bpoints(self, 1 + i*3)

    # ---------------------------------------------------------------------------
    # Adjust the size of the bpoints array

    def adjust_bpoints(self, length):
        target = ((length-1)//10 + 1) * 10
        size = (1+target)*3
        if self.bpoints is not None:
            if len(self.bpoints) >= size:
                return

        if self.bpoints is None:
            self.bpoints = np.zeros((size, 2), np.float)
        else:
            self.bpoints = np.resize(self.bpoints, (size, 2))

    def interp_index(self, index):
        return 1 + index*3

    # ---------------------------------------------------------------------------
    # Append a new interpolation

    def append(self, interp):

        self.adjust_bpoints(len(self)+1)

        if len(self) == 0:
            self.interpolations = [interp]
            interp.capture_bpoints(self, 1)
            return interp

        if abs(interp.x0 - self.x1) > Interpolation.RESOL:
            raise RuntimeError(
                error_title % "append" +
                "Interpolations append error: the x0 of a new interpolation must equal the x1 to the last one." +
                f"Interpolations: {self}" +
                f"Interpolation to insert: {interp}" +
                f"Interpolations.x1 = {self.x1:.2f}, Interpolation.x0 = {interp.x0:.2f}"
                )

        interp.x0 = self.x1
        interp.y0 = self.y1
        self.interpolations.append(interp)

        interp.capture_bpoints(self, self.interp_index(len(self)-1))

        return interp


    @property
    def x0(self):
        """Starting x value of the function."""
        if len(self) == 0:
            return 0.
        return self.bpoints[1, 0]
        #return self[0].x0

    @property
    def x1(self):
        """Ending x value of the function."""
        if len(self) == 0:
            return 1.
        return self.bpoints[self.interp_index(len(self)), 0]
        #return self[-1].x1

    @property
    def y0(self):
        """Starting y value of the function."""
        if len(self) == 0:
            return 0.
        return self.bpoints[1, 1]
        #return self[0].y0

    @property
    def y1(self):
        """Ending y value of the function."""
        if len(self) == 0:
            return 1.
        return self.bpoints[self.interp_index(len(self)), 1]
        #return self[-1].y1

    @property
    def deltas(self):
        return np.array([itp.delta for itp in self.interpolations])

    @property
    def x0s(self):
        return np.array([itp.x0 for itp in self.interpolations])

    @property
    def x1s(self):
        return np.array([itp.x1 for itp in self.interpolations])

    # ====================================================================================================
    # Call

    def __call__(self, x):

        # ---------------------------------------------------------------------------
        # Empty curve

        if len(self) == 0:
            return np.array(x)

        # ---------------------------------------------------------------------------
        # A single value

        if np.array(x).size == 1:

            if self.extrapolation == 'CYCLIC':
                x = self.x0 + (x-self.x0)%(self.x1 - self.x0)

            if self.extrapolation == 'CONSTANT':
                if x <= self.x0:
                    return self.y0
                if x >= self.x1:
                    return self.y1
            else:
                if x <= self.x0:
                    return self.y0 + (x-self.x0)*self[0].left_tangent
                if x >= self.x1:
                    return self.y1 + (x-self.x1)*self[-1].right_tangent

            for interp in self.interpolations:
                if interp.x0 + interp.delta >= x:
                    return interp(x)

        # ---------------------------------------------------------------------------
        # Not that many values

        if len(x) < 100:
            return [self(X) for X in x]

        # ---------------------------------------------------------------------------
        # Vectorisation

        # Cyclic extrapolation
        if self.extrapolation == 'CYCLIC':
            xs = self.x0 + (np.array(x) - self.x0)%(self.x1 - self.x0)
        else:
            xs = np.array(x)

        # The resulting array
        ys = np.full(len(xs), 9., np.float)

        # ----- Points which are below the definition interval

        i_inf = np.where(xs <= self.x0)[0]
        if self.extrapolation == 'CONSTANT':
            ys[i_inf] = self.y0
        else:
            ys[i_inf] = self.y0 + (xs[i_inf]-self.x0)*self.interpolations[0].left_tangent

        # ----- Points which are above the definition interval

        i_sup = np.where(xs >= self.x1)[0]
        if self.extrapolation == 'CONSTANT':
            ys[i_sup] = self.y1
        else:
            ys[i_sup] = self.y1 + (xs[i_sup]-self.x1)*self.interpolations[-1].right_tangent

        # ----- Remaining points are within the definition interval

        idx = np.delete(np.arange(len(xs)), np.concatenate((i_inf, i_sup)))

        # Duplicaton of the xs for each of the bezier points
        axs = np.full((len(self), len(idx)), xs[idx]).transpose()

        # Deltas
        deltas = np.full((len(idx), len(self)), self.deltas)

        # Differences
        diffs = (axs - self.x0s)
        ix, tx = np.where(np.logical_and(np.greater_equal(diffs, 0), np.less(diffs, deltas)))

        # differences in a linear array (not useful here)
        # ts = diffs[ix, tx]

        # Array of the remaining x
        rem_x = xs[idx]

        # Let's loop on the easing to compute on the x which are in the interval
        # Note the this algorithm supposes that the number of easings is low
        # Compared to the number of x to compute

        interpolations = self.interpolations
        yints = np.full(len(idx), 8, np.float)
        for i in range(len(interpolations)):

            i_filter = np.where(tx==i)[0]
            vals = interpolations[i](rem_x[i_filter])

            yints[i_filter] = vals

        ys[idx] = yints

        return ys

    # ====================================================================================================
    # Integral

    def integral(self, x):

        xs = np.array(x)
        single = len(xs.shape) == 0

        # Array of values

        if not single:
            r = np.zeros(len(xs), np.float)
            for i in range(len(xs)):
                r[i] = self.integral(xs[i])
            return r

        # Below initial value of the interval

        if x <= self.x0:
            return 0.

        # With an interpolation

        r = 0.

        for itp in self:
            if x >= itp.x1:
                r += itp.integral()
            else:
                r += itp.integral(x)
                return r

        # Extrapolation
        # CAUTION: whatever the extrapolation we use constant

        extr = (x - self.x1) * self.y1

        return r + extr


    # ====================================================================================================
    # Scale along X and Y

    def scale(self, scale):
        sc = np.array(scale)
        if len(sc.shape) == 0:
            scx = scale
            scy = 1.
        else:
            scx = sc[0]
            scy = sc[1]


        a = self.bpoints[2:] - self.bpoints[1]
        a[:, 0] *= scx
        a[:, 1] *= scy

        self.bpoints[2:] = self.bpoints[1] + a

    # ====================================================================================================
    # Translate along X and Y

    def translate(self, vec):
        v = np.array(vec)
        if len(v.shape) == 0:
            vx = vec
            vy = 0.
        else:
            vx = v[0]
            vy = v[1]

        self.bpoints += (vx, vy)

    # ====================================================================================================
    # Call

    def _plot(self, count=100, margin=0., fcomp=None, integ=False):
        
        import matplotlib.pyplot as plt

        x0 = self.x0
        x1 = self.x1
        amp = x1-x0

        x0 -= margin*amp
        x1 += margin*amp
        dx = (x1-x0)/(count-1)

        xs = np.arange(x0, x1+dx, dx, dtype=np.float)

        fig, ax = plt.subplots()
        ys = self(xs)
        ax.plot(xs, ys)

        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])

        if integ:
            ax.plot(xs, [self.integral(x) for x in xs])

        ax.set(xlabel='x', ylabel='easing',
               title=f"{self}"[:60])
        ax.grid()

        fig.savefig("test.png")
        plt.show()


def test_c(count=10):

    interps = list(Interpolation.INTERPS.keys())
    easings = Interpolation.EASINGS

    wfc = Interpolations()
    x0 = 0.
    y0 = 0.
    for i in range(count):
        x1 = x0 + 1
        y1 = y0 + (np.random.random_sample()-0.5)*2
        itp = Interpolation(
            interps[np.random.randint(len(interps))],
            easings[np.random.randint(len(easings))],
            input=input, output=output
            )
        wfc.append(itp)
        x0 = x1
        y0 = y1

    print(wfc)
    wfc._plot(count=1000, integ=True)
    #wfc.scale((0.5, 2))
    #wfc._plot(count=1000, integ=True)


#test_c()

def test_int():
    itp = Interpolation.Bezier(240, 340, 0, 1.2)
    itp._plot(integ=False)

#test_int()
