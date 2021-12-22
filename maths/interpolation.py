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

try:
    from ..core.commons import WError
except:
    WError = RuntimeError
    

# =============================================================================================================================
# Useful constants

zero  = 1e-6
twopi = pi*2
hlfpi = pi/2

# =============================================================================================================================
# Rectangle 

class Rect():
    """A rectangular zone.
    
    Initialized witn two points. Give simple access to x, y locations and amplitudes.
    Provides conversion to and from square (0, 0) (1, 1)
    """
    
    def __init__(self, P0=(0., 0.), P1=(1., 1.)):
        """Rectangular zone
        

        Parameters
        ----------
        P0 : couple of floats, optional
            The low left corner. The default is (0., 0.).
        P1 : couple of floats, optional
            The upper right corner. The default is (1., 1.).

        Returns
        -------
        None.
        """
        self.x0 = P0[0]
        self.y0 = P0[1]
        self.x1 = P1[0]
        self.y1 = P1[1]
        
    def __repr__(self):
        return f"Rect[P0 ({self.x0:.2f} {self.y0:.2f}) P1 ({self.x1:.2f} {self.y1:.2f})]"
        
    @property
    def P0(self):
        """The low left corner.
        """
        return np.array((self.x0, self.y0))
    
    @P0.setter
    def P0(self, P):
        self.x0 = P[0]
        self.y0 = P[1]

    @property
    def P1(self):
        """The upper right corner.
        """
        
        return np.array((self.x1, self.y1))

    @P1.setter
    def P1(self, P):
        self.x1 = P[0]
        self.y1 = P[1]
        
    @property
    def x_amp(self):
        """The x amplitude of the rectangle.
        """
        
        return self.x1 - self.x0
    
    @property
    def y_amp(self):
        """The y amplitude of the rectangle.
        """
        
        return self.y1 - self.y0
    
    @property
    def A(self):
        """The 2-D amplitude of the rectangle.
        """
        
        return np.array((self.x1 - self.x0, self.y1 - self.y0))
    
    def norm_x(self, x, clip=True):
        """Convert a x value into the [0, 1] interval.

        Parameters
        ----------
        x : float
            The x value to convert.
        clip : bool, optional
            If true, the returne value is clipped in [0, 1]. The default is True.

        Returns
        -------
        float
            Converted value.
        """
        
        r = (x - self.x0)/self.x_amp
        if clip:
            return np.clip(r, 0, 1)
        else:
            return r
        
    def norm_y(self, y, clip=True):
        """Convert a y value into the [0, 1] interval.

        Parameters
        ----------
        y : float
            The y value to convert.
        clip : bool, optional
            If true, the returne value is clipped in [0, 1]. The default is True.

        Returns
        -------
        float
            Converted value.
        """
        
        r = (y - self.y0)/self.y_amp
        if clip:
            return np.clip(r, 0, 1)
        else:
            return r
    
    def exp_x(self, x):
        """Convert a x value from [0, 1] to rectangular interval [x0, x1].

        Parameters
        ----------
        x : float
            The x value to expand.

        Returns
        -------
        float
            Converted value.
        """
        
        return self.x0 + x*self.x_amp
    
    def exp_y(self, y):
        """Convert a y value from [0, 1] to rectangular interval [y0, x1].

        Parameters
        ----------
        y : float
            The y value to expand.

        Returns
        -------
        float
            Converted value.
        """

        return self.y0 + y*self.y_amp
    
    def normalize(self, P, clip=True):
        """Convert a point into the unitary square.

        Parameters
        ----------
        P : couple of floats
            The point to convert.
        clip : bool, optional
            If true, the returne value is clipped in the unitary square. The default is True.

        Returns
        -------
        array of floats
            Converted point.
        """
        
        base = np.array(P) - self.P0
        r = base/self.A
        r[np.isnan(r)] = base[np.isnan(r)]
        if clip:
            return np.clip(r, 0, 1)
        else:
            return r
    
    def expand(self, P):
        """Convert a point from the unitary form to the rectangle.

        Parameters
        ----------
        P : couple of floats
            The point to convert.

        Returns
        -------
        array of floats
            Converted point.
        """
        
        return self.P0 + P*self.A

# =============================================================================================================================
# Interpolation function = function such as f(0) = 0 and f(1) = 1
# Immplemented interpolation functions
#
#        'CONSTANT'  : 'tangents': [0, 0]},
#        'LINEAR'    : 'tangents': [1, 1]},
#        'BEZIER'    : 'tangents': [0, 0]},
#        'SINE'      : 'tangents': [0, 1]},
#        'QUAD'      : 'tangents': [0, 1]},
#        'CUBIC'     : 'tangents': [0, 1]},
#        'QUART'     : 'tangents': [0, 1]},
#        'QUINT'     : 'tangents': [0, 1]},
#        'EXPO'      : 'tangents': [10*np.log(2), 0]},
#        'CIRC'      : 'tangents': [0, 0]},
#        'BACK'      : 'EASE_IN',  'tangents': [0, 0]},
#        'BOUNCE'    : 'EASE_OUT', 'tangents': [0, 0]},
#        'ELASTIC'   : 'tangents': [0, 0]},

# ----------------------------------------------------------------------------------------------------
# Interpolation function

class Easing():
    """A standard easing function.
    
    The Easing class from and to the unitary square. The __call__ function
    can be called with a np.array.
    """
    
    EASINGS = {
            'CONSTANT'  : {'can': 'constant',   'left': 0.,   'right': 0.,   'auto':'EASE_IN'}, 
            'LINEAR'    : {'can': 'linear',     'left': 1.,   'right': 1.,   'auto':'EASE_IN'},
            'BEZIER'    : {'can': 'bezier',     'left': None, 'right': None, 'auto':'EASE_IN'},
            'SINE'      : {'can': 'sine',       'left': 0.,   'right': 1.,   'auto':'EASE_IN'},
            'QUAD'      : {'can': 'quadratic',  'left': 0.,   'right': 2.,   'auto':'EASE_IN'},
            'CUBIC'     : {'can': 'cubic',      'left': 0.,   'right': 3.,   'auto':'EASE_IN'},
            'QUART'     : {'can': 'quartic',    'left': 0.,   'right': 4.,   'auto':'EASE_IN'},
            'QUINT'     : {'can': 'quintic',    'left': 0.,   'right': 5.,   'auto':'EASE_IN'},
            'EXPO'      : {'can': 'exponential','left': 0.,   'right': None, 'auto':'EASE_IN'},
            'CIRC'      : {'can': 'circular',   'left': 0.,   'right': 0.,   'auto':'EASE_IN'},
            
            'BACK'      : {'can': 'back',       'left': 0.,   'right': None, 'auto':'EASE_OUT'},
            'BOUNCE'    : {'can': 'bounce',     'left': 0.,   'right': 0.,   'auto':'EASE_OUT'},
            'ELASTIC'   : {'can': 'elastic',    'left': 0.,   'right': None, 'auto':'EASE_OUT'},
           }
    
    def __init__(self, name='LINEAR', ease='AUTO'):
        """A standard Eassing.
        
        Parameters
        ----------
        name : str, optional
            A valid Easing code. The default is 'LINEAR'.
        ease : str, optional
            A valid ease code. The default is 'AUTO'.

        Returns
        -------
        None.

        """
        
        self.name = name
        self.ease = ease
        
        # Easing parameters
        self.factor    = 10       # Exponential
        self.back      = 1.70158  # Back
        self.bounces   = 3        # Bounce
        self.period    = 0.3      # Elastic
        self.amplitude = 0.4      # Elastic
        self.P1        = np.array((1/3, 0.)) # Bezier
        self.P2        = np.array((2/3, 1.)) # Bezier
        
    @staticmethod
    def check_easing(name, **kwargs):

        synos = {
            'QUADRATIC':   'QUAD',
            'QUARTIC':     'QUART',
            'QUINTIC':     'QUINT',
            'EXPONENTIAL': 'EXPO',
            'CIRCULAR':    'CIRC'
            }
        syno = synos.get(name)
        if syno is not None: name = syno
        
        if not name in Easing.EASINGS:
            raise WError(f"Incorrect easing code: {name}. Valid codes are {Easing.EASINGS.keys()}", **kwargs)
            
        return name
        
    @property
    def name(self):
        """The easing code.

        Returns
        -------
        str
            The easing code.
        """
        
        return self.name_
    
    @name.setter
    def name(self, value):
        
        self.name_ = Easing.check_easing(value, Class="Easing", Method="name", value=value)
        
        easing = self.EASINGS.get(self.name_)
        if easing is None:
            raise WError(f"Unkwnow easing name: {value}",
                    Class = "Easing",
                    Method = "name",
                    value = value)
            
        setattr(self, 'canonic', getattr(self, '_' + easing['can']))
        self.left_     = easing['left']
        self.right_    = easing['right']
        self.auto_ease = easing['auto']
        
    def __repr__(self):
        s = f"<Easing {self.name} {self.auto_ease}"
        if self.name == 'BEZIER':
            s += f" P1: [{self.P1[0]:.2f}, {self.P1[1]:.2f}], P2: [{self.P2[0]:.2f}, {self.P2[1]:.2f}]"
        return s + ">"
        
    # ----- Left and right tangents
    
    @property
    def tangents(self):
        """Compute left and right tangents.
        
        Left and right tangents are used for extrapolation.

        Returns
        -------
        float
            The left tangent.
        float
            The right tangent.

        """
        
        # ----- Bezier curve dont use left_ & right_ attributes
        
        if self.name == 'BEZIER':
            left  = 0.
            right = 0.
            if abs(self.P1[0]) > 0.001: left  = self.P1[1]/self.P1[0]
            if abs(1 - self.P2[0]) > 0.001: right = (1 - self.P2[1]) / (1 - self.P2[0])
            
            return left, right
        
        # ----- Non Bezier
            
        ease = self.auto_ease if self.ease == 'AUTO' else self.ease
        
        left = self.left_
        right = self.right_

        if left is None:  left  = (self(0.01) - self(0.))*100
        if right is None: right = (self(1.) - self(0.99))*100
        
        if ease == 'EASE_IN':
            return left, right
        elif ease == 'EASE_OUT':
            return right, left
        else:
            return left, left
        
    @property
    def left(self):
        """The left tangent.
        """
        
        left, right = self.tangents
        return left
    
    @property
    def right(self):
        """The right tangent.
        """
        
        left, right = self.tangents
        return right
    
    # ----- The call to the function
    
    def __call__(self, t):
        """Compute the easing function.
        
        Parameters
        ----------
        t : float or array of floats
            The x values where to compute the easing.

        Returns
        -------
        float or array of floats
            The computed values.
        """

        if not hasattr(t, '__len__'):
            return self(np.array([t]))[0]
        
        t = np.array(t)
        
        ease = self.auto_ease if self.ease == 'AUTO' else self.ease

        if (ease == 'EASE_IN') or (self.name == 'BEZIER'):
            return self.canonic(t)
        
        elif ease == 'EASE_OUT':
            return 1 - self.canonic(1 - t)
        
        else:
            y = np.zeros(len(t), np.float)
                
            idx = t < 0.5
            y[idx] = self.canonic(t[idx]*2)/2
            
            idx = np.logical_not(idx)
            y[idx] = 1 - self.canonic(2 - t[idx]*2)/2
                
            return y
        
    # ===========================================================================
    # The easing functions
    # EASE_IN implementation
    # Argument must be a np.array
    
    # ---------------------------------------------------------------------------
    # Linear

    def _linear(self, t):
        return t

    # ---------------------------------------------------------------------------
    # Constant interpolation
    
    def _constant(self, t):
        y = np.zeros_like(t)
        y[t == 1.] = 1
        return y

    # ---------------------------------------------------------------------------
    # Sine interpolation
    
    def _sine(self, t):
        return 1 - np.cos(t * hlfpi)

    # ---------------------------------------------------------------------------
    # Quadratic interpolation
    
    def _quadratic(self, t):
        return t*t
    
    # ---------------------------------------------------------------------------
    # Cubic interpolation
    
    def _cubic(self, t):
        return t*t*t
    
    # ---------------------------------------------------------------------------
    # Quartic interpolation
    
    def _quartic(self, t):
        t2 = t*t
        return t2*t2

    # ---------------------------------------------------------------------------
    # Quintic interpolation
    
    def _quintic(self, t):
        t2 = t*t
        return t2*t2*t
    
    # ---------------------------------------------------------------------------
    # Exponential interpolation
    
    def _exponential(self, t):
        return np.power(2, -self.factor * (1. - t))
    
    # ---------------------------------------------------------------------------
    # Circular interpolation
    
    def _circular(self, t):
        return 1 - np.sqrt(1 - t*t)

    # ---------------------------------------------------------------------------
    # Back interpolation
    
    def _back(self, t):
        return t*t*((self.back + 1)*t - self.back)
    
    # ---------------------------------------------------------------------------
    # Bounce interpolation
    
    def _bounce(self, t):

        # Number of bounces
        n = len(self.di)
    
        # Duplicaton of the t for each of the intervals starting abscissa
        # NOTE: 1-t because the computation is made on falling from 1 to 0
        # but the default ease in is time reversed from 0 to 1
    
        ats = np.full((n, len(t)), 1 - t).transpose()
    
        # Distances to the parabola centers
        y = ats - self.ci
    
        # Parabola equation
        # a and di are supposed to be correctly initialized :-)
        y = self.a*y*y + self.di
    
        # Return the max values (normally, only one positive value per line)
    
        return np.max(y, axis=1)  
    
    # ---------------------------------------------------------------------------
    # Bounce utilities
    
    @property
    def bounces(self):
        return len(self.di)
    
    @bounces.setter
    def bounces(self, n):
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
        self.xi = np.insert(xi[:-1], 0, 0)          # 0, q/2, q, ...
        self.di = np.insert(a * di * di / 4, 0, 1)  # Parabola equation : a*x*x + di
        self.ci = np.insert(xi[:-1] + di/2, 0, 0)            

    # ---------------------------------------------------------------------------
    # Elastic interpolation
    
    def _elastic(self, t):

        period    = .3 if self.period < 0.0001 else self.period
        amplitude = self.amplitude
        
        # t = -np.array(t)  # OUT
        t = -np.array(1-t)  # IN
        f = np.ones_like(t)
        
        if period < 0.0001:
            period = .3
            
        if amplitude < 1:
            
            s = period / 4
            f *= amplitude
                
            ids = np.where(abs(t) < s)[0]
            l = abs(t[ids])/s
            f[ids] = (f[ids]*l) + (1. - l)
            
            amplitude = 1.
            
        else:
            
            s = period / twopi * np.arcsin(1/amplitude)
            
        # return (f * (amplitude * np.power(2, 10*ts) * np.sin((t - s) * twopi / period))) + 1 # OUT
        return -(f * (amplitude * np.power(2, 10*t) * np.sin((t - s) * twopi / period)))      # IN

    # ---------------------------------------------------------------------------
    # Elastic utilities
        
    def set_peramp(self, rect, period=None, amplitude=None):
        if period is not None:
            self.period = period/rect.x_amp
        if amplitude is not None:
            self.amplitude= amplitude/rect.y_amp
            
    def get_peramp(self, rect):
        return self.period*rect.x_amp, self.amplitude*rect.y_amp
    
    # ---------------------------------------------------------------------------
    # Bezier interpolation
    
    def _bezier(self, t):
        
        # Bounds of dychotomy computation
        t0 = np.zeros(len(t), np.float)
        t1 = np.ones(len(t), np.float)
        
        # t --> (x, y) points
        # Loop until x = t with a certain precision

        # ----- Bezier computation        
        # P = _u3*P0 + 3*_u2*u*P1 + 3*_u*u2*P2 +  u3*P3
        # P0 = (0, 0) -> No term in P0
        # P# = (1, 1) -> Non P3
        
        u = np.array(t)
        x1 = self.P1[0]
        x2 = self.P2[0]
        
        zero = 0.001
        
        for i in range(15):

            u2   = u*u
            u3   = u2*u

            _u  = 1 - u
            _u2 = _u*_u

            x = 3*_u2*u*x1 +  3*_u*u2*x2 + u3
            ds = x - t
            if max(abs(ds)) < zero:
                break
            
            ineg = ds < -zero
            t0[ineg] = u[ineg] 
            
            ipos = ds > zero
            t1[ipos] = u[ipos] 
            
            ich = np.logical_or(ineg, ipos)
            u[ich] = (t0[ich] + t1[ich])/2

        return 3*_u2*u*self.P1[1] +  3*_u*u2*self.P2[1] + u3
    
    # ---------------------------------------------------------------------------
    # Bezier utilities
    
    def get_bpoints(self, rect=Rect((0., 0.), (1., 1.))):
        return rect.expand(((0., 0.), self.P1, self.P2, (1., 1.)))
    
    def set_bpoints(self, bp):
        rect = Rect(bp[0], bp[3])
        self.P1 = rect.normalize(bp[1])
        self.P2 = rect.normalize(bp[2])

    # ===========================================================================
    # From / to blender keyframe
    
    def from_keyframes(self, kf0, kf1):
        """Create an Interpolation from two Blender key frame structure
        
        Parameters
        ----------
        kf0 : Blender Keyframe
            The first key frame
            
        hf1 : Blender Keyframe
            The second key frame
        """
        
        self.name = kf0.interpolation
        
        rect = Rect(kf0.co, kf1.co)
        if self.name == 'BEZIER':
            
            self.P1 = rect.normalize(kf0.handle_right, clip=False)
            self.P2 = rect.normalize(kf1.handle_left, clip=False)
            
        else:
            #self.factor    = 10
            self.back      = kf0.back
            #self.bounces   = 3
            self.set_peramp(rect, kf0.period, kf0.amplitude)
            
    @classmethod
    def FromKFPoints(cls, kf0, kf1):
        easing = cls(name=kf0.interpolation, ease=kf0.easing)
        easing.from_keyframes(kf0, kf1)
        return easing
        
    # ===========================================================================
    # Develop
    
    # ---------------------------------------------------------------------------
    # Plot
    
    def plot(self, ease='AUTO', xax=None, count=100):
        
        import matplotlib.pyplot as plt
        
        if xax is None:
            fig, ax = plt.subplots()
        else:
            ax = xax
            
        x = np.linspace(0., 1., count)
        ax.plot(x, self(x, ease=ease))
        ax.set(title=self.name + '- ' + ease)
        
        if xax is None:
            plt.show()
            
    @classmethod
    def plot_all(cls, ease='AUTO', codes=None, count=100):
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        for name in cls.EASINGS:
            if codes is None or name in codes:
                easing = Easing(name)
                easing.plot(ease=ease, xax=ax, count=count)
                print(easing(0.25))
            
        plt.show()
        
# =============================================================================================================================
# Interpolation

class BCurve():
    """A collection of easing functions mapped on intervals.
    """
    
    def __init__(self, extrapolation='CONSTANT'):
        """A collection of easing functions mapped on intervals.        

        Parameters
        ----------
        extrapolation : str, optional
            A valid extrapolation code. The default is 'CONSTANT'.

        Returns
        -------
        None.
        """
        
        self.easings = []
        self.points  = np.zeros((0, 2), np.float)
        self.extrapolation = extrapolation
        
    def __repr__(self):
        s = f"<BCurve {len(self.easings)} easing(s), extrapolation: {self.extrapolation}"
        if len(self.points) == 0:
            s += " no points"
        elif len(self.points) == 1:
            s += f" single point: ({self.points[0][0]:.1f} {self.points[0][1]:.1f})]"
        else:
            s += f" into points:\n{self.points}"
            
        s += "\neasings"
        for e in self.easings:
            s += f"\n{e}"
            
        return s + ">"
        
    # ---------------------------------------------------------------------------
    # A simple easing
    
    @classmethod
    def Single(cls, P0, P1, name='LINEAR', ease='AUTO', extrapolation='CONSTANT'):
        """Initialize a new BCurve with a unique easing function.        

        Parameters
        ----------
        P0 : point
            The low left corner of the function rectangle.
        P1 : point
            The high right corner of the function rectangle.
        name : str, optional
            The code of the easing function to use. The default is 'LINEAR'.
        ease : str, optional
            The ease ease to use. The default is 'AUTO'.
        extrapolation : str, optional
            The extrapolation outside the input interval. The default is 'CONSTANT'.

        Returns
        -------
        BCurve
            A BCurve instance.
        """
        
        bc = BCurve(extrapolation=extrapolation)
        bc.points = np.array((P0, P1))
        bc.easings = [Easing(name, ease)]
        return bc
        
    # ---------------------------------------------------------------------------
    # Build from a Blender fcurve
        
    @classmethod
    def FromFCurve(cls, fcurve):
        """Generate a BCurve from a Blender fcuruve.        

        Parameters
        ----------
        fcurve : Blender fcurbe
            The fcurve to copy.

        Returns
        -------
        BCurve
            A Bcurve instance.
        """

        bcurve = cls()
        bcurve.extrapolation = fcurve.extrapolation
        
        # ----- Load the points
        
        count = len(fcurve.keyframe_points)
        
        bcurve.points = np.zeros(count*2, np.float)
        fcurve.keyframe_points.foreach_get('co', bcurve.points)
        bcurve.points = bcurve.points.reshape(count, 2)
        
        # ---- Create the easings
        
        for i in range(count-1):
            kf0 = fcurve.keyframe_points[i]
            kf1 = fcurve.keyframe_points[i+1]
            bcurve.easings.append(Easing.FromKFPoints(kf0, kf1))

        return bcurve
    
    # ---------------------------------------------------------------------------
    # To fcurve
    # Not a true fcurve but an array of keyframe like

    @property
    def keyframe_points(self):
        """Convert the BCurve into an array of Blender keyframes.        

        Returns
        -------
        array of keyframes
            Each item contains attributes required to initialze a valid Blender keyframe.
        """

        class Kf():
            pass

        kfs = []
        for i in range(len(self.points)):
            P = self.points[i]
            if i == len(self.points)-1:
                easing = self[i-1]
                rect   = self.easing_rect(i-1)
            else:
                easing = self[i]
                rect   = self.easing_rect(i)
                
            kf = Kf()
            kf.handle_left  = (-1., 0.)
            kf.handle_right = ( 1., 0.)
            
            if i > 0:
                kf.handle_left = self.easing_rect(i-1).expand(self[i-1].P2)
                
            kf.co            = np.array(P)
            kf.handle_right  = rect.expand(easing.P1)

            kf.interpolation = easing.name
            kf.easing        = easing.ease
            
            p, a = easing.get_peramp(rect)
            kf.amplitude     = a
            kf.period        = p
            
            kf.back          = easing.back
            
            if i == 0:
                kf.handle_left = P - np.array(kf.handle_right - P)
            if i == len(self.points)-1:
                kf.handle_right = P + np.array(P - kf.handle_left)

            kfs.append(kf)

        return kfs
        
        
        
        for i in range(len(self)):
            easing = self[i]
            rect   = self.easing_rect(i)
            
            kf = Kf()
            
            if i > 0:
                kf.handle_left = self.easing_rect(i-1).expand(self[i-1].P2)
            else:
                kf.handle_left = -rect.expand(easing.P1)
                
                
            kf.co            = rect.P0
            print("keyframe_points", type(easing), easing)
            kf.handle_right  = rect.expand(easing.P1)

            kf.interpolation = easing.name
            kf.easing        = easing.ease
            
            p, a = easing.get_peramp(rect)
            kf.amplitude     = a
            kf.period        = p
            
            kf.back          = easing.back

            kfs.append(kf)

        return kfs
    
    # ---------------------------------------------------------------------------
    # Intervals access
    
    def __len__(self):
        return len(self.easings)
    
    def __getitem__(self, index):
        return self.easings[index]
    
    @property
    def rect(self):
        """The full rectangle of the BCurve.        

        Returns
        -------
        Rect
            Starting ending points of the curve.
        """
        
        if len(self.points) < 2:
            return Rect((0, 0), (1, 1))
        else:
            return Rect( (self.points[ 0, 0], np.min(self.points[:, 1])),
                         (self.points[-1, 0], np.max(self.points[:, 1])) )
        
    def easing_rect(self, index):
        """The rect of a given easing function.        

        Parameters
        ----------
        index : int
            Index of the curve.

        Returns
        -------
        Rect
            The rect used to compute the easing function.
        """
        
        if len(self.points) < 2:
            return Rect((0, 0), (1, 1))
        
        if index is None:
            return Rect(self.points[0], self.points[-1])
        else:
            return Rect(self.points[index], self.points[index+1])
        
    # ---------------------------------------------------------------------------
    # Starting point
    
    def set_start_point(self, point):
        """Initialize the BCurve with a starting point.

        Note that this function reset the full BCurve to an empty curve.        

        Parameters
        ----------
        point : point
            The starting point.

        Returns
        -------
        None.
        """
        
        self.easings = []
        #self.points = np.expand_dims(np.array(point, np.float), axis=0)
        self.points = np.array(point).reshape(1, 2)
        
    # ---------------------------------------------------------------------------
    # Add an easing the curve
    # End point is used as start point if it is located before the first existing one
    
    def add_easing(self, end_point, easing=Easing('LINEAR', 'AUTO')):
        """Add and easing to the curve.

        Typical use is to append a new easing at the end of existing ones,
        but if the end point is not after the current end point, the easing
        is insert within the existing ones.

        Parameters
        ----------
        end_point : point
            End of the easing.
        easing : Easing, optional
            The easing to use. The default is Easing('LINEAR', 'AUTO').

        Returns
        -------
        int
            The index of the created easing with the BCurve.
        """
        
        if type(easing) is str:
            easing = Easing(easing, 'AUTO')
        
        zero = 0.0001
        point = np.array(end_point, np.float)
        xp = point[0]
        
        # ----- First point : no easing
        
        if len(self.points) == 0:
            self.points = np.array((point)).reshape((1, 2))
            print(self.points)
            return 0
        
        # ---- First easing
        
        if len(self.points) == 1:
            self.easings = [easing]
            if xp < self.points[0, 0]:
                self.points = np.array((point, self.points[0]))
            else:
                self.points = np.array((self.points[0], point))
            return 0
        
        # ---- Easings exist: let's see if the abscissa is too close 
        # of an existing abscissa
        
        for i, x in enumerate(self.points[:-1, 0]):
            if abs(xp-x) < zero:
                return i
            
        # Equal to the last x
        if abs(self.points[-1, 0] - xp ) < zero:
            return len(self)-1
        
        # ----- So, there is something to insert
        
        for i in range(len(self)):
            if xp < self.points[i, 0]:
                self.easings.append(i, easing)
                self.points = np.insert(self.points, point)
                return i
        
        # ----- Append
        
        self.easings.append(easing)
        self.points = np.append(self.points, [point], axis=0)
        return len(self)-1
    
    # ---------------------------------------------------------------------------
    # Add an easing the curve
    # End point is used as start point if it is located before the first existing one
    
    def add(self, end_point, interpolation='BEZIER', easing='AUTO'):
        """Add and easing to the curve.

        Typical use is to append a new easing at the end of existing ones,
        but if the end point is not after the current end point, the easing
        is insert within the existing ones.

        Parameters
        ----------
        end_point : point
            End of the easing.
        interpolation : str, optional
            The interpolation type. The default is 'BEZIER'.
        easing : str, optional
            The easing. The default is 'AUTO'.

        Returns
        -------
        int
            The index of the created easing with the BCurve.
        """
        
        return self.add_easing(end_point, Easing(interpolation, easing))
    
    
    # ---------------------------------------------------------------------------
    # Compute
    
    def __call__(self, t, xbounds=None, ybounds=None):
        """Compute the blender curve on a array of values.
        
        xBounds and yBounds can provide bounds per value in t.
        This allow to use a curve on various intervals
        
        Parameters
        ----------
        t : array of floats
            The abscissa of the curve
        xbounds : array of couple of floats, optional
            One interval per value to use as the replacement of the default interval.
        ybounds : array of couple of floats, optional
            One interval per value to use as the replacement of the default y interval.
            
        Returns
        -------
        array of float
            The curve values
        """
        
        if not hasattr(t, '__len__'):
            y = self(np.array([t]), xbounds, ybounds)
            if len(y) == 1:
                return y[0]
            else:
                return y

        # Make sure is an array        
        t = np.array(t, np.float)
        y = np.zeros_like(t)
        
        # No curve
        if len(self) == 0:
            return y

        # Default bounds
        bounds = self.rect
        
        # ----- if xbounds exists: convert to default bounds
        if xbounds is not None:
            xbounds = np.resize(xbounds, np.shape(t) + (2,))
            t = bounds.x0 + (t - xbounds[..., 0]) / (xbounds[..., 1] - xbounds[..., 0]) * bounds.x_amp

        # ----- Periodic
        if self.extrapolation == 'PERIODIC':
            t = bounds.x0 + ((t - bounds.x0) % bounds.x_amp)
        
        # ----- Loop on the easings
        
        for i in range(len(self.easings)):
            rect = self.easing_rect(i)
            ts = rect.norm_x(t, clip=False)
            idx = np.logical_and(ts >= 0, ts <= 1)
            if np.any(idx):
                y[idx] = rect.exp_y(self[i](ts[idx]))
                
        # ----- Extrapolation
        
        if self.extrapolation == 'CONSTANT':
            
            y[t<bounds.x0] = self.points[0, 1]
            y[t>bounds.x1] = self.points[-1, 1]
            
        else:
            tg = bounds.y_amp/bounds.x_amp
            
            idx = t < bounds.x0
            y[idx] = bounds.y0 + (t[idx] - bounds.x0)*self[0].left*tg
            
            idx = t > bounds.x1
            y[idx] = bounds.y1 + (t[idx] - bounds.x1)*self[-1].right*tg
                
        # ----- if ybounds exists: convert to target bounds
        
        if ybounds is None:
            return y
        else:
            ybounds = np.resize(ybounds, np.shape(t) + (2,))
            return ybounds[..., 0] + (y - bounds.y0) / bounds.y_amp * (ybounds[..., 1] - ybounds[..., 0])
        
    # ---------------------------------------------------------------------------
    # Integral
    # When the y values are a speed relative to x, the integral allows
    # to get the location at time x

    def integral(self, t0, t1, count=100):
        """Compute the integral of the BCurve between to values.

        Note that if t1 is an array of float, t0 can be a single float value.
        If t0 is an array, it must be of the same shape as t1.        

        Parameters
        ----------
        t0 : float or array of floats
            Start value.
        t1 : float or array of floats
            End value.
        count : int, optional
            Number of steps to use to compute the integral. The default is 100.

        Returns
        -------
        float or array of floats
            The integral or array of integrals.
        """
        
        if np.size(t1) > np.size(t0):
            shape = np.shape(t1)
            size  = np.size(t1)
        else:
            shape = np.shape(t0)
            size  = np.size(t0)
            
        if not hasattr(shape, '__len__'):
            shcount = (shape, count)
        else:
            shcount = shape + (count,)
            
        t = np.linspace(np.resize(t0, size), np.resize(t1, size), count).transpose().reshape(shcount)
        dt = (t1-t0)/(count-1)
        
        # Compute the values and sum on the last axis
        return np.sum(self(t), axis=-1) * dt
                
    # ---------------------------------------------------------------------------
    # Develop
    
    @classmethod
    def Random(cls, count=3):
        
        bcurve = BCurve()
        bcurve.points = np.zeros((count+1, 2), np.float)
        if True: # Random else (0 -> 1)
            bcurve.points[:, 0] = np.linspace(10., 110, count+1)
            bcurve.points[:, 1] = np.random.uniform(10, 20, count+1)
        else:
            bcurve.points[:, 0] = np.linspace(0., 1, count+1)
            bcurve.points[:, 1] = np.linspace(0., 1, count+1)
        
        for i in range(count):
            name = list(Easing.EASINGS)[np.random.randint(0, len(Easing.EASINGS))]
            bcurve.easings.append(Easing(name, ease='EASE_IN' if np.random.randint(2) == 1 else 'EASE_OUT'))

        return bcurve
    
    @classmethod
    def Build(cls):
        bc = BCurve()
        
        bc.set_start_point((10, 5))

        bc.add((15, 7), Easing('CIRCULAR'))
        bc.add((20, -3), Easing('ELASTIC'))
        
        return bc
    
    def plot(self, margin=0., integral=False, count=1000):
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        
        if len(self) > 0:
            
            dx = margin*self.rect.x_amp
            x = np.linspace(self.rect.x0 - dx, self.rect.x1 + dx, count)
            ax.plot(x, self(x))
            ax.plot(self.points[:,0], self.points[:, 1], 's')
            ax.set(title=self[0].name + ' ' + self[0].ease)
            
            if integral:
                ax.plot(x, self.integral(0, x))
        
        plt.show()
        
    def plot_bounds(self, repls=3, count=100):
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        xbounds = np.zeros((repls, 2), np.float)
        ybounds = np.zeros((repls, 2), np.float)
        for i in range(repls):
            xbounds[i] = (i*4, i*4+3)
            ybounds[i] = (i*3, i*5+5)
            
        xbounds += 10
        ybounds -= 10
        
        xr = np.zeros((repls, count), np.float)
        yr = np.zeros((repls, count), np.float)
        
        for i in range(repls):
            xr[i] = np.linspace(xbounds[i, 0], xbounds[i, 1], count)
            
        for i in range(count):
            yr[:, i] = self(xr[:, i], xbounds= xbounds, ybounds= ybounds)
            
        for i in range(repls):
            ax.plot(xr[i], yr[i])
        
        plt.show()
        
        
# =============================================================================================================================
# Usefull interpolation

def interpolation_function(shape="/", name=None):
    
    valid_shapes = ["/", "\\", "V", "^", "S", "2"]
    
    if not shape in valid_shapes:
        raise WError(f"Incorrect shape code {shape}. Valid shapes are {valid_shapes}",
                Function = "interpolation_function",
                shape = shape,
                name = name)
        
    if name is None:
        if shape in ["/", "\\"]:
            name = 'LINEAR'
        else:
            name = 'BEZIER'

    name = Easing.check_easing(name, Function="interpolation_function", shape=shape)
    
    bc = BCurve()
    
    if shape in ["\\", "V", "2"]:
        low  = 1
        high = 0
    else:
        low  = 0
        high = 1
            
    if shape in ['V', '^']:
        ease1 = 'EASE_OUT'
        ease2 = 'EASE_IN'
    elif shape in ['/', '\\']:
        ease1 = 'EASE_IN'
    else:
        ease1 = 'EASE_OUT'
    
            
    bc.set_start_point((0, low))
    bc.add((1, high), interpolation=name, easing=ease1)
    
    if shape in ['V', '^']:
        bc.add((2, low), interpolation=name, easing=ease2)

    return bc

# =============================================================================================================================
# Interpolate function

def interpolate(x, x_min=0., x_max=1., y_min=0., y_max=1., interpolation='LINEAR', extrapolation='CONSTANT'):
    
    bc = BCurve.Single((0., 0.), (1., 1.), name=interpolation, extrapolation=extrapolation)
    
    if np.shape(x) == ():
        return bc(x, [x_min, x_max], [y_min, y_max])
    
    xbounds = np.empty(np.shape(x)+(2,), float)
    ybounds = np.empty(np.shape(x)+(2,), float)
    
    xbounds[:, 0] =  x_min
    xbounds[:, 1] =  x_max
    ybounds[:, 0] =  y_min
    ybounds[:, 1] =  y_max
    
    return bc(x, xbounds, ybounds)

# =============================================================================================================================
# Vector interpolation

def norm_interpolate(v, x_min=0., x_max=1., y_min=0., y_max=1., interpolation='LINEAR', extrapolation='CONSTANT'):
    nrms = np.linalg.norm(v, axis=-1)
    new_nrms = interpolate(nrms, x_min, x_max, y_min, y_max, interpolation, extrapolation)
    nrms[nrms < 1e-8] = 1
    return v * np.expand_dims(new_nrms / nrms, axis=-1)

def test_interpolate():

    import matplotlib.pyplot as plt
    
    count = 50
    x = np.linspace(-2, 3, count)
    
    np.random.seed(0)
    x0 = np.random.normal(-1, .01, count)
    x1 = np.random.normal(2, .01, count)
    y0 = np.random.normal(10, .1, count)
    y1 = np.random.normal(20, .1, count)
    
    y = interpolate(x, x0, x1, y0, y1, 'BEZIER')
    
    fig, ax = plt.subplots()
    
    ax.plot(x, y)
    
    plt.show()

    
    
    


