#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:25:51 2021

@author: alain
"""

import numpy as np

# ====================================================================================================
# Bezier curves are made of control points:
# - points
# - left control points
# - right control points
#
# Left and right control points can be computed from points
# The curves can be 2D or 3D
# One curve is made of count points
# Curves can be passed in an array with any user shape
#
# The shape of a set of Bezier curves is (shape, count, vdim), for instance:
#
# - (2, 3, 10, 3) : array (2, 3) of Bezier curves made of 10 3D-points
#
# The computation of the point at value t returns an array of shape (shape, vdim):
# - (2, 3, 10, 3) --> (2, 3, 3)
#
# The computation can be made with an array of t. If n is the length of t,
# the returned shape is (n, shape, vdim)
# - (2, 3, 10, 3) (100) --> (100, 2, 3, 3)

# ====================================================================================================
# Compute the tangent of a function
#
# The function returns either a value or an array of values

def tangent(f, t, dt=0.01):
    """Compute the tangent of a function
    
    Parameters
    ----------
    f : function
        The function to derive. Template is f(t) = float or array of floats
        
    t : float or array of floats
        The points where to compute the derivatives
        
    dt : float, default = 0.01
        Value to use to compute the tangent
        
    Returns
    -------
    same type as f(t) or array of this type
    """
    
    return (f(t + dt/2) - f(t - dt/2)) / dt

# ====================================================================================================
# Compute bezier control points from a function

def control_points(f, count, t0=0., t1=1., dt=None):
    """Compute the Bezier control points from a function.
    
    For one single value t, f returns a single float or a shaped array of floats:
    If f returns a single value, the shaped of the return array is (count, vdim)
    If f returns a shaped array, the return array is (shape, count, vdim)
    
    Parameters
    ----------
    f : function of template f(t) = point or array of points
        The function to use
        
    count : int
        The number of control points to compute
        
    t0 : float, default = 0.
        The starting value
        
    t1 : float, detault = 1.
        The ending value
        
    dt : float, default = 0.0001
        The value to use to compute the derivative
        
    Returns
    -------
    points, left control points, right control points
    """
    
    if dt is None:
        dt=(t1-t0)/10000
    
    count  = max(2, count)
    delta  = (t1 - t0) / (count - 1)
    ts     = np.linspace(t0, t1, count)
    
    try:
        points = f(ts)
        ders   = (f(ts+dt) - f(ts-dt)) /2 /dt

    except:
        points = np.array([f(t)    for t in ts])
        d1     = np.array([f(t+dt) for t in ts])
        d0     = np.array([f(t-dt) for t in ts])
        ders   = (d1 - d0) /2 /dt

    ders *= delta / 3

    return points, points - ders, points + ders    

    
# ====================================================================================================
# Bezier interpolations from an array of series of Bezier control points

class Beziers():
    """Bezier interpolations from an array of series of Bezier control points.
    
    Can manage an array of curves:
        The shape of the points is (shape, count, 3)
        
    Left anf right interpolation points are optional. If they are not provided,
    they are computed.
    """

    def __init__(self, points, lefts=None, rights=None, t0=0., t1=1.):
        """
        Parameters
        ----------
        points : series of vectors or array of series of vectors
            The series of points to interpolate
            
        lefts : series of vectors or array of series of vectors, optional
            The left control points
            
        rights : series of vectors or array of series of vectors, optional
            The right control points 
        """

        self.points = np.array(points)
        self.t0 = t0
        self.t1 = t1
        
        # Compute the derivatives if lefts and rights are not given
        
        der = np.empty(self.points.shape, float)
        der[..., 1:-1, :] = self.points[..., 2:, :] - self.points[..., :-2, :]
        der[..., 0, :]    = (self.points[..., 1,:] - self.points[..., 0, :])/2
        der[..., -1, :]   = (self.points[..., -1, :] - self.points[..., -2, :])/2
        
        nrm       = np.linalg.norm(der, axis=-1)
        nrm[abs(nrm) < 0.001] = 1.
        der = der / np.expand_dims(nrm, axis=-1)
        
        dists = np.expand_dims(np.linalg.norm(self.points[..., 1:, :] - self.points[..., :-1, :], axis=-1), axis=-1)
        
        self.lefts = np.array(self.points)
        if lefts is None:
            self.lefts[..., 1:, :] -= der[..., 1:, :]*dists/3
            self.lefts[..., 0, :]  -= der[..., 0, :]*dists[..., 0, :]/3
        else:
            self.lefts[:] = lefts

        self.rights = np.array(self.points)
        if rights is None:
            self.rights[..., :-1, :] += der[..., :-1,:]*dists/3
            self.rights[..., -1, :]  += der[..., -1, :]*dists[..., -1, :]/3
        else:
            self.rights[:] = rights
            
        # ---------------------------------------------------------------------------
        # Compute the computation arrays
        #
        # P = (1-t)^3P0 + 3t(1-t)^2P1 + 3t^2(1-t)P2 + t^3P3
        # P0 : 1 - 3t + 3t^2 - t^3
        # P1 : 3t - 6t^2 + 3t^3
        # P2 : 3t^2 - 3t^3
        # P3 : t^3
        #
        # P = t^3(-P0 + 3P1 - 3P2 + P3) + t^2.3(P0 - 2P1 + P2) + t.3(-P0 + P1) + P0
        # P(0) = P0
        # P(1) = P3
        #
        # Let's note:
        # B0 = P3 - 3P2 + 3P1 - P0
        # B1 = 3(P0 - 2P1 + P2)
        # B2 = 3(P1 - P0)
        # B3 = P0
        #
        # We have:
        #
        # P = t^3.B0 + t^2.B1 + t.B2 + P0 = t(t(t.B0 + B1) + B2) + P0
        #
        # Hence:
        # P' = 3t^2.B0 + 2.t.B1 + B2 = t(3t.B0 + 2B1) + B2
        # P'(0) = B2 = 3(P1 - P0)
        # P(1)  = 3B0 + 2B1 + B2 = 3P3-9P2+9P1-3P0 + 6P0-12P1+6P2 + 3P1-3P0 = 3(P3 - P2) 
        #
        # And for acceleration
        # P'' = 6t.B0 + 2.B1
        #
        # ------- With custom parameter x from x0 to x1
        #
        # Q(x) = P(t(x)) with t = (x-x0)/(x1-x0)
        # dQ/dx = t'.P'(t)
        # Q' = P'/(x1-x0)
        
        
        self.B3 = self.points[..., :-1, :]
        
        self.B0 = self.points[..., 1:, :] - 3*self.lefts[..., 1:, :] + 3*self.rights[..., :-1, :] - self.B3
        self.B1 = 3*(self.B3 - 2*self.rights[..., :-1, :] + self.lefts[..., 1:, :])
        self.B2 = 3*(self.rights[..., :-1, :] - self.B3)
        
        
    @classmethod
    def FromFunction(cls, f, count, t0=0., t1=1., dt=None):
        v, l, f = control_points(f, count, t0, t1, dt)
        return cls(v, l, f, t0=t0, t1=t1)
        
            
    def __repr__(self):
        return f"<Array{self.shape} of Bezier curves ({self.size}) made of {self.count} {self.vdim}D-points>"
            
    # ====================================================================================================
    # Dimensions of the curves
            
    @property
    def shape(self):
        """Shape of array of curves.
        
        The shape of the control points array is (shape, count, vdim)

        Returns
        -------
        tuple
            User shape of the array of Bezier curves.
        """
        
        return self.points.shape[:-2]
    
    @property
    def size(self):
        """Size of array of curves.
        
        Size of tyhe shape of the control points array is (shape, count, vdim)

        Returns
        -------
        int
            Number of Bezier curves.
        """
        
        shape = self.shape
        return 1 if shape == () else np.product(shape)
    
    @property
    def count(self):
        """Number of control points per Bezier curve.

        Returns
        -------
        int
            How many control points in each curve.
        """
        
        return self.points.shape[-2]
    
    @property
    def vdim(self):
        """Dimension of the control points vectors.

        Returns
        -------
        int
            Dimension of the control points vectors.
        """
        
        return self.points.shape[-1] 
    
    # ====================================================================================================
    # Compute the points on a value or array of values
            
    def __call__(self, t, individuals=False, der=0):
        """Compute the interpolation.
        
        individuals indicates if there is one time per curve:
            
        - individuals = False (default)
            - points shape: (shape, count, 3)
            - t shape:      (n,)
            - Result shape: (shape, n, 3)
        
        - individuals = True
            - points shape: (shape, count, 3)
            - t shape:      (shape)
            - Result shape: (shape, 3)
        
        Parameters
        ----------
        t : float or array of floats
            The values where to compute the interpolation
        individuals : bool, default = False
            True if there is one time per curve.
        der : int. Default 0
            Return the value (0), the tangent (1) or the acceleration (2)
        
        Returns
        -------
        vector or array of vectors
        """
        
        # ---------------------------------------------------------------------------
        # Check that individuals argument is correct
        
        if individuals:
            if self.shape != np.shape(t):
                s = "Beziers call error: individuals=True requires the argument t to have the same shape as the Beziers class.\n"
                s += f"   Beziers.shape: {self.shape}\n"
                s += f"   t.shape:       {np.shape(t)}\n"
                s += f"Beziers: {self}"
                raise RuntimeError(s)
                
            if self.shape == ():
                return self(t, individuals=False)

        # ---------------------------------------------------------------------------
        # Parameter is a single float
                
        else:
            if not hasattr(t, '__len__'):
                return self([t])[..., 0, :]

        n = np.product(np.shape(t))
            
        # ---------------------------------------------------------------------------
        # Conversion from custom range to [0, 1]
        
        DT = self.t1 - self.t0
        ts = (np.reshape(np.array(t, float), (n,)) - self.t0)/DT
        
        # ---------------------------------------------------------------------------
        # Ensure the parameter t is within interval [0, 1]
        
        ts[np.where(ts < 0)] = 0
        ts[np.where(ts > 1)] = 1
        
        # ---------------------------------------------------------------------------
        # Get the indices of the points intervals
        
        count = self.count 
        
        inds = (ts*(count-1)).astype(int)
        inds[inds == count-1] = count-2   # Ensure inds+1 won't crash
        
        # Location within the interval
        delta = 1/(count - 1)  

        # ---------------------------------------------------------------------------
        # Beziers coefficients
        
        ps = ((ts - inds*delta) / delta).reshape(n, 1) # 1 for inds == count-1 shifted to count-2 !
        
        if individuals:
            # Shape to linear array of control points (count, count... count)
            lsh = ((count-1)*n, 3)
            
            # Inds to the linear list of control points
            # Reshape to get the target shape
            inds = np.reshape(inds + np.arange(n)*(count-1), self.shape)
            
            if der == 2:
                # P'' = 6t.B0 + 2.B1
                return (6*ps*self.B0.reshape(lsh)[inds] + 2*self.B1.reshape(lsh)[inds])*(DT*DT)
            elif der == 1:
                # P' = t(3t.B0 + 2B1) + B2
                return (ps*(3*ps*self.B0.reshape(lsh)[inds] + 2*self.B1.reshape(lsh)[inds]) + self.B2.reshape(lsh)[inds])*DT
            else:
                # P = t(t(t.B0 + B1) + B2) + B3
                return ps*(ps*(ps*self.B0.reshape(lsh)[inds] + self.B1.reshape(lsh)[inds]) + self.B2.reshape(lsh)[inds]) + self.B3.reshape(lsh)[inds]
            
        else:
            if der == 2:
                # P'' = 6t.B0 + 2.B1
                return (6*ps*self.B0[..., inds, :] + 2*self.B1[..., inds, :])*(DT*DT)
            elif der == 1:
                # P' = t(3t.B0 + 2B1) + B2
                return (ps*(3*ps*self.B0[..., inds, :] + 2*self.B1[..., inds, :]) + self.B2[...,inds, :])*DT
            else:
                # P = t(t(t.B0 + B1) + B2) + B3
                return ps*(ps*(ps*self.B0[..., inds, :] + self.B1[..., inds, :]) + self.B2[..., inds, :]) + self.B3[..., inds, :]
    
    # ====================================================================================================
    # Derivative
    
    def tangent(self, t, individuals=False):
        return self(t, individuals=individuals, der=1)
    
    def curvature(self, t, individuals=False):
        return self(t, individuals=individuals, der=2)
    
    # ====================================================================================================
    # Param of the closest point
    
    def closest_param(self, point, resolution=1000):
        i = np.argmin(np.linalg.norm(self(np.linspace(self.t0, self.t1, resolution)) - point, axis=-1))
        return self.t0 + i/(resolution-1)*(self.t1-self.t0)
    
    
    # ====================================================================================================
    # The control points

    def control_points(self):
        """Return the points and the left and rigth control points"""
        
        return self.points, self.lefts, self.rights
    
    # ====================================================================================================
    # The function giving the bezier parameter from the distance on the curve
    # Return the function and the length of the curve
    
    def distance_to_param(self, resolution=1000):

        ts     = np.linspace(self.t0, self.t1, resolution)
        pts    = self(ts)
        
        ds     = np.zeros(resolution, float)
        ds[1:] = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
        
        return Polynoms(np.cumsum(ds), ts, linear=False), np.sum(ds)
    
    
    # ====================================================================================================
    # Write / read
    
    def to_text(self):
        n = len(self.points)
        s = f"{n}\n"
        for i in range(n):
            s += f"{self.points[i, 0]}; {self.points[i, 1]}; {self.points[i, 2]};"
            s += f"{self.lefts [i, 0]}; {self.lefts [i, 1]}; {self.lefts [i, 2]};"
            s += f"{self.rights[i, 0]}; {self.rights[i, 1]}; {self.rights[i, 2]}\n"
        return s
        
    @classmethod
    def FromText(cls, text):
        
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line != "":
                n = int(line)
                
                p = np.zeros((n, 3), float)
                l = np.zeros((n, 3), float)
                r = np.zeros((n, 3), float)
                
                i_start = i+1
                break
        
        for i in range(n):
            v = lines[i_start + i].split(";")
            p[i, 0] = float(v[0])
            p[i, 1] = float(v[1])
            p[i, 2] = float(v[2])
    
            l[i, 0] = float(v[3])
            l[i, 1] = float(v[4])
            l[i, 2] = float(v[5])
    
            r[i, 0] = float(v[6])
            r[i, 1] = float(v[7])
            r[i, 2] = float(v[8])
            
        return cls(p, l, r)
    

    # ====================================================================================================
    # To Blender curve vertices
    # A Blender curve can be initialized in one shot with the profile structure
    # Profile is an array with one entry per curve. Each entry has 3 values:
    # - 1 or 3: multiplicator of the number of vertices (3 for Bezier)
    # - n     : number of vertices
    # - type  : curve type, 0 for Bezier
    
    def blender_vertices(self):
        
        profile = np.empty((self.size, 3), int)
        profile[:] = (3, self.count, 0)
        
        verts = np.zeros((self.shape + (3, self.count, 3)), float)
        verts[..., 0, :, :self.vdim] = self.points
        verts[..., 1, :, :self.vdim] = self.lefts
        verts[..., 2, :, :self.vdim] = self.rights
        
        return verts.reshape(self.size*self.count*3, 3), profile
    
    # ====================================================================================================
    # DEBUG : plot the curves
    
    def plot(self, count=100, points=True, controls=False, curves=True, title=None):
        """Plot the interpolation"""
        
        import matplotlib.pyplot as plt
        
        pts = self.points.reshape(self.size, self.count, self.vdim)
        lfs = self.lefts.reshape(self.size, self.count, self.vdim)
        rgs = self.rights.reshape(self.size, self.count, self.vdim)
        
        fig, ax = plt.subplots()
        if points:
            for i in range(self.size):
                ax.plot(pts[i, :, 0], pts[i, :, 1], 'o')
                if controls:
                    ax.plot(lfs[i, :, 0], lfs[i, :, 1], 'o')
                    ax.plot(rgs[i, :, 0], rgs[i, :, 1], 'o')
        
        t = np.linspace(self.t0, self.t1, 100)
        p = self(t)
        
        p = p.reshape(self.size, len(t), self.vdim)
        
        if curves:
            for i in range(self.size):
                ax.plot(p[i, :, 0], p[i, :, 1], '-')
                
        if title is not None:
            ax.set_title(title)
        
        plt.show()
        
    # ====================================================================================================
    # DEBUG : a demo
        
    @classmethod
    def demo(cls, shape=(), ctl_points=True, seed=0):
        """A simple demo
        
        Parameters
        ----------
        ctl_points : bool, default = True
            Compute the left and right control points
            
        Returns
        -------
        plot with the control points and the interpolated curve
        
        """
        
        # ---------------------------------------------------------------------------
        # A damped sinusoid
        
        def f(x, seed=0):
            
            np.random.seed(seed)
            a = np.random.normal(1, .4)
            w = np.random.normal(1, .4)
            
            n = len(x) if hasattr(x, '__len__') else 1 
            
            p = np.zeros((n, 2), float)
            p[:, 0] = x
            p[:, 1] = a*np.sin(w*x)/x
            
            if hasattr(x, '__len__'):
                return p
            else:
                return p[0]
            
        # ---------------------------------------------------------------------------
        # A shaped array of damped sinusoid
        
        def fa(x, shape, seed=0):
            
            if not hasattr(shape, '__len__'): shape = (shape,)
            size = int(np.product(shape))
            n = len(x) if hasattr(x, '__len__') else 1 
            
            r = np.empty((size, n, 2), float)
            
            for i in range(size):
                r[i] = f(x, seed=seed+i)
                
            if hasattr(x, '__len__'):
                return r.reshape(shape + (n, 2))
            else:
                return r.reshape((n, 2))
            
        if seed is None: seed = np.random.randint(10000000)
                
        
        # Shape
        size = 1 if shape == () else np.product(shape)
        
        # Number of points
        count= 10

        if ctl_points:
            
            p, l, r = control_points(lambda x : fa(x, shape, seed=seed), count, 0.001, 10.)
            
        else:
            p = np.zeros((size,) + (count, 2), float)
            x = np.linspace(0.01, 10., count)
            for i in range(size):
                p[i] = f(x, seed+i)
            p = p.reshape(shape + (count, 2))
            l = None
            r = None
            
        beziers = cls(p, l, r)
        beziers.plot()
        
        return beziers
    
# ====================================================================================================
# Return Bezier control points from Bezier control points

def bezier_control_points(count, verts, lefts=None, rights=None):
    """Compute count control points from any series of Bezier control points.
    
    Used both to reduce the number of points for interpolation and
    to compute left and rigth control points.
    
    Parameters
    ----------
    count : int
        Number of control points to compute
        
    verts : array of vectors
        The interpolations points
        
    lefts : array of vectors, default None
        Left interpolations points
        
    rights : array of vectors, default None
        Right interpolations points
        
    Returns
    -------
    3 arrays of count vectors
        control points, left control points, right control points        
    """        
        
    beziers = Beziers(verts, lefts, rights)
    return control_points(beziers, count)


# ====================================================================================================
# Interpolation 2D polynom

class Polynoms():
    """Interpolate x, y values with polynoms
    
    Each interval is interpolated by a polynom of degree 3 to 
    produce a continuous curve between intervals.
    First and last interval are only of degree 2.
    
    The set of polynoms is a shaped array:
        x & y: array(shape, count) of floats
        
    The call Polynoms(t) return an array(shape) of floats
    The call Polynoms(array(n) of t) retuens an array(shape, n) of floats
    """
    
    def __init__(self, x, y, periodic=False, linear=False):
        """
        Parameters
        ----------
        x : array of floats
            The x values
            
        y : array of floats
            the y values
            
        periodic : bool
            The interpolation is periodic
        """
        
        self.y = np.array(y)
        self.x = np.empty(self.y.shape, float)
        self.x[:] = x
        
        self.periodic = periodic
        self.linear   = linear
        
        self.a, self.b, self.c, self.d = Polynoms.coefficients(self.x, self.y, linear=self.linear)
        
    def __repr__(self):
        return f"<Array{self.shape} of polynomial curves ({self.size}) made of {self.count} points>"
        
    # ====================================================================================================
    # Dimensions of the polynomial curves
    
    @property
    def shape(self):
        return self.x.shape[:-1]
    
    @property
    def size(self):
        shape = self.shape
        return 1 if self.shape == () else np.product(shape)
    
    @property
    def count(self):
        return self.x.shape[-1]
    
    # ====================================================================================================
    # The x intervals
        
    @property
    def delta_x(self):
        """The lengths of the intervals
        
        Returns
        -------
        array of floats
            The dim of the array is one lesser than the number of points
        """
        return self.x[..., -1] - self.x[..., 0]
    
    # ====================================================================================================
    # Inverted functions
    
    @property
    def invert(self):
        """Return the invert interpolation
        
        Returns
        -------
        InterpolationPolynom
            Interpolations computed with y, x
        """
        
        return Polynoms(self.y, self.x, periodic=self.periodic, linear=self.linear)
    
    # ====================================================================================================
    # Compute the polynomial coefficients
        
    @staticmethod
    def coefficients(x, y, linear=False):
        """Compute the coefficients of the polynoms from the values
        
        Parameters
        ----------
        x : array of floats
            The x values
            
        y : array of floats
            The y values
            
        Returns
        -------
        4 array of floats
            The a, b, c, d coefficients of the intervals
        """
        
        shape = np.shape(x)[:-1]
        n = np.shape(x)[-1]
        
        # For each curve, there are:
        # n-1 : polynoms from 0 to n-2
        # n-3 : polynoms of degree 3 from 1 to n-3
        # 1   : polynom of degree 2 at 0
        # 1   : polynom of degree 2 at n-2
        
        # ----- The four polynoms coefficients
        
        a = np.zeros(shape + (n-1,), float)
        b = np.zeros(shape + (n-1,), float)
        c = np.zeros(shape + (n-1,), float)
        d = np.zeros(shape + (n-1,), float)
        
        
        # ===========================================================================
        # Linear coefficients
        # f(t) = y0 + (t-x0)/(x1-x0)*(y1-y0)
        # c = (y1-y0)/(x1-x0)
        # f(t) = t*c + y0 - x0*c
        
        if linear:
            c = (y[..., 1:] - y[..., :-1])/(x[..., 1:] - x[..., :-1])
            d = y[..., :-1] - x[...,:-1]*c
            
            return a, b, c, d
        
        # ===========================================================================
        # Polynomial coefficients
        
        # ----- Compute the n-2 derivatives (excluding 0 and n-1 extremity points)
        
        ders = (y[..., 2:] - y[..., :-2])/(x[..., 2:] - x[..., :-2])
        
        # ----- Helpers

        d1 = ders[..., :-1]
        d2 = ders[..., 1:]
        
        x_2 = x*x
        
        x1 = x[..., 1:-2]
        x2 = x[..., 2:-1]
        y1 = y[..., 1:-2]
        y2 = y[..., 2:-1]
        
        x21 = x2 - x1
        
        # ----- n-3 polynoms of degree 3
        
        a[..., 1:-1] = (d1 + d2 - 2*(y2 - y1)/x21)  / x21**2
        b[..., 1:-1] = -1.5*a[..., 1:-1]*(x2 + x1) + 0.5*(d2 - d1)/x21
        c[..., 1:-1] = 3*a[..., 1:-1]*x1*x2 + (d1*x2 - d2*x1)/x21
        
        # ----- First polynom of degree 2 (a[0] is initialized to 0)
        
        dx = x[..., 1] - x[..., 0]
        b[..., 0] = (ders[..., 0] - (y[..., 1] - y[..., 0])/dx)/dx
        c[..., 0] = ders[..., 0] - 2*b[..., 0]*x[..., 1]
        
        dx = (x[..., -1] - x[..., -2])
        b[..., -1] = - (ders[..., -1] - (y[..., -1] - y[..., -2])/dx)/dx
        c[..., -1] = ders[..., -1] - 2*b[..., -1]*x[..., -2]
        
        # ----- Computation of d
        
        d = y[..., :-1] - a*x_2[..., :-1]*x[..., :-1] - b*x_2[..., :-1] - c*x[..., :-1]   
        
        # ----- We are good
        
        return a, b, c, d            
            

        
    # ====================================================================================================
    # Compute the values
    
    def __call__(self, t):
        """Compute the interpolation of values.
        
        If x is a single value, return an array of shape floats.
        If x is an array of n floats, return an array (shape, n) floats
        
        Parameters
        ----------
        x : array of floats
            The values where to compute the interpolation
            
        Returns
        -------
        array of floats
        """
        
        # Only a single value
        if not hasattr(t, "__len__"):
            return self([t])[..., 0]
        
        t = np.array(t)
        n = len(t)
        
        # Modulo
        if self.periodic:
            t = np.mod(t, self.delta_x)
            
        # results
        y = np.zeros(self.shape + (n,), float)
        
        # Loop on the curves
        ncurves = self.count - 1   #len(self.x) - 1
        xshape = self.shape + (1,)
        for i in range(ncurves):
            if i == 0:
                wh = t <= np.reshape(self.x[..., 1], xshape)
            elif i == ncurves-1:
                wh = t > np.reshape(self.x[..., ncurves-1], xshape)
            else:
                wh = np.logical_and(t > np.reshape(self.x[..., i], xshape) , t <= np.reshape(self.x[..., i+1], xshape))
            
            k = np.argwhere(wh)
            t_ind = k[..., -1]
            s_ind = k[..., :-1]
                
            if len(k) > 0:
                
                t_2 = t[t_ind] * t[t_ind]
                
                s = tuple(np.insert(s_ind, len(self.shape), i, axis=-1).transpose())
                
                aa = np.squeeze(self.a[s])*t_2*t[t_ind]
                bb = np.squeeze(self.b[s])*t_2
                cc = np.squeeze(self.c[s])*t[t_ind]
                dd = np.squeeze(self.d[s])
                
                y[wh] = aa + bb + cc + dd
                
        return y
    
    # ====================================================================================================
    # Debug: plot the functions
    
    def plot(self, x, points=True, curves=True):
        """Plot the interpolation curve
        
        Parameters
        ----------
        dx : float
            Starts and ends the curve outside the x bounds
        """
        
        import matplotlib.pyplot as plt
        
        # ----- The points
        
        fig, ax = plt.subplots(1, 1)
        if points:
            for i in range(self.size):
                ax.plot(np.reshape(self.x, (self.size, self.count))[i], np.reshape(self.y, (self.size, self.count))[i], 'o')
        
        # ----- The curve
        
        #if dx is None:
        #    dx = 0.1*self.delta_x
            
        #x = np.linspace(self.x[0] - dx, self.x[-1] + dx, 100)
        y = self(x)
        size = int(np.product(np.shape(y)[:-1]))
        if curves:
            if size == 0:
                ax.plot(x, y)
            else:
                y = np.reshape(y, (size, len(x)))
                for i in range(size):
                    ax.plot(x, y[i])
        
        plt.show()
        
    # ====================================================================================================
    # Debug: plot the functions
        
    @classmethod
    def demo(cls, linear=False):
        x = np.linspace(0, 2*np.pi,7)
        y = np.sin(x)
        
        f = cls(x, y, linear=linear)
        f.plot(np.linspace(-1, 7, 100))
        
        return f
    

