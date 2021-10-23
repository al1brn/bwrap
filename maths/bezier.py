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
# Bezier interpolations from an array of seroes of Bezier control points

class Beziers():
    """Bezier interpolations from an array of seroes of Bezier control points.
    
    Can manage an array of curves:
        The shape of the points is (shape, count, 3)
        
    Left anf right interpolation points are optional. If they are not provided,
    they are computed.
    """

    def __init__(self, points, lefts=None, rights=None):
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
            
    def __call__(self, t):
        """Compute the interpolation.
        
        - points shape: (shape, count, 3)
        - t shape:      (n,)
        - Result shape: (shape, n, 3)
        
        Parameters
        ----------
        t : float or array of floats
            The values where to compute the interpolation
        
        Returns
        -------
        vector or array of vectors
        """
    
        # Parameter is not an array
        if not hasattr(t, '__len__'):
            return self([t])[..., 0, :]
        
        # Make sure it is an np array
        ts = np.array(t, float)
        n = len(ts)
        
        # And that it is between 0 and 1
        ts[np.where(ts < 0)] = 0
        ts[np.where(ts > 1)] = 1
        
        # Number of bezier points
        count = self.count 
        
        # Indices
        inds = (ts*(count-1)).astype(int)
        inds[inds == count-1] = count-2   # Ensure inds+1 won't crash
        
        # Location within the interval
        delta = 1/(count - 1)             
        
        # Bezier computation
        
        ps = ((ts - inds*delta) / delta).reshape(n, 1) # 1 for inds == count-1 shifted to count-2 !

        ps2  = ps*ps
        ps3  = ps2*ps
        _ps  = 1 - ps
        _ps2 = _ps*_ps
        _ps3 = _ps2*_ps
        
        v  = self.points[..., inds, :]*_ps3
        v += 3*self.rights[..., inds, :]*_ps2*ps
        v += 3*self.lefts[..., inds+1, :]*_ps*ps2
        v += self.points[..., inds+1, :]*ps3
        
        return v
    
    # ====================================================================================================
    # The control points

    def control_points(self):
        """Return the points and the left and rigth control points"""
        
        return self.points, self.lefts, self.rights
    
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
    
    def plot(self, count=100, points=True, controls=False, curves=True):
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
        
        t = np.linspace(0, 1, 100)
        p = self(t)
        
        p = p.reshape(self.size, len(t), self.vdim)
        
        if curves:
            for i in range(self.size):
                ax.plot(p[i, :, 0], p[i, :, 1], '-')
        
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
    
    def __init__(self, x, y, periodic=False):
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
        
        self.a, self.b, self.c, self.d = Polynoms.coefficients(self.x, self.y)
        
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
        
        return Polynoms(self.y, self.x, self.periodic)
    
    # ====================================================================================================
    # Compute the polynomial coefficients
        
    @staticmethod
    def coefficients(x, y):
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
        
        if True:
            
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
            
            
        else:
            n = len(x)                  
            
            # There are:
            # n-1 : polynoms from 0 to n-2
            # n-3 : polynoms of degree 3 from 1 to n-3
            # 1   : polynom of degree 2 at 0
            # 1   : polynom of degree 2 at n-2
            
            # ----- The four polynoms coefficients
            
            a = np.zeros(n-1, np.float)
            b = np.zeros(n-1, np.float)
            c = np.zeros(n-1, np.float)
            d = np.zeros(n-1, np.float)
            
            # ----- Compute the n-2 derivatives (excluding 0 and n-1 extremity points)
            
            ders = (y[2:] - y[:-2])/(x[2:] - x[:-2])
            
            # ----- Helpers
    
            d1 = ders[:-1]
            d2 = ders[1:]
            
            x_2 = x*x
            
            x1 = x[1:-2]
            x2 = x[2:-1]
            y1 = y[1:-2]
            y2 = y[2:-1]
            
            x21 = x2 - x1
            
            # ----- n-3 polynoms of degree 3
            
            a[1:-1] = (d1 + d2 - 2*(y2 - y1)/x21)  / x21**2
            b[1:-1] = -1.5*a[1:-1]*(x2 + x1) + 0.5*(d2 - d1)/x21
            c[1:-1] = 3*a[1:-1]*x1*x2 + (d1*x2 - d2*x1)/x21
            
            # ----- First polynom of degree 2 (a[0] is initialized to 0)
            
            dx = x[1] - x[0]
            b[0] = (ders[0] - (y[1] - y[0])/dx)/dx
            c[0] = ders[0] - 2*b[0]*x[1]
            
            dx = (x[-1] - x[-2])
            b[-1] = - (ders[-1] - (y[-1] - y[-2])/dx)/dx
            c[-1] = ders[-1] - 2*b[-1]*x[-2]
            
            # ----- Computation of d
            
            d = y[:-1] - a*x_2[:-1]*x[:-1] - b*x_2[:-1] - c*x[:-1]   
            
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
                #ks = ks.reshape(len(ks))
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
    def demo(cls):
        x = np.linspace(0, 2*np.pi, 5)
        y = np.sin(x)
        
        f = cls(x, y, True)
        f.plot(7)
        
        return f

