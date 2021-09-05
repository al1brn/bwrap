#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:25:51 2021

@author: alain
"""

import numpy as np

# ---------------------------------------------------------------------------
# Compute the tangent of a function

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
    
    if hasattr(t, '__len__'):
        return (f(t + dt/2) - f(t - dt/2)) / dt
    else:
        return (np.array(f(t + dt/2)) - np.array(f(t - dt/2))) / dt
    
# ---------------------------------------------------------------------------
# Bezier from a function

def control_points(f, count, t0=0., t1=1., dt=0.0001):
    """Compute the Bezier control points of a function
    
    Parameters
    ----------
    f : function of template f(t) = point
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

    count  = max(2, count)
    delta  = (t1 - t0) / (count - 1)
    ts     = t0 + np.arange(count) * delta

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
    
# ---------------------------------------------------------------------------
# List of points interpolated as a curve

class PointsInterpolation():
    """Class to interpolate a series of points
    
    """

    def __init__(self, points, lefts=None, rights=None):
        """
        Parameters
        ----------
        points : array of vectors
            The array of points to interpolate
            
        lefts : array of vectors, optional
            The left control points
            
        rights : array of vectors, optional
            The right control points 
        """

        self.points = np.array(points)
        
        # Compute the derivatives if lefts and rights are not given

        der       = np.array(self.points)
        der[1:-1] = self.points[2:] - self.points[:-2]
        der[0]    = (self.points[1] - self.points[0])/2
        der[-1]   = (self.points[-1] - self.points[-2])/2

        der *= 2/(len(self.points)-1)/3

        if lefts is None:
            self.lefts = self.points - der
        else:
            self.lefts = lefts

        if rights is None:
            self.rights = self.points + der
        else:
            self.rights = rights

    @property
    def count(self):
        """Number of control points"""
        
        return len(self.points)

    def __call__(self, t):
        """Compute the interpolation
        
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
            return self([t])[0]

        # Ok, we have an array as an input
        n     = len(self.points)
        delta = 1./(n - 1)
        
        # Make sure it is an np array
        ts = np.array(t)
        
        # And that it is between 0 and 1
        ts[np.where(ts < 0)] = 0
        ts[np.where(ts > 1)] = 1
        
        # Indices
        inds = (ts*(n-1)).astype(np.int)
        
        # Indices bounds
        inds[np.greater(inds, n-2)] = n-2
        inds[np.less(inds, 0)]      = 0
        
        # Bezier computation
        ps = (ts - inds*delta) / delta

        ps2  = ps*ps
        ps3  = ps2*ps
        _ps  = 1 - ps
        _ps2 = _ps*_ps
        _ps3 = _ps2*_ps

        return self.points[inds]*_ps3[:,np.newaxis] + 3*self.rights[inds]*(_ps2*ps)[:,np.newaxis] + 3*self.lefts[inds+1]*(_ps*ps2)[:,np.newaxis] + self.points[inds+1]*ps3[:,np.newaxis]

    def bezier(self):
        """Return the points and the left and rigth control points"""
        
        return self.points, self.lefts, self.rights
    
    def plot(self, count=100):
        """Plot the interpolation"""
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot(self.points[:, 0], self.points[:, 1], 's')
        
        t = np.linspace(-0.1, 1.1, 100)
        p = self(t)
        
        ax.plot(p[:, 0], p[:, 1])
        
        plt.show()
        
    @staticmethod
    def demo(ctl_points=True):
        """A simple demo
        
        Parameters
        ----------
        ctl_points : bool, default = True
            Compute the left and right control points
            
        Returns
        -------
        plot with the control points and the interpolated curve
        
        """

        # Interpolate a damped sinusoid
        def f(x):
            p = np.zeros((len(x), 2), float)
            p[:, 0] = x
            p[:, 1] = np.sin(x)/x
            return p
        
        # Number of points
        count= 10

        if ctl_points:
            
            p, l, r = control_points(f, count, 0.001, 10.)
            
        else:
            x = np.linspace(0.01, 10., count)
            p = f(x)
            l = None
            r = None
        
        intrp = PointsInterpolation(p, l, r)
        intrp.plot()
        
# ---------------------------------------------------------------------------
# User friendly

def from_points(count, verts, lefts=None, rights=None):
    """Compute control points from interpolation points
    
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
        
    vf = PointsInterpolation(verts, lefts, rights)
    return control_points(vf, count)

def from_function(count, f, t0, t1):
    """Compute control points from a function
    
    Parameters
    ----------
    count : int
        Number of control points to compute
        
    f : function, template f(float) -> vector
        The function to interpolate
        
    t0 : float
        The starting value
        
    t1 : float
        The ending value
        
    Returns
    -------
    3 arrays of count vectors
        control points, left control points, right control points    
    """
    
    return control_points(f, count, t0, t1, dt=(t1-t0)/10000)

# ---------------------------------------------------------------------------
# Function

class InterpolationPolynom():
    """Interpolate x, y values with polynoms
    
    Each interval is interpolated by a polynom of degree 3 to 
    produce a continuous curve between intervals.
    First and last interval are only of degree 2.
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
        
        self.x = np.array(x)
        self.y = np.array(y)
        self.periodic = periodic
        
        self.a, self.b, self.c, self.d = InterpolationPolynom.computePolynoms(self.x, self.y)
        
    @property
    def delta_x(self):
        """The lengths of the intervals
        
        Returns
        -------
        array of floats
            The dim of the array is one lesser than the number of points
        """
        return self.x[-1] - self.x[0]
    
    @property
    def invert(self):
        """Return the invert interpolation
        
        Returns
        -------
        InterpolationPolynom
            Interpolations computed with y, x
        """
        
        return InterpolationPolynom(self.y, self.x, self.periodic)
        
    @staticmethod
    def computePolynoms(x, y):
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
        
        # ----- Number of points
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
    
    def __call__(self, x):
        """Compute the interpolation of values
        
        Parameters
        ----------
        x : array of floats
            The values where to compute the interpolation
            
        Returns
        -------
        array of floats
        """
        
        # Only a single value
        if not hasattr(x, "__len__"):
            return self([x])[0]
        
        
        # Modulo
        if self.periodic:
            x = np.mod(x, self.delta_x)
        
        # results
        y = np.zeros(len(x), np.float)
        
        # Loop on the curves
        ncurves = len(self.x) - 1
        for i in range(ncurves):
            if i == 0:
                ks = np.argwhere(x <= self.x[1])
            elif i == ncurves-1:
                ks = np.argwhere(x > self.x[ncurves-1])
            else:
                ks = np.argwhere(np.logical_and(x > self.x[i], x <= self.x[i+1]))
                
            if len(ks) > 0:
                #ks = ks.reshape(len(ks))
                x_2 = x[ks] * x[ks]
                y[ks] = self.a[i]*x_2*x[ks] + self.b[i]*x_2 + self.c[i]*x[ks] + self.d[i]
                
        return y
    
    # ----- For debug
    
    def plot(self, dx=None):
        """Plot the interpolation curve
        
        Parameters
        ----------
        dx : float
            Starts and ends the curve outside the x bounds
        """
        
        import matplotlib.pyplot as plt
        
        # ----- The points
        
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.x, self.y, 'o-')
        
        # ----- The curve
        
        if dx is None:
            dx = 0.1*self.delta_x
        x = np.linspace(self.x[0] - dx, self.x[-1] + dx, 100)
        ax.plot(x, self(x))
        
        plt.show()
        
    @staticmethod
    def demo():
        x = np.linspace(0, 2*np.pi, 5)
        y = np.sin(x)
        
        f = InterpolationPolynom(x, y, True)
        f.plot(7)

