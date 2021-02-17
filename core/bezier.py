#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:25:51 2021

@author: alain
"""

import numpy as np

def tangent(f, t, dt=0.01):
    return ( (np.array(f(t + dt/2)) - np.array(f(t - dt/2))) / dt)

# ---------------------------------------------------------------------------
# List of points interpolated as a function

class PointsInterpolation():

    def __init__(self, points, lefts=None, rights=None):

        self.points = np.array(points)

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
        return len(self.points)

    @property
    def delta(self):
        return 1/(len(self.points)-1)

    def __call__(self, t):

        # Parameter is not an array
        if len(np.shape(t)) == 0:
            return self([t])[0]

        # Ok, we have an array as an input
        n     = len(self.points)
        delta = self.delta

        ts = np.array(t)
        inds = (ts*(n-1)).astype(np.int)

        inds[np.greater(inds, n-2)] = n-2
        inds[np.less(inds, 0)]      = 0

        ps = (ts - inds*delta) / delta

        ps2  = ps*ps
        ps3  = ps2*ps
        _ps  = 1 - ps
        _ps2 = _ps*_ps
        _ps3 = _ps2*_ps

        return self.points[inds]*_ps3[:,np.newaxis] + 3*self.rights[inds]*(_ps2*ps)[:,np.newaxis] + 3*self.lefts[inds+1]*(_ps*ps2)[:,np.newaxis] + self.points[inds+1]*ps3[:,np.newaxis]

    def bezier(self):
        return self.points, self.lefts, self.rights

# ---------------------------------------------------------------------------
# Bezier from a function

def control_points(f, count, t0=0., t1=1., dt=0.0001):

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
# User friendly

def from_points(count, verts, lefts=None, rights=None):
    vf = PointsInterpolation(verts, lefts, rights)
    return control_points(vf, count)

def from_function(count, f, t0, t1):
    dt = (t1-t0)/10000
    return control_points(f, count, t0, t1, dt)
