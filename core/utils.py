#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

import numpy as np


# Dichotomy solver

def dicho(f, value, t0=0., t1=1., zero=0.001):

    v0   = f(t0)
    v1   = f(t1)

    if v1 < v0:
        return dicho(lambda t: -f(t), -value, t0, t1, zero)

    if value <= v0 + zero:
        return t0
    if value >= v1 - zero:
        return t1

    for i in range(20):
        t = (t0+t1)/2
        v = f(t)

        if abs(value-v) < zero:
            return t

        if value < v:
            t1 = t
        else:
            t0 = t

    return (t0+t1)/2

def npdicho(f, values, t0=1., t1=0., zero=0.001):

    vals = np.array(values)

    # ----- Single value, simple algorithm

    if len(vals.shape) == 0:
        return dicho(f, values, t0, t1, zero)

    # ---- Preparation
    count = len(vals)

    # ---- Compute the bounds
    # Takes the opportunity to test if the function must be vectorized

    vf  = f
    ts0 = np.full(count, t0, np.float)
    ts1 = np.full(count, t1, np.float)

    try:
        vs0 = vf(ts0)
    except:
        print("npdicho: function vertorization")
        vf  = np.vectorize(f)
        vs0 = vf(ts0)

    vs1 = vf(ts1)

    # Function is decreasing
    if vs1[0] < vs0[0]:
        return npdicho(lambda t: -vf(t), -vals, t0, t1, zero)

    # Ok, we are sure the function is increasing
    # Let's remove the values outside the bounds

    # Result array
    res = np.empty(count, np.float)

    i_bef = np.where(vals <= vs0 + zero)[0]
    i_aft = np.where(vals >= vs1 - zero)[0]
    res[i_bef] = t0
    res[i_aft] = t1

    # Remaining indices to work with
    inds = np.delete(np.arange(count), np.concatenate((i_bef, i_aft)))

    # Bounds can be limited to the useful values@
    ts0 = ts0[inds]
    ts1 = ts1[inds]

    # Loop
    for i in range(20):
        if len(inds) == 0:
            break

        # Step
        ts = (ts0 + ts1)/2

        # Compute the values
        vs = vf(ts)

        # Where it is ok
        i_ok  = np.where(np.abs(vs-vals[inds])<=zero)[0]

        # Where it is below and above
        i_bel = np.where(vs < vals[inds])[0]
        i_abo = np.where(vs > vals[inds])[0]

        # Update the bounds
        ts0[i_bel] = ts[i_bel]
        ts1[i_abo] = ts[i_abo]

        # Correct values
        res[inds[i_ok]] = ts[i_ok]

        # Update the indices
        i_keep = np.delete(np.arange(len(inds)), i_ok)
        ts0    = ts0[i_keep]
        ts1    = ts1[i_keep]
        inds   = inds[i_keep]

    # At last !
    return res






















    zero = 0.001
