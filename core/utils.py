#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

import numpy as np


# ---------------------------------------------------------------------------
# Dichotomy solver

def dicho(f, value, t0=0., t1=1., zero=0.001):
    """Solves an equation by dicchotomy
    
    The equation is a function of shape f(x) = y. We try to find
    x such as f(x) = v
    
    Parameters
    ----------
    f : function, template: f(x) = y
        Equation to solve
        
    value : float
        Target value
        
    t0 : float
        Beginning of search interval
        
    t1 : float
        End of search interval
        
    zero : float, default = 0.001
        Precision
        
    Results
    -------
    float
        Equation solution 
    """

    v0   = f(t0)
    v1   = f(t1)
    
    # Algorithm only fo growing functions
    if v1 < v0:
        return dicho(lambda t: -f(t), -value, t0, t1, zero)
    
    # Solution is one of the bounds
    if value <= v0 + zero:
        return t0
    if value >= v1 - zero:
        return t1
    
    # Solving loop
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

# ---------------------------------------------------------------------------
# Vectorized version of the dichotomy solver

def npdicho(f, values, t0=1., t1=0., zero=0.001):
    """Solve an equations for an array of target values
    
    The equation is a function of shape f(x) = y. We try to find
    x such as f(x) = v for several v values
    
    Parameters
    ----------
    f : function, template: f(x) = y
        Equation to solve
        
    values : array of floats
        Target values
        
    t0 : float
        Beginning of search interval
        
    t1 : float
        End of search interval
        
    zero : float, default = 0.001
        Precision
        
    Results
    -------
    array of floats
        Equation solutions 
    """

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
    res = np.zeros(count, np.float)

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
    loops = 20
    for i in range(loops):

        # Step
        ts = (ts0 + ts1)/2
        
        # End of the story ?
        if len(inds) == 0:
            break
        else:
            # For last loop
            res[inds] = ts

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
        
        # For last loop
            

    # At last !
    return res

# ---------------------------------------------------------------------------
# Demo

def demo(f= lambda x: np.sin(x), t0=-np.pi/2, t1=np.pi/2, count=50):
    
    # The target values
    ys = np.random.uniform(f(t0), f(t1), max(1, count))
    if count <= 1:
        ys = ys[0]
    
    # Solvers
    xs = npdicho(f, ys, t0, t1)
    
    # Draw the results
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    
    # The curve
    x = np.linspace(t0, t1, 100)
    ax.plot(x, f(x))
    
    # The points
    ax.plot(xs, ys, 's')
    
    # Show
    plt.show()

