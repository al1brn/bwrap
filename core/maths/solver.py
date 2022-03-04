#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:14:47 2021

@author: alain
"""

import numpy as np

# ---------------------------------------------------------------------------
# Solve by dichotomy

def dichotomy(f, target=0, start=0, prec=0.0001, t0=None, t1=None):
    
    shape = np.shape(target)
    size  = np.size(target)
    
    if np.size(start) > size:
        shape = np.shape(start)
        size  = np.size(start)
        
    v0 = f(start)
    if np.size(v0) > size:
        shape = np.shape(v0)
        size  = np.size(v0)
        
    single = shape == ()
    if single:
        shape = (1,)
        
    # ----- The target
        
    vt = np.empty(shape, float)
    vt[:] = target
    
    v1 = np.array(v0)
    
    # ----- Lowest limit
        
    if t0 is None:
        t0 = np.empty(shape, float)
        t0[:] = start
        
        e = 1
        for i in range(30):
            inds = np.where(v0 > vt)[0]
            if len(inds) == 0:
                break
            t0[inds] -= e
            e *= 2
            v0 = f(t0)
    else:
        t0 = np.resize(t0, shape).astype(float)
        
    # ----- Highest limit
        
    if t1 is None:
        t1 = np.empty(shape, float)
        t1[:] = start
        
        e = 1
        for i in range(30):
            inds = np.where(v1 < vt)[0]
            if len(inds) == 0:
                break
            t1[inds] += e
            e *= 2
            v1 = f(t1)
    else:
        t1 = np.resize(t1, shape).astype(float)

    # ---------------------------------------------------------------------------
    # Dichotomy loop
    
    for i in range(40):
        
        t = (t0 + t1)/2
        
        v = f(t)
        
        if len(np.where(abs(v - vt) > prec)[0]) == 0:
            break
        
        inds = np.where(v < vt)
        t0[inds] = t[inds]
        
        inds = np.where(v > vt)
        t1[inds] = t[inds]
        
    # ---------------------------------------------------------------------------
    # Return the result        
        
    if single:
        return t[0]
    else:
        return t    
    
    
def demo():

    import matplotlib.pyplot as plt
    from timeit import timeit

    def f(t):
        return t**3

    target = np.linspace(f(-100), f(100), 10000000)
    
    def test():
        return dichotomy(f, target)
    
    dt = timeit(test, number=1)

    print(f"timeit: {dt:.2f}")

    fig, ax = plt.subplots()

    x = test()

    y = f(x)
    
    ax.plot(x, target, 'r')
    ax.plot(x, y, '.g')
    
    
    plt.show()


        

    