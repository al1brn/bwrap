#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:37:15 2022

@author: alain.bernard@loreal.com
"""

import time
import numpy as np


from core.maths.geoarrays import Vectors, Matrices, Eulers, Quaternions, AxisAngles, TMatrices


v = Vectors(0, shape=(10,))
print(v)

v[:] = (1, 2, 3)
print(v)


Matrices.test()

def test():
    
    q = Eulers(np.radians(((10, 20, 30)))).quaternions()
    #vs = Vectors(np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))))
    vs = Vectors(np.array(((1, 1, 1))))
    
    w = q @ vs
    qi = ~q
    
    v1 = qi @ w
    
    print(vs)
    print(w)
    print(v1)
    print()
    print(qi)
    print(">", qi @ w)

    print(~(q.matrices()) @ w)
    print(~(q.eulers()) @ w)
    print(~(q.quaternions()) @ w)
    
    print(qi.matrices() @ w)
    print(qi.eulers() @ w)
    print(qi.quaternions() @ w)
    
    print(qi)
    print(qi.quaternions())

    print(">", qi @ w)
    print(">", qi.quaternions() @ w)
    
    
    


#test()
    





def debug():

    q = Quaternions.Identity(shape=1)
    q = AxisAngles((0, 0, -1), angles=330).quaternions()
    q = Quaternions([  -0.966,   -0.000,    0.000,   -0.259]).normalize()
    c = Vectors((-10, -10, -10))
    
    
    q0 = q
    eulers = q0.eulers()
    q = eulers.quaternions()
    
    print("q0", q0)
    print("q ", q)
    print()
    
    ref = eulers @ c
    
    d = q @ c.quaternions()
    e = (~q) @ d
    
    print("rot   ", eulers)
    print("quat  ", q)
    print("base  ", c)
    print("target", ref)
    print("res   ", d.vectors())
    print("back  ", e.vectors())
    
    print(d.vectors())
    print(e.vectors())
    
    
    #Eulers.test(stop=True)
    
    """
    tmat = TMatrices(shape=3,scales=Vectors(2))
    
    print(tmat)
    
    vect = Vectors((1, 2, 3), shape=(3, 4))
    print(vect)
    
    print(tmat @ vect)
    """
    
#debug()





