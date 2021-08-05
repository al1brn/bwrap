#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:58:47 2021

@author: alain
"""

import numpy as np

if True:
    from .shapes import  get_full_shape, get_main_shape
    from ..core.commons import WError
    from ..maths.interpolation import BCurve, interpolation_function
else:
    from shapes import  get_full_shape, get_main_shape
    #from ..core.commons import WError

# ---------------------------------------------------------------------------
# Interpolation curve
    
def get_bcurve(name, cyclic):

    bc = BCurve()

    bc.set_start_point((0, 0))
    bc.add((1, 1), interpolation=name)

    if cyclic:
        bc.add((2, 0), interpolation=name)
        
    return bc

# ---------------------------------------------------------------------------
# Check an array is acceptable as shape keys
# return the number of steps and the number of vertices per shape

def check_shape_keys(shape_keys, **kwargs):
    
    sk_shape = get_main_shape(np.shape(shape_keys), 3) # Could alo be 4 vectors !
    
    if len(sk_shape) == 0:
        raise WError("The shape keys must be an array of arrays of floats, not a single value",
                shape_keys_shape = sk_shape,
                shape_keys = shape_keys,
                **kwargs)

    if len(sk_shape) > 2:
        raise WError(f"The shape keys must be an array of arrays of floats. Shape {sk_shape} is not valid.",
                shape_keys_shape = sk_shape,
                **kwargs
                )
        
    steps = sk_shape[0]
    if steps < 2:
        raise WError(f"The shape keys must have at least two steps. {steps} is not enough.",
                shape_keys_shape = sk_shape,
                **kwargs
                )
        
    return steps, sk_shape[1]


# ---------------------------------------------------------------------------
# Morphing from a series of intermediary shapes
#
# - shape_keys is a array of vertices of shapes (steps, n, 3). steps is the number
#   of keyshapes. n is the number of vertices per shape.
# - t is a float or an array of floats. One morphing is retured per t with 
#   the shame shape as t:
#
# t(p, q) --> (p, q, n, 3)

def morph(shape_keys, t, interpolation_shape=None, interpolation_name=None):

    # ---------------------------------------------------------------------------
    # Check that shape_keys as an acceptable shape
    
    steps, n = check_shape_keys(shape_keys, Function="Morph")
    dim = np.shape(shape_keys)[-1] # 3 of 4-vectors
    
    # ----- Only one vertex per shape key !
    
    if len(np.shape(shape_keys)) == 1:
        return morph(np.reshape(shape_keys, (steps, 1)), t, interpolation_shape, interpolation_name)

    # ----- Let's make sure we at least a a shape of length 2 for t
    
    shape = np.shape(t)
    if shape == ():
        return morph(shape_keys, np.reshape(t, (1,)), interpolation_shape, interpolation_name)[0]
    count = np.product(shape)
    
    
    # ---------------------------------------------------------------------------
    # We are good:
    # steps key shapes of n vertices to compute with t values to get a result
    # of shape (shape, n, 3)
    
    # steps indices
    t_mod = np.mod(t, steps).reshape(count)

    # ----- Interpolation
    
    if interpolation_shape is not None:
        ifunc = interpolation_function(interpolation_shape, interpolation_name)
        t_mod = ifunc(t_mod, xbounds=(0, steps), ybounds=(0, steps))
    
    # ----- int inidices
    
    t0 = np.floor(t_mod).astype(int).reshape(count)
    t0[t0 == steps]   = 0
    t0[t0 == steps-1] = steps-2
    
    # ----- percentage between indices
    
    p = np.resize((t_mod - t0).transpose(), (dim, n, count)).transpose()
    
    verts = shape_keys[t0]*(1-p) + shape_keys[t0+1]*p
    
    return verts.reshape(get_full_shape(shape, (n, dim)))
    
    