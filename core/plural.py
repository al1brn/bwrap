#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:28:31 2021

@author: alain
"""

import numpy as np

from .commons import base_error_title

error_title = base_error_title % "plural.%s"


# ---------------------------------------------------------------------------
# Resize an array to a given shape, possibly by broadcasting the passed array

def to_shape(a, shape):
    
    size = np.product(shape)
            
    na = np.array(a)
    if na.size == size:
        return np.reshape(na, shape)
    elif size % na.size == 0:
        return np.resize(na, shape)
    else:
        raise RuntimeError(error_title % "to_shape" +
            f"target shape: {shape}\n" +
            f"input size:   {na.size}")
        
# ---------------------------------------------------------------------------
# Compute the target shape of an array based on the shape of the items
# - 1      --> count
# - (1)    --> (count, (1))
# - (2, 3) --> (count, 2, 3)

def target_shape(count, shape):
    
    size  = np.product(shape)
    
    if size == 1:
        return count
    
    dims =  np.size(shape)
    
    if dims == 1:
        return (count, shape)

    if dims == 2:
        return (count, shape[0], shape[1])
    
    a = [count]
    a.extend(shape)
    
    return list(a)

# ---------------------------------------------------------------------------
# Plural getter
# Use the quick method foreach_get if exists for coll
# Otherwise loop on the items in the collection

def getattrs(coll, name, shape, nptype=np.float):
    
    count = len(coll)
    size  = np.product(shape)
    
    # ----- Quick method
    
    if hasattr(coll, 'foreach_get'):

        vals = np.empty(count*size, nptype)
        
        coll.foreach_get(name, vals)
        
        # Matrices must be transposed
        if np.size(shape) == 2:
            vals = np.reshape(vals, (count, shape[0], shape[1]))
            return np.transpose(vals, axes=(0, 2, 1))
        
        # Otherwise it is ok
        if size > 1:
            return np.reshape(vals, target_shape(count, shape))
        else:
            return vals
        
    # ----- Loop
        
    vals = np.empty(target_shape(count, shape), nptype)
    for i, item in enumerate(coll):
        vals[i] = getattr(item, name)
        
    return vals


# ---------------------------------------------------------------------------
# Plural setter
# As for plural getter
        
def setattrs(coll, name, value, shape):
    
    count  = len(coll)
    size   = np.product(shape)    
        
    val    = np.array(value)
    vals   = to_shape(val, count*size)
    
    if hasattr(coll, 'foreach_set'):

        # Matrices must be transposed
        if np.size(shape) == 2:
            vals = np.reshape(vals, (count, shape[0], shape[1]))
            vals = np.transpose(vals, axes=(0, 2, 1))
            vals = np.reshape(vals, count*size)
        
        coll.foreach_set(name, vals)
        
    else:
        if size > 1:
            vals = np.reshape(vals, target_shape(count, shape))

        for i, item in enumerate(coll):
            setattr(item, name, vals[i])
            
            