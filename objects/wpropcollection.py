#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:20:40 2021

@author: alain
"""

import numpy as np

from ..wrappers.wrap_function import wrap


# ===========================================================================
# Blender prop_collection wrapper
#
# The PropCollection keeps a reference to a wrapper which must implement
# the objects property

class WPropCollection():
    
    def __init__(self, collection):
        self.collection = collection
        
    def find(self, key):
        return wrap(self.collection.find(key))
    
    def get(self, key):
        return wrap(self.collection.get(key))

    # ---------------------------------------------------------------------------
    # As an array of objects
    
    def __len__(self):
        return len(self.collection)
    
    def __getitem__(self, index):
        return wrap(self.collection[index])
    
    def keys(self):
        return self.collection.keys()
    
    def values(self):
        return [wrap(o) for o in self.collection.values()]
    
    def items(self):
        return [(k, wrap(o)) for k, o in self.collection.items()]
    
    # ---------------------------------------------------------------------------
    # Get float attributes    
    
    def get_float_attrs(self, name, dim=1):
        
        objs = self.collection
        shape = len(objs) if dim == 1 else (len(objs), dim)
        
        a = np.empty(len(objs)*dim, np.float)
        objs.foreach_get(name, a)
        
        return a.reshape(shape)
            
        
    # ---------------------------------------------------------------------------
    # Set float attributes    
        
    def set_float_attrs(self, name, array, dim=1):

        objs = self.collection
        shape = len(objs) if dim == 1 else (len(objs), dim)
        
        a = np.empty(shape, np.float)
        a[:] = array
        
        objs.foreach_set(name, a.reshape(len(objs)*dim))

            
    # ---------------------------------------------------------------------------
    # Get int attributes    
        
    def get_int_attrs(self, name, dim=1):
        
        objs = self.collection
        shape = len(objs) if dim == 1 else (len(objs), dim)
        
        a = np.empty(len(objs)*dim, int)
        objs.foreach_get(name, a)
        
        return a.reshape(shape)
        
    # ---------------------------------------------------------------------------
    # Set float attributes    
        
    def set_int_attrs(self, name, array, dim=1):

        objs = self.collection
        shape = len(objs) if dim == 1 else (len(objs), dim)
        
        a = np.empty(shape, np.float)
        a[:] = array
        
        objs.foreach_set(name, a.reshape(len(objs)*dim))
            
    # ---------------------------------------------------------------------------
    # Get matrices
    
    def get_matrices(self, name, dim=4):
        return self.get_float_attrs(name, dim=dim*dim).reshape(len(self.collection), dim, dim)
    
    # ---------------------------------------------------------------------------
    # Set matrices
    
    def set_matrices(self, name, value, dim=4):
        size = np.size(value)
        return self.set_float_attrs(name, np.reshape(value, (size, dim*dim)), dim=dim*dim)
    
    # ---------------------------------------------------------------------------
    # Get a particular value
    
    def get_value(self, name, index):
        return getattr(self.collection[index], name)
        
    # ---------------------------------------------------------------------------
    # Set a particular value  
        
    def set_value(self, name, index, value):
        setattr(self.collection[index], name, value)

    # ---------------------------------------------------------------------------
    # Get plural values
    
    def get_values(self, name, index):
        objs = self.collection
        return [getattr(obj, name) for obj in objs]
        
    # ---------------------------------------------------------------------------
    # Set plural values
        
    def set_values(self, name, index, values, unique=False):
        objs = self.collection
        for i, obj in enumerate(objs):
            if unique:
                setattr(obj, name, values)
            else:
                setattr(obj, name, values[i])
     

    
    

        
        
        
        
    
        
        
    
    
        
        
        
        
        
            
