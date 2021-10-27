#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 12:01:19 2021

@author: alain
"""

from numpy import ndarray

class SymArray(ndarray):
    
    def __new__(cls, length, dtype=float, *args):
        a = ndarray.__new__(cls, (length, length), dtype, *args)
        a[:] = 0
        return a
    
    def __setitem__(self, index, value):
        
        # ----- If index is length 3 this is the second assignment
        
        try:
            if index[2] == 1:
                super().__setitem__((index[0], index[1]), value)
            return
        except:
            pass
        
        # ----- Let's assign the value at the required location
        
        super().__setitem__(index, value)
        
        # ---- Symmetric assignment with a third component to avoid infinite reccuring calls
        
        try:
            self.__setitem__((index[1], index[0], 1), value)
        except:
            pass
        
