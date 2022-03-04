#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:12:42 2021

@author: alain
"""

# -----------------------------------------------------------------------------------------------------------------------------
# Metrics interface
#
# Used by aligner
# Implemented by fonts

# By default, implements a metrics with a ratio

class Metrics():
    
    def __init__(self, metrics, ratio=1.):
        
        self.metrics = metrics
        self.ratio   = ratio
        
    @property
    def space_width(self):
        return self.metrics.space_width*self.ratio
        
        
        
    
    