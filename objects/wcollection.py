#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:43:51 2021

@author: alain
"""

import bpy

from ..wrappers.wid import WID
from .wobjects import WObjects

from ..core.commons import WError

class WCollection(WID, WObjects):
    
    def __init__(self, wrapped, world = False, shape = None):
        
        if type(wrapped) is str:
            name = wrapped
        else:
            name = wrapped.name
        
        super().__init__(name=name, coll=bpy.data.collections)
        WObjects.__init__(self, self.wrapped.objects)
        
        self.world = world
        if shape is not None:
            self.reshape(shape)
        
    @property
    def objects(self):
        return self.wrapped.objects
        
        
    
    




