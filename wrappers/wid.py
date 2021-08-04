#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:49:24 2021

@author: alain
"""

import bpy
from .wstruct import WStruct

from ..blender import depsgraph 


# ---------------------------------------------------------------------------
# Root wrapper
# wrapped = ID

class WID(WStruct):
    
    """Wrapper for the Blender ID Struct.
    
    Implements the evaluated property to give access to the evaluated object.
    """

    # ---------------------------------------------------------------------------
    # Initialization by name
    
    def __init__(self, wrapped, is_evaluated=None):
        
        if type(wrapped) is str:
            name = wrapped
            if is_evaluated is None:
                is_evaluated = False
        else:
            name = wrapped.name
            is_evaluated = wrapped.is_evaluated
            
        super().__init__()
        
        self.name = name
        self.is_evaluated = is_evaluated
            
    # ---------------------------------------------------------------------------
    # Access to the object
    
    @property
    def blender_object(self):
        return depsgraph.get_object(self.name, self.is_evaluated)
    
    # ---------------------------------------------------------------------------
    # Evaluated ID

    @property
    def evaluated(self):
        """Wraps the evaluated object.

        Returns
        -------
        WID
            Wrapped of the evaluated object.
        """
        
        if self.is_evaluated:
            return self
        
        return type(self)(self.name, is_evaluated=True)
    
    # ---------------------------------------------------------------------------
    # Evaluated ID

    @property
    def evaluated_DEPR(self):
        """Wraps the evaluated object.

        Returns
        -------
        WID
            Wrapped of the evaluated object.
        """
        
        if self.wrapped.is_evaluated:
            return self
        
        if depsgraph.render_mode():
            return 
        
        graph = depsgraph.depsgraph()
        
        
        return depsgraph.get_evaluated(self.wrapped)
        
        if DEPSGRAPH is None:
            return self
        else:
            return self.__class__(self.wrapped.evaluated_get(DEPSGRAPH))
            
        # OLD
            
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return self.__class__(self.wrapped.evaluated_get(depsgraph))

