#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:49:24 2021

@author: alain
"""

import bpy
from .wstruct import WStruct

# ---------------------------------------------------------------------------
# Root wrapper
# wrapped = ID

class WID(WStruct):
    """Wrapper for the Blender ID Struct.
    
    Implements the evaluated property to give access to the evaluated object.
    """

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
        
        if self.wrapped.is_evaluated:
            return self

        else:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            return self.__class__(self.wrapped.evaluated_get(depsgraph))

