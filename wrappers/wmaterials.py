#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:12:24 2021

@author: alain
"""

import numpy as np

# Interface for materials management

class WMaterials():
    
    # ---------------------------------------------------------------------------
    # Materials
    
    def copy_materials_from(self, other):
        """Copy the list of materials from another object.

        Parameters
        ----------
        other : object with materials
            The object to copy the materials from.

        Returns
        -------
        None.
        """
        
        self.wrapped.materials.clear()
        #wother = WMesh.get_mesh(other, Class="WMesh", Method="copy_materials_from")
        for mat in other.materials:
            self.wrapped.materials.append(mat)
   
            
