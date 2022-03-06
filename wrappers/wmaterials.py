#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:12:24 2021

@author: alain
"""

import numpy as np

import bpy

from ..core.commons import WError

# Interface for materials management
# Can be initialized either from an object name or directly from the data structure

class WMaterials():
    
    def __init__(self, object_name=None, data=None):
        
        if object_name is None:
            if data is None:
                raise WError("WMaterials initialization error. Either object name or data structure must be provided.",
                    Class = "WMaterials",
                    Mehod = "__init__")
                
            self.data_       = data
            self.object_name = None
            
        else:
            if type(object_name) is str:
                self.object_name = object_name
            else:
                self.object_name = object_name.name
            
    @property
    def name(self):
        if self.object_name is None:
            return f"{type(self.data).__name__} {self.data.name}"
        else:
            return f"Object {self.object_name}"
        
    def __repr__(self):
        s = f"<WMaterials of '{self.name}': "
        if len(self) == 0:
            s += "0 material"
        elif len(self) == 1:
            s += f"1 material: '{self[0].name}'"
            
        else:
            s += f"{len(self)} materials:\n"
            for i, mat in enumerate(self):
                s += f"   {i:2d}: '{self[i].name}'\n"
        return s + ">"

    # ---------------------------------------------------------------------------
    # Access to the data structure
    
    @property
    def data(self):
        if self.object_name is None:
            return self.data_
        else:
            return bpy.data.objects[self.object_name].data

    # ---------------------------------------------------------------------------
    # Access to the blender structure
    
    @property
    def materials(self):
        return self.data.materials
    
    # ---------------------------------------------------------------------------
    # As an array of materials
        
    def __len__(self):
        return len(self.materials)
    
    def __getitem__(self, index):
        return self.materials[index]
        
    def clear(self):
        self.materials.clear()
        
    # ---------------------------------------------------------------------------
    # Data type
    
    @property
    def data_type(self):
        return type(self.data).__name__

    # ---------------------------------------------------------------------------
    # Access to the structure owning the material_index attribute
    
    @property
    def indices_structure(self):

        dtype = self.data_type
        if dtype == 'Mesh':
            return self.data.polygons
        
        elif dtype == 'Curve':
            return self.data.splines
            
        elif dtype == 'TextCurve':
            return self.data.body_format
            
        else:
            return None
    
    # ---------------------------------------------------------------------------
    # Get / set the materials indices
    
    @property
    def material_indices(self):
        
        props = self.indices_structure
        if props is None:
            return None
        else:
            a = np.empty(len(props), int)
            props.foreach_get('material_index', a)
            return a
            
    @material_indices.setter
    def material_indices(self, indices):
        
        props = self.indices_structure
        if props is not None:
            a = np.empty(len(props), int)
            a[:] = indices
            a = np.clip(a, 0, len(self)-1)
            props.foreach_set('material_index', a)
        
    # ---------------------------------------------------------------------------
    # Materials names
    
    def append(self, names):
        for name in names:
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
            
            self.materials.append(mat)
    
        
    @property
    def mat_names(self):
        return [mat.name for mat in self.materials]
    
    @mat_names.setter
    def mat_names(self, names):
        
        mat_i = self.mat_indices

        self.clear()
        self.append(names)
            
        self.mat_indices = mat_i
        
            
        
                
