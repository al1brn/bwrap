#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:12:24 2021

@author: alain
"""

import bpy
from .wstruct import WStruct

# Interface for materials management

class WMaterials(WStruct):
    
    def __init__(self, name):
        if not type(name) is str:
            name = name.name
            
        super().__init__(name=name)
        
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

    @property
    def wrapped(self):
        return bpy.data.objects[self.name].data.materials
        
    def __len__(self):
        return len(self.wrapped)
    
    def __getitem__(self, index):
        return self.wrapped[index]
        
    def clear(self):
        self.wrapped.clear()
        
    @property
    def mat_names(self):
        return [mat.name for mat in self.wrapped]
        
    def indices_by_name(self, name):
        indices = []
        for i, mat in enumerate(self.wrapped):
            if mat.name == name:
                indices.append(i)
        return indices
        
    def copy_materials_from(self, other, append=False):
        mat_names = self.mat_names
        wmat = WMaterials(other)
        for mat in wmat.wrapped:
            if append or (mat.name not in mat_names):
                self.wrapped.append(mat)
                
    def replace(self, names):
        self.clear()
        for name in names:
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
            
            self.wrapped.append(mat)
            
            
