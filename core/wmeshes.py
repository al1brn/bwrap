#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np

from .wrappers import WObject, wrap
from .geometry import *


class Crowd():
    def __init__(self, name, model, count=100):
        
        # Wrap the object and checks it is a mesh
        self.wobject = wrap(name)
        if not self.wobject.is_mesh:
            raise RuntimeError(f"Crowd init error: {name} must be a mesh object")
            
        # Do the same with the model    
        self.wmodel = wrap(model)
        if not self.wmodel.is_mesh:
            raise RuntimeError(f"Cfrowd init error {model} must be a mesh object")
            
        # Initialize the meshes
        self._count = 0
        self.count = count
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        return np.arange(self.v_count)*self.count + index
    
    @property
    def count(self):
        return self._count
    
    @count.setter
    def count(self, value):

        # Nothing to do
        if self._count == value:
            return
        
        # The new number of particules
        self._count = value
        
        # Build the reference particules
        
        # ----- Vertices
        # The vertices (1, 2, 3, ...) are duplicated in (1, 1, 1... 2, 2, 2... 3, 3, 3...)
        # This allows to operate on them by simply resizing the operand
        
        verts = self.wmodel.verts
        self.v_count = len(verts)
        
        self.total_verts = self._count * self.v_count
        self.base_verts  = np.ones((self.total_verts, 4), np.float)
        for i, v in enumerate(verts):
            self.base_verts[i*self._count:(i+1)*self._count, :3] = v

        # ----- Polygons
        # A vertex index in a polygon tuple is :
        # - scaled by its value * count
        # - shifted by the index of the particule
        
        polys = self.wmodel.poly_indices
        self.p_count = len(polys)
        
        # Faces are ordered particle after particle
        faces = [[index*self._count + i for index in face] for i in range(self._count) for face in polys]

        # ----- New geometry
        
        self.wobject.new_geometry(self.base_verts[:, :3], faces)
        
        # ----- uv mapping
        
        for name in self.wmodel.uvmaps:
            uvs = self.wmodel.get_uvs(name)
            self.wobject.create_uvmap(name)
            self.wobject.set_uvs(name, np.resize(uvs, (self._count*len(uvs), 2)))
        
        
        # ----- Transformation matrices
        
        self.tmat = tmatrix(count=self._count)
        
    @property
    def verts(self):
        return self.wobject.verts
    
    @verts.setter
    def verts(self, value):
        self.wobject.verts = value
        
    def apply_modifier(self):
        self.wobject.verts = np.einsum('...jk,...k', 
            np.resize(self.tmat, (self.total_verts, 4, 4)),
            self.base_verts)[:, :3]
        
    @property
    def locations(self):
        return translation_from_tmat(self.tmat)
    
    @locations.setter
    def locations(self, value):
        self.tmat[:, :3, 3] = np.resize(value, (self.count, 3))
        self.apply_modifier()
    
    @property
    def scales(self):
        return scale_from_tmat(self.tmat)
    
    @scales.setter
    def scales(self, value):
        t, m, s = decompose_tmat(self.tmat)
        self.tmat = tmatrix(t, m, np.resize(value, (self.count, 3)))
        self.apply_modifier()
        
    @property
    def rotations(self):
        return mat_from_tmat(self.tmat)
    
    @rotations.setter
    def rotations(self, value):
        t, m, s = decompose_tmat(self.tmat)
        self.tmat = tmatrix(t, np.resize(value, (self.count, 3, 3)), s)
        self.apply_modifier()
        
    @property
    def eulers(self):
        return m_to_euler(self.tmat, self.wobject.rotation_mode)
    
    @eulers.setter
    def eulers(self, value):
        self.rotations = e_to_matrix(value, self.wobject.rotation_mode)
        
    @property
    def eulersd(self):
        return np.degrees(m_to_euler(self.tmat, self.wobject.rotation_mode))
    
    @eulersd.setter
    def eulersd(self, value):
        self.rotations = e_to_matrix(np.radians(value), self.wobject.rotation_mode)
        
        
    
        
    
        
        