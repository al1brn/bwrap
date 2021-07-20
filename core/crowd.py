#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np

from .wrappers import wrap
from .transformations import Transformations

# ====================================================================================================
# Crowd: duplicates a mesh in a single mesh

class Crowd(Transformations):
    def __init__(self, name, model, count=100):
        
        super().__init__(count=count)
        
        # ----- Wrap the objects and check they are meshes
        
        self.wobject = wrap(name)
        if not self.wobject.is_mesh:
            raise RuntimeError(f"Crowd init error: {name} must be a mesh object")
            
        self.wmodel = wrap(model)
        if not self.wmodel.is_mesh:
            raise RuntimeError(f"Cfrowd init error {model} must be a mesh object")
        
        # ----- Build the new geometry made of stacking vertices and polygons
        
        # Vertices
        verts = np.array(self.wmodel.verts)
        self.v_count = len(verts)
        self.total_verts = len(self) * self.v_count
        
        self.base_verts = np.column_stack((verts, np.ones(self.v_count)))

        # Polygons
        polys = self.wmodel.poly_indices
        self.p_count = len(polys)
        
        faces = [[index + i*self.v_count for index in face] for i in range(len(self)) for face in polys]

        # New geometry
        self.wobject.new_geometry(np.resize(verts, (self.total_verts, 3)), faces)
        
        # ----- Materials
        
        self.wobject.copy_materials_from(self.wmodel)
        self.wobject.material_indices = self.wmodel.material_indices # Will be properly broadcasted
        
        # ----- uv mapping
        
        for name in self.wmodel.uvmaps:
            uvs = self.wmodel.get_uvs(name)
            self.wobject.create_uvmap(name)
            self.wobject.set_uvs(name, np.resize(uvs, (len(self)*len(uvs), 2)))
            
        # ----- No animation
        self.animated = False
            
    def __repr__(self):
        s = "<"
        s += f"Crowd of {len(self)} meshes of {self.v_count} vertices, vertices: {self.total_verts}"
        if self.animated:
            s += "\n"
            s += f"Animation of {self.steps} steps: {self.base_verts.shape}"
        return s + ">"    
        
        
    @property
    def verts(self):
        return self.wobject.verts
    
    @verts.setter
    def verts(self, value):
        self.wobject.verts = value
        
    # ---------------------------------------------------------------------------
    # Overrides matrices transformations
        
    def apply(self):
        self.wobject.wdata.verts = self.transform_verts43(self.base_verts)

    # ---------------------------------------------------------------------------
    # Euler order
    
    @property
    def euler_order(self):
        return self.wmodel.rotation_euler.order
    
    # ---------------------------------------------------------------------------
    # Set an animation
    # The animation is made of steps set of vertices. The number of vertices
    # must match the number of vertices used for initialization.
    # The shape of verts is hence: (steps, v_count, 3)
    
    def set_animation(self, animation, phases=None, speeds=1, seed=0):
        
        self.animated  = True
        self.steps     = len(animation)
        n              = self.steps*self.v_count
        self.animation = np.resize(np.column_stack(
                    (animation.reshape(n, 3), np.ones(n))
                ), (self.steps, self.v_count, 4))
        
        if seed is not None:
            np.random.seed(seed)
            
        if phases is None:
            self.phases = np.random.randint(0, self.steps, len(self))
        else:
            self.phases = np.resize(phases, len(self))
            
        if speeds is None:
            self.speeds = np.random.randint(1, self.steps, len(self))
        else:
            self.speeds = np.resize(speeds, len(self))
            
        # ----- Reshape the base vertices
        self.base_verts = np.resize(self.base_verts, (len(self), self.v_count, 4))
            
    # ---------------------------------------------------------------------------
    # Set an animation
    
    def animate(self, frame):
        if self.animated:
            indices = (self.phases + frame*self.speeds) % self.steps
            self.base_verts = self.animation[indices]
            self.lock_apply()
            
        
    
        
    
    
    
    
    

