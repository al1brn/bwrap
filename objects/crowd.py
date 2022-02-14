#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np
import bpy

from ..maths.transformations import Transformations, SlaveTransformations

from ..wrappers.wrap_function import wrap
from ..wrappers.geometry import Geometry

from ..core.commons import WError

# ====================================================================================================
# A group transformation offers a transformation matrices array for a group of vertices
# It is initialized with a group name
# The vertices indices are captured from the geometry.groups of the crowd
# The pivot can take three values:
# - None:       The center of the group of vertices is taken as the pivot point
# - Group name: The center of this group name
# - vector:     The direct value of the pivot

class GroupTransformations():
    
    def __init__(self, crowd, group_name, pivot=None):
        
        if not crowd.geometry.groups.group_exists(group_name):
            raise WError(f"Group '{group_name}' doesn't exist in the geometry.",
                    Class = "GroupTransformations",
                    Method = "__init__",
                    existing_groups = f"{crowd.geometry.groups.groups}")
            
        self.crowd    = crowd
        self.indices  = crowd.geometry.groups.vertices(group_name)
        self.transfos = Transformations(count=crowd.shape)
        
        self.pivot = np.zeros(4, np.float) # No one at the end !
        if pivot is None or type(pivot) is str:
            if pivot is None:
                verts = crowd.geometry.verts[self.indices]
            else:
                verts = crowd.geometry.verts[crowd.geometry.groups.vertices(pivot)]
                
            n = len(verts)
            if n != 0:
                self.pivot[:3] = np.sum(verts, axis=0)/n
        else:
            self.pivot[:3] = pivot
            
    def __repr__(self):
        return f"<GroupTransfo: pivot: {self.pivot[:3]}, indices: {len(self.indices)}>"


# ====================================================================================================
# A crowd is an array of transformation matrices which can be shaped
#
# It is made of similar or different geometries
#
# To build a crowd, the following steps are necessary
# - build the full geometry made of instances of base geometries
# - create a single Blender object
# - animate the vertices by updating the object
#
# In order to benefit from numpy optimization, geometries instanced several times
# are stacked in one single array:
# - count instances of geo(n, 3) --> (count, n, 3)
#
# There is a slicing mechanism to go from global coordinates to local coordinates


# ====================================================================================================
# A simple crowd simple duplicates a single geometry

class Crowd(Transformations):
    
    def __init__(self, geometry, shape=(1,)):
        
        super().__init__(count=shape)
        
        self.geometry = geometry
        
        # ----- Shape keys animation
        
        self.shape_keys     = None
        self.animation_     = np.zeros(self.shape, float) 
        self.back_animation = False
        
        # ----- Transformations
        
        self.pre_transformations  = []
        self.post_transformations = []
        
        # ----- Groups transformatios
        
        self.group_transformations = {}
        
    # ---------------------------------------------------------------------------
    # Content
    
    def __repr__(self):
        s  = f"<{type(self).__name__} of shape {self.shape} = {self.size} instance(s)\n"
        s += f"   geometry:   {self.geometry.verts_count} vertices, total: {self.size * self.geometry.verts_count} vertices\n"
        s +=  "   shape keys: "
        if self.shape_keys is None:
            s += "None\n"
        else:
            s += f"{len(self.shape_keys)}\n"
        s += f"   animations: pre= {len(self.pre_transformations)}, post={len(self.post_transformations)}\n"
        s += f"   group transformations: {len(self.group_transformations)}\n"
        for group_name, trf in self.group_transformations.items():
            s += f"       {group_name:10s}: {trf}\n"
        
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # Initialize from an object
    
    @classmethod
    def FromObject(cls, object, shape=(1,), crowd_name=None):
        sc = cls(Geometry.FromObject(object), shape=shape)
        wobj = wrap(object)
        sc.shape_keys = wobj.wdata.wshape_keys.verts()
        
        if crowd_name is not None:
            sc.build_object(crowd_name)
            
        return sc
    
    # ---------------------------------------------------------------------------
    # Build the object
    
    def build_object(self, crowd_name):
        self.crowd_name = crowd_name
        return self.full_geometry.set_to(crowd_name)

    # ---------------------------------------------------------------------------
    # Update the object
    
    def update(self, object=None):
        if object is None:
            if hasattr(self, "crowd_name"):
                wobj = wrap(self.crowd_name)
            else:
                return
        else:
            wobj = wrap(object)
            
        wobj.verts = self.transform()
    
    # ---------------------------------------------------------------------------
    # Size has chnaged
    
    def size_changed(self):
        self.animation_ = np.resize(self.animation_, self.shape)
        for gt in self.group_transformations.values():
            gt.transfos.resize(self.shape)
    
    # ---------------------------------------------------------------------------
    # Total verts count
    
    @property
    def verts_total(self):
        return self.geometry.verts_count * self.size
    
    # ---------------------------------------------------------------------------
    # Animation
    
    @property
    def animation(self):
        return self.animation_
    
    @animation.setter
    def animation(self, value):
        self.animation_[:] = value
    
    # ---------------------------------------------------------------------------
    # Add a transformation
    
    def add_transformation(self, transfo, pre=True):
        if pre:
            self.pre_transformations.append(transfo)
        else:
            self.post_transformations.append(transfo)
            
    # ---------------------------------------------------------------------------
    # Add a group transformatin
    
    def add_group_transformation(self, group_name, pivot=None):
        self.group_transformations[group_name] = GroupTransformations(self, group_name, pivot)
        
    # ---------------------------------------------------------------------------
    # group transformation
    
    def group_transform(self, group_name):
        return self.group_transformations[group_name].transfos
        
    # ---------------------------------------------------------------------------
    # Build the full geometry
        
    @property
    def full_geometry(self):
        return Geometry.Array(self.geometry, count=self.size)
    
    # ---------------------------------------------------------------------------
    # Set shape keys
    
    def set_shape_keys(self, shape_keys):
        self.shape_keys = shape_keys
    
    # ---------------------------------------------------------------------------
    # Transform the vertices
    
    def transform(self):
        
        # ----- Empty array of 4-vertices

        verts4 = np.ones(self.shape + (self.geometry.verts_count, 4), float)
        
        # ----- Initialize with geometry vertices or with an interpolation of shape keys
        
        if self.shape_keys is None:
            
            verts4[..., :3] = self.geometry.verts
            
        else:
            sk_count = len(self.shape_keys)
            
            if self.back_animation:
                fs = self.animation_ % (2*sk_count)
                after = fs > sk_count
                fs[after] = 2*sk_count - fs[after] 
            else:
                fs = self.animation_ % sk_count
                
            i0 = np.clip(np.floor(fs).astype(int), 0, sk_count-2)
            fs = np.reshape(fs - i0, self.shape + (1, 1))

            verts4[..., :3] = (1-fs)*self.shape_keys[i0] + fs*self.shape_keys[i0+1]

        # ----- Group transformations
        
        for group_name, g_trf in self.group_transformations.items():
            verts4[..., g_trf.indices, :] = g_trf.pivot + g_trf.transfos.transform_verts4(verts4[..., g_trf.indices, :] - g_trf.pivot)
        
        # ----- Pre transformations
        
        for trf in self.pre_transformations:
            trf(verts4[..., :3])
            
        # ----- Transformation
        
        verts = self.transform_verts43(verts4)
        del verts4
            
        # ----- Post transformations
        
        for trf in self.post_transformations:
            trf(verts)
            
        # ----- Return the result
        
        return verts.reshape(self.size*self.geometry.verts_count, 3)
    
    
# ====================================================================================================
# Multiple crowds
#
# The indices array is shaped (n) and gives the crowd index corresponding to the global instance index

class Crowds(Transformations):
    
    def __init__(self):
        super().__init__()

        self.empty       = True
        
        self.crowds      = []
        self.counts      = []
        self.indices     = np.zeros((), int)
        self.geo_names   = []
        
        self.shape_      = ()
        
    # ---------------------------------------------------------------------------
    # Representation
    
    def __repr__(self):
        s = f"<Crowds of shape {self.shape} ({self.size} instances) with {len(self.geo_names)} geometries.\n"
        for i, c in enumerate(self.crowds):
            s += f"   {self.geo_names[i]:20s}: {c.size:3d} instances of {c.geometry.verts_count:3d} vertices, total = {c.verts_total}\n"
        s += f" total: {self.verts_total} vertices\n"
        return s + ">"
        
    # ---------------------------------------------------------------------------
    # Build the object
    
    def build_object(self, crowd_name):
        self.crowd_name = crowd_name
        return self.full_geometry.set_to(crowd_name)

    # ---------------------------------------------------------------------------
    # Update the object
    
    def update(self, object=None):
        
        if self.empty:
            return
        
        if object is None:
            if hasattr(self, "crowd_name"):
                wobj = wrap(self.crowd_name)
            else:
                return
        else:
            wobj = wrap(object)
            
        wobj.verts = self.transform()
        
    # ----------------------------------------------------------------------------------------------------
    # Ovverides
        
    @property
    def size(self):
        return sum(self.counts)
    
    @property
    def shape(self):
        return self.shape_
        
    def reshape(self, new_shape):
        if int(np.product(new_shape)) == self.size:
            self.shape_ = new_shape
            super().reshape(new_shape)
        else:
            raise WError(f"Impossible de to reshape to {new_shape} transformation of size {self.size}",
                        Class = "Crowds",
                        Method = "reshape")
            
    # ----------------------------------------------------------------------------------------------------
    # Crowd name to index
    
    def geo_index(self, geo_name):

        if type(geo_name) is int:
            return geo_name
        
        try:
            return self.geo_names.index(geo_name)
        except:
            return None
            
    # ----------------------------------------------------------------------------------------------------
    # mat indices
    # global tmat indices for the geometry passed in argument
    
    def geo_indices(self, geo_name):
        return np.where(self.indices == self.geo_index(geo_name))[0]
    
    # ----------------------------------------------------------------------------------------------------
    # Add a geometry
    
    def add_geometry(self, geometry, count, name=None):
        
        crowd = Crowd(geometry, shape=(count,))
        
        if self.empty:

            geo_index = 0
            
            self.crowds  = [crowd]
            self.counts  = [count]
            self.indices = np.zeros(count, int)
            
            self.empty   = False
            
        else:
        
            geo_index = len(self.crowds)
        
            self.crowds.append(crowd)
            self.counts.append(count)
            self.indices = np.append(self.indices, np.ones(count, int)*geo_index)
            
        # ----- Slice
        
        crowd.set_slice_of(self, self.geo_indices(geo_index))
        
        # ----- Names
        
        if name is None:
            name = f"Geometry {geo_index}"
        
        self.geo_names.append(name)

        # ----- Shape
        
        self.shape_ = (self.size,)
        self.resize(self.shape)

        # ----- Return the index
        
        return geo_index
    
    # ----------------------------------------------------------------------------------------------------
    # Add an object
    
    def add_object(self, object, count):
        
        wobj = wrap(object)
        
        geo_index = self.add_geometry(Geometry.FromObject(object), count, name=wobj.name)
        crowd = self.crowds[geo_index]
        crowd.shape_keys = wobj.wdata.wshape_keys.verts()
        
        return geo_index
    
    
    # ----------------------------------------------------------------------------------------------------
    # Add an item in a geometry
    
    def add_instance(self, geo_name, count=1):
        
        geo_index = self.geo_index(geo_name)
        
        self.counts[geo_index] += count
        
        # ----- Shape and resize

        self.shape_ = (self.size,)
        self.resize(self.shape)
        
        # ----- Append the index
        
        self.indices = np.append(self.indices, np.ones(count, int)*geo_index)
        
        # ----- Update the crowd
        
        crowd = self.crowds[geo_index] 
        crowd.set_slice(self.geo_indices(geo_index))
        crowd.size_changed()
        
        
    # ----------------------------------------------------------------------------------------------------
    # full geometry
    
    @property
    def full_geometry(self):
        
        geo = None
        for crowd in self.crowds:
            if geo is None:
                geo = crowd.full_geometry
            else:
                geo.join(crowd.full_geometry)
                
        return geo
    
    # ---------------------------------------------------------------------------
    # Total number of vertices
    
    @property
    def verts_total(self):
        vc = 0
        for crowd in self.crowds:
            vc += crowd.verts_total
            
        return vc
    
    # ---------------------------------------------------------------------------
    # Transform the vertices
    
    def transform(self):
        
        verts = np.zeros((self.verts_total, 3), float)
        verts_offset = 0
        
        lin_tmat = np.reshape(self.tmat, (self.size, 4, 4))

        for geo_index, crowd in enumerate(self.crowds):
            
            # ----- Update the crowd matrices with the global matrices
            
            crowd.tmat = lin_tmat[self.geo_indices(geo_index)]
            
            # ----- Transformed vertices
            
            verts[verts_offset:verts_offset + crowd.verts_total] = crowd.transform()
            
            verts_offset += crowd.verts_total
            
        return verts
    
    
