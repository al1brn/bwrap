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
        self.transfo  = Transformations(count=crowd.shape)
        
        if pivot is None or type(pivot) is str:
            if pivot is None:
                verts = crowd.geometry.verts[:, 0, self.indices]
            else:
                verts = crowd.geometry.verts[:, 0, crowd.geometry.groups.vertices(pivot)]
                
            self.pivot = np.insert(np.average(verts, axis=1), 3, 0, axis=-1)
        else:
            self.pivot = np.zeros((crowd.geometry.parts_count, 4), np.float) # No one at the end !
            self.pivot[:, :3] = pivot
            
    def __repr__(self):
        return f"<GroupTransfo: pivot: {self.pivot[:, :3]}, indices: {len(self.indices)}>"


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
# A simple crowd duplicates a single geometry
#
# The number of indivual geometries can be obtained by two means:
# - The base geometry has already parts
# - The crowd instance has a shape different from (1,)
# These two can be combined
#
# Arrayed geometries store faces, uvmap ans mat indices for the whole size
# For big crowds, this can be avoided by managing the base geometry an using
# a bigger shape for the crowd
# The faces, uvmaps... are computed once when build the target object

class Crowd(Transformations):
    
    def __init__(self, geometry, shape=None, name=None):
        
        # ----- Initialize the matrices
        
        super().__init__(count=Crowd.acceptable_shape(shape, geometry.parts_count))
        self.wobject = None
        
        self.geometry = geometry
        
        # ----- Shape animation
        
        self.animation_              = None
        self.animation_relative      = False 
        self.animation_extrapolation = 'CLIP'  # 'CLIP', 'LOOP', 'BACK'
        
        # ----- Transformations
        
        self.deformation_time  = 0.
        self.pre_deformations  = []
        self.post_deformations = []
        
        # ----- Groups transformatios
        
        self.group_transformations = {}
        
        # ----- Build the object
        
        self.rebuild = True
        
        if name is not None:
            self.build_object(name)
        
        
    # ---------------------------------------------------------------------------
    # Content
    
    def __repr__(self):
        s  = f"<{type(self).__name__} of shape {self.shape} = {self.size} instance(s) of geometry\n"
        s += f"{self.geometry}\n\n"
        s += f"   animations: pre= {len(self.pre_deformations)}, post={len(self.post_deformations)}\n"
        s += f"   group transformations: {len(self.group_transformations)}\n"
        for group_name, trf in self.group_transformations.items():
            s += f"       {group_name:10s}: {trf}\n"
        
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # Initialize from an object
    
    @classmethod
    def FromObject(cls, object, shape=(1,), name=None):
        
        return cls(Geometry.FromObject(object), shape=shape, name=name)
    
    # ---------------------------------------------------------------------------
    # Make sure that the shape is compatible with the number of geometry parts
    
    @staticmethod
    def acceptable_shape(shape, parts_count):
        
        #print("Crowd.acceptable_shape", shape, parts_count)
        
        if shape is None:
            return (parts_count,)
        
        if hasattr(shape, '__len__'):
            shape = tuple(shape)
        else:
            shape = (shape,)
        
        size = int(np.product(shape))
        if size % parts_count == 0:
            return shape
        else:
            raise WError(f"Crowd shape {shape} can't be used with a geometry of {parts_count} parts. " + 
                         f"The size {size} of the shape should be a multiple of the number of parts.",
                         Class = "Crowd"
                         )
        
    # ---------------------------------------------------------------------------
    # Resize the crowd
    
    def resize(self, shape):        
        super().resize(self.acceptable_shape(shape, self.geometry.parts_count))
        self.size_changed()
        
    # ---------------------------------------------------------------------------
    # Build the full geometry
        
    @property
    def full_geometry(self):
        if self.geometry.parts_count == 0:
            return Geometry()
        
        return Geometry.Array(self.geometry, count=self.size // self.geometry.parts_count)
        
    # ---------------------------------------------------------------------------
    # Build the object
    
    def build_object(self, object=None):
        if object is None:
            wobj = self.wobject
        else:
            wobj = wrap(object, create=self.geometry.type)
            
        if wobj is None:
            return
        
        self.wobject = wobj
        
        if self.geometry.parts_count == 0:
            return
        
        self.rebuild = False
        
        return self.full_geometry.set_to(wobj, verts=self.transform())

    # ---------------------------------------------------------------------------
    # Update the object
    
    def update(self, object=None):
        
        if self.rebuild:
            
            self.build_object(object)
            
        else:
            if object is None:
                wobj = self.wobject
            else:
                wobj = wrap(object)
                
            if wobj is None:
                return
                
            wobj.verts = self.transform()    
        
    # ---------------------------------------------------------------------------
    # Apply
    
    def apply(self):
        super().apply()
        self.update()
    
    # ---------------------------------------------------------------------------
    # Size has chnaged
    
    def size_changed(self):
        
        # ----- Animation
        if self.geometry.shapes_count == 1:
            self.animation_ = None
            
        elif self.animation_ is not None:
            if self.animation_relative:
                self.animation_ = np.resize(self.animation_, self.shape + (self.geometry.shapes_count,))
            else:
                self.animation_ = np.resize(self.animation_, self.shape)
                
        # ----- Group transformations
                
        for gt in self.group_transformations.values():
            gt.transfo.resize(self.shape)
    
    # ---------------------------------------------------------------------------
    # Total verts count
    
    @property
    def verts_total(self):
        return self.geometry.verts_count * self.size
    
    # ---------------------------------------------------------------------------
    # Parts management
    
    def add_geometry(self, geometry):
        self.geometry.join(geometry, as_part=True)
        self.resize((self.geometry.parts_count,))
        
    # ---------------------------------------------------------------------------
    # Locate the parts to their geometrical center
    
    def origin_to_geometry(self):
        
        centers = self.geometry.centers()
        self.geometry.translate(-centers)
        self.location += centers
        
    # ---------------------------------------------------------------------------
    # Direct access to geometry attributes
    
    def faces_indices(self, part_index):
        
        if hasattr(part_index, '__len__'):
            faces = []
            for p_i in part_index:
                faces.extend(self.faces_indices(p_i))
            return faces
        
        return self.geometry.get_part_info(part_index)["faces_indices"]
    
    # ---------------------------------------------------------------------------
    # Animation
    
    @property
    def is_animatable(self):
        return self.geometry.shapes_count > 1
    
    @property
    def is_animated(self):
        return self.animation_ is not None
    
    def init_animation(self):
        self.animation = 0.
    
    @property
    def animation(self):
        return self.animation_
    
    @animation.setter
    def animation(self, value):
        
        if self.geometry.shapes_count == 1:
            return
        
        if value is None:
            self.animation_ = None
        else:
            if self.animation_ is None:
                if self.animation_relative:
                    self.animation_ = np.zeros(self.shape + (self.geometry.shapes_count-1,), float)
                else:
                    self.animation_ = np.zeros(self.shape, float)
                
            self.animation_[:] = value
    
    # ---------------------------------------------------------------------------
    # Add a transformation
    
    def add_deformation(self, deform, pre=True):
        if pre:
            self.pre_deformations.append(deform)
        else:
            self.post_deformations.append(deform)
            
    # ---------------------------------------------------------------------------
    # Add a group transformatin
    
    def add_group_transformation(self, group_name, pivot=None):
        gt = GroupTransformations(self, group_name, pivot)
        self.group_transformations[group_name] = gt
        return gt
        
    # ---------------------------------------------------------------------------
    # group transformation
    
    def group_transform(self, group_name):
        return self.group_transformations[group_name].transfo
    
    # ---------------------------------------------------------------------------
    # Transform the vertices
    
    def transform(self):
        
        if self.geometry.parts_count == 0:
            return None
        
        # ----- The shaped vertices
        
        verts4 = np.insert(
            self.geometry.shaped_verts(
                shape         = self.shape,
                relative      = self.animation_relative,
                shapes        = self.animation,
                extrapolation = self.animation_extrapolation
            ), 3, 1, axis=-1)

        # ----- Group transformations
        
        for group_name, g_trf in self.group_transformations.items():
            verts4[..., g_trf.indices, :] = g_trf.pivot + g_trf.transfo.transform_verts4(verts4[..., g_trf.indices, :] - g_trf.pivot)
        
        # ----- Pre transformations
        
        for deform in self.pre_deformations:
            deform(self.deformation_time, verts4[..., :3])
            
        # ----- Transformation
        
        verts = self.transform_verts43(verts4)
        del verts4
            
        # ----- Post transformations
        
        for deform in self.post_deformations:
            deform(self.deformation_time, verts)
            
        # ----- Return the result
        
        return self.geometry.true_verts(verts)
    
    
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
        
        self.crowd_name = None
        
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
    
    def build_object(self, crowd_name=None):
        
        if crowd_name is not None:
            self.crowd_name = crowd_name
        
        return self.full_geometry.set_to(self.crowd_name, verts=self.transform())

    # ---------------------------------------------------------------------------
    # Update the object
    
    def update(self, crowd_name=None):
        
        if self.empty:
            return
        
        if crowd_name is not None:
            self.crowd_name = crowd_name
            
        if self.crowd_name is None:
            return
            
        wobj = wrap(self.crowd_name)
        if wobj is None:
            return
            
        wobj.verts = self.transform()
            
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

        # ----- Resize
        
        self.resize((sum(self.counts),))

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

        self.resize((sum(self.counts),))
        
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
    
    
