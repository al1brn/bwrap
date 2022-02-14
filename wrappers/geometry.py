#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 18:55:52 2022

@author: alain
"""


            
import numpy as np

from .wvertexgroups import WVertexGroups
from .wrap_function import wrap

# ----------------------------------------------------------------------------------------------------
# A utility to clone a simple structure

def clone(p):
    class Q():
        pass
    
    q = Q()
    for k in dir(p):
        if k[:2] != '__':
            setattr(q, k, getattr(p, k))
            
    return q

# ----------------------------------------------------------------------------------------------------
# The Geometry class

class Geometry():
    
    def __init__(self, type = 'Mesh'):
        self.type      = type
        self.verts     = []
        self.mat_i     = []
        self.materials = []
        
        if self.type == 'Mesh':
            self.faces  = []
            self.uvmaps = {}
            self.groups = WVertexGroups()
            
        elif self.type == 'Curve':
            self.profile = None
        
        
    def __repr__(self):
        return f"<Geometry of type {self.type} with {self.verts_count} vertices>"
    
    # ----------------------------------------------------------------------------------------------------
    # Initialize from an object, Mehs or Curve
        
    @classmethod
    def FromObject(cls, object):

        wobj = wrap(object, create=False)
        
        # ---------------------------------------------------------------------------
        # Import from a Mesh
        
        if wobj.object_type == 'Mesh':
            
            geo = cls(type='Mesh')
            
            wmesh = wobj.wdata
            
            geo.verts = wmesh.verts
            geo.faces = wmesh.faces
            geo.mat_i = wmesh.material_indices

            geo.uvmaps = {}
            for name in wobj.uvmaps:
                geo.uvmaps[name] = wmesh.get_uvs(name)
                
            geo.groups = wobj.wvertex_groups
            
        # ---------------------------------------------------------------------------
        # Import from a Curve
        
        elif wobj.object_type == 'Curve':
            
            geo = cls(type='Curve')
            
            wcurve = wobj.wdata
            
            geo.verts   = wcurve.verts
            geo.profile = wcurve.profile
            geo.mat_i   = wcurve.material_indices
            
            geo.curve_properties = wcurve.curve_properties
            geo.splines_properties = wcurve.splines_properties
            
        else:
            
            return None
        
        # ---------------------------------------------------------------------------
        # Materials
        
        geo.materials = wobj.wmaterials.mat_names

        # ---------------------------------------------------------------------------
        # Done

        return geo
    
    # ----------------------------------------------------------------------------------------------------
    # Number of vertices
    
    @property
    def verts_count(self):
        return len(self.verts)
    
    # ----------------------------------------------------------------------------------------------------
    # Materials are defined
    
    @property
    def materials_count(self):
        return len(self.materials)
    
    @property
    def has_mat_i(self):
        return len(self.mat_i) > 0
    
    def ensure_mat_i(self):
        if self.type == 'Mesh':
            target = len(self.faces)
        else:
            target = len(self.profile)
            
        if len(self.mat_i) <target:
            self.mat_i = np.append(self.mat_i, np.zeros(target - len(self.mat_i)))
    
    
    # ----------------------------------------------------------------------------------------------------
    # Array
    
    @classmethod
    def Array(cls, model, count=1):
        
        count = max(1, count)
        
        geo = Geometry(type=model.type)
        
        # ----- Materials
        
        geo.materials = list(model.materials)
        
        geo.mat_i = np.zeros((count, len(model.mat_i)), int)
        geo.mat_i[:] = model.mat_i
        geo.mat_i = np.reshape(geo.mat_i, count*len(model.mat_i))
        
        # ----- Vertices
        
        geo.verts = np.empty((count, model.verts_count, 3), float)
        geo.verts[:] = model.verts
        geo.verts = np.reshape(geo.verts, (count*model.verts_count, 3))        

        # ----- Mesh
        
        if model.type == 'Mesh':
            
            # ----- Faces
            
            geo.faces = []
            offset = 0
            for i in range(count):
                geo.faces.extend([[offset + f for f in face] for face in model.faces])
                offset += model.verts_count
            
            # ----- uv maps
            
            geo.uvmaps = {}
            for name, uvs in model.uvmaps.items():
                uvmap = np.empty((count, len(uvs), 2), float)
                uvmap[:] = uvs
                geo.uvmaps[name] = np.reshape(uvmap, (count*len(uvs), 2))
                
            # ----- vertex groups
            
            geo.groups = model.groups.clone()
            geo.groups.array(count, model.verts_count)
                
        # ----- Curve
                
        elif model.type == 'Curve':
            
            # ----- Profile
            
            n = len(model.profile)
            geo.profile = np.empty((count, n, 3), int)
            geo.profile[:] = model.profile
            geo.profile = np.reshape(geo.profile, (count*n, 3))

            # ----- Properties
            
            if hasattr(model, 'curve_properties'):
                geo.curve_properties = model.curve_properties
                
            if hasattr(model, 'splines_properties'):
                geo.splines_properties = [None] * (count*len(model.splines_properties))
                offset = 0 
                for i in range(count):
                    for j, props in enumerate(model.splines_properties):
                        geo.splines_properties[offset + j] = clone(props)
                    offset += len(model.splines_properties)
                    
        # ----- Return the geometry
                    
        return geo
    
    # ----------------------------------------------------------------------------------------------------
    # Join another geometry
    
    def join(self, other):
        
        # ----- Materials
        
        # Map the other material indices to the new materials list
        # Append new material if required
        
        mat_inds = []
        for i, name in enumerate(other.materials):
            try:
                index = self.materials.index(name)
                mat_inds.append(index) 
            except:
                mat_inds.append(len(self.materials))
                self.materials.append(name)
                
        # Append the new indices
        if len(mat_inds) > 0:
            self.mat_i = np.append(self.mat_i, np.array(mat_inds)[other.mat_i])
        
        # ----- Vertices
        
        vcount = len(self.verts)
        self.verts = np.append(self.verts, other.verts, axis=0)

        # ----- Mesh
        
        if self.type == 'Mesh':
            
            # ----- Faces

            for face in other.faces:
                self.faces.append([vcount + i for i in face])
            
            # ----- uv maps
            # Can join only maps which exit in both geometries
            
            new_uvs = {}
            for name, uvs in enumerate(self.uvmaps):
                o_uvs = other.uvmaps.get(name)
                if o_uvs is not None:
                    new_uvs[name] = np.append(o_uvs, axis=0)
            
            del self.uvmaps
            self.uvmaps = new_uvs
            
            # ----- Vertex groups
            
            self.groups.join(other.groups, vcount)
            
        # ----- Curve
                
        elif self.type == 'Curve':
            
            # ----- Profile
            
            self.profile = np.append(self.profile, other.profile, axis=0)
            
            # ----- Properties
            
            if not hasattr(self, 'curve_properties'):
                if hasattr(other, 'curve_properties'):
                    self.curve_properties = other.curve_properties
                
            if hasattr(self, 'splines_properties'):
                if hasattr(other, 'splines_properties'):
                    self.splines_properties.extend([clone(props) for props in other.splines_properties])
                else:
                    del self.splines_properties
                    
                    
        # ----- Ensure material indices are consistent
        
        self.ensure_mat_i()
        
        # ----- Enable chaining
        
        return self
    
    
    # ----------------------------------------------------------------------------------------------------
    # Set to an object
    
    def set_to(self, object):
        
        wobj = wrap(object, create=self.type.upper())
        
        # ----- Materials
        
        wobj.wmaterials.replace(self.materials)
        
        # ----- Mesh
        
        
        if self.type == 'Mesh':
            
            wobj.new_geometry(self.verts, self.faces)
            wobj.material_indices = self.mat_i
            for name, uvs in self.uvmaps.items():
                wobj.create_uvmap(name)
                wobj.set_uvs(name, uvs)
                
            wobj.wvertex_groups = self.groups
            
        elif self.type == 'Curve':
            
            wobj.wdata.profile = self.profile
            wobj.wdata.verts   = self.verts
            wobj.wdata.material_indices = self.mat_i

            # ----- Properties

            if hasattr(self, 'curve_properties'):
                wobj.wdata.curve_properties = self.curve_properties
                
            if hasattr(self, 'splines_properties'):
                wobj.wdata.splines_properties = self.splines_properties
                
        return wobj
    

        