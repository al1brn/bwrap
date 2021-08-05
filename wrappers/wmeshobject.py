#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:20:47 2021

@author: alain
"""

import numpy as np

from .wmesh import WMesh
from .wobject import WObject
from .wshapekey import WMeshShapeKeys
from ..core.class_enhance import expose

from ..core.commons import WError

class WMeshObject(WObject):
    
    def __init__(self, wrapped, is_evaluated=None):
        super().__init__(wrapped, is_evaluated)
        self.wmesh = WMesh(self.name, self.is_evaluated)
    
    def set_evaluated(self, value):
        self.is_evaluated = value
        self.wmesh.is_evaluated = value

    # ---------------------------------------------------------------------------
    # Origin to geometry

    def origin_to_geometry(self):
        """Utility to set the mesh origin to the geometry center.

        Raises
        ------
        RuntimeError
            If the object is not a mesh.

        Returns
        -------
        None.
        """

        wmesh = self.wmesh

        verts = wmesh.verts
        origin = np.sum(verts, axis=0)/len(verts)
        wmesh.verts = verts - origin

        self.location = np.array(self.location) + origin
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Get animation from shape keys
    
    @property
    def wshape_keys(self):
        return WMeshShapeKeys(self.name)
    
    def get_sk_animation(self, eval_times):
        memo = self.eval_time
        
        verts = np.empty((len(eval_times), self.verts_count, 3), np.float)
        for i, evt in enumerate(eval_times):
            self.eval_time = evt
            verts[i] = self.evaluated.verts
        
        self.eval_time = memo
        
        return verts

    # -----------------------------------------------------------------------------------------------------------------------------
    # The available groups
        
    def groups(self):
        return [group.name for group in self.wrapped.vertex_groups]
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Vertex group index
    
    def group_index(self, group_name):
        group = self.wrapped.vertex_groups.get(group_name)
        if group is None:
            raise WError(f"Vertex group '{group_name}' doesn't exist in the mesh object '{self.name}'")
        else:
            return group.index

    # -----------------------------------------------------------------------------------------------------------------------------
    # Indices of a group

    def group_indices(self, group_name=None):

        if group_name is None:
            idx = {}
            for name in self.groups():
                idx[name] = self.group_indices(name)
            return idx

        else:

            return self.wmesh.group_indices(self.group_index(group_name))
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Center of a group
    
    def group_center(self, group_name):
        inds = self.group_indices(group_name)
        if len(inds) > 0:
            return np.sum(self.verts[inds], axis=0)/len(inds)
        else:
            return np.array((0., 0., 0.))
        
    

# ===========================================================================
# Expose wmesh methods and properties  

expose(WMeshObject, WMesh, "wmesh", WMesh.exposed_methods(), WMesh.exposed_properties())  
    
    
    
    