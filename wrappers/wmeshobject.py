#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:20:47 2021

@author: alain
"""

import numpy as np

from .wmesh import WMesh
from .wobject import WObject

class WMeshObject(WObject):
    
    @property
    def wmesh(self):
        wo = self.wrapped
        return WMesh(wo.data, wo.is_evaluated)

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
    
    