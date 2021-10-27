#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:19:41 2021

@author: alain
"""

import numpy as np

from .wstruct import WStruct

from ..core.commons import WError
from ..maths.bezier import Beziers

# ---------------------------------------------------------------------------
# Spline wrapper
# wrapped : Spline

class WSpline(WStruct):
    """Spline wrapper.
    
    The wrapper gives access to the points.
    For Bezier curves, gives access to the left and right handles.
    For nurbs curves, the points are 4D vectors.
    """

    @property
    def use_bezier(self):
        """Use bezier or nurbs.

        Returns
        -------
        bool
            True if Bezier curve, False otherwise.
        """
        
        return self.wrapped.type == 'BEZIER'
    
    def __len__(self):
        """The number of points.

        Returns
        -------
        int
            Number of points in the Spline.
        """
        
        if self.use_bezier:
            return len(self.wrapped.bezier_points)
        else:
            return len(self.wrapped.points)
        
    def __getitem__(self, index):
        if self.use_bezier:
            return self.wrapped.bezier_points[index]
        else:
            return self.wrapped.points[index]
    
    @property
    def points(self):
        """The blender points of the spline.
        
        returns bezier_points or points depending on use_bezier

        Returns
        -------
        Collection
            The Blender collection corresponding to the curve type
        """
        
        if self.use_bezier:
            return self.wrapped.bezier_points
        else:
            return self.wrapped.points
        
    # ---------------------------------------------------------------------------
    # Vertices
    
    def get_vertices(self, with_handles=True, with_w=True):
        
        n = len(self.points)
        if self.use_bezier:
                
            a = np.empty(n*3, float)
            self.points.foreach_get('co', a)

            if with_handles:
                v = np.empty((3, n, 3), float)
                v[0] = np.reshape(a, (n, 3))
                
                self.points.foreach_get('handle_left', a)
                v[1] = np.reshape(a, (n, 3))
                
                self.points.foreach_get('handle_right', a)
                v[2] = np.reshape(a, (n, 3))
            else:
                v = np.reshape(a, (n, 3))
                
        else:
            a = np.empty(n*4, float)
            self.points.foreach_get('co', a)
            
            if with_w:
                v = np.reshape(a, (n, 4))
            else:
                v = np.array(np.reshape(a, (n, 4))[:, :3])
                
        return v
    
    def set_vertices(self, verts):
        
        shape = np.shape(verts)
        n = len(self.points)
        
        if self.use_bezier:
            if len(shape) == 3:
                ex_shape = (3, n, 3)
                if shape != ex_shape:
                    raise WError(f"Vertices array has an incorrect shape. Expected is {ex_shape}, passed is {shape}",
                                 Class = "Spline", Method="set_vertices")
                    
                a = np.array(verts[0]).reshape(n*3)
                self.points.foreach_set('co', a)
                a = np.array(verts[1]).reshape(n*3)
                self.points.foreach_set('handle_left', a)
                a = np.array(verts[2]).reshape(n*3)
                self.points.foreach_set('handle_right', a)
                
            else:
                ex_shape = (n, 3)
                if shape != ex_shape:
                    raise WError(f"Vertices array has an incorrect shape. Expected is {ex_shape}, passed is {shape}",
                                 Class = "Spline", Method="set_vertices")
                    
                a = np.array(verts).reshape(n*3)
                self.points.foreach_set('co', a)
                
                v, l, r = Beziers(verts).control_points()
                
                self.points.foreach_set('handle_left', np.reshape(l, n*3))
                self.points.foreach_set('handle_right', np.reshape(r, n*3))
                
                
        else:
            ex_shape = [(n, 3), (n, 4)]
            if shape not in ex_shape:
                raise WError(f"Vertices array has an incorrect shape. Expected is {ex_shape}, passed is {shape}",
                             Class = "Spline", Method="set_°vertices")
                
            if shape[2] == 3:
                a = np.ones((n, 4), float)
                a[:, :3] = verts
            else:
                a = verts
                
            self.points.foreach_set('co', np.reshape(a, n*4))
        
    # ---------------------------------------------------------------------------
    # Copy from another spline
    
    def copy_from(self, other):
        
        self.tilt_interpolation   = other.tilt_interpolation
        self.radius_interpolation = other.radius_interpolation
        self.type                 = other.type
        self.order_u              = other.order_u
        self.order_v              = other.order_v
        self.resolution_u         = other.resolution_u
        self.resolution_v         = other.resolution_v
        self.use_cyclic_u         = other.use_cyclic_u
        self.use_cyclic_v         = other.use_cyclic_v
        self.use_endpoint_u       = other.use_endpoint_u
        self.use_endpoint_v       = other.use_endpoint_v
        self.use_bezier_u         = other.use_bezier_u
        self.use_bezier_v         = other.use_bezier_v
        self.use_smooth           = other.use_smooth
        self.material_index       = other.material_index
        
    # ---------------------------------------------------------------------------
    # Get a Bezier function
    
    def get_bezier(self):
        
        if self.use_bezier:
            verts = self.get_vertices(with_handles=True)
            return Beziers(verts[0], verts[1], verts[2])
        else:
            return Beziers(self.get_vertices(with_w=False))
        
        
    # ===========================================================================
    # Generated source code for WSpline class

    @property
    def rna_type(self):
        return self.wrapped.rna_type

    @property
    def bezier_points(self):
        return self.wrapped.bezier_points

    @property
    def tilt_interpolation(self):
        return self.wrapped.tilt_interpolation

    @tilt_interpolation.setter
    def tilt_interpolation(self, value):
        self.wrapped.tilt_interpolation = value

    @property
    def radius_interpolation(self):
        return self.wrapped.radius_interpolation

    @radius_interpolation.setter
    def radius_interpolation(self, value):
        self.wrapped.radius_interpolation = value

    @property
    def type(self):
        return self.wrapped.type

    @type.setter
    def type(self, value):
        self.wrapped.type = value

    @property
    def point_count_u(self):
        return self.wrapped.point_count_u

    @property
    def point_count_v(self):
        return self.wrapped.point_count_v

    @property
    def order_u(self):
        return self.wrapped.order_u

    @order_u.setter
    def order_u(self, value):
        self.wrapped.order_u = value

    @property
    def order_v(self):
        return self.wrapped.order_v

    @order_v.setter
    def order_v(self, value):
        self.wrapped.order_v = value

    @property
    def resolution_u(self):
        return self.wrapped.resolution_u

    @resolution_u.setter
    def resolution_u(self, value):
        self.wrapped.resolution_u = value

    @property
    def resolution_v(self):
        return self.wrapped.resolution_v

    @resolution_v.setter
    def resolution_v(self, value):
        self.wrapped.resolution_v = value

    @property
    def use_cyclic_u(self):
        return self.wrapped.use_cyclic_u

    @use_cyclic_u.setter
    def use_cyclic_u(self, value):
        self.wrapped.use_cyclic_u = value

    @property
    def use_cyclic_v(self):
        return self.wrapped.use_cyclic_v

    @use_cyclic_v.setter
    def use_cyclic_v(self, value):
        self.wrapped.use_cyclic_v = value

    @property
    def use_endpoint_u(self):
        return self.wrapped.use_endpoint_u

    @use_endpoint_u.setter
    def use_endpoint_u(self, value):
        self.wrapped.use_endpoint_u = value

    @property
    def use_endpoint_v(self):
        return self.wrapped.use_endpoint_v

    @use_endpoint_v.setter
    def use_endpoint_v(self, value):
        self.wrapped.use_endpoint_v = value

    @property
    def use_bezier_u(self):
        return self.wrapped.use_bezier_u

    @use_bezier_u.setter
    def use_bezier_u(self, value):
        self.wrapped.use_bezier_u = value

    @property
    def use_bezier_v(self):
        return self.wrapped.use_bezier_v

    @use_bezier_v.setter
    def use_bezier_v(self, value):
        self.wrapped.use_bezier_v = value

    @property
    def use_smooth(self):
        return self.wrapped.use_smooth

    @use_smooth.setter
    def use_smooth(self, value):
        self.wrapped.use_smooth = value

    @property
    def material_index(self):
        return self.wrapped.material_index

    @material_index.setter
    def material_index(self, value):
        self.wrapped.material_index = value

    @property
    def character_index(self):
        return self.wrapped.character_index

    @property
    def bl_rna(self):
        return self.wrapped.bl_rna

    def calc_length(self, *args, **kwargs):
        return self.wrapped.calc_length(*args, **kwargs)

    # End of generation
    # ===========================================================================
        
        
        
        
       