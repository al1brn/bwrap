#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:19:41 2021

@author: alain
"""

from .wstruct import WStruct

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
    
    @property
    def count(self):
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
        
        
        
        
       