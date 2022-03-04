#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:19:41 2021

@author: alain
"""

import numpy as np

from .wstruct import WStruct

from ..core.commons import WError
from ..core.maths.bezier import Beziers

# ---------------------------------------------------------------------------
# Spline wrapper
# wrapped : Spline

class WSpline(WStruct):
    """Spline wrapper.
    
    The wrapper gives access to the points.
    For Bezier curves, gives access to the left and right handles.
    For nurbs curves, the points are 4D vectors.
    """
    
    # ---------------------------------------------------------------------------
    # [‘POLY’, ‘BEZIER’, ‘BSPLINE’, ‘CARDINAL’, ‘NURBS’], default ‘POLY’
    
    @staticmethod
    def type_to_code(type):
        if type == 'BEZIER':
            return 0
        elif type == 'POLY':
            return 1
        elif type == 'BSPLINE':
            return 2
        elif type == 'CARDINAL':
            return 3
        else: # NURBS
            return 4
        
    @staticmethod
    def code_to_type(code):
        return ['BEZIER', 'POLY', 'BSPLINE', 'CARDINAL', 'NURBS'][code]
    
    @property
    def type_code(self):
        return WSpline.type_to_code(self.wrapped.type)

    # ---------------------------------------------------------------------------
    # A Bezier curve
    
    @property
    def use_bezier(self):
        """Use bezier or nurbs.

        Returns
        -------
        bool
            True if Bezier curve, False otherwise.
        """
        
        return self.wrapped.type == 'BEZIER'
    
    # ---------------------------------------------------------------------------
    # The points array
    
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
    # Spline profile: 1 triplets made of
    # - 3 for Bezier curves (point, left, right)
    # - points count
    # - type code
    
    @property
    def profile(self):
        if self.type == 'BEZIER':
            return [3, len(self.wrapped.bezier_points), 0]
        else:
            return [1, len(self.wrapped.points), WSpline.type_to_code(self.wrapped.type)]
        
    # ---------------------------------------------------------------------------
    # As an array of points or bezier points

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
        
    # ---------------------------------------------------------------------------
    # Verts counts
    
    @property
    def verts_count(self):
        return len(self.points)
    
    @verts_count.setter
    def verts_count(self, value):
        
        pts = self.points
        if len(pts) < value:
            pts.add(value - len(pts))

        if len(pts) > value:
            raise WError("Impossible to reduce the number of points of a spline.",
                    Splines_points = len(pts),
                    Input_points = value,
                    Class = "WSpline",
                    Method = "verts_count")
        
    # ---------------------------------------------------------------------------
    # Get all the vertices
    # - Bezier : vertices plus left and right handles if required
    # - Other  : 3-vertices or 4 vertices if required
    
    def get_vertices(self, with_handles=True, with_w=True):
        
        wrapped = self.wrapped
        
        if wrapped.type == 'BEZIER':
            
            n = len(wrapped.bezier_points)
                
            a = np.empty(n*3, float)
            wrapped.bezier_points.foreach_get('co', a)

            if with_handles:
                v = np.empty((3, n, 3), float)
                v[0] = np.reshape(a, (n, 3))
                
                wrapped.bezier_points.foreach_get('handle_left', a)
                v[1] = np.reshape(a, (n, 3))
                
                wrapped.bezier_points.foreach_get('handle_right', a)
                v[2] = np.reshape(a, (n, 3))
                
            else:
                v = np.reshape(a, (n, 3))
                
        else:
            
            n = len(wrapped.points)
            
            a = np.empty(n*4, float)
            wrapped.points.foreach_get('co', a)
            
            if with_w:
                v = np.reshape(a, (n, 4))
            else:
                v = np.array(np.reshape(a, (n, 4))[:, :3])
                
        return v
    
    # ---------------------------------------------------------------------------
    # Set the vertices
    
    def set_vertices(self, verts):
        
        wrapped = self.wrapped
        shape   = np.shape(verts)
        
        # ----- Adjust the number of vertices
        
        if len(shape) == 3:
            self.verts_count = shape[1]
        else:
            self.verts_count = shape[0]

        n = self.verts_count
        
        # ----- Bezier
        
        if self.use_bezier:
            
            if len(shape) == 3:
                    
                ex_shape = (3, n, 3)
                if shape != ex_shape:
                    raise WError(f"Vertices array has an incorrect shape. Expected is {ex_shape}, passed is {shape}",
                                 Class = "Spline", Method="set_vertices")
                    
                a = np.array(verts[0]).reshape(n*3)
                wrapped.bezier_points.foreach_set('co', a)
                a = np.array(verts[1]).reshape(n*3)
                wrapped.bezier_points.foreach_set('handle_left', a)
                a = np.array(verts[2]).reshape(n*3)
                wrapped.bezier_points.foreach_set('handle_right', a)
                
            else:
                
                ex_shape = (n, 3)
                if shape != ex_shape:
                    raise WError(f"Vertices array has an incorrect shape. Expected is {ex_shape}, passed is {shape}",
                                 Class = "Spline", Method="set_vertices")
                    
                #a = np.array(verts).reshape(n*3)
                #wrapped.bezier_points.foreach_set('co', a)
                
                v, l, r = Beziers(verts).control_points()
                
                wrapped.bezier_points.foreach_set('co',           np.reshape(v, n*3))
                wrapped.bezier_points.foreach_set('handle_left',  np.reshape(l, n*3))
                wrapped.bezier_points.foreach_set('handle_right', np.reshape(r, n*3))
                
        # ----- Non Bezier
                
        else:
            
            ex_shape = [(n, 3), (n, 4)]
            if shape not in ex_shape:
                raise WError(f"Vertices array has an incorrect shape. Expected is {ex_shape}, passed is {shape}",
                             Class = "Spline", Method="set_°vertices")
                
            a = np.ones((n, 4), float)
            if shape[1] == 3:
                a[:, :3] = verts
            else:
                a[:] = verts
                
            wrapped.points.foreach_set('co', np.reshape(a, n*4))    
            
        self.mark_update()
   
    # ---------------------------------------------------------------------------
    # Vertices
    
    @property
    def verts(self):
        return self.get_vertices(with_handles=True, with_w = False)
        
    @verts.setter
    def verts(self, value):
        self.set_vertices(value)
        
    # ---------------------------------------------------------------------------
    # Vertices with w
    
    @property
    def verts4(self):
        return self.get_vertices(with_handles=True, with_w = True)
    
    @verts4.setter
    def verts4(self, value):
        self.set_vertices(value)
        
    # ---------------------------------------------------------------------------
    # w value only
    
    @property
    def ws(self):
        return self.verts4[:, -1]
    
    @ws.setter
    def ws(self, value):
        v4 = self.verts4
        v4[:, -1] = value
        self.verts4 = v4
        
    # ---------------------------------------------------------------------------
    # For stacking
    
    @property
    def stack_verts(self):
        verts = self.get_vertices(with_handles=True, with_w=False)
        if self.use_bezier:
            return np.reshape(verts, (len(self)*3, 3))
        else:
            return verts
        
    @stack_verts.setter
    def stack_verts(self, value):
        if self.use_bezier:
            self.set_vertices(np.reshape(value, (3, len(self), 3)))
        else:
            self.set_vertices(value)
            
    # ---------------------------------------------------------------------------
    # Get an attribute
    
    def get_points_attr(self, attr_name):
        
        if attr_name == 'W':
            return self.ws
            
        pts = self.points
        n   = len(pts)
        a   = np.empty(n, float)
        pts.foreach_get(attr_name, a)
        return a

    # ---------------------------------------------------------------------------
    # Set an attribute
    
    def set_points_attr(self, attr_name, value):
        
        if attr_name == 'W':
            self.ws  =value
            return
        
        pts = self.points
        n   = len(pts)
        a   = np.empty(n, float)
        a[:] = value
        pts.foreach_set(attr_name, a)
        
    # ---------------------------------------------------------------------------
    # Tilt
    
    @property
    def tilts(self):
        return self.get_points_attr('tilt')
    
    @tilts.setter
    def tilts(self, value):
        self.set_points_attr('tilt', value)
        
    # ---------------------------------------------------------------------------
    # Radius
    
    @property
    def radius(self):
        return self.get_points_attr('radius')
    
    @radius.setter
    def radius(self, value):
        self.set_points_attr('radius', value)
        
    # ---------------------------------------------------------------------------
    # Weight soft body
    
    @property
    def weight_softbodies(self):
        return self.get_points_attr('weight_softbody')
    
    @weight_softbodies.setter
    def weight_softbodies(self, value):
        self.set_points_attr('weight_softbody', value)
        
    # ---------------------------------------------------------------------------
    # Weight
    
    @property
    def weights(self):
        return self.get_points_attr('weight')
    
    @weights.setter
    def weights(self, value):
        self.set_points_attr('weight', value)
        
    # ---------------------------------------------------------------------------
    # Common:
    # - 0 : tilt
    # - 1 : radius
    # - 2 : weight_softbody
    # Non Bezier
    # - 3 : weight
    # - 4 : W
    
    @property
    def trws(self):
        n = len(self)
        shape = (n, 3) if self.use_bezier else (n, 5)
        a = np.empty(shape, float)
        a[0] = self.tilts
        a[1] = self.radius
        a[2] = self.weight_softbodies
        if shape[1] > 3:
            a[3] = self.weights
            a[4] = self.ws
            
        return a
    
    @trws.setter
    def trws(self, value):
        n = len(self)
        shape = (n, 3) if self.use_bezier else (n, 5)
        a = np.empty(shape, float)
        try:
            a[:] = value
        except:
            raise WError(f"Setting tilt, radius, weight softbody with array of shape {np.shape(value)} is not possible.\n" +
                         f"Expected shape is {shape}.",
                         Class = "Spline", Method="trws")
        
        self.tilts  = a[0]
        self.radius = a[1]
        self.weight_softbodies = a[2]
        if shape[1] > 3:
            self.weights = a[3]
            self.ws      = a[4]
        
    # ---------------------------------------------------------------------------
    # Full points
    # - Bezier     : (verts, lefts, rights), trws
    # - Non Bezier : verts, trws
    
    @property
    def full_points(self):
        return self.get_vertices(with_handles=True, with_w=True), self.trws
    
    @full_points.setter
    def full_points(self, value):
        self.set_vertices(value[0])
        self.trws = value[1]
        
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
        #self.material_index       = other.material_index

    # ---------------------------------------------------------------------------
    # Copy from another spline
    
    def copy_to(self, other):
        
        other.tilt_interpolation   = self.tilt_interpolation
        other.radius_interpolation = self.radius_interpolation
        other.type                 = self.type
        other.order_u              = self.order_u
        other.order_v              = self.order_v
        other.resolution_u         = self.resolution_u
        other.resolution_v         = self.resolution_v
        other.use_cyclic_u         = self.use_cyclic_u
        other.use_cyclic_v         = self.use_cyclic_v
        other.use_endpoint_u       = self.use_endpoint_u
        other.use_endpoint_v       = self.use_endpoint_v
        other.use_bezier_u         = self.use_bezier_u
        other.use_bezier_v         = self.use_bezier_v
        other.use_smooth           = self.use_smooth
        #other.material_index       = self.material_index
        
        return other
        
    # ---------------------------------------------------------------------------
    # Parameters
    
    @property
    def spline_properties(self):
        class Props():
            pass
        
        return self.copy_to(Props())
    
    @spline_properties.setter
    def spline_properties(self, value):
        self.copy_from(value)
        
        
    # ---------------------------------------------------------------------------
    # Set a function
    
    def set_function(self, f, t0=0., t1=1., dt=None):
        n = len(self)
        if self.use_bezier:
            v, l, r = Beziers.FromFunction(f, n, t0, t1, dt).control_points()
            self.set_vertices(np.array((v, l, r)))
        else:
            self.verts = f[np.linspace(t0, t1, n)]
        
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
        
        
        
        
       