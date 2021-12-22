#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:20:43 2021

@author: alain
"""

import numpy as np

from .wspline import WSpline

from ..maths.bezier import  control_points, Beziers


from ..core.commons import WError


# ---------------------------------------------------------------------------
# Bezier Spline wrapper
    
class WBezierSpline(WSpline):
    pass

class WBezierSpline_OLD(WSpline):
    """Wraps a Bezier spline.
    
    The points of the curve can be managed with the left and right handles or not:
        - curve.verts = np.array(...)
        - curve.set_handles(points, lefts, rights)
        
    When lefts and rights handles are not given they are computed.
    """
    
    def copy_from(self, other):
        
        super().copy_from(other)
        
        self.bezier_points.add(len(other.bezier_points) - len(self.bezier_points))
        
        for p, o in zip(self.bezier_points, other.bezier_points):
            p.co                = o.co
            p.handle_left       = o.handle_left
            p.handle_right      = o.handle_right
            p.handle_left_type  = o.handle_left_type
            p.handle_right_type = o.handle_right_type
            p.radius            = o.radius
            p.tilt              = o.tilt
    
    @property
    def verts(self):
        """Vertices of the curve.

        Returns
        -------
        array of vertices
            The vertices of the curve.
        """
        
        bpoints = self.wrapped.bezier_points
        count   = len(bpoints)
        pts     = np.empty(count*3, np.float)
        bpoints.foreach_get("co", pts)
        return pts.reshape((count, 3))

    @verts.setter
    def verts(self, verts):
        self.set_handles(verts)

    @property
    def lefts(self):
        """Left handles of the curve.
        
        Left handles can't be set solely. Use set_handles.
        
        Returns
        -------
        array of vertices
            The left handfles.
        """

        bpoints = self.wrapped.bezier_points
        count   = len(bpoints)
        pts     = np.empty(count*3, np.float)
        bpoints.foreach_get("handle_left", pts)
        return pts.reshape((count, 3))

    @property
    def rights(self):
        """Right handles of the curve.
        
        Right handles can't be set solely. Use set_handles.
        
        Returns
        -------
        array of vertices
            The right handfles.
        """

        bpoints = self.wrapped.bezier_points
        count   = len(bpoints)
        pts     = np.empty(count*3, np.float)
        bpoints.foreach_get("handle_right", pts)
        return pts.reshape((count, 3))

    # ---------------------------------------------------------------------------
    # Get the points and handles for bezier curves
    
    def get_handles(self):
        """Get the vertices and the handles of the curve.

        Returns
        -------
        3 arrays of vertices
            Vertices, left and right handles.
        """
        
        bl_points = self.wrapped.bezier_points
        count  = len(bl_points)

        pts    = np.empty(count*3, np.float)
        lfs    = np.empty(count*3, np.float)
        rgs    = np.empty(count*3, np.float)

        bl_points.foreach_get("co", pts)
        bl_points.foreach_get("handle_left", lfs)
        bl_points.foreach_get("handle_right", rgs)

        return pts.reshape((count, 3)), lfs.reshape((count, 3)), rgs.reshape((count, 3))


    # ---------------------------------------------------------------------------
    # Set the points and possibly handles for bezier curves

    def set_handles(self, verts, lefts=None, rights=None):
        """Set the vertices and the handles of the curve.
        
        The number of vertices can be greater than the number of existing vertices
        but it can't be lower. If lower, an exception is raised.
        
        To decrease the number of vertices, the spline must be replaced.
        To replace a curve without loosing the control points, the save and restore
        methods can be used.

        Parameters
        ----------
        verts : array of vertices
            The vertices to set.
        lefts : array of vertices, optional
            The left handles. The length of this array, if given, must match the one
            of the verts array. The default is None.
        rights : array of vertices, optional
            The rights handles. The length of this array, if given, must match the one
            of the verts array. The default is None.

        Raises
        ------
        RuntimeError
            Raise an error if the number of given vertices is less than the number
            of existing vertices.

        Returns
        -------
        None.
        """

        nvectors = np.array(verts)
        count = len(nvectors)

        bl_points = self.wrapped.bezier_points
        if len(bl_points) < count:
            bl_points.add(len(verts) - len(bl_points))

        if len(bl_points) > count:
            raise WError("The number of points to set is not enough:",
                         Splines_points = len(bl_points),
                         Input_points = count,
                         Class = "WBezierSpline",
                         Method = "set_handles")

        bl_points.foreach_set("co", np.reshape(nvectors, count*3))

        if lefts is not None:
            pts = np.array(lefts).reshape(count*3)
            bl_points.foreach_set("handle_left", np.reshape(pts, count*3))

        if rights is not None:
            pts = np.array(rights).reshape(count*3)
            bl_points.foreach_set("handle_right", np.reshape(pts, count*3))

        if (lefts is None) and (rights is None):
            for bv in bl_points:
                bv.handle_left_type  = 'AUTO'
                bv.handle_right_type = 'AUTO'

        self.mark_update()

    # ---------------------------------------------------------------------------
    # As an interpolated function
        
    @property
    def function(self):
        """Returns a function interpolating this curve.        

        Returns
        -------
        PointsInterpolation
            The class can be called from 0. to 1. to get any vector on the curve..        
        """
        
        points, lefts, rights = self.get_handles()
        return Beziers(points, lefts, rights)
        
    # ---------------------------------------------------------------------------
    # Save and restore points when changing the number of vertices

    def save(self):
        """Save the vertices within a dictionnary.
        
        The result can be used with the restore function.

        Returns
        -------
        dictionnary {'type', 'verts', 'lefts', 'rights'}
            The vertices, left and right handles.
        """
        
        verts, lefts, rights = self.get_handles()
        return {"type": 'BEZIER', "verts": verts, "lefts": lefts, "rights": rights}
    
    def restore(self, data, count=None):
        """Restore the vertices from a dictionnary previously created with save.
        
        This method is typically used with a newly created Bezier curve with no point.

        Contrarily to set_handles, the number of vertices (and handles) to restore
        can be controlled with the count argument. The save / restore couple can be used to
        lower the number of control points.
        
        Typical use from WCurve:
            spline = self[index]
            data = spline.save
            self.delete(index)
            spline = self.new('BEZIER')
            spline.restore(date, target_count)
            
        Parameters
        ----------
        data : dictionnary {'type', 'verts', 'lefts', 'rights'}
            A valid dictionnary of type BEZIER.
        count : int, optional
            The number of vertices to restore. The default is None.

        Returns
        -------
        None.
        """

        if count is None:
            count = len(data["verts"])

        points = np.resize(data["verts"],  (count, 3))
        
        lefts = data.get("lefts")
        if lefts is not None:
            lefts  = np.resize(lefts,  (count, 3))
            
        rights = data.get("rights")
        if rights is not None:
            rights = np.resize(rights, (count, 3))
            
        self.set_handles(points, lefts, rights)

    # ---------------------------------------------------------------------------
    # Geometry from points

    def from_points(self, verts, lefts=None, rights=None, length=None):
        """Create a curve from a series of vertices.
        
        This function is similar to set_handles but here the number of control points
        can be controled with the count argument. The control points of the curve are
        computed to match the target number.

        Parameters
        ----------
        verts : array of vertices
            Interpolation vertices.
        lefts : array of vertices, optional
            Left handles. The default is None.
        rights : array of vertices, optional
            Right handles. The default is None.
        length : int
            The number of vertices for the curve. Default is None

        Returns
        -------
        None.
        """
        
        if length is None:
            vf = Beziers(verts, lefts, rights)
            vs = vf.points
            ls = vf.lefts
            rs = vf.rights
        else:
            vs, ls, rs = control_points(vf, length)

        self.set_handles(vs, ls, rs)

    # ---------------------------------------------------------------------------
    # Geometry from function

    def from_function(self, count, f, t0=0, t1=1):
        """Create a curve from a function.

        Parameters
        ----------
        count : int
            The number of vertices to create.
        f : function of template f(t) --> vertex
            The function to use to create the curve.
        t0 : float, optional
            Starting value to use to compute the curve. The default is 0.
        t1 : float, optional
            Ending valud to use to compute the curve. The default is 1.

        Returns
        -------
        None.
        """
        
        dt = (t1-t0)/1000
        verts, lefts, rights = control_points(f, count, t0, t1, dt)

        self.set_handles(verts, lefts, rights)
        
