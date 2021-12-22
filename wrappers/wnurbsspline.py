#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:25:08 2021

@author: alain
"""

import numpy as np

from .wspline import WSpline

from ..maths.bezier import  control_points, Beziers


from ..core.commons import WError


# ---------------------------------------------------------------------------
# Nurbs Spline wrapper
        
class WNurbsSpline(WSpline):
    pass

class WNurbsSpline_OLD(WSpline):
    """Nurbs spline wrapper.
    
    Caution: verts are 3-vectors. To set and get the 4-verts, use verts4 property.
    """
    
    def copy_from(self, other):
        
        super().copy_from(other)
        
        self.points.add(len(other.points) - len(self.points))
        
        for p, o in zip(self.points, other.points):
            p.co              = o.co
            p.radius          = o.radius
            p.tilt            = o.tilt
            p.weight          = o.weight
            p.weight_softbody = o.weight_softbody
    

    @property
    def verts4(self):
        """The 4-vertices of the curve.

        Returns
        -------
        array of 4-vertices
            The control points of the nurbs.
        """
        
        bpoints = self.wrapped.points
        count   = len(bpoints)
        pts     = np.empty(count*4, np.float)
        bpoints.foreach_get("co", pts)
        return pts.reshape((count, 4))

    @verts4.setter
    def verts4(self, verts):
        nverts = np.array(verts)
        count = len(nverts)

        bpoints = self.wrapped.points
        if len(bpoints) < count:
            bpoints.add(count - len(bpoints))

        if len(bpoints) > count:
            raise WError("The number of points to set is not enough",
                    Splines_points = len(bpoints),
                    Input_points = count,
                    Class = "WNurbsSpline",
                    Method = "verts4")

        bpoints.foreach_set("co", np.reshape(nverts, count*4))

        self.mark_update()
        
    @property
    def verts(self):
        """The vertices of the curve.
        
        Note that this property doesn't return the w component of the vertices.

        Returns
        -------
        array of vertices
            The control vertices of the curve.
        """
        
        return self.verts4[:, :3]
    
    @verts.setter
    def verts(self, vs):
        n = np.size(vs)//3
        v4 = np.ones((n, 4), np.float)
        v4[:, :3] =  np.reshape(vs, (n, 3))
        self.verts4 = v4
        
    # ---------------------------------------------------------------------------
    # Save and restore points when changing the number of vertices

    def save(self):
        """Save the vertices within a dictionnary.
        
        The result can be used with the restore function.

        Returns
        -------
        dictionnary {'type', 'verts4'}
            The 4-vertices.
        """
        
        return {"type": 'NURBS',  "verts4": self.verts4}
    
    def restore(self, data, count=None):
        """Restore the vertices from a dictionnary previously created with save.
        
        This method is typically used with a newly created curve with no point.

        Contrarily to verts4, the number of vertices to restore can be controlled
        with the count argument. The save / restore couple can be used to lower the
        number of control points.
        
        Typical use from WCurve:
            spline = self[index]
            data = spline.save
            self.delete(index)
            spline = self.new('NURBS')
            spline.restore(date, target_count)
            
        Parameters
        ----------
        data : dictionnary {'type', 'verts4'}
            A valid dictionnary of type NURBS.
        count : int, optional
            The number of vertices to restore. The default is None.

        Returns
        -------
        None.
        """

        if count is None:
            count = len(data["verts"])

        self.verts4 = np.resize(data["verts4"], (count, 4))

    # ---------------------------------------------------------------------------
    # Geometry from points

    def from_points(self, count, verts, lefts=None, rights=None):
        """Create a curve from a series of vertices.
        
        This function is similar to set_handles but here the number of control points
        can be controled with the count argument. The control points of the curve are
        computed to match the target number.

        Parameters
        ----------
        count : int
            The number of vertices for the curve.
        verts : array of vertices
            Interpolation vertices.
        lefts : array of vertices, optional
            Left handles. The default is None.
        rights : array of vertices, optional
            Right handles. The default is None.

        Returns
        -------
        None.
        """
        
        vf = Beziers(verts, lefts, rights)
        vs, ls, rs = control_points(vf, count)

        self.verts = vs

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

        self.verts = verts