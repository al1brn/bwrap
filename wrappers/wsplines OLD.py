#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 07:11:35 2021

@author: alain
"""

import numpy as np

import bpy

from .wstruct import WStruct
from .wspline import WSpline
from .wbezierspline import WBezierSpline
from .wnurbsspline import WNurbsSpline

from ..core.profile import Profile
from ..core.maths.bezier import Beziers

from ..core.commons import WError


class WSplines(WStruct):
    
    # ---------------------------------------------------------------------------
    # Initalization
    
    def __init__(self, data):
        super().__init__(wrapped=data.splines)
        
        self.obj_name = None
        for obj in bpy.data.objects:
            if obj.data == data:
                self.obj_name = obj.name
                break
            
        self.profile_ = None
                
    def update_tag(self):
        obj = bpy.data.objects[self.obj_name]
        obj.data.update_tag()
        obj.update_tag()

    # ---------------------------------------------------------------------------
    # WCurve is a collection of splines

    def __len__(self):
        """Number of splines.

        Returns
        -------
        int
            Number fo splines.
        """
        
        return len(self.wrapped)

    def __getitem__(self, index):
        """The wrapper of the indexed spline.

        Parameters
        ----------
        index : int
            Valid index within the colleciton of spines.

        Returns
        -------
        WSpline
            Wrapper of the indexed spline.
        """
        
        return WSplines.spline_wrapper(self.wrapped[index])

    # ---------------------------------------------------------------------------
    # Utility function to wrap a BEZIER ot NURBS spline.
    
    @staticmethod
    def spline_wrapper(spline):
        """Utility function to wrap a BEZIER ot NURBS spline.
    
        Parameters
        ----------
        spline : Blender spline
            The spline to wrap with a WBezierSpline ot WNurbsSpline wrapper.
    
        Returns
        -------
        WSpline
            The wrapper of the spline.
        """
        
        return WBezierSpline(spline) if spline.type == 'BEZIER' else WNurbsSpline(spline)

    # ---------------------------------------------------------------------------
    # Add a spline

    def new(self, spline_type='BEZIER', verts_count=None):
        """Create a new spline of the given type.

        Parameters
        ----------
        spline_type : str, optional
            A valide spline type. The default is 'BEZIER'.

        Returns
        -------
        spline : WSpline
            Wrapper of the newly created spline.
        """
        
        splines = self.wrapped
        spline  = WSplines.spline_wrapper(splines.new(spline_type))
        if verts_count is not None:
            spline.verts_count = verts_count
        
        self.update_tag()
        
        return spline

    # ---------------------------------------------------------------------------
    # Delete a spline

    def delete(self, index):
        """Delete the spline at the given index.

        Parameters
        ----------
        index : int
            Index of the spline to delete.

        Returns
        -------
        None.
        """
        
        splines = self.wrapped
        if index <= len(splines)-1:
            splines.remove(splines[index])

        self.update_tag()
        
    # ===========================================================================
    # Profile
    
    def cache_profile(self):
        self.profile_ = None
        self.profile_ = self.profile
        return self.profile_
    
    def uncache_profile(self):
        self.profile_ = None
    
    @property
    def profile(self):
        if self.profile_ is not None:
            return self.profile_
        else:
            return Profile.FromSplines(self)
    
    # ===========================================================================
    # Change the profile
    #
    # As soon as the requested type of spline doesn't match the existing,
    # the spline and the following are deleted
    #
    # Nothing happens if the requested profile matches the existing one
    
    @profile.setter
    def profile(self, profile):
        
        self.profile_ = profile
        
        splines = self.wrapped
        
        # ---------------------------------------------------------------------------
        # From bottom to top while we can change without deleting splines
        
        index = 0
        for (ctype, length), spline in zip(profile, self):
            
            spline_type = Profile.ctype_code(spline.type)
            
            # ----- Tupes are different and one is Bezier
            # we must exit
            
            if spline_type != ctype:
                if spline_type == 0 or ctype == 0:
                    break
            
            points = spline.points
            n = len(points)
            if n < length:
                points.add(length - n)
                
            if len(points) != length:
                break
                
            index += 1
        
        # ---------------------------------------------------------------------------
        # index is the number of splines which are ok
        # splines after must be deleted
        
        for i in range(index, len(splines)):
            splines.remove(splines[len(splines)-1])
            
        # ---------------------------------------------------------------------------
        # We must then create the missing splines with the right number of points
        
        for i in range(index, len(profile)):
            spline = self.new(profile.ctype(i, as_str=True),  profile.length(i))
                
        self.update_tag()
        
                
    # ===========================================================================
    # A user friendly version of the profile setting
    #
    # set_profile('BEZIER', 3, 10) : Create 10 BEZIER splines of 3 points
    
    def set_profile(self, ctype = 'BEZIER', length=2, count=1):
        self.profile = Profile(ctype, length, count)
        
    # ===========================================================================
    # Initialize with a set of points
    
    def set_beziers(self, points, lefts=None, rights=None):
        
        bzs    = Beziers(points, lefts, rights)
        count  = 1 if bzs.shape == () else bzs.shape[0]
        length = bzs.count
        
        self.set_profile('BEZIER', length, count)
        v, l, r = bzs.control_points()
        
        if bzs.shape == ():
            self[0].set_vertices(np.array((v, l, r)))
        else:
            for i in range(count):
                self[i].set_vertices(np.array((v[i], l[i], r[i])))
                
        return
            
    # ===========================================================================
    # Initialize with a set of functions
            
    def set_function(self, f, t0=0, t1=1, length=100):
        
        self.set_profile('BEZIER', length, 1)
        self[0].set_function(f, t0, t1)
            
    # ===========================================================================
    # Initialize with a set of functions
            
    def set_functions(self, fs, t0=0, t1=1, length=100):
        
        self.set_profile('BEZIER', length, len(fs))
        for i, f in enumerate(fs):
            self[i].set_function(f, t0, t1)
        
    # ===========================================================================
    # Some helpers
    
    @property
    def only_bezier(self):
        return self.profile.only_bezier

    @property
    def only_nurbs(self):
        return self.profile.only_nurbs
    
    @property
    def has_bezier(self):
        return self.profile.has_bezier

    @property
    def has_nurbs(self):
        return self.profile.has_nurbs

    @property
    def is_mix(self):
        return self.has_bezier and self.is_mix
    
    @property
    def verts_count(self):
        return self.profile.verts_count
    
    @property
    def points_count(self):
        return self.profile.points_count
    
    # ===========================================================================
    # Get the vertices
    
    def get_vertices(self, ndim=3):
        
        profile = self.profile

        ndim = np.clip(ndim, 3, 4)

        verts = np.zeros((profile.verts_count, ndim), float)

        for (ctype, offset, length, nverts), spline in zip(profile.verts_iter(), self.wrapped):

            if ctype == 0:
                a = np.empty((3, length * 3), float)

                spline.bezier_points.foreach_get('co',           a[0])
                spline.bezier_points.foreach_get('handle_left',  a[1])
                spline.bezier_points.foreach_get('handle_right', a[2])

                verts[offset:offset+nverts, :3] = np.reshape(a, (nverts, 3))

            else:
                a = np.empty(length*4, float)
                spline.points.foreach_get('co', a)

                verts[offset:offset +
                      nverts] = np.reshape(a, (length, 4))[:, :ndim]

        return verts
    
    @property
    def verts4(self):
        return self.get_vertices(ndim=4)
    
    @property
    def verts(self):
        return self.get_vertices()

    # ===========================================================================
    # Set the vertices, either ndim=3 ort 4
    
    @verts.setter
    def verts(self, verts):
        
        profile = self.profile

        ndim = np.shape(verts)[-1]

        for (ctype, offset, length, nverts), spline in zip(profile.verts_iter(), self.wrapped):

            if ctype == 0:
                a = np.reshape(verts[offset:offset+nverts, :3], (3, length*3))

                spline.bezier_points.foreach_set('co',           a[0])
                spline.bezier_points.foreach_set('handle_left',  a[1])
                spline.bezier_points.foreach_set('handle_right', a[2])

            else:
                a = np.ones((length, 4), float)
                a[:, :ndim] = verts[offset:offset+nverts]
                spline.points.foreach_set('co', np.reshape(a, (length*4)))

        self.wrapped.data.update_tag()

    # ===========================================================================
    # Read attributes

    def get_attrs(self, attrs=['radius', 'tilt', 'weight', 'weight_softbody']):
        
        profile = self.profile

        n = len(attrs)
        if n == 0:
            return None

        values = {}
        for attr in attrs:
            values[attr] = np.zeros(profile.points_count, float)

        for (ctype, offset, length), spline in zip(profile.points_iter(), self.wrapped):

            for attr, vals in values.items():

                pts = spline.bezier_points if ctype == 0 else spline.points

                if (ctype != 0) or (attr != 'weight'):
                    pts.foreach_get(attr, vals[offset:offset+length])

        return values

    # ===========================================================================
    # Write attributes

    def set_attrs(self, values):
        
        profile = self.profile

        offset = 0

        for (ctype, offset, length), spline in zip(profile.points_iter(), self.wrapped):

            for attr, vals in values.items():

                pts = spline.bezier_points if ctype == 0 else spline.points

                if (ctype != 0) or (attr != 'weight'):
                    pts.foreach_set(attr, vals[offset:offset+length])    
    
            
    # ===========================================================================
    
    def get_attr(self, name):
        return self.get_attrs(name)[name]

    def set_attr(self, name, values):
        return self.set_attrs({name: values})
    
    # ---------------------------------------------------------------------------
    # Tilt
    
    @property
    def tilts(self):
        return self.get_attr('tilt')
    
    @tilts.setter
    def tilts(self, value):
        self.set_points_attr('tilt', value)
        
    # ---------------------------------------------------------------------------
    # Radius
    
    @property
    def radius(self):
        return self.get_attr('radius')
    
    @radius.setter
    def radius(self, value):
        self.set_attr('radius', value)

    # ---------------------------------------------------------------------------
    # Weight soft body
    
    @property
    def weight_softbodies(self):
        return self.get_attr('weight_softbody')
    
    @weight_softbodies.setter
    def weight_softbodies(self, value):
        self.set_attr('weight_softbody', value)
        
    # ---------------------------------------------------------------------------
    # Weight
    
    @property
    def weights(self):
        return self.get_attr('weight')
    
    @weights.setter
    def weights(self, value):
        self.set_attr('weight', value)
        

    # ---------------------------------------------------------------------------
    # Spline parameters
    
    @property
    def splines_properties(self):
        return [self[i].spline_properties for i in range(len(self))]
    
    @splines_properties.setter
    def splines_properties(self, value):
        for i, props in enumerate(value):
            self[i].spline_properties = props
   
    
    