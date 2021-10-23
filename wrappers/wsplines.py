#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 07:11:35 2021

@author: alain
"""

import numpy as np

import bpy

from .wstruct import WStruct
from .wbezierspline import WBezierSpline
from .wnurbsspline import WNurbsSpline

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

    def new(self, spline_type='BEZIER'):
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
        self.blender_object.data.update_tag()
        
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
    # Types
    
    @property
    def types(self):
        splines = self.wrapped
        return [spline.type for spline in splines]
        
    @types.setter
    def types(self, value):
        splines = self.wrapped
        
        if type(value) is str:
            values = [value for i in range(len(splines))]
        else:
            values = value
            
        if len(values) != len(splines):
            raise WError(f"WSplines error: the number of splines types {len(values)} doesn't match the number of splines {len(splines)}.",
                         Class="WSpline", Method="types", types=value)
            
        for spline, v in zip(splines, values):
            spline.type = v
            
        self.update_tag()
        
        
    # ===========================================================================
    # Splines can be of two types:
    # - Bezier     : type 3
    # - Non Bezier : type 1
    # 
    # The curve profile is  an array of triplets : (spline type, vertices count, spline type index)
    # Note that BEZIER splines are necessary (3, ?, 0)
    # 
    # This array can then be used to read the vertices
    # Note that:
    # np.min(profile[:, 0]) == 3          : True if only bezier splines
    # np.max(profile[:, 0]) == 3          : True if only non bezier splines
    # np.sum(profile[:, 0] * profile[:, 1]) : gives the total number of vertices
    
    @staticmethod
    def spline_type_index(type):
        if type == 'BEZIER':
            return 0
        elif type == 'POLY':
            return 1
        elif type == 'BSPLINE':
            return 2
        elif type == 'CARDINAL':
            return 3
        else:
            return 4
        
    @staticmethod
    def spline_type(index):
        return ['BEZIER', 'POLY', 'BSPLINE', 'CARDINAL', 'NURBS'][index]
    
    @property
    def profile(self):
        
        splines = self.wrapped
        
        profile = np.zeros((len(splines), 3), int)
        for i, spline in enumerate(splines):
            if spline.type == 'BEZIER':
                profile[i] = (3, len(spline.bezier_points), 0)
            else:
                profile[i] = (1, len(spline.points), WSplines.spline_type_index(spline.type))
                
        return profile
    
    # ===========================================================================
    # Change the profile
    #
    # As soon as the requested type of spline doesn't match the existing,
    # the spline and the following are deleted
    #
    # Nothing happens if the requested profile matches the existing one
    
    @profile.setter
    def profile(self, profile):
        
        splines = self.wrapped
        
        # ---------------------------------------------------------------------------
        # From bottom to top while we can change without deleting splines
        
        index = 0
        for i_spline in range(min(len(splines), len(profile))):
            
            spline = splines[i_spline]
            
            if spline.type != WSplines.spline_type(profile[i_spline, 2]):
                break
            
            if spline.type == 'BEZIER':
                points = spline.bezier_points
            else:
                points = spline.points
                
            n = len(points)
            if n < profile[i_spline, 1]:
                points.add(n - len(points))
                
            if len(points) != profile[i_spline, 1]:
                break
                
            index += 1
            
        # ---------------------------------------------------------------------------
        # index is the number of splines which are ok
        # splines after must be deleted
        
        for i in range(index, len(splines)):
            splines.remove(splines[len(splines)-1])
            
        # ---------------------------------------------------------------------------
        # We must then create the missing splines
        
        for i in range(index, len(profile)):
            
            spline = splines.new(WSplines.spline_type(profile[i, 2]))

            if profile[i, 0] == 3:
                points = spline.bezier_points
            else:
                points = spline.points
                
            n = profile[i, 1]
            if n > len(points):
                points.add(n - len(points))
                
        self.update_tag()

                
    # ===========================================================================
    # A user friendly version of the profile setting
    #
    # set_profile('BEZIER', 3, 10) : Create 10 BEZIER splines of 3 points
    
    def set_profile(self, types='BEZIER', lengths=2, count=None):
        
        if type(types) is str:
            atypes = [WSplines.spline_type_index(types)]
        else:
            atypes = [WSplines.spline_type_index(t) for t in types]
            
        acount = [3 if i == 0 else 1 for i in atypes]

            
        if hasattr(lengths, '__len__'):
            alengths = lengths
        else:
            alengths = [lengths]
            
        if count is None:
            count = max(len(atypes), len(alengths))
        
        profile = np.array((count, 3), int)
        profile[:, 0] = acount
        profile[:, 1] = alengths
        profile[:, 2] = atypes
        
        self.profile = profile
        
        self.update_tag()
        
        
    # ===========================================================================
    # Some helpers
    
    def only_bezier(self, profile=None):
        if profile is None:
            profile = self.profile
        return np.min(profile[:, 0]) == 3
    
    def only_nurbs(self, profile=None):
        if profile is None:
            profile = self.profile
        return np.max(profile[:, 0]) == 1
    
    def has_bezier(self, profile=None):
        if profile is None:
            profile = self.profile
        return np.max(profile[:, 0]) == 3
        
    def has_nurbs(self, profile=None):
        if profile is None:
            profile = self.profile
        return np.min(profile[:, 0]) == 1
    
    def verts_count(self, profile=None):
        if profile is None:
            profile = self.profile
        return np.sum(profile[:, 0] * profile[:, 1])
    
    @property
    def verts_dim(self):
        return 5 if self.has_nurbs else 3
    
    def set_types(self, type='BEZIER', lengths=2):
        pass
    
    # ===========================================================================
    # Read the splines vertices
    # 
    # The verts are organized in a single array which is meaning full only with a profile
    # - Bezier     : 3 arrays of 3-vectors
    # - Non Bezier : 1 array of 3-vectors 
    # 
    # The profile is passed as a parameter to avoid to read it if available by the caller
    #
    # Extended add additional dimensions to the vertices for non bezier
    # - Radius
    # - Tilt
    # - W
    #
    # The extended vertices for non bezier splines are:
    # - 0:3 = the vertices
    # - 3   = radius
    # - 4   = tilt
    # - 5   = W
    #
    # Hence, by using verts[..., :3], one can manipulate the vertices while keeping the extra info
    

    @staticmethod
    def verts_slice(verts):
        return verts[..., :3]
    
    @staticmethod
    def radius_slice(verts):
        return verts[..., 3]

    @staticmethod
    def tilt_slice(verts):
        return verts[..., 4]
    
    @staticmethod
    def w_slice(verts):
        return verts[..., 5]
    
    def get_vertices(self, profile=None, extended=True):
        
        splines = self.wrapped
        
        if profile is None:
            profile = self.profile
            
        nverts = np.sum(profile[:, 0] * profile[:, 1])
        only_bezier = np.min(profile[:, 0]) == 3
                
        if only_bezier or (not extended):
            verts = np.zeros((nverts, 3), np.float)
        else:
            verts = np.zeros((nverts, 6), np.float)
        
        index = 0
        for spline in splines:
            
            if spline.type == 'BEZIER':
            
                n = len(spline.bezier_points)
                
                a = np.zeros(n*3, np.float)
                for attr in ['co', 'handle_left', 'handle_right']:
                    spline.bezier_points.foreach_get(attr, a)
                    verts[index:index+n, :3] = a.reshape(n, 3)
                    index += n
                    
            else:
                
                n = len(spline.points)
                
                a = np.zeros(n*4, np.float)
                spline.points.foreach_get('co', a)
                a = a.reshape(n, 4)
                verts[index:index+n, :3] = a[:, :3]
                
                if extended:
                    
                    verts[index:index+n, 5]  = a[:, 3]
                    
                    a = np.zeros(n, np.float)
                    
                    spline.points.foreach_get('radius', a)
                    verts[index:index+n, 3] = a
                    spline.points.foreach_get('tilt', a)
                    verts[index:index+n, 4] = a
                
                index += n
                
        return verts
    
    # ===========================================================================
    # Default access to the vertices
    
    @property
    def ext_verts(self):
        return self.get_vertices(extended=True)

    @property
    def verts(self):
        return self.get_vertices(extended=False)

    @verts.setter
    def verts(self, verts):
        
        splines = self.wrapped
        
        extended = verts.shape[-1] >= 5
        
        index = 0
        for spline in splines:
            
            if spline.type == 'BEZIER':
                n = len(spline.bezier_points)
                for attr in ['co', 'handle_left', 'handle_right']:
                    a = verts[index:index+n, :3]
                    index += n
                    spline.bezier_points.foreach_set(attr, a.reshape(n*3))
                    
            else:
                n = len(spline.points)
                a = np.zeros((n, 4), np.float)
                
                # ----- Get or read the w component
                
                if verts.shape[-1] >= 6:
                    a[:, 3] = verts[index:index+n, 5]
                else:
                    spline.points.foreach_get('co', a.reshape(n*4))

                # ----- The 3-vertices
                    
                a[:, :3] = verts[index:index+n, :3]
                spline.points.foreach_set('co', a.reshape(n*4))

                # ----- Radius and tilt
                
                if extended:
                    rt = np.array(verts[index:index+n, 3])
                    spline.points.foreach_set('radius', rt)
                    
                    rt = np.array(verts[index:index+n, 4])
                    spline.points.foreach_set('tilt',   rt)
                    
                # -----  Next segment

                index += n
    
        self.update_tag()
        
    # ===========================================================================
    # Only 3D vertices
    
        
    
    