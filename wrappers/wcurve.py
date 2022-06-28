#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:38:15 2021

@author: alain
"""

import numpy as np

from .wid import WID

from .wbezierspline import WBezierSpline
from .wnurbsspline import WNurbsSpline

from ..core.profile import Profile
from ..core.maths.bezier import Beziers

from .wmaterials import WMaterials
from .wshapekeys import WShapeKeys

from ..core.commons import WError

        
# ---------------------------------------------------------------------------
# Curve wrapper
# wrapped : Curve

class WCurve(WID):
    """Curve data wrapper.
    
    In addition to wrap the Curve class, the wrapper also behaves as an array
    to give easy access to the splines.
    
    The items are wrapper of splines.
    """
    
    # ---------------------------------------------------------------------------
    # Initialization by name
    
    def __init__(self, wrapped, is_evaluated=None):
        super().__init__(wrapped, is_evaluated=is_evaluated)
        
        self.profile_ = None
    

    @property
    def wrapped(self):
        """The wrapped Blender instance.

        Returns
        -------
        Struct
            The wrapped object.
        """
        
        return self.blender_object.data
    
    # ===========================================================================
    # ===========================================================================
    # Splines wrapper

    # ---------------------------------------------------------------------------
    # WCurve is a collection of splines

    def __len__(self):
        return len(self.wrapped.splines)

    def __getitem__(self, index):
        return self.spline_wrapper(self.wrapped.splines[index])

    # ---------------------------------------------------------------------------
    # Utility function to wrap a BEZIER ot NURBS spline.
    
    @staticmethod
    def spline_wrapper(spline):
        return WBezierSpline(spline) if spline.type == 'BEZIER' else WNurbsSpline(spline)

    # ---------------------------------------------------------------------------
    # Add a spline

    def new(self, spline_type='BEZIER', verts_count=None):
        splines = self.splines
        spline  = self.spline_wrapper(splines.new(spline_type))
        if verts_count is not None:
            spline.verts_count = verts_count
        
        self.update_tag()
        
        self.uncache_profile()
        
        return spline

    # ---------------------------------------------------------------------------
    # Delete a spline

    def delete(self, index):
        splines = self.splines
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
        
        splines = self.splines
        
        # ---------------------------------------------------------------------------
        # From bottom to top while we can change without deleting splines
        
        index = 0
        for (ctype, length), spline in zip(profile, self):
            
            spline_type = Profile.ctype_code(spline.type)
            
            # ----- Types are different and one is Bezier
            # we must exit
            
            if spline_type != ctype:
                if spline_type == 0 or ctype == 0:
                    break
                spline.type = Profile.ctype_name(ctype)

            # ----- We can only add points, otherwise
            # we must exit
            
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

        for (ctype, offset, length, nverts), spline in zip(profile.verts_iter(), self.splines):

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

        for (ctype, offset, length, nverts), spline in zip(profile.verts_iter(), self.splines):

            if ctype == 0:
                a = np.reshape(verts[offset:offset+nverts, :3], (3, length*3))

                spline.bezier_points.foreach_set('co',           a[0])
                spline.bezier_points.foreach_set('handle_left',  a[1])
                spline.bezier_points.foreach_set('handle_right', a[2])

            else:
                a = np.ones((length, 4), float)
                a[:, :ndim] = verts[offset:offset+nverts]
                spline.points.foreach_set('co', np.reshape(a, (length*4)))

        self.update_tag()

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

        for (ctype, offset, length), spline in zip(profile.points_iter(), self.splines):

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

        for (ctype, offset, length), spline in zip(profile.points_iter(), self.splines):

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
    
    # ===========================================================================
    # ===========================================================================
    # Shape keys
    
    # ---------------------------------------------------------------------------
    # Shape keys wrapper
    
    @property
    def wshape_keys(self):
        return WShapeKeys(self.blender_object)
    
    # ---------------------------------------------------------------------------
    # Get the shape keys vertices
    
    def get_shape_keys_verts(self, name=None):
        
        # --------------------------------------------------
        # Get the blocks to read
        
        keys = self.wshape_keys.get_keys(name)
        if len(keys) == 0:
            return None
            
        blocks = self.wshape_keys[keys]
        count = len(blocks)
        if count == 0:
            return None
        
        # --------------------------------------------------
        # Let's read the blocks
        
        # --------------------------------------------------
        # Need to read the alternance bezier / not bezier
        
        profile     = self.profile
        
        only_bezier = profile.only_bezier
        only_nurbs  = profile.only_nurbs
        
        nverts      = profile.verts_count
        verts       = np.zeros((count, nverts, 3), float)
            
        # --------------------------------------------------
        # Loop on the shapes

        for i_sk, sk in enumerate(blocks):
            
            # --------------------------------------------------
            # Load bezier when only bezier curves
            
            if only_bezier:
                
                co = np.empty(nverts*3, float)
                le = np.empty(nverts*3, float)
                ri = np.empty(nverts*3, float)
                
                sk.data.foreach_get('co',           co)
                sk.data.foreach_get('handle_left',  le)
                sk.data.foreach_get('handle_right', ri)
                
                for _, index, n in profile.points_iter():
                    i3 = index*3
                    verts[i_sk, i3       :i3 + n  ] = co.reshape(nverts, 3)[index: index+n]
                    verts[i_sk, i3 + n   :i3 + 2*n] = le.reshape(nverts, 3)[index: index+n]
                    verts[i_sk, i3 + 2*n :i3 + 3*n] = ri.reshape(nverts, 3)[index: index+n]
                    
            # --------------------------------------------------
            # Load nurbs when no bezier curves
            
            elif only_nurbs:
                
                co = np.empty(nverts*3, float)
                
                sk.data.foreach_get('co', co)
                
                for _, index, n in profile.points_iter():
                    verts[i_sk, index :index + n] = co.reshape(nverts, 3)[index: index+n]
                    
            # --------------------------------------------------
            # We have to loop :-(
            
            else:
                index  = 0
                
                for ctype, offset, n in profile.points_iter():
                    
                    if ctype == 0:
                        for i in range(n):
        
                            usk = sk.data[offset + i]
        
                            verts[i_sk, index       + i] = usk.co
                            verts[i_sk, index +   n + i] = usk.handle_left
                            verts[i_sk, index + 2*n + i] = usk.handle_right
                            
                        index += 3*n
                        
                    else:
                        for i in range(n):
        
                            usk = sk.data[offset + i]
                            verts[i_sk, index + i] = usk.co
                            
                        index += n

        return verts

    # ===========================================================================
    # Get the curve vertices
    
    def set_shape_keys_verts(self, verts, name="Key"):
        
        # ----- Check the validity of the vertices shape
        
        shape = np.shape(verts)
        if len(shape) == 2:
            count = 1
        elif len(shape) == 3:
            count = shape[0]
        else:
            raise WError(f"Impossible to set curve vertices with shape {shape}. The vertices must be shaped either in two ot three dimensions.",
                        Class = "WCurve",
                        Method = "set_shape_keys_verts",
                        Object      = self.name,
                        verts_shape = np.shape(verts),
                        name        = name)
            
        # --------------------------------------------------
        # Let's write the blocks
        
        # --------------------------------------------------
        # Need to read the alternance bezier / not bezier
        
        profile     = self.profile
        only_bezier = profile.only_bezier
        only_nurbs  = profile.only_nurbs
        
        nverts      = profile.verts_count
        if nverts != shape[-2]:
            raise WError(f"Impossible to set curve vertices with shape {shape}. The number of vertices per shape must be {nverts}, not {shape[-2]}.",
                        Class = "WShapeKeys",
                        Method = "set_curve_vertices",
                        Object      = self.name,
                        verts_shape = np.shape(verts),
                        name        = name)
            
        # --------------------------------------------------
        # Loop on the shapes
        
        # ----- The list of keys
            
        keys = self.wshape_keys.series_key(name, range(count))
        
        for i_sk, key in enumerate(keys):
            
            sk = self.wshape_keys.shape_key(key, create=True)
            
            # --------------------------------------------------
            # Only bezier curves
            
            if only_bezier:
                
                co = np.empty((nverts, 3), float)
                le = np.empty((nverts, 3), float)
                ri = np.empty((nverts, 3), float)
                
                for _, index, n in profile.points_iter():
                    i3 = index*3
                    co[index:index + n] = verts[i_sk, i3       :i3 + n  ]
                    le[index:index + n] = verts[i_sk, i3 + n   :i3 + 2*n]
                    ri[index:index + n] = verts[i_sk, i3 + 2*n :i3 + 3*n]
                        
                sk.data.foreach_set('co',           co.reshape(nverts * 3))
                sk.data.foreach_set('handle_left',  le.reshape(nverts * 3))
                sk.data.foreach_set('handle_right', ri.reshape(nverts * 3))
                    
            # --------------------------------------------------
            # No bezier curve at all
            
            elif only_nurbs:
                
                co = np.empty((nverts, 3), float)
                
                for _, index, n in profile.points_iter():
                    co[index: index+n] = verts[i_sk, index :index + n]

                sk.data.foreach_set('co', co.reshape(nverts * 3))

            # --------------------------------------------------
            # We have to loop :-(        
            
            else:
                index = 0
                for ctype, offset, n in profile.points_iter():
                    if ctype == 3:
                        
                        for i in range(n):
        
                            usk = sk.data[offset + i]
        
                            usk.co           = verts[i_sk, index       + i]
                            usk.handle_left  = verts[i_sk, index +   n + i]
                            usk.handle_right = verts[i_sk, index + 2*n + i]
                            
                        index += 3*n
                        
                    else:
                    
                        for i in range(n):
        
                            usk = sk.data[offset + i]
        
                            usk.co = verts[i_sk, index + i]
                            
                        index += n
    

    # ===========================================================================
    # Materials
    
    @property
    def wmaterials(self):
        return WMaterials(data=self)
        
    @property
    def material_indices(self):
        """Material indices from the splines.
        """
        
        inds = np.zeros(len(self.wrapped.splines), int)
        self.wrapped.splines.foreach_get("material_index", inds)
        return inds
    
    @material_indices.setter
    def material_indices(self, value):
        inds = np.zeros(len(self.wrapped.splines), int)
        inds[:] = value
        self.wrapped.splines.foreach_set("material_index", inds)            

    # ===========================================================================
    # Copy main properties
    
    def copy_from(self, other):

        self.wrapped.bevel_depth            = other.bevel_depth 
        self.wrapped.bevel_factor_end       = other.bevel_factor_end 
        self.wrapped.bevel_factor_mapping_end = other.bevel_factor_mapping_end 
        self.wrapped.bevel_factor_mapping_start = other.bevel_factor_mapping_start 
        self.wrapped.bevel_factor_start     = other.bevel_factor_start 
        self.wrapped.bevel_mode             = other.bevel_mode 
        self.wrapped.bevel_object           = other.bevel_object 
        self.wrapped.bevel_resolution       = other.bevel_resolution 
        self.wrapped.dimensions             = other.dimensions 
        self.wrapped.extrude                = other.extrude 
        self.wrapped.fill_mode              = other.fill_mode 
        self.wrapped.offset                 = other.offset 
        self.wrapped.path_duration          = other.path_duration 
        self.wrapped.render_resolution_u    = other.render_resolution_u 
        self.wrapped.render_resolution_v    = other.render_resolution_v 
        self.wrapped.resolution_u           = other.resolution_u 
        self.wrapped.resolution_v           = other.resolution_v 
        self.wrapped.taper_object           = other.taper_object 
        self.wrapped.taper_radius_mode      = other.taper_radius_mode 
        self.wrapped.twist_mode             = other.twist_mode 
        self.wrapped.twist_smooth           = other.twist_smooth 
        self.wrapped.use_auto_texspace      = other.use_auto_texspace 
        self.wrapped.use_deform_bounds      = other.use_deform_bounds 
        self.wrapped.use_fill_caps          = other.use_fill_caps 
        self.wrapped.use_map_taper          = other.use_map_taper 
        self.wrapped.use_path               = other.use_path 
        self.wrapped.use_path_clamp         = other.use_path_clamp 
        self.wrapped.use_path_follow        = other.use_path_follow 
        self.wrapped.use_radius             = other.use_radius 
        self.wrapped.use_stretch            = other.use_stretch   
        
    def copy_to(self, other):

        other.bevel_depth            = self.wrapped.bevel_depth 
        other.bevel_factor_end       = self.wrapped.bevel_factor_end 
        other.bevel_factor_mapping_end = self.wrapped.bevel_factor_mapping_end 
        other.bevel_factor_mapping_start = self.wrapped.bevel_factor_mapping_start 
        other.bevel_factor_start     = self.wrapped.bevel_factor_start 
        other.bevel_mode             = self.wrapped.bevel_mode 
        other.bevel_object           = self.wrapped.bevel_object 
        other.bevel_resolution       = self.wrapped.bevel_resolution 
        other.dimensions             = self.wrapped.dimensions 
        other.extrude                = self.wrapped.extrude 
        other.fill_mode              = self.wrapped.fill_mode 
        other.offset                 = self.wrapped.offset 
        other.path_duration          = self.wrapped.path_duration 
        other.render_resolution_u    = self.wrapped.render_resolution_u 
        other.render_resolution_v    = self.wrapped.render_resolution_v 
        other.resolution_u           = self.wrapped.resolution_u 
        other.resolution_v           = self.wrapped.resolution_v 
        other.taper_object           = self.wrapped.taper_object 
        other.taper_radius_mode      = self.wrapped.taper_radius_mode 
        other.twist_mode             = self.wrapped.twist_mode 
        other.twist_smooth           = self.wrapped.twist_smooth 
        other.use_auto_texspace      = self.wrapped.use_auto_texspace 
        other.use_deform_bounds      = self.wrapped.use_deform_bounds 
        other.use_fill_caps          = self.wrapped.use_fill_caps 
        other.use_map_taper          = self.wrapped.use_map_taper 
        other.use_path               = self.wrapped.use_path 
        other.use_path_clamp         = self.wrapped.use_path_clamp 
        other.use_path_follow        = self.wrapped.use_path_follow 
        other.use_radius             = self.wrapped.use_radius 
        other.use_stretch            = self.wrapped.use_stretch   
        
        return other
    
    
    @property
    def curve_properties(self):
        class Props():
            pass
        
        return self.copy_to(Props())
        
    @curve_properties.setter
    def curve_properties(self, value):
        self.copy_from(value)
    
    
    # ===========================================================================
    # Properties and methods to expose to WMeshObject
    
    @classmethod
    def exposed_methods(cls):
        return ["new", "delete", "set_profile", "set_beziers", "set_function", "set_functions",
                "get_attrs", "set_attrs", "cache_profile", "uncache_profile",
                "get_shape_keys_verts", "set_shape_keys_verts"]

    @classmethod
    def exposed_properties(cls):
        return {"materials": 'RO', "profile": 'RW', "verts4": 'RO', "verts": 'RW',
                "only_bezier": 'RO', "only_nurbs": 'RO', "has_bezier": 'RO', "has_nurbs": 'RO', "is_mix": 'RO', 
                "material_indices": 'RW', "verts_count": 'RO', "points_count": 'RO', "wshape_keys": 'RO',
                "curve_properties": 'RW', "splines_properties": 'RW',
                "tilts": 'RW', "radius": 'RW', "weights": 'RO', "weight_softbodies": 'RW'}
    
    # ===========================================================================
    # Generated source code for WCurve class

    @property
    def rna_type(self):
        return self.wrapped.rna_type

    @property
    def name_full(self):
        return self.wrapped.name_full

    @property
    def original(self):
        return self.wrapped.original

    @property
    def users(self):
        return self.wrapped.users

    @property
    def use_fake_user(self):
        return self.wrapped.use_fake_user

    @use_fake_user.setter
    def use_fake_user(self, value):
        self.wrapped.use_fake_user = value

    @property
    def is_embedded_data(self):
        return self.wrapped.is_embedded_data

    @property
    def tag(self):
        return self.wrapped.tag

    @tag.setter
    def tag(self, value):
        self.wrapped.tag = value

    @property
    def is_library_indirect(self):
        return self.wrapped.is_library_indirect

    @property
    def library(self):
        return self.wrapped.library

    @property
    def asset_data(self):
        return self.wrapped.asset_data

    @property
    def override_library(self):
        return self.wrapped.override_library

    @property
    def preview(self):
        return self.wrapped.preview

    @property
    def shape_keys(self):
        return self.wrapped.shape_keys

    @property
    def splines(self):
        return self.wrapped.splines

    @property
    def path_duration(self):
        return self.wrapped.path_duration

    @path_duration.setter
    def path_duration(self, value):
        self.wrapped.path_duration = value

    @property
    def use_path(self):
        return self.wrapped.use_path

    @use_path.setter
    def use_path(self, value):
        self.wrapped.use_path = value

    @property
    def use_path_follow(self):
        return self.wrapped.use_path_follow

    @use_path_follow.setter
    def use_path_follow(self, value):
        self.wrapped.use_path_follow = value

    @property
    def use_path_clamp(self):
        return self.wrapped.use_path_clamp

    @use_path_clamp.setter
    def use_path_clamp(self, value):
        self.wrapped.use_path_clamp = value

    @property
    def use_stretch(self):
        return self.wrapped.use_stretch

    @use_stretch.setter
    def use_stretch(self, value):
        self.wrapped.use_stretch = value

    @property
    def use_deform_bounds(self):
        return self.wrapped.use_deform_bounds

    @use_deform_bounds.setter
    def use_deform_bounds(self, value):
        self.wrapped.use_deform_bounds = value

    @property
    def use_radius(self):
        return self.wrapped.use_radius

    @use_radius.setter
    def use_radius(self, value):
        self.wrapped.use_radius = value

    @property
    def bevel_mode(self):
        return self.wrapped.bevel_mode

    @bevel_mode.setter
    def bevel_mode(self, value):
        self.wrapped.bevel_mode = value

    @property
    def bevel_profile(self):
        return self.wrapped.bevel_profile

    @property
    def bevel_resolution(self):
        return self.wrapped.bevel_resolution

    @bevel_resolution.setter
    def bevel_resolution(self, value):
        self.wrapped.bevel_resolution = value

    @property
    def offset(self):
        return self.wrapped.offset

    @offset.setter
    def offset(self, value):
        self.wrapped.offset = value

    @property
    def extrude(self):
        return self.wrapped.extrude

    @extrude.setter
    def extrude(self, value):
        self.wrapped.extrude = value

    @property
    def bevel_depth(self):
        return self.wrapped.bevel_depth

    @bevel_depth.setter
    def bevel_depth(self, value):
        self.wrapped.bevel_depth = value

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
    def render_resolution_u(self):
        return self.wrapped.render_resolution_u

    @render_resolution_u.setter
    def render_resolution_u(self, value):
        self.wrapped.render_resolution_u = value

    @property
    def render_resolution_v(self):
        return self.wrapped.render_resolution_v

    @render_resolution_v.setter
    def render_resolution_v(self, value):
        self.wrapped.render_resolution_v = value

    @property
    def eval_time(self):
        return self.wrapped.eval_time

    @eval_time.setter
    def eval_time(self, value):
        self.wrapped.eval_time = value

    @property
    def bevel_object(self):
        return self.wrapped.bevel_object

    @bevel_object.setter
    def bevel_object(self, value):
        self.wrapped.bevel_object = value

    @property
    def taper_object(self):
        return self.wrapped.taper_object

    @taper_object.setter
    def taper_object(self, value):
        self.wrapped.taper_object = value

    @property
    def dimensions(self):
        return self.wrapped.dimensions

    @dimensions.setter
    def dimensions(self, value):
        self.wrapped.dimensions = value

    @property
    def fill_mode(self):
        return self.wrapped.fill_mode

    @fill_mode.setter
    def fill_mode(self, value):
        self.wrapped.fill_mode = value

    @property
    def twist_mode(self):
        return self.wrapped.twist_mode

    @twist_mode.setter
    def twist_mode(self, value):
        self.wrapped.twist_mode = value

    @property
    def taper_radius_mode(self):
        return self.wrapped.taper_radius_mode

    @taper_radius_mode.setter
    def taper_radius_mode(self, value):
        self.wrapped.taper_radius_mode = value

    @property
    def bevel_factor_mapping_start(self):
        return self.wrapped.bevel_factor_mapping_start

    @bevel_factor_mapping_start.setter
    def bevel_factor_mapping_start(self, value):
        self.wrapped.bevel_factor_mapping_start = value

    @property
    def bevel_factor_mapping_end(self):
        return self.wrapped.bevel_factor_mapping_end

    @bevel_factor_mapping_end.setter
    def bevel_factor_mapping_end(self, value):
        self.wrapped.bevel_factor_mapping_end = value

    @property
    def twist_smooth(self):
        return self.wrapped.twist_smooth

    @twist_smooth.setter
    def twist_smooth(self, value):
        self.wrapped.twist_smooth = value

    @property
    def use_fill_deform(self):
        return self.wrapped.use_fill_deform

    @use_fill_deform.setter
    def use_fill_deform(self, value):
        self.wrapped.use_fill_deform = value

    @property
    def use_fill_caps(self):
        return self.wrapped.use_fill_caps

    @use_fill_caps.setter
    def use_fill_caps(self, value):
        self.wrapped.use_fill_caps = value

    @property
    def use_map_taper(self):
        return self.wrapped.use_map_taper

    @use_map_taper.setter
    def use_map_taper(self, value):
        self.wrapped.use_map_taper = value

    @property
    def use_auto_texspace(self):
        return self.wrapped.use_auto_texspace

    @use_auto_texspace.setter
    def use_auto_texspace(self, value):
        self.wrapped.use_auto_texspace = value

    @property
    def texspace_location(self):
        return self.wrapped.texspace_location

    @texspace_location.setter
    def texspace_location(self, value):
        self.wrapped.texspace_location = value

    @property
    def texspace_size(self):
        return self.wrapped.texspace_size

    @texspace_size.setter
    def texspace_size(self, value):
        self.wrapped.texspace_size = value

    @property
    def materials(self):
        return self.wrapped.materials

    @property
    def bevel_factor_start(self):
        return self.wrapped.bevel_factor_start

    @bevel_factor_start.setter
    def bevel_factor_start(self, value):
        self.wrapped.bevel_factor_start = value

    @property
    def bevel_factor_end(self):
        return self.wrapped.bevel_factor_end

    @bevel_factor_end.setter
    def bevel_factor_end(self, value):
        self.wrapped.bevel_factor_end = value

    @property
    def is_editmode(self):
        return self.wrapped.is_editmode

    @property
    def cycles(self):
        return self.wrapped.cycles

    def animation_data_clear(self, *args, **kwargs):
        return self.wrapped.animation_data_clear(*args, **kwargs)

    def animation_data_create(self, *args, **kwargs):
        return self.wrapped.animation_data_create(*args, **kwargs)

    @property
    def bl_rna(self):
        return self.wrapped.bl_rna

    def copy(self, *args, **kwargs):
        return self.wrapped.copy(*args, **kwargs)

    def evaluated_get(self, *args, **kwargs):
        return self.wrapped.evaluated_get(*args, **kwargs)

    def make_local(self, *args, **kwargs):
        return self.wrapped.make_local(*args, **kwargs)

    def override_create(self, *args, **kwargs):
        return self.wrapped.override_create(*args, **kwargs)

    def override_template_create(self, *args, **kwargs):
        return self.wrapped.override_template_create(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.wrapped.transform(*args, **kwargs)

    def update_gpu_tag(self, *args, **kwargs):
        return self.wrapped.update_gpu_tag(*args, **kwargs)

    def update_tag(self, *args, **kwargs):
        return self.wrapped.update_tag(*args, **kwargs)

    def user_clear(self, *args, **kwargs):
        return self.wrapped.user_clear(*args, **kwargs)

    def user_of_id(self, *args, **kwargs):
        return self.wrapped.user_of_id(*args, **kwargs)

    def user_remap(self, *args, **kwargs):
        return self.wrapped.user_remap(*args, **kwargs)

    def validate_material_indices(self, *args, **kwargs):
        return self.wrapped.validate_material_indices(*args, **kwargs)

    # End of generation
    # ===========================================================================

