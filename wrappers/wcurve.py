#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:38:15 2021

@author: alain
"""

import bpy

from .wid import WID
from .wbezierspline import WBezierSpline
from .wnurbsspline import WNurbsSpline
from .wmaterials import WMaterials
        
# ---------------------------------------------------------------------------
# Curve wrapper
# wrapped : Curve

class WCurve(WID, WMaterials):
    """Curve data wrapper.
    
    In addition to wrap the Curve class, the wrapper also behaves as an array
    to give easy access to the splines.
    
    The items are wrapper of splines.
    """

    @property
    def wrapped(self):
        """The wrapped Blender instance.

        Returns
        -------
        Struct
            The wrapped object.
        """
        
        return self.blender_object.data
        
        if self.wrapped_ is None:
            return bpy.data.curves[self.name_]
        else:
            return self.wrapped_
            

    # ---------------------------------------------------------------------------
    # WCurve is a collection of splines

    def __len__(self):
        """Number of splines.

        Returns
        -------
        int
            Number fo splines.
        """
        
        return len(self.wrapped.splines)

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
        
        return WCurve.spline_wrapper(self.wrapped.splines[index])

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
        
        splines = self.wrapped.splines
        spline  = WCurve.spline_wrapper(splines.new(spline_type))
        self.wrapped.id_data.update_tag()
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
        
        splines = self.wrapped.splines
        if index <= len(splines)-1:
            splines.remove(splines[index])
        self.wrapped.id_data.update_tag()
        return
    
    # ---------------------------------------------------------------------------
    # Set the number of splines
    
    def set_splines_count(self, count, spline_type='BEZIER'):
        """Set the number of splines within the curve.
        
        The current number fo splines is either reduced or increased to 
        match the target count.

        Parameters
        ----------
        count : int
            Number of splines.
        spline_type : TYPE, optional
            DESCRIPTION. The default is 'BEZIER'.

        Returns
        -------
        None.
        """

        current = len(self)
        
        # Delete the splines which are too numerous
        for i in range(current - count):
            self.delete(len(self)-1)
            
        # Create new splines
        for i in range(current, count):
            self.new(spline_type)
            
        self.wrapped.id_data.update_tag()
        
    # ---------------------------------------------------------------------------
    # Set the number of vertices per spline
    
    def set_verts_count(self, count, index=None):
        """Set the number of vertices for one or all the splines.

        Parameters
        ----------
        count : int
            Number of vertices.
        index : int, optional
            Index of the spline to manage. All the splines if None. The default is None.

        Returns
        -------
        array of ints
            Indices of the modified splines.
        """
        
        # ----- All the splines or only one
        if index is None:
            splines = self
            created = [i for i in range(len(self))]
        else:
            splines = [self[index]]
            created = [index]
        
        # ----- Nothing to do
        ok = True
        for ws in splines:
            if len(ws) != count:
                ok = False
                break
            
        if ok: return created
        
        # ----- Save the existing splines
        saves = [spline.save() for spline in splines]

        # ----- Clear
        if index is None:
            self.wrapped.splines.clear()
        else:
            self.delete(index)

        # ----- Rebuild the splines
        created = []
        for save in saves:
            spline = self.new(save["type"])
            spline.restore(save, count)
            created.append(len(self)-1)

        # ---- OK
        self.wrapped.id_data.update_tag()
        
        return created
    
    # ===========================================================================
    # Properties and methods to expose to WMeshObject
    
    @classmethod
    def exposed_methods(cls):
        return ["copy_materials_from", "new", "delete", "set_splines_count", "set_verts_count"]

    @classmethod
    def exposed_properties(cls):
        return {"materials": 'RO'}
    
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

