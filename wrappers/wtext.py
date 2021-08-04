#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:42:08 2021

@author: alain
"""

import bpy

from .wid import WID

# ---------------------------------------------------------------------------
# Text wrapper
# wrapped : TextCurve

class WText(WID):
    """TextCurve wrapper.
    
    Simple wrapper limited to provide the text attribute.
    Other attributes come from the Blender TextCurve class.
    """

    def __init__DEPR(self, wrapped, evaluated=False):
        if evaluated:
            super().__init__(wrapped, name=wrapped.name)
        else:
            super().__init__(name=wrapped.name, coll=bpy.data.curves)
            
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
            

    @property
    def text(self):
        """The text displayed by the object.

        Returns
        -------
        str
            Text.
        """
        
        return self.wrapped.body

    @text.setter
    def text(self, value):
        self.wrapped.body = value
        
    # ===========================================================================
    # Properties and methods to expose to WMeshObject
    
    @classmethod
    def exposed_methods(cls):
        return []

    @classmethod
    def exposed_properties(cls):
        return {"text": 'RW'}
        
        
    # ===========================================================================
    # Generated source code for WText class

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

    @property
    def align_x(self):
        return self.wrapped.align_x

    @align_x.setter
    def align_x(self, value):
        self.wrapped.align_x = value

    @property
    def align_y(self):
        return self.wrapped.align_y

    @align_y.setter
    def align_y(self, value):
        self.wrapped.align_y = value

    @property
    def overflow(self):
        return self.wrapped.overflow

    @overflow.setter
    def overflow(self, value):
        self.wrapped.overflow = value

    @property
    def size(self):
        return self.wrapped.size

    @size.setter
    def size(self, value):
        self.wrapped.size = value

    @property
    def small_caps_scale(self):
        return self.wrapped.small_caps_scale

    @small_caps_scale.setter
    def small_caps_scale(self, value):
        self.wrapped.small_caps_scale = value

    @property
    def space_line(self):
        return self.wrapped.space_line

    @space_line.setter
    def space_line(self, value):
        self.wrapped.space_line = value

    @property
    def space_word(self):
        return self.wrapped.space_word

    @space_word.setter
    def space_word(self, value):
        self.wrapped.space_word = value

    @property
    def space_character(self):
        return self.wrapped.space_character

    @space_character.setter
    def space_character(self, value):
        self.wrapped.space_character = value

    @property
    def shear(self):
        return self.wrapped.shear

    @shear.setter
    def shear(self, value):
        self.wrapped.shear = value

    @property
    def offset_x(self):
        return self.wrapped.offset_x

    @offset_x.setter
    def offset_x(self, value):
        self.wrapped.offset_x = value

    @property
    def offset_y(self):
        return self.wrapped.offset_y

    @offset_y.setter
    def offset_y(self, value):
        self.wrapped.offset_y = value

    @property
    def underline_position(self):
        return self.wrapped.underline_position

    @underline_position.setter
    def underline_position(self, value):
        self.wrapped.underline_position = value

    @property
    def underline_height(self):
        return self.wrapped.underline_height

    @underline_height.setter
    def underline_height(self, value):
        self.wrapped.underline_height = value

    @property
    def text_boxes(self):
        return self.wrapped.text_boxes

    @property
    def active_textbox(self):
        return self.wrapped.active_textbox

    @active_textbox.setter
    def active_textbox(self, value):
        self.wrapped.active_textbox = value

    @property
    def family(self):
        return self.wrapped.family

    @family.setter
    def family(self, value):
        self.wrapped.family = value

    @property
    def body(self):
        return self.wrapped.body

    @body.setter
    def body(self, value):
        self.wrapped.body = value

    @property
    def body_format(self):
        return self.wrapped.body_format

    @property
    def follow_curve(self):
        return self.wrapped.follow_curve

    @follow_curve.setter
    def follow_curve(self, value):
        self.wrapped.follow_curve = value

    @property
    def font(self):
        return self.wrapped.font

    @font.setter
    def font(self, value):
        self.wrapped.font = value

    @property
    def font_bold(self):
        return self.wrapped.font_bold

    @font_bold.setter
    def font_bold(self, value):
        self.wrapped.font_bold = value

    @property
    def font_italic(self):
        return self.wrapped.font_italic

    @font_italic.setter
    def font_italic(self, value):
        self.wrapped.font_italic = value

    @property
    def font_bold_italic(self):
        return self.wrapped.font_bold_italic

    @font_bold_italic.setter
    def font_bold_italic(self, value):
        self.wrapped.font_bold_italic = value

    @property
    def edit_format(self):
        return self.wrapped.edit_format

    @property
    def use_fast_edit(self):
        return self.wrapped.use_fast_edit

    @use_fast_edit.setter
    def use_fast_edit(self, value):
        self.wrapped.use_fast_edit = value

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
