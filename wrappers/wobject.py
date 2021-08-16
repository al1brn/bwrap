#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:44:35 2021

@author: alain
"""

import numpy as np

from .wid import WID
from .wmesh import WMesh
from .wcurve import WCurve
from .wtext import WText

from ..maths.transformations import ObjectTransformations

from ..core.commons import WError


# ---------------------------------------------------------------------------
# Object wrapper
# wrapped: Object

class WObject(WID, ObjectTransformations):
    """Blender object wrapper.
    
    Provides the wdata attributes which is the proper wrapper of the data block.
    WObject captures attributes of wdata. The following expressions are equivalent:
        - wobject.wdata.verts
        - wobject.verts
        
    In particular, wrapper of a curve object implements array access to splines wrappers.
    """

    def __init__(self, wrapped, is_evaluated = None):
        
        
        # ----- WID initialization

        super().__init__(wrapped, is_evaluated)
            
        # ----- ObjectTransformations initialization
        
        ObjectTransformations.__init__(self, world=False)
        
        if self.wrapped.data is None:
            self.object_type = 'Empty'
        else:
            self.object_type = type(self.wrapped.data).__name__
        
        
    @property
    def wrapped(self):
        """The wrapped Blender instance.

        Returns
        -------
        Struct
            The wrapped object.
        """
        
        return self.blender_object
    
    def set_evaluated(self, value):
        self.is_evaluated = value
            
    # ---------------------------------------------------------------------------
    # Data
    
    def __repr__(self):
        return f"[Wrapper {self.__class__.__name__} of {self.object_type} object '{self.name}']"
            
    # ---------------------------------------------------------------------------
    # Data
    
    @property
    def supported(self):
        return self.object_type in ['Mesh', 'Curve', 'Text', 'Empty']

    @property
    def is_mesh(self):
        """Type is mesh.

        Returns
        -------
        bool
            True if object type is Mesh.
        """
        
        return self.object_type == 'Mesh'

    @property
    def wdata(self):
        """Returns the wrapper of the data block.
        
        The type of wrapper depends upon the object type.

        Raises
        ------
        RuntimeError
            If the type is not yet supported.

        Returns
        -------
        WID
            Data wrapper.
        """

        # Empty object -> return None
        wo = self.wrapped
        data = wo.data
        if data is None:
            return None
        
        # Supported types
        name = self.object_type
        if name == 'Mesh':
            return WMesh(self.name, self.is_evaluated)
        
        elif name in ['Curve', 'SurfaceCurve']:
            return WCurve(self.name, self.is_evaluated)
        
        elif name == 'TextCurve':
            return WText(self.name, self.is_evaluated)
        
        else:
            return None
        
            # Doesn't raise an error
        
            raise WError("Data class '{name}' not yet supported !",
                    Class_name = WObject,
                    Method = "wdata")

    # ---------------------------------------------------------------------------
    # Snapshot

    def snapshot(self, key="Wrap"):
        """Store the matrix basis in a key. 

        Parameters
        ----------
        key : str, optional
            Snapshot key. The default is "Wrap".

        Returns
        -------
        None.
        """
        
        m = np.array(self.wrapped.matrix_basis).reshape(16)
        self.wrapped[key] = m

    def restore_snapshot(self, key, mandatory=False):
        """Restore the snapshot.

        Parameters
        ----------
        key : str
            The sbapshot key.
        mandatory : bool, optional
            Raises an error if the key doesn't exist. The default is False.

        Raises
        ------
        RuntimeError
            If mandatory and the snapshot doesn't exist.

        Returns
        -------
        None.
        """
        
        m = self.wrapped.get(key)
        if m is None:
            if mandatory:
                raise WError(f"The snapshot key '{key}' doesn't exist for object '{self.name}'.",
                    Class = "WObject",
                    Method = "restore_snapshot",
                    key = key,
                    mandatory = mandatory)
                
            return

        m = np.reshape(m, (4, 4))
        self.wrapped.matrix_basis = np.transpose(m)

        self.mark_update()

    # -----------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # Shape keys management

    # -----------------------------------------------------------------------------------------------------------------------------
    # Has been the shape_keys structure created ?
    
    @property
    def keyshapable(self):
        return self.object_type in ['Mesh', 'Curve', 'SurfaceCurve']


    # ===========================================================================
    # Generated source code for WObject class

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
    def data(self):
        return self.wrapped.data

    @data.setter
    def data(self, value):
        self.wrapped.data = value

    @property
    def type(self):
        return self.wrapped.type

    @property
    def mode(self):
        return self.wrapped.mode

    @property
    def bound_box(self):
        return self.wrapped.bound_box

    @property
    def parent(self):
        return self.wrapped.parent

    @parent.setter
    def parent(self, value):
        self.wrapped.parent = value

    @property
    def parent_type(self):
        return self.wrapped.parent_type

    @parent_type.setter
    def parent_type(self, value):
        self.wrapped.parent_type = value

    @property
    def parent_vertices(self):
        return self.wrapped.parent_vertices

    @parent_vertices.setter
    def parent_vertices(self, value):
        self.wrapped.parent_vertices = value

    @property
    def parent_bone(self):
        return self.wrapped.parent_bone

    @parent_bone.setter
    def parent_bone(self, value):
        self.wrapped.parent_bone = value

    @property
    def use_camera_lock_parent(self):
        return self.wrapped.use_camera_lock_parent

    @use_camera_lock_parent.setter
    def use_camera_lock_parent(self, value):
        self.wrapped.use_camera_lock_parent = value

    @property
    def track_axis(self):
        return self.wrapped.track_axis

    @track_axis.setter
    def track_axis(self, value):
        self.wrapped.track_axis = value

    @property
    def up_axis(self):
        return self.wrapped.up_axis

    @up_axis.setter
    def up_axis(self, value):
        self.wrapped.up_axis = value

    @property
    def proxy(self):
        return self.wrapped.proxy

    @property
    def proxy_collection(self):
        return self.wrapped.proxy_collection

    @property
    def material_slots(self):
        return self.wrapped.material_slots

    @property
    def active_material(self):
        return self.wrapped.active_material

    @active_material.setter
    def active_material(self, value):
        self.wrapped.active_material = value

    @property
    def active_material_index(self):
        return self.wrapped.active_material_index

    @active_material_index.setter
    def active_material_index(self, value):
        self.wrapped.active_material_index = value

    @property
    def rotation_axis_angle(self):
        return self.wrapped.rotation_axis_angle

    @rotation_axis_angle.setter
    def rotation_axis_angle(self, value):
        self.wrapped.rotation_axis_angle = value

    @property
    def rotation_euler(self):
        return self.wrapped.rotation_euler

    @rotation_euler.setter
    def rotation_euler(self, value):
        self.wrapped.rotation_euler = value

    @property
    def rotation_mode(self):
        return self.wrapped.rotation_mode

    @rotation_mode.setter
    def rotation_mode(self, value):
        self.wrapped.rotation_mode = value

    @property
    def dimensions(self):
        return self.wrapped.dimensions

    @dimensions.setter
    def dimensions(self, value):
        self.wrapped.dimensions = value

    @property
    def delta_location(self):
        return self.wrapped.delta_location

    @delta_location.setter
    def delta_location(self, value):
        self.wrapped.delta_location = value

    @property
    def delta_rotation_euler(self):
        return self.wrapped.delta_rotation_euler

    @delta_rotation_euler.setter
    def delta_rotation_euler(self, value):
        self.wrapped.delta_rotation_euler = value

    @property
    def delta_rotation_quaternion(self):
        return self.wrapped.delta_rotation_quaternion

    @delta_rotation_quaternion.setter
    def delta_rotation_quaternion(self, value):
        self.wrapped.delta_rotation_quaternion = value

    @property
    def delta_scale(self):
        return self.wrapped.delta_scale

    @delta_scale.setter
    def delta_scale(self, value):
        self.wrapped.delta_scale = value

    @property
    def lock_location(self):
        return self.wrapped.lock_location

    @lock_location.setter
    def lock_location(self, value):
        self.wrapped.lock_location = value

    @property
    def lock_rotation(self):
        return self.wrapped.lock_rotation

    @lock_rotation.setter
    def lock_rotation(self, value):
        self.wrapped.lock_rotation = value

    @property
    def lock_rotation_w(self):
        return self.wrapped.lock_rotation_w

    @lock_rotation_w.setter
    def lock_rotation_w(self, value):
        self.wrapped.lock_rotation_w = value

    @property
    def lock_rotations_4d(self):
        return self.wrapped.lock_rotations_4d

    @lock_rotations_4d.setter
    def lock_rotations_4d(self, value):
        self.wrapped.lock_rotations_4d = value

    @property
    def lock_scale(self):
        return self.wrapped.lock_scale

    @lock_scale.setter
    def lock_scale(self, value):
        self.wrapped.lock_scale = value

    @property
    def matrix_world(self):
        return self.wrapped.matrix_world

    @matrix_world.setter
    def matrix_world(self, value):
        self.wrapped.matrix_world = value

    @property
    def matrix_local(self):
        return self.wrapped.matrix_local

    @matrix_local.setter
    def matrix_local(self, value):
        self.wrapped.matrix_local = value

    @property
    def matrix_basis(self):
        return self.wrapped.matrix_basis

    @matrix_basis.setter
    def matrix_basis(self, value):
        self.wrapped.matrix_basis = value

    @property
    def matrix_parent_inverse(self):
        return self.wrapped.matrix_parent_inverse

    @matrix_parent_inverse.setter
    def matrix_parent_inverse(self, value):
        self.wrapped.matrix_parent_inverse = value

    @property
    def modifiers(self):
        return self.wrapped.modifiers

    @property
    def grease_pencil_modifiers(self):
        return self.wrapped.grease_pencil_modifiers

    @property
    def shader_effects(self):
        return self.wrapped.shader_effects

    @property
    def constraints(self):
        return self.wrapped.constraints

    @property
    def vertex_groups(self):
        return self.wrapped.vertex_groups

    @property
    def face_maps(self):
        return self.wrapped.face_maps

    @property
    def empty_display_type(self):
        return self.wrapped.empty_display_type

    @empty_display_type.setter
    def empty_display_type(self, value):
        self.wrapped.empty_display_type = value

    @property
    def empty_display_size(self):
        return self.wrapped.empty_display_size

    @empty_display_size.setter
    def empty_display_size(self, value):
        self.wrapped.empty_display_size = value

    @property
    def empty_image_offset(self):
        return self.wrapped.empty_image_offset

    @empty_image_offset.setter
    def empty_image_offset(self, value):
        self.wrapped.empty_image_offset = value

    @property
    def image_user(self):
        return self.wrapped.image_user

    @property
    def empty_image_depth(self):
        return self.wrapped.empty_image_depth

    @empty_image_depth.setter
    def empty_image_depth(self, value):
        self.wrapped.empty_image_depth = value

    @property
    def show_empty_image_perspective(self):
        return self.wrapped.show_empty_image_perspective

    @show_empty_image_perspective.setter
    def show_empty_image_perspective(self, value):
        self.wrapped.show_empty_image_perspective = value

    @property
    def show_empty_image_orthographic(self):
        return self.wrapped.show_empty_image_orthographic

    @show_empty_image_orthographic.setter
    def show_empty_image_orthographic(self, value):
        self.wrapped.show_empty_image_orthographic = value

    @property
    def show_empty_image_only_axis_aligned(self):
        return self.wrapped.show_empty_image_only_axis_aligned

    @show_empty_image_only_axis_aligned.setter
    def show_empty_image_only_axis_aligned(self, value):
        self.wrapped.show_empty_image_only_axis_aligned = value

    @property
    def use_empty_image_alpha(self):
        return self.wrapped.use_empty_image_alpha

    @use_empty_image_alpha.setter
    def use_empty_image_alpha(self, value):
        self.wrapped.use_empty_image_alpha = value

    @property
    def empty_image_side(self):
        return self.wrapped.empty_image_side

    @empty_image_side.setter
    def empty_image_side(self, value):
        self.wrapped.empty_image_side = value

    @property
    def pass_index(self):
        return self.wrapped.pass_index

    @pass_index.setter
    def pass_index(self, value):
        self.wrapped.pass_index = value

    @property
    def color(self):
        return self.wrapped.color

    @color.setter
    def color(self, value):
        self.wrapped.color = value

    @property
    def field(self):
        return self.wrapped.field

    @property
    def collision(self):
        return self.wrapped.collision

    @property
    def soft_body(self):
        return self.wrapped.soft_body

    @property
    def particle_systems(self):
        return self.wrapped.particle_systems

    @property
    def rigid_body(self):
        return self.wrapped.rigid_body

    @property
    def rigid_body_constraint(self):
        return self.wrapped.rigid_body_constraint

    @property
    def hide_viewport(self):
        return self.wrapped.hide_viewport

    @hide_viewport.setter
    def hide_viewport(self, value):
        self.wrapped.hide_viewport = value

    @property
    def hide_select(self):
        return self.wrapped.hide_select

    @hide_select.setter
    def hide_select(self, value):
        self.wrapped.hide_select = value

    @property
    def hide_render(self):
        return self.wrapped.hide_render

    @hide_render.setter
    def hide_render(self, value):
        self.wrapped.hide_render = value

    @property
    def show_instancer_for_render(self):
        return self.wrapped.show_instancer_for_render

    @show_instancer_for_render.setter
    def show_instancer_for_render(self, value):
        self.wrapped.show_instancer_for_render = value

    @property
    def show_instancer_for_viewport(self):
        return self.wrapped.show_instancer_for_viewport

    @show_instancer_for_viewport.setter
    def show_instancer_for_viewport(self, value):
        self.wrapped.show_instancer_for_viewport = value

    @property
    def instance_type(self):
        return self.wrapped.instance_type

    @instance_type.setter
    def instance_type(self, value):
        self.wrapped.instance_type = value

    @property
    def use_instance_vertices_rotation(self):
        return self.wrapped.use_instance_vertices_rotation

    @use_instance_vertices_rotation.setter
    def use_instance_vertices_rotation(self, value):
        self.wrapped.use_instance_vertices_rotation = value

    @property
    def use_instance_faces_scale(self):
        return self.wrapped.use_instance_faces_scale

    @use_instance_faces_scale.setter
    def use_instance_faces_scale(self, value):
        self.wrapped.use_instance_faces_scale = value

    @property
    def instance_faces_scale(self):
        return self.wrapped.instance_faces_scale

    @instance_faces_scale.setter
    def instance_faces_scale(self, value):
        self.wrapped.instance_faces_scale = value

    @property
    def instance_collection(self):
        return self.wrapped.instance_collection

    @instance_collection.setter
    def instance_collection(self, value):
        self.wrapped.instance_collection = value

    @property
    def is_instancer(self):
        return self.wrapped.is_instancer

    @property
    def display_type(self):
        return self.wrapped.display_type

    @display_type.setter
    def display_type(self, value):
        self.wrapped.display_type = value

    @property
    def show_bounds(self):
        return self.wrapped.show_bounds

    @show_bounds.setter
    def show_bounds(self, value):
        self.wrapped.show_bounds = value

    @property
    def display_bounds_type(self):
        return self.wrapped.display_bounds_type

    @display_bounds_type.setter
    def display_bounds_type(self, value):
        self.wrapped.display_bounds_type = value

    @property
    def show_name(self):
        return self.wrapped.show_name

    @show_name.setter
    def show_name(self, value):
        self.wrapped.show_name = value

    @property
    def show_axis(self):
        return self.wrapped.show_axis

    @show_axis.setter
    def show_axis(self, value):
        self.wrapped.show_axis = value

    @property
    def show_texture_space(self):
        return self.wrapped.show_texture_space

    @show_texture_space.setter
    def show_texture_space(self, value):
        self.wrapped.show_texture_space = value

    @property
    def show_wire(self):
        return self.wrapped.show_wire

    @show_wire.setter
    def show_wire(self, value):
        self.wrapped.show_wire = value

    @property
    def show_all_edges(self):
        return self.wrapped.show_all_edges

    @show_all_edges.setter
    def show_all_edges(self, value):
        self.wrapped.show_all_edges = value

    @property
    def use_grease_pencil_lights(self):
        return self.wrapped.use_grease_pencil_lights

    @use_grease_pencil_lights.setter
    def use_grease_pencil_lights(self, value):
        self.wrapped.use_grease_pencil_lights = value

    @property
    def show_transparent(self):
        return self.wrapped.show_transparent

    @show_transparent.setter
    def show_transparent(self, value):
        self.wrapped.show_transparent = value

    @property
    def show_in_front(self):
        return self.wrapped.show_in_front

    @show_in_front.setter
    def show_in_front(self, value):
        self.wrapped.show_in_front = value

    @property
    def pose_library(self):
        return self.wrapped.pose_library

    @pose_library.setter
    def pose_library(self, value):
        self.wrapped.pose_library = value

    @property
    def pose(self):
        return self.wrapped.pose

    @property
    def show_only_shape_key(self):
        return self.wrapped.show_only_shape_key

    @show_only_shape_key.setter
    def show_only_shape_key(self, value):
        self.wrapped.show_only_shape_key = value

    @property
    def use_shape_key_edit_mode(self):
        return self.wrapped.use_shape_key_edit_mode

    @use_shape_key_edit_mode.setter
    def use_shape_key_edit_mode(self, value):
        self.wrapped.use_shape_key_edit_mode = value

    @property
    def active_shape_key(self):
        return self.wrapped.active_shape_key

    @property
    def active_shape_key_index(self):
        return self.wrapped.active_shape_key_index

    @active_shape_key_index.setter
    def active_shape_key_index(self, value):
        self.wrapped.active_shape_key_index = value

    @property
    def use_dynamic_topology_sculpting(self):
        return self.wrapped.use_dynamic_topology_sculpting

    @property
    def is_from_instancer(self):
        return self.wrapped.is_from_instancer

    @property
    def is_from_set(self):
        return self.wrapped.is_from_set

    @property
    def display(self):
        return self.wrapped.display

    @property
    def lineart(self):
        return self.wrapped.lineart

    @property
    def use_mesh_mirror_x(self):
        return self.wrapped.use_mesh_mirror_x

    @use_mesh_mirror_x.setter
    def use_mesh_mirror_x(self, value):
        self.wrapped.use_mesh_mirror_x = value

    @property
    def use_mesh_mirror_y(self):
        return self.wrapped.use_mesh_mirror_y

    @use_mesh_mirror_y.setter
    def use_mesh_mirror_y(self, value):
        self.wrapped.use_mesh_mirror_y = value

    @property
    def use_mesh_mirror_z(self):
        return self.wrapped.use_mesh_mirror_z

    @use_mesh_mirror_z.setter
    def use_mesh_mirror_z(self, value):
        self.wrapped.use_mesh_mirror_z = value

    @property
    def animation_visualization(self):
        return self.wrapped.animation_visualization

    @property
    def motion_path(self):
        return self.wrapped.motion_path

    @property
    def cycles_visibility(self):
        return self.wrapped.cycles_visibility

    @property
    def cycles(self):
        return self.wrapped.cycles

    @property
    def ant_landscape(self):
        return self.wrapped.ant_landscape

    def animation_data_clear(self, *args, **kwargs):
        return self.wrapped.animation_data_clear(*args, **kwargs)

    def animation_data_create(self, *args, **kwargs):
        return self.wrapped.animation_data_create(*args, **kwargs)

    @property
    def bl_rna(self):
        return self.wrapped.bl_rna

    def cache_release(self, *args, **kwargs):
        return self.wrapped.cache_release(*args, **kwargs)

    def calc_matrix_camera(self, *args, **kwargs):
        return self.wrapped.calc_matrix_camera(*args, **kwargs)

    def camera_fit_coords(self, *args, **kwargs):
        return self.wrapped.camera_fit_coords(*args, **kwargs)

    @property
    def children(self):
        return self.wrapped.children

    def closest_point_on_mesh(self, *args, **kwargs):
        return self.wrapped.closest_point_on_mesh(*args, **kwargs)

    def convert_space(self, *args, **kwargs):
        return self.wrapped.convert_space(*args, **kwargs)

    def copy(self, *args, **kwargs):
        return self.wrapped.copy(*args, **kwargs)

    def evaluated_get(self, *args, **kwargs):
        return self.wrapped.evaluated_get(*args, **kwargs)

    def find_armature(self, *args, **kwargs):
        return self.wrapped.find_armature(*args, **kwargs)

    def generate_gpencil_strokes(self, *args, **kwargs):
        return self.wrapped.generate_gpencil_strokes(*args, **kwargs)

    def hide_get(self, *args, **kwargs):
        return self.wrapped.hide_get(*args, **kwargs)

    def hide_set(self, *args, **kwargs):
        return self.wrapped.hide_set(*args, **kwargs)

    def holdout_get(self, *args, **kwargs):
        return self.wrapped.holdout_get(*args, **kwargs)

    def indirect_only_get(self, *args, **kwargs):
        return self.wrapped.indirect_only_get(*args, **kwargs)

    def is_deform_modified(self, *args, **kwargs):
        return self.wrapped.is_deform_modified(*args, **kwargs)

    def is_modified(self, *args, **kwargs):
        return self.wrapped.is_modified(*args, **kwargs)

    def local_view_get(self, *args, **kwargs):
        return self.wrapped.local_view_get(*args, **kwargs)

    def local_view_set(self, *args, **kwargs):
        return self.wrapped.local_view_set(*args, **kwargs)

    def make_local(self, *args, **kwargs):
        return self.wrapped.make_local(*args, **kwargs)

    def override_create(self, *args, **kwargs):
        return self.wrapped.override_create(*args, **kwargs)

    def override_template_create(self, *args, **kwargs):
        return self.wrapped.override_template_create(*args, **kwargs)

    def ray_cast(self, *args, **kwargs):
        return self.wrapped.ray_cast(*args, **kwargs)

    def select_get(self, *args, **kwargs):
        return self.wrapped.select_get(*args, **kwargs)

    def select_set(self, *args, **kwargs):
        return self.wrapped.select_set(*args, **kwargs)

    def shape_key_add(self, *args, **kwargs):
        return self.wrapped.shape_key_add(*args, **kwargs)

    def shape_key_clear(self, *args, **kwargs):
        return self.wrapped.shape_key_clear(*args, **kwargs)

    def shape_key_remove(self, *args, **kwargs):
        return self.wrapped.shape_key_remove(*args, **kwargs)

    def to_curve(self, *args, **kwargs):
        return self.wrapped.to_curve(*args, **kwargs)

    def to_curve_clear(self, *args, **kwargs):
        return self.wrapped.to_curve_clear(*args, **kwargs)

    def to_mesh(self, *args, **kwargs):
        return self.wrapped.to_mesh(*args, **kwargs)

    def to_mesh_clear(self, *args, **kwargs):
        return self.wrapped.to_mesh_clear(*args, **kwargs)

    def update_from_editmode(self, *args, **kwargs):
        return self.wrapped.update_from_editmode(*args, **kwargs)

    def update_tag(self, *args, **kwargs):
        return self.wrapped.update_tag(*args, **kwargs)

    def user_clear(self, *args, **kwargs):
        return self.wrapped.user_clear(*args, **kwargs)

    def user_of_id(self, *args, **kwargs):
        return self.wrapped.user_of_id(*args, **kwargs)

    def user_remap(self, *args, **kwargs):
        return self.wrapped.user_remap(*args, **kwargs)

    @property
    def users_collection(self):
        return self.wrapped.users_collection

    @property
    def users_scene(self):
        return self.wrapped.users_scene

    def visible_get(self, *args, **kwargs):
        return self.wrapped.visible_get(*args, **kwargs)

    def visible_in_viewport_get(self, *args, **kwargs):
        return self.wrapped.visible_in_viewport_get(*args, **kwargs)

    # End of generation
    # ===========================================================================
    
    