#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

bl_info = {
    "name":     "Blender wrap",
    "author":   "Alain Bernard",
    "version":  (1, 0),
    "blender":  (2, 80, 0),
    "location": "View3D > Sidebar > Wrap",
    "description": "Wrapanime commands and custom parameters",
    "warning":   "",
    "wiki_url":  "",
    "category":  "3D View"}

import numpy as np

import bpy

from .core.wrappers import wrap

from .core.utils import dicho
from .core.utils import npdicho

from .core.interpolation import Interpolation

from .core.blender import create_collection
from .core.blender import get_collection
from .core.blender import get_object_collections
from .core.blender import put_object_in_collection

from .core.blender import wrap_collection
from .core.blender import control_collection

from .core.blender import get_frame

from .core.blender import create_object
from .core.blender import get_object
from .core.blender import get_create_object
from .core.blender import get_control_object
from .core.blender import copy_modifiers
from .core.blender import delete_object
from .core.blender import smooth_object
from .core.blender import hide_object
from .core.blender import show_object
from .core.blender import set_material

from .core.duplicator import Duplicator

from .core.animation import register as animation_register
from .core.animation import unregister as animation_unregister
from .core.animation import Interval
from .core.animation import Engine

from .core.interpolation import Interpolation

from .core.bezier import PointsInterpolation

from .core.curspaces import CurvedSpace

from .core.meshes import arrow, curved_arrow

from .core.meshbuilder import MeshBuilder
from .core.markers import markers

from .core import d4

from .core.commons import base_error_title
error_title = base_error_title % "main.%s"


# ==========================================================================================
# UI

class ClearParamsOperator(bpy.types.Operator):
    """Delete the user parameters"""
    bl_idname = "wrap.clear_params"
    bl_label = "Clear parameters"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        del_params()
        return {'FINISHED'}


class SetupOperator(bpy.types.Operator):
    """Execute the initial set up functions"""
    bl_idname = "wrap.setup"
    bl_label = "Setup"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        Engine.setup()
        return {'FINISHED'}


class ExecOperator(bpy.types.Operator):
    """Execute the udate function"""
    bl_idname = "wrap.exec"
    bl_label = "Update"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        Engine.execute()
        return {'FINISHED'}

def ui_param_full_name(name, group):
    return f"{group}.{name}"

def ui_param_list(ctl):

    params = {}
    def add(group, name, key):
        ps = params.get(group)
        if ps is None:
            params[group] = [(name, key)]
        else:
            ps.append((name, key))

    for key in ctl.keys():
        if key[0] != "_":
            gn = key.split('.')
            if len(gn) == 1:
                add("_main", key, key)
            else:
                add("_main" if gn[0] == "" else gn[0], gn[1], key)

    return params

def scalar_param(name, default=0., min=0., max=1., group="", description="Wrapanime parameter"):

    ctl = get_control_object()
    rna = ctl['_RNA_UI']

    fname = ui_param_full_name(name, group)
    prm   = ctl.get(fname)
    if prm is None:
        ctl[fname] = default

    rna[fname] = {
        "description": description,
        "default":     default,
        "min":         min,
        "max":         max,
        "soft_min":    min,
        "soft_max":    max,
        }

def bool_param(name, default=True, group="", description="Wrapanime parameter"):

    ctl = get_control_object()
    rna = ctl['_RNA_UI']

    fname = ui_param_full_name(name, group)
    prm   = ctl.get(fname)
    if prm is None:
        ctl[fname] = default

    rna[fname] = {
        "description": description,
        "default":     default,
        }

def vector_param(name, default=(0., 0., 0.), group="", description="Wrapanime parameter"):

    ctl = get_control_object()
    rna = ctl['_RNA_UI']

    fname = ui_param_full_name(name, group)
    prm   = ctl.get(fname)
    if prm is None:
        ctl[fname] = default

    rna[fname] = {
        "description": description,
        "default":     default,
        }

def get_param(name, group=""):
    ctl = get_control_object()
    val = ctl.get(ui_param_full_name(name, group))
    if val is None:
        print(f"Wrapanime WARNING: param {name} doesn't exist.")
    return val

def del_params():
    ctl = get_control_object()
    keys = ctl.keys()
    for k in keys:
        del ctl[k]


class WAMainPanel(bpy.types.Panel):
    """Wrapanime commands"""
    bl_label        = "Commands"
    bl_category     = "Wrap"
    #bl_idname       = "SCENE_PT_layout"
    bl_space_type   = 'VIEW_3D'
    bl_region_type  = 'UI'
    #bl_context      = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.operator("wrap.setup", icon='FACE_MAPS')
        layout.operator("wrap.exec", icon='FILE_REFRESH')

        if context.screen.is_animation_playing:
            layout.operator("screen.animation_play", text="Pause", icon='PAUSE')
        else:
            layout.operator("screen.animation_play", text="Play", icon='PLAY')

        row = layout.row()
        row.prop(scene, "wa_frame_exec", text="Frame change")
        #row.label(text=f"Frame {scene.frame_current_final:6.1f}")
        row.prop(scene, "wa_hide_viewport", text="Hide in VP")

        layout.operator("wrap.clear_params", icon='CANCEL')



class WAControlPanel(bpy.types.Panel):

    """User parameters to control animation"""
    bl_label        = "User parameters"
    bl_category     = "Wrap"
    #bl_idname       = "SCENE_PT_layout"
    bl_space_type   = 'VIEW_3D'
    bl_region_type  = 'UI'
    #bl_context      = "scene"

    def draw(self, context):
        layout = self.layout

        ctl = get_control_object()
        params = ui_param_list(ctl)

        def draw_group(prms):
            for pf in prms:
                name  = pf[0]
                fname = pf[1]

                if np.size(ctl[fname]) > 1:
                    box = layout.box()
                    box.label(text=name)
                    col = box.column()
                    col.prop(ctl,f'["{fname}"]',text = '')
                else:
                    layout.prop(ctl,f'["{fname}"]', text=pf[0])

        prms = params.get("_main")
        if prms is not None:
            draw_group(prms)

        for key,prms in params.items():
            if key != "_main":
                row = layout.row()
                row.label(text=key)
                draw_group(prms)


def menu_func(self, context):
    #self.layout.operator(AddMoebius.bl_idname, icon='MESH_ICOSPHERE')
    pass


def register():

    animation_register()

    bpy.utils.register_class(ClearParamsOperator)
    bpy.utils.register_class(SetupOperator)
    bpy.utils.register_class(ExecOperator)

    bpy.utils.register_class(WAMainPanel)
    bpy.utils.register_class(WAControlPanel)


def unregister():
    bpy.utils.unregister_class(WAControlPanel)
    bpy.utils.unregister_class(WAMainPanel)

    bpy.utils.unregister_class(ExecOperator)
    bpy.utils.unregister_class(SetupOperator)
    bpy.utils.unregister_class(ClearParamsOperator)

    animation_register()

if __name__ == "__main__":
    register()
