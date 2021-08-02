#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:12:33 2021

@author: alain
"""
import bpy
import numpy as np

from .core import animation
from .core.animation import Interval, Animator, Engine
from .blender.blender import get_control_object

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
    print("Registering Blender Wrap...")

    animation.register()

    bpy.utils.register_class(ClearParamsOperator)
    bpy.utils.register_class(SetupOperator)
    bpy.utils.register_class(ExecOperator)

    bpy.utils.register_class(WAMainPanel)
    bpy.utils.register_class(WAControlPanel)


def unregister():
    print("Unregistering Blender Wrap...")

    bpy.utils.unregister_class(WAControlPanel)
    bpy.utils.unregister_class(WAMainPanel)

    bpy.utils.unregister_class(ExecOperator)
    bpy.utils.unregister_class(SetupOperator)
    bpy.utils.unregister_class(ClearParamsOperator)

    animation.unregister()

if __name__ == "__main__":
    register()
