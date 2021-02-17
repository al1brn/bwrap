#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:24:15 2021

@author: alain
"""

import bpy

from .wrappers import wrap
from .frames import get_frame

from .commons import base_error_title
error_title = base_error_title % "blender.%s"


# =============================================================================================================================
# Collections

# -----------------------------------------------------------------------------------------------------------------------------
# Get a collection
# Collection can be a name or the collection itself
# return None if it doesn't exist

def collection_by_name(collection):
    if type(collection) is str:
        return bpy.data.collections.get(collection)
    else:
        return collection

# -----------------------------------------------------------------------------------------------------------------------------
# Create a collection if it doesn't exist

def create_collection(name, parent=None):

    new_coll = bpy.data.collections.get(name)

    if new_coll is None:

        new_coll = bpy.data.collections.new(name)

        cparent = collection_by_name(parent)
        if cparent is None:
            cparent = bpy.context.scene.collection

        cparent.children.link(new_coll)

    return new_coll

# -----------------------------------------------------------------------------------------------------------------------------
# Get a collection
# Can create it if it doesn't exist

def get_collection(collection, create=True, parent=None):

    if type(collection) is str:
        coll = bpy.data.collections.get(collection)
        if (coll is None) and create:
            return create_collection(collection, parent)
        return coll
    else:
        return collection

# -----------------------------------------------------------------------------------------------------------------------------
# Get the collection of an object

def get_object_collections(obj):
    colls = []
    for coll in bpy.data.collections:
        if obj.name in coll.objects:
            colls.append(coll)

    return colls

# -----------------------------------------------------------------------------------------------------------------------------
# Link an object to a collection
# Unlink all the linked collections

def put_object_in_collection(obj, collection=None):

    colls = get_object_collections(obj)
    for coll in colls:
        coll.objects.unlink(obj)

    coll = get_collection(collection, create=False)

    if coll is None:
        bpy.context.collection.objects.link(obj)
    else:
        coll.objects.link(obj)

    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Get a collection used by wrapanime
# The top collection is WrapAnime
# All other collections are children of WrapAnime
# The name is prefixed by "W " to avoid names collisions

def wrap_collection(name=None):

    # Make sure the top collection exists
    top = get_collection("WrapAnime", create=True)

    if name is None:
        return top

    # Prefix with "W "

    cname = name if name[:2] == "W " else "W " + name

    return create_collection(cname, parent=top)

# -----------------------------------------------------------------------------------------------------------------------------
# Get the top collection used by the add-on

def control_collection():
    return wrap_collection("W Control")

# =============================================================================================================================
# Objects management utilities

# -----------------------------------------------------------------------------------------------------------------------------
# Create an object

def create_object(name, what='CUBE', collection=None, parent=None, **kwargs):

    generics = ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME', 'ARMATURE', 'LATTICE',
                'EMPTY', 'GPENCIL', 'CAMERA', 'LIGHT', 'SPEAKER', 'LIGHT_PROBE']
    typeds = ['CIRCLE', 'CONE', 'CUBE', 'GIZMO_CUBE', 'CYLINDER', 'GRID', 'ICOSPHERE', 'MONKEY', 'PLANE', 'TORUS', 'UVSPHERE',
              'BEZIERCIRCLE', 'BEZIERCURVE', 'NURBSCIRCLE', 'NURBSCURVE', 'NURBSPATH']


    if what in generics:

        bpy.ops.object.add(type=what, **kwargs)

    elif what == 'CIRCLE':
        bpy.ops.mesh.primitive_circle_add(**kwargs)
    elif what == 'CONE':
        bpy.ops.mesh.primitive_cone_add(**kwargs)
    elif what == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(**kwargs)
    elif what == 'GIZMO_CUBE':
        bpy.ops.mesh.primitive_cube_add_gizmo(**kwargs)
    elif what == 'CYLINDER':
        bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    elif what == 'GRID':
        bpy.ops.mesh.primitive_grid_add(**kwargs)
    elif what in ['ICOSPHERE', 'ICO_SPHERE']:
        bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
    elif what == 'MONKEY':
        bpy.ops.mesh.primitive_monkey_add(**kwargs)
    elif what == 'PLANE':
        bpy.ops.mesh.primitive_plane_add(**kwargs)
    elif what == 'TORUS':
        bpy.ops.mesh.primitive_torus_add(**kwargs)
    elif what in ['UVSPHERE', 'UV_SPHERE', 'SPHERE']:
        bpy.ops.mesh.primitive_uv_sphere_add(**kwargs)


    elif what in ['BEZIERCIRCLE', 'BEZIER_CIRCLE']:
        bpy.ops.curve.primitive_bezier_circle_add(**kwargs)
    elif what in ['BEZIERCURVE', 'BEZIER_CURVE', 'BEZIER']:
        bpy.ops.curve.primitive_bezier_curve_add(**kwargs)
    elif what in ['NURBSCIRCLE', 'NURBS_CIRCLE']:
        bpy.ops.curve.primitive_nurbs_circle_add(**kwargs)
    elif what in ['NURBSCURVE', 'NURBS_CURVE', 'NURBS']:
        bpy.ops.curve.primitive_nurbs_curve_add(**kwargs)
    elif what in ['NURBSPATH', 'NURBS_PATH']:
        bpy.ops.curve.primitive_nurbs_path_add(**kwargs)

    else:
        raise RuntimeError(
            error_title % "create_object" +
            f"Invalid object creation name: '{what}' is not valid.",
            f"Valid codes are {generics + typeds}")

    obj             = bpy.context.active_object
    obj.name        = name
    obj.parent      = parent
    obj.location    = bpy.context.scene.cursor.location

    # Links exclusively to the requested collection

    if collection is not None:
        bpy.ops.collection.objects_remove_all()
        get_collection(collection).objects.link(obj)

    return wrap(obj)

# -----------------------------------------------------------------------------------------------------------------------------
# Get an object by name or object itself
# The object can also be a WObject
# If otype is not None, the type of the object must the given value

def get_object(obj_or_name, mandatory=True, otype=None):

    if type(obj_or_name) is str:
        obj = bpy.data.objects.get(obj_or_name)

    elif hasattr(obj_or_name, 'name'):
        obj = bpy.data.objects.get(obj_or_name.name)

    else:
        obj = obj_or_name

    if (obj is None) and mandatory:
        raise RuntimeError(
            error_title % "get_object" +
            f"Object '{obj_or_name}' doesn't exist")

    if (obj is not None) and (otype is not None):
        if obj.type != otype:
            raise RuntimeError(
                error_title % "get_object" +
                    f"Object type error: '{otype}' is expected",
                    f"The type of the Blender object '{obj.name}' is '{obj.type}."
                    )

    return wrap(obj)

# -----------------------------------------------------------------------------------------------------------------------------
# Get an object and create it if it doesn't exist
# if create is None -> no creation
# For creation, the create argument must contain a valid object creation name

def get_create_object(obj_or_name, create=None, collection=None, **kwargs):

    obj = get_object(obj_or_name, mandatory = create is None)
    if obj is not None:
        return obj

    return wrap(create_object(obj_or_name, what=create, collection=collection, parent=None, **kwargs))


def get_control_object():

    wctl = get_create_object("W Control", create='EMPTY', collection=control_collection())
    ctl = wctl.wrapped

    # Ensure _RNA_UI prop exists
    rna = ctl.get('_RNA_UI')
    if rna is None:
        ctl['_RNA_UI'] = {}

    return ctl



# -----------------------------------------------------------------------------------------------------------------------------
# Copy modifiers

def copy_modifiers(source, target):

    for mSrc in source.modifiers:

        mDst = target.modifiers.get(mSrc.name, None)
        if not mDst:
            mDst = target.modifiers.new(mSrc.name, mSrc.type)

        # collect names of writable properties
        properties = [p.identifier for p in mSrc.bl_rna.properties if not p.is_readonly]

        # copy those properties
        for prop in properties:
            setattr(mDst, prop, getattr(mSrc, prop))


# -----------------------------------------------------------------------------------------------------------------------------
# Duplicate an object and its hierarchy

def duplicate_object(obj, collection=None, link=False, modifiers=False, children=False):

    # ----- Object copy
    dupl = obj.copy()

    # ----- Data copy
    if obj.data is not None:
        if not link:
            dupl.data = obj.data.copy()

    # ----- Modifiers
    if modifiers:
        copy_modifiers(obj, dupl)

    # ----- Collection to place the duplicate into
    if collection is None:
        colls = get_object_collections(obj)
        for coll in colls:
            coll.objects.link(dupl)
    else:
        collection.objects.link(dupl)

    # ----- Children copy
    if children:
        for child in obj.children:
            duplicate_object(child, collection=collection, link=link).parent = dupl

    # ----- Done !
    return dupl

# -----------------------------------------------------------------------------------------------------------------------------
# Delete an object and its children

def delete_object(obj_or_name, children=True):

    wobj = get_object(obj_or_name, mandatory=False)
    if wobj is None:
        return

    obj = wobj.wrapped

    def add_to_coll(o, coll):
        for child in o.children:
            add_to_coll(child, coll)
        coll.append(o)

    coll = []
    if children:
        add_to_coll(obj, coll)
    else:
        coll = [obj]

    for o in coll:
        bpy.data.objects.remove(o)


# -----------------------------------------------------------------------------------------------------------------------------
# Smooth

def smooth_object(obj):

    mesh = obj.data
    for f in mesh.bm.faces:
        f.smooth = True
    mesh.done()

    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Hide / Show

def hide_object(obj, value=True, frame=None, viewport=True):

    if viewport:
        obj.hide_viewport = value

    obj.hide_render = value

    if frame is not None:
        iframe = get_frame(frame)
        if viewport:
            obj.keyframe_insert(data_path="hide_viewport", frame=iframe)
        obj.keyframe_insert(data_path="hide_render", frame=iframe)

def show_object(obj, value=False, frame=None, viewport=True):
    hide_object(obj, not value, frame=frame, viewport=viewport)

# -----------------------------------------------------------------------------------------------------------------------------
# Assign a texture

def set_material(obj, material_name):

    # Get material
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        return
        # mat = bpy.data.materials.new(name="Material")

    # Assign it to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat


# =============================================================================================================================
# Registering the module

def register():
    # Ensure control object is created
    get_control_object()

def unregister():
    pass

if __name__ == "__main__":
    register()
