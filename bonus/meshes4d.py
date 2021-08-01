"""
4D management
hypermeshes generation

Version 1.0
Date: May 31st 2020
Author: Alain Bernard
"""

import bpy
import bmesh
import math
import mathutils

# ----------------------------------------------------------------------------------------------------
# Slicer

class Slicer():

    def __init__(self, slice='NONE', **kwargs):

        # Make a new BMesh
        self.bm = bmesh.new()

        # Add the 4D layers
        floats = self.bm.verts.layers.float

        floats.new("4D X")
        floats.new("4D Y")
        floats.new("4D Z")
        self.layer_w = floats.new("4D W")

        # Add the init layer
        self.layer_init = self.bm.verts.layers.int.new("init")

        # Standard slice type

        def arg(name, default):
            if name in kwargs:
                return kwargs[name]
            else:
                return default

        self.slice = slice

        if slice == 'UVSPHERE':
            self.segments = max(3, min(512, arg('segments', 32)))
            self.rings = max(3, min(512, arg('rings', 16)))

        elif slice == 'ICOSPHERE':
            self.subdivisions = max(1, min(9, arg('subdivisions', 2)))

    # ---------------------------------------------------------------------------
    # After adding vertices, set the w coordinates
    # to the uninitialized vertices

    def set_w(self, w):

        # fw = bm.verts.layers.float["4D W"]
        # ok = bm.verts.layers.int["init"]

        self.bm.verts.ensure_lookup_table()
        for i in range(len(self.bm.verts)):
            if self.bm.verts[i][self.layer_init] == 0:
                self.bm.verts[i][self.layer_w] = w
                self.bm.verts[i][self.layer_init] = 1

    # ---------------------------------------------------------------------------
    # New slice

    def new_slice(self, size, w):

        # Null size slice
        if size < 0.01:
            bmesh.ops.create_vert(self.bm, co=(0., 0., 0.))
            self.set_w(w)
            return

        # ----- UVSPHERE
        if self.slice == 'UVSPHERE':
            bmesh.ops.create_uvsphere(
                self.bm,
                u_segments=self.segments,
                v_segments=self.rings,
                diameter=size,
                matrix=mathutils.Matrix.Identity(4),
                calc_uvs=True)


        # ----- ICOSPHERE
        elif self.slice == 'ICOSPHERE':
            bmesh.ops.create_icosphere(
                self.bm,
                subdivisions=self.subdivisions,
                diameter=size,
                matrix=mathutils.Matrix.Identity(4),
                calc_uvs=True)

        # ----- CUBE
        elif self.slice == 'CUBE':
            bmesh.ops.create_cube(
                self.bm,
                size=size,
                matrix=mathutils.Matrix.Identity(4),
                calc_uvs=True)

        else:
            raise RuntimeError(f"4D Slicer: unknown slice type: '{self.slice}'")

        # The w coordinate
        self.set_w(w)

    # ---------------------------------------------------------------------------
    # Generates slices from a profile
    # The profile returns a size from a w coordinate

    def profile(self, f, slices=5, w0=0., w1=1.):

        slices = max(1, min(21, slices))
        amp = w1 - w0
        dw = 0 if slices == 1 else amp / (slices - 1)

        # Loop on the slices
        for i in range(slices):
            w = w0 + i*dw
            self.new_slice(f(w), w)

    # ---------------------------------------------------------------------------
    # Create the mesh object
    # NOTE: after calling this method, the bm attribute is not valid anymore

    def create_object(self, name="Object"):

        # Delete the init layer
        self.bm.verts.layers.int.remove(self.bm.verts.layers.int["init"])

        # Write the bmesh into a new mesh
        me = bpy.data.meshes.new("Mesh")
        self.bm.to_mesh(me)
        self.bm.free()
        self.bm = None

        # Add the mesh to the scene
        obj = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(obj)

        # Active as a 4D mesh
        obj.d4.is4D = True
        obj.d4.object_type = 'SURFACE'
        obj.d4.mesh_to_layers()

        # Update in the context
        bpy.context.scene.d4.update_objects([obj])

        # Select and make active
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        return obj


# ----------------------------------------------------------------------------------------------------
# Hyper uv sphere

def hypersphere(radius=1., slices=5, slice='UVSPHERE', segments=32, rings=16, subdivisions=2):

    def f(w):
        r2 = radius * radius
        w2 = min(r2, w * w)
        return math.sqrt(radius * radius - w * w)

    slicer = Slicer(slice=slice, segments=segments, rings=rings, subdivisions=subdivisions)
    slicer.profile(f, slices=slices, w0=-radius, w1=radius)

    return slicer.create_object("Hypersphere")


# ----------------------------------------------------------------------------------------------------
# Hypercone

def hypercone(radius=1., height=1., slices=5, slice='UVSPHERE', segments=32, rings=16, subdivisions=2):

    def f(w):
        return radius / height * (height - w)

    slicer = Slicer(slice=slice, segments=segments, rings=rings, subdivisions=subdivisions)
    slicer.profile(f, slices=slices, w0=0., w1=height)

    return slicer.create_object("Hypercone")

# ----------------------------------------------------------------------------------------------------
# Hypercylinder

def hypercylinder(radius=1., height=2., slices=5, slice='UVSPHERE', segments=32, rings=16, subdivisions=2):
    def f(w):
        return radius

    slicer = Slicer(slice=slice, segments=segments, rings=rings, subdivisions=subdivisions)
    slicer.profile(f, slices=slices, w0=-height / 2, w1=height / 2)
    # Create the object

    return slicer.create_object("Hypercylinder")

# ----------------------------------------------------------------------------------------------------
# Hypercube (an hypercylinder of 2 slices :-)

def hypercube(size=2.):
    obj = hypercylinder(radius=size, height=size, slices=2, slice='CUBE')
    obj.name = "Hypercube"
    return obj

# ----------------------------------------------------------------------------------------------------
# Hyper hyperbola

def hyper_hyperbola(radius=1., height=2., slices=5, slice='UVSPHERE', segments=32, rings=16, subdivisions=2):
    def f(w):
        return math.sqrt(radius*radius + w*w)

    slicer = Slicer(slice=slice, segments=segments, rings=rings, subdivisions=subdivisions)
    slicer.profile(f, slices=slices, w0=-height / 2, w1=height / 2)
    # Create the object

    return slicer.create_object("Hyper hyperbola")













