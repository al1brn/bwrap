"""
4D management

Version 1.0
Date: May 31st 2020
Author: Alain Bernard
"""

import numpy as np

import bpy
import bmesh
from bpy.props import BoolProperty, FloatProperty, EnumProperty, FloatVectorProperty, StringProperty, IntProperty, \
    PointerProperty
from mathutils import Vector, Matrix, Euler
import collections
import math
from math import cos, sin, asin, atan2, radians, degrees, pi, sqrt

import gpu
from gpu_extras.batch import batch_for_shader

ZERO = 0.00001
ei4 = np.array((1., 0., 0., 0.))
ej4 = np.array((0., 1., 0., 0.))
ek4 = np.array((0., 0., 1., 0.))
el4 = np.array((0., 0., 0., 1.))

# ======================================================================================================================================================
# 4D Geometry
#
# - cross4D         : Cross product of two 4D vectors
# - theta_phi       : Compute theta and phi angles of a vector (in degrees)
# - matrix_4D_uvw   : Compute the projection matrix with 3 angles u, v, w
# - matrix_4D_axis  : Compute the projection matrix along an axis
# - matrix_4D_axis3 : Compute a projection matrix along a 3D axis and an angle


# --------------------------------------------------------------------------------------------------
# Normalize a 4D vector

def normalize(v):
    vs = np.array(v)
    if len(vs.shape) == 1:
        return vs / np.linalg.norm(vs)
    else:
        return vs / np.linalg.norm(vs, axis=len(vs.shape)-1)[:, np.newaxis]

# --------------------------------------------------------------------------------------------------
# Get an axis from spec

def get_axis(axis):
    if type(axis) is str:
        uc = axis.upper()[-1]
        if uc == 'X':
            v = np.array(ei4)
        elif uc == 'Y':
            v = np.array(ej4)
        elif uc == 'Z':
            v = np.array(ek4)
        elif uc in ['W', 'T']:
            v = np.array(el4)
        else:
            raise RuntimeError(f"Unknown axis spec '{axis}': must be in ['X', 'Y', 'Z', 'W' or 'T']")

        if axis[0] == '-':
            v = -1. * v

        return v

    else:
        v = np.array(axis)
        n = np.linalg.norm(v)
        if n < ZERO:
            raise RuntimeError(f"The length of the vector is too small to get an axis'{axis}'")
        else:
            return v / n

# --------------------------------------------------------------------------------------------------
# Cross product of three 4D vectors

def cross4D(v1, v2, v3):
    """4D cross product

    Implement the cross product between 3 4D Vectors
    """

    u1 = np.array(v1)
    u2 = np.array(v2)
    u3 = np.array(v3)

    x = np.linalg.det(np.array((u1, u2, u3, ei4)))
    y = np.linalg.det(np.array((u1, u2, u3, ej4)))
    z = np.linalg.det(np.array((u1, u2, u3, ek4)))
    t = np.linalg.det(np.array((u1, u2, u3, el4)))

    return np.array((x, y, z, t))

# --------------------------------------------------------------------------------------------------
# 3D spherical to cartesian

def sph_to_cart3(s):

    ss = np.array(s)

    # Only one vector
    if len(ss.shape) == 1:
        return sph_to_cart3(ss[np.newaxis])[0]

    # Conversion
    rs = ss[:, 0]*np.cos(ss[:, 2])

    return np.array( (rs*np.cos(ss[:, 1]), rs*np.sin(ss[:, 1]), ss[:, 0]*np.sin(ss[:, 2])) ).transpose()

# --------------------------------------------------------------------------------------------------
# 3D cartesian to spherical

def cart_to_sph3(v):

    vs = np.array(v)

    # only one vector
    if len(vs.shape) == 1:
        return cart_to_sph3(vs[np.newaxis])[0]

    # ---- Several vectors

    # Normalize
    rs = np.linalg.norm(vs, axis=1)

    phis   = np.arcsin(vs[:, 2] / rs)
    thetas = np.arctan2(vs[:, 1], vs[:, 0])

    return np.array((rs, thetas, phis)).transpose()

# --------------------------------------------------------------------------------------------------
# 4D hyperspherical to cartesian
# rho, theta, phi, alpha

def sph_to_cart4(c):
    cs = np.array(c)

    if len(cs.shape) == 1:
        return sph_to_cart4(cs[np.newaxis])[0]

    cal = np.cos(cs[:, 3])
    cph = np.cos(cs[:, 2])
    cc  = cal*cph

    carts = np.array((cc * np.cos(cs[:, 1]), cc * np.sin(cs[:, 1]), cal * np.sin(cs[:, 2]), np.sin(cs[:, 3]))).transpose()
    carts = carts * cs[:, 0][:, np.newaxis]

    return carts

# --------------------------------------------------------------------------------------------------
# 4D cartesian to hyperspherical

def cart_to_sph4(v):

    vs  = np.array(v)

    if len(vs.shape) == 1:
        return cart_to_sph4(vs[np.newaxis])[0]

    count = len(vs)

    # ----- The 3 angles
    # Default value set for null length

    thetas = np.zeros(count, np.float)
    phis   = np.ones(count, np.float)*pi/2
    alphas = np.array(phis)

    # ----- The norms and the normalized vectors
    rs   = np.linalg.norm(vs, axis=1)

    # ----- Vectors for which the norm is not null
    inds = np.where(rs > ZERO)[0]

    # ----- Alpha is simple to compute
    alphas[inds] = np.arcsin(vs[inds, 3] / rs[inds])

    # ----- The rest is 3D spherical coordinates

    sph3 = cart_to_sph3(vs[inds, :3])
    thetas[inds] = sph3[:, 1]
    phis[inds]   = sph3[:, 2]

    # ---- The result
    return np.array((rs, thetas, phis, alphas)).transpose()

# --------------------------------------------------------------------------------------------------
# Compute the projection matrix along an axis

def matrix_4D_axis4(axis=None):
    """Compute the projection matrix along a 4D axis

    if the axis parameter is None, return default projection
    """

    # Default projection
    if axis is None:
        return np.array((ei4, ej4, el4))

    # Normalized projection axis
    L = get_axis(axis)

    # Axis is equal to L
    if abs(abs(np.dot(L, el4)) - 1.) < ZERO:
        return np.array((ei4, ej4, ek4))

    # Axis is equal to K
    if abs(abs(np.dot(L, ek4)) - 1.) < ZERO:
        return np.array((ei4, ej4, el4))

    # New vectors

    # I^J^K  =  L
    # J^K^L  = -I
    # K^L^I  =  J
    # L^I^J  = -K

    I = -cross4D(ek4, el4, L)
    J = cross4D(el4, L, I)
    K = -cross4D(L, I, J)

    return np.array((I, J, K))

# --------------------------------------------------------------------------------------------------
# Compute the projection matrix with 3 angles u, v, w

def matrix_4D_uvw(u, v, w):
    """Compute the projection matrix with 3 angles u, v and w"""

    return np.array((
        (-sin(w), cos(w), 0., 0.),
        (-cos(w) * sin(u), -sin(w) * sin(u), cos(u), 0.),
        (-cos(w) * cos(u) * sin(v), -sin(w) * cos(u) * sin(v), -sin(u) * sin(v), cos(v))
    ))


# --------------------------------------------------------------------------------------------------
# Compute a projection matrix along a 3D axis and an angle

def matrix_4D_axis3(V, w):
    sph = cart_to_sph3(V)
    return matrix_4D_uvw(sph[1], sph[2], w)

# ======================================================================================================================================================
# 4D Scene extension settings

class FourD_settings(bpy.types.PropertyGroup):
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Toggle display 4D / 3D

    fourD_display_: BoolProperty(default=False) = False

    def fourD_display_get(self):
        return self.fourD_display_

    def fourD_display_set(self, value):
        self.fourD_display_ = value
        self.update_projection()
        # Grids
        bpy.ops.spaceview3d.showgridbutton()

    fourD_display: BoolProperty(
        name="4D display",
        description="Switch from standard to 4D display",
        set=fourD_display_set,
        get=fourD_display_get,
    )

    # ----------------------------------------------------------------------------------------------------
    # Projection of all the objects within the scene

    def update_projection(self):
        scene = self.id_data

        if self.fourD_display:
            M_proj = self.M_proj()
            for obj in scene.objects:
                obj.fourD_param.projection(M_proj)
        else:
            for obj in scene.objects:
                obj.fourD_param.display_4D = False

    # ----------------------------------------------------------------------------------------------------
    # Projection mode

    fourD_proj_: IntProperty(default=0) = 0

    def fourD_proj_get(self):
        return self.fourD_proj_

    def fourD_proj_set(self, value):
        self.fourD_proj_ = value
        self.update_projection()

    fourD_proj: EnumProperty(
        items=[
            ('ORTH', "Orth", "Orthogonal projection"),
            ('VECTOR', "Vector", "Along a vector"),
            ('ANGLES', "Angles", "Define 3 angles"),
        ],
        set=fourD_proj_set,
        get=fourD_proj_get,
    )

    # ----------------------------------------------------------------------------------------------------
    # Orthogonal mode : the orthogonal axis

    orth_axis_: IntProperty(default=3) = 3

    def orth_axis_get(self):
        return self.orth_axis_

    def orth_axis_set(self, value):
        self.orth_axis_ = value
        self.update_projection()

    orth_axis: EnumProperty(
        items=[
            ('X', "X", "Orthogonal to X axis"),
            ('Y', "Y", "Orthogonal to Y axis"),
            ('Z', "Z", "Orthogonal to Z axis"),
            ('W', "W", "Orthogonal to W axis"),
        ],
        set=orth_axis_set,
        get=orth_axis_get,
    )

    # ----------------------------------------------------------------------------------------------------
    # Vector mode : the projection vector

    proj_vector_: FloatVectorProperty(
        description="Projection axis",
        default=(10.0, 10.0, 0.0, 10.0),
        subtype='XYZ',
        size=4,
    )

    def proj_vector_get(self):
        return self.proj_vector_

    def proj_vector_set(self, value):
        self.proj_vector_ = value
        self.update_projection()

    proj_vector: FloatVectorProperty(
        name="Vector",
        subtype='XYZ',
        size=4,
        get=proj_vector_get,
        set=proj_vector_set,
    )

    # ----------------------------------------------------------------------------------------------------
    # Angles mode : the rotation angles

    proj_angles_: FloatVectorProperty(
        default=(0.0, 0.0, 0.0),
        subtype='EULER',
        size=3,
    )

    def proj_angles_get(self):
        return self.proj_angles_

    def proj_angles_set(self, value):
        self.proj_angles_ = value
        self.update_projection()

    proj_angles: FloatVectorProperty(
        name="Angles",
        description="Projection angles along XY, YZ, ZW",
        subtype='EULER',
        size=3,
        step=100,
        get=proj_angles_get,
        set=proj_angles_set,
    )

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Show grids

    show_grid_XY: BoolProperty(
        name="XY",
        description="Display a grid on the XY plan",
        default=True
    )
    show_grid_YZ: BoolProperty(
        name="YZ",
        description="Display a grid on the YZ plan",
        default=False
    )
    show_grid_ZW: BoolProperty(
        name="ZW",
        description="Display a grid on the ZW plan",
        default=True
    )
    show_grid_WX: BoolProperty(
        name="WX",
        description="Display a grid on the WX plan",
        default=False
    )

    # ----------------------------------------------------------------------------------------------------
    # Show 4D grids toggle management

    show_floor_3D: BoolProperty(default=True)
    show_axis_x_3D: BoolProperty(default=True)
    show_axis_y_3D: BoolProperty(default=True)
    show_axis_z_3D: BoolProperty(default=False)

    show_4D_grids_: BoolProperty(default=False)

    def show_4D_grids__get(self):
        return self.show_4D_grids_

    def show_4D_grids__set(self, value):
        self.show_4D_grids_ = value
        bpy.ops.spaceview3d.showgridbutton()

    show_4D_grids: BoolProperty(
        name="4D grids",
        get=show_4D_grids__get,
        set=show_4D_grids__set,
        description="Show / hide the 4D grids"
    )

    # ----------------------------------------------------------------------------------------------------
    # 4D --> 3D projection matrix

    def M_proj(self):

        # ---------------------------------------------------------------------------
        # Orthogonal projection

        if self.fourD_proj == 'ORTH':

            if self.orth_axis == 'X':
                return Matrix((
                    (0., 1., 0., 0.),
                    (0., 0., 1., 0.),
                    (0., 0., 0., 1.)
                ))
            elif self.orth_axis == 'Y':
                return Matrix((
                    (1., 0., 0., 0.),
                    (0., 0., 1., 0.),
                    (0., 0., 0., 1.)
                ))
            elif self.orth_axis == 'Z':
                return Matrix((
                    (1., 0., 0., 0.),
                    (0., 1., 0., 0.),
                    (0., 0., 0., 1.)
                ))
            else:
                return Matrix((
                    (1., 0., 0., 0.),
                    (0., 1., 0., 0.),
                    (0., 0., 1., 0.)
                ))

        # ---------------------------------------------------------------------------
        # Projection following a 4D vector

        elif self.fourD_proj == 'VECTOR':
            return matrix_4D_axis4(self.proj_vector)

        # ---------------------------------------------------------------------------
        # Projection according angles

        elif self.fourD_proj == 'ANGLES':
            return matrix_4D_uvw(self.proj_angles[0], self.proj_angles[1], self.proj_angles[2])

        else:
            print("Unknown projection code : ", self.fourD_proj)
            return None


# ======================================================================================================================================================
# Projection 4D --> 3D operator

class FourDProjectionOperator(bpy.types.Operator):
    """4D Space"""
    bl_idname = "scene.fourd_projection"
    bl_label = "Update 4D"

    def execute(self, context):
        scene = context.scene
        scene.fourD_settings.update_projection()

        return {'FINISHED'}


# ======================================================================================================================================================
# 4D settings scene panel

class FourDProjectionPanel(bpy.types.Panel):
    """4D extension"""

    bl_label = "4D projection"
    bl_idname = "FOURD_PT_fourd_projection"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '4D'
    bl_order = 1

    def draw(self, context):

        settings = context.scene.fourD_settings

        layout = self.layout

        # ----- 4D Display switch

        # 4D Display switch
        mbox = layout.box()
        row = mbox.row(align=True)
        row.prop(settings, "fourD_display", toggle=True)

        # 4D Projection parameters

        # Projection mode
        mbox.label(text="Projection 4D")
        row = mbox.row(align=True)
        row.prop_enum(settings, "fourD_proj", 'ORTH')
        row.prop_enum(settings, "fourD_proj", 'VECTOR')
        row.prop_enum(settings, "fourD_proj", 'ANGLES')

        # Modes parameters
        if settings.fourD_proj == 'ORTH':
            row = mbox.row(align=True)
            row.prop_enum(settings, "orth_axis", 'X')
            row.prop_enum(settings, "orth_axis", 'Y')
            row.prop_enum(settings, "orth_axis", 'Z')
            row.prop_enum(settings, "orth_axis", 'W')

        if settings.fourD_proj == 'VECTOR':
            row = mbox.column()
            row.prop(settings, "proj_vector")

        if settings.fourD_proj == 'ANGLES':
            row = mbox.column()
            row.prop(settings, "proj_angles")

        # Update the projection
        if settings.fourD_display:
            mbox.operator("scene.fourd_projection")

        # ----- 4D grids

        # show 4D grids switch
        # The button is not used directly but though the toggle boolean

        if settings.fourD_display:
            mbox = layout.box()

            row = mbox.row(align=True)
            row.prop(settings, "show_4D_grids", toggle=True)

            mbox.label(text="Display grids")
            row = mbox.row(align=True)
            row.prop(settings, "show_grid_XY", toggle=True)
            row.prop(settings, "show_grid_YZ", toggle=True)
            row.prop(settings, "show_grid_ZW", toggle=True)
            row.prop(settings, "show_grid_WX", toggle=True)


# ======================================================================================================================================================
# Create the 4D layer to store the 4th coordinate

def get_4D_layer(bm):
    id4D = bm.verts.layers.float.get('4D coord')
    if id4D is None:
        id4D = bm.verts.layers.float.new('4D coord')
    return id4D


# ======================================================================================================================================================
# 4D object extension
# The fourth coordinates is stored in an additional layer named '4D coord'
# The object is plunged in one of the four sub spaces: XYZ, XYW, XWY, WYZ

class SubSpace():
    modes = ['XYZ', 'XYW', 'WYZ', 'XWZ']
    perps = ['W', 'Z', 'W', 'Y']

    def __init__(self, mode='XYZ'):
        if type(mode) is int:
            mode = SubSpace.modes[mode]
        if not mode in SubSpace.modes:
            raise NameError("Unknow, sub space mode: '{}'".format(mode))
        self.mode = mode

    def vector_4D(self, V3D, fourth):
        if self.mode == 'XYZ':
            return Vector((V3D[0], V3D[1], V3D[2], fourth))
        if self.mode == 'XYW':
            return Vector((V3D[0], V3D[1], fourth, V3D[2]))
        if self.mode == 'WYZ':
            return Vector((fourth, V3D[1], V3D[2], V3D[0]))
        if self.mode == 'XWZ':
            return Vector((V3D[0], fourth, V3D[2], V3D[1]))

    def vector_3D(self, V4D):
        if self.mode == 'XYZ':
            return Vector((V4D[0], V4D[1], V4D[2])), V4D[3]
        if self.mode == 'XYW':
            return Vector((V4D[0], V4D[1], V4D[3])), V4D[2]
        if self.mode == 'WYZ':
            return Vector((V4D[3], V4D[1], V4D[2])), V4D[0]
        if self.mode == 'XWZ':
            return Vector((V4D[0], V4D[3], V4D[2])), V4D[1]

    def get_fourth_axis_name(self):
        return SubSpace.perps[SubSpace.modes.index(self.mode)]

    def get_fourth_axis_index(self):
        return ['X', 'Y', 'Z', 'W'].index(self.get_fourth_axis_name())

    # Axis perpendicular to the 3D sub space
    def get_perp(self):
        return get_axis(self.get_fourth_axis_name())

    # Get the 4D axis mapped in the 3D XYZ space
    def get_mapped_axis(self, axis='Z'):
        return self.mode[['X', 'Y', 'Z'].index(axis)]


# -----------------------------------------------------------------------------------------------------------------------------
# 4D object extension

class FourD_Param(bpy.types.PropertyGroup):

    # ----------------------------------------------------------------------------------------------------
    # Utility: bmesh handling

    def bmesh_begin(self):
        obj = self.id_data
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
        return bm

    def bmesh_end(self, bm):
        obj = self.id_data
        if obj.mode == 'EDIT':
            bmesh.update_edit_mesh(obj.data, True)
        else:
            bm.to_mesh(obj.data)
            bm.free()

    # ----------------------------------------------------------------------------------------------------
    # The object is a 4D object

    is4D: BoolProperty(
        name="4D object",
        description="This is a 4D object",
        default=False
    )

    # 4D type

    type: EnumProperty(
        items=[
            ('POINT', "Point", "The shape of the object is not deformed by the 4D projection, just its location"),
            ('AXIS', "Axis",
             "The vertical axis, with its Blender 3D rotation, is rotated by the 4D projection without deforming the shape"),
            ('SURFACE', "Surface", "The shape is deformed by the 4D projection"),
        ],
        default="SURFACE",
    )

    # Modelling space

    modelling_space_: IntProperty(default=0)

    # Change the modelling space

    def modelling_space_set(self, value):

        # ----- Change the subspace

        oldsubs = SubSpace(self.modelling_space_)
        newsubs = SubSpace(value)

        self.modelling_space_ = value

        # ----- Object location

        self.id_data.location, self.fourD_loc = newsubs.vector_3D(
            oldsubs.vector_4D(self.id_data.location, self.fourD_loc))

        # ----- Vertices

        if self.type == 'SURFACE':

            bm = self.bmesh_begin()
            id4D = get_4D_layer(bm)

            for v in bm.verts:
                v.co, v[id4D] = newsubs.vector_3D(oldsubs.vector_4D(v.co, v[id4D]))

            self.bmesh_end(bm)

    def modelling_space_get(self):
        return self.modelling_space_

    modelling_space: EnumProperty(
        items=[
            ('XYZ', "XYZ", "3D modelling is made in hyperspace XYZ"),
            ('XYW', "XYW", "3D modelling is made in hyperspace XYW (Z is used as W)"),
            ('WYZ', "WYZ", "3D modelling is made in hyperspace WYZ (X is used as W)"),
            ('XWZ', "XWZ", "3D modelling is made in hyperspace XWZ (Y is used as W)"),
        ],
        default="XYZ",
        set=modelling_space_set,
        get=modelling_space_get,
    )

    # Additional coordinate

    fourD_loc: FloatProperty(
        name="Coordinate",
        description="Coordinate of the modelling space along the fourth dimension",
        default=0.0,
    )

    # -----------------------------------------------------------------------------------------------------------------------------
    # Set / Get the 4D Vector

    store_location_4D: FloatVectorProperty(size=4)
    store_location_3D: FloatVectorProperty(size=3)
    store_rotation_euler: FloatVectorProperty(size=3)
    store_rotation_mode: StringProperty()

    def location_4D_get(self):

        if self.display_4D:
            return Vector(self.store_location_4D)

        loc = self.id_data.location
        loc_w = self.fourD_loc

        return SubSpace(self.modelling_space_).vector_4D(loc, loc_w)

    def location_4D_set(self, loc4D):
        o = self.id_data
        o.location, self.fourD_loc = SubSpace(self.modelling_space_).vector_3D(loc4D)
        # self.vector_3D(loc4D)

    # The 4D location

    location_4D: FloatVectorProperty(
        name="4D location",
        description="4D location",
        subtype='XYZ',
        size=4,
        get=location_4D_get,
        set=location_4D_set
    )

    # -----------------------------------------------------------------------------------------------------------------------------
    # Get the 4D coordinates of a vertex from 3D coordinates relative to the center

    def get_4D_coordinates(self, coord_3D):

        # Center in 3D space

        if self.display_4D:
            center_3D = Vector(self.store_location_3D)
        else:
            center_3D = Vector(self.id_data.location)

        # Absolute coordinate

        loc = center_3D + coord_3D

        # Transformation

        # return self.vector_4D(loc, self.fourD_loc)
        return SubSpace(self.modelling_space_).vector_4D(loc, self.fourD_loc)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Utilities

    def vector_4D(self, V3D, fourth):
        return SubSpace(self.modelling_space).vector_4D(V3D, fourth)

    def vector_3D(self, V4D):
        return SubSpace(self.modelling_space).vector_3D(V4D)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Toggle 4D / 3D display

    display_4D_: BoolProperty(default=False)

    def display_4D_get(self):
        return self.display_4D_

    def display_4D_set(self, value):

        if self.display_4D_ == value:
            return

        loc4D = Vector(self.location_4D)

        self.display_4D_ = value
        if self.display_4D_:
            self.store_location_4D = loc4D
            self.store_location_3D = self.id_data.location
            self.store_rotation_euler = self.id_data.rotation_euler
            self.store_rotation_mode = self.id_data.rotation_mode

            if self.type == 'SURFACE':
                self.shape_key_4D(True)
        else:
            self.location_4D = loc4D
            self.id_data.rotation_euler = self.store_rotation_euler
            self.id_data.rotation_mode = self.store_rotation_mode

            if self.type == 'SURFACE':
                self.shape_key_4D(False)

    display_4D: BoolProperty(
        set=display_4D_set,
        get=display_4D_get,
    )

    # -----------------------------------------------------------------------------------------------------------------------------
    # Object rotation

    rotation0_: FloatVectorProperty(
        default=(0.0, 0.0),
        unit='ROTATION',
        size=2,
    )
    rotation1_: FloatVectorProperty(
        default=(0.0, 0.0),
        unit='ROTATION',
        size=2,
    )

    def rotation0_get(self):
        return self.rotation0_

    def rotation0_set(self, value):
        self.rotation0_ = value
        if self.display_4D:
            bpy.context.scene.fourD_settings.update_projection()

    def rotation1_get(self):
        return self.rotation1_

    def rotation1_set(self, value):
        self.rotation1_ = value
        if self.display_4D:
            bpy.context.scene.fourD_settings.update_projection()

    rotation0: FloatVectorProperty(
        name="XY-ZW rotation",
        description="Projection angles along planes XY and ZW",
        unit='ROTATION',
        step=100,
        size=2,
        get=rotation0_get,
        set=rotation0_set,
    )

    rotation1: FloatVectorProperty(
        name="XZ-YW rotation",
        description="Projection angles along planes XZ and YW",
        unit='ROTATION',
        step=100,
        size=2,
        get=rotation1_get,
        set=rotation1_set,
    )

    def get_rotation_matrix(self):

        def mrot(angle, i, j):
            c = cos(angle)
            s = sin(angle)
            M = Matrix.Identity(4)
            M[i][i] = c
            M[j][j] = c
            M[i][j] = -s
            M[j][i] = s
            return M

        M0 = mrot(self.rotation0[0], 0, 1)
        M1 = mrot(self.rotation0[1], 2, 3)
        M2 = mrot(self.rotation1[0], 0, 3)
        M3 = mrot(self.rotation1[1], 1, 2)

        return (M3 @ (M2 @ M1)) @ M0

    # -----------------------------------------------------------------------------------------------------------------------------
    # Shape key

    def get_shape_key(self, fourD=False):

        obj = self.id_data

        if obj.data.shape_keys is None:
            base = obj.shape_key_add()
            base.name = "3D"
            obj.data.shape_keys.use_relative = False

        name = "4D" if fourD else "3D"
        sk = obj.data.shape_keys.key_blocks.get(name)

        if sk is None:
            sk = obj.shape_key_add()
            sk.name = name

        return sk

    # -----------------------------------------------------------------------------------------------------------------------------
    # Shape key in 4D mode

    def shape_key_4D(self, fourD=False):

        obj = self.id_data

        if obj.data.shape_keys is None:
            if not fourD:
                return

        sk = self.get_shape_key(fourD)
        obj.data.shape_keys.eval_time = sk.frame

    # -----------------------------------------------------------------------------------------------------------------------------
    # Projection

    def projection(self, MProj=None):

        if not self.is4D:
            return

        if MProj is None:
            MProj = bpy.context.scene.fourD_settings.M_proj()

        obj = self.id_data

        self.display_4D = True
        obj.location = MProj @ self.location_4D

        # ----------------------------------------------------------------------------------------------------
        # Axis object: change the orientation

        if self.type == 'AXIS':
            # Z Oriented axis by default with the object euler rotation
            base_axis = Vector((0., 0., 1.))
            axis = base_axis.copy()
            axis.rotate(Euler(self.store_rotation_euler, obj.rotation_mode))

            # Transform in the mapping sub space
            axis4D = SubSpace(self.modelling_space).vector_4D(axis, 0.)

            # Project in the 3D space
            target = MProj @ axis4D

            # Compute the new rotation of the axis
            q = base_axis.rotation_difference(target)
            obj.rotation_euler = q.to_euler(obj.rotation_mode)

        # ----------------------------------------------------------------------------------------------------
        # Surface object: deform the shape

        if self.type == 'SURFACE':

            self.shape_key_4D(True)  # Ensure 4D shape key exists
            bm = self.bmesh_begin()  # bmesh object
            id4D = get_4D_layer(bm)  # 4th dimension layer
            sh = bm.verts.layers.shape.get("4D")  # Shape key layer

            center = Vector(self.location_4D)  # Location 4D of the object
            M_rot = self.get_rotation_matrix()  # Local rotation matrix

            # Loop on the vertice

            for v in bm.verts:
                v4d = self.vector_4D(v.co, v[id4D])  # The local 4D position of the vertex
                co = center + (M_rot @ v4d)  # Rotation and world location
                v[sh] = MProj @ co - obj.location  # Shape vertex

            self.bmesh_end(bm)

    # ----------------------------------------------------------------------------------------------------
    # Count the number of selected vertices

    def selected_count(self):

        obj = self.id_data
        if obj.mode != 'EDIT':
            return 0

        bm = bmesh.from_edit_mesh(obj.data)

        n = 0
        for v in bm.verts:
            if v.select:
                n += 1

        return n

    # ----------------------------------------------------------------------------------------------------
    # Get the 4D location of vertice in edit mode

    def vertex_4D_location_get(self):

        obj = self.id_data
        if obj.mode != 'EDIT':
            return None

        bm = bmesh.from_edit_mesh(obj.data)
        id4D = get_4D_layer(bm)

        loc = Vector((0., 0., 0., 0.))
        n = 0
        for v in bm.verts:
            if v.select:
                loc = loc + self.vector_4D(v.co, v[id4D])
                n += 1

        if n == 0:
            return None

        return loc / n

    def vertex_4D_location_set(self, value):

        obj = self.id_data
        if obj.mode != 'EDIT':
            return

        # Get the current value

        cur_loc = self.vertex_4D_location_get()

        # Uses the difference

        print(value)

        diff = Vector(value) - cur_loc

        # Loop on the vertice

        bm = bmesh.from_edit_mesh(obj.data)
        id4D = get_4D_layer(bm)

        for v in bm.verts:
            if v.select:
                loc = self.vector_4D(v.co, v[id4D])
                loc = loc + diff
                v.co, v[id4D] = self.vector_3D(loc)

        bmesh.update_edit_mesh(obj.data, True)

    # The 4D vertice location property (edit mode only)

    vertex_4D_location: FloatVectorProperty(
        name="Vertex",
        size=4,
        subtype='XYZ',
        step=1,
        get=vertex_4D_location_get,
        set=vertex_4D_location_set)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Extrusion along the perpendicular axis

    def extrude_4D(self, w_min=-1., w_max=1., slices=9):

        # ----- The object is used as a model
        bm_ref = self.bmesh_begin()

        bm_ref.verts.index_update()
        bm_ref.verts.ensure_lookup_table()
        count = len(bm_ref.verts)

        bm_ref.faces.index_update()
        bm_ref.faces.ensure_lookup_table()
        faces_count = len(bm_ref.faces)

        uv_ref = bm_ref.loops.layers.uv.active

        # ----- Create a new object
        # mesh = bpy.data.meshes.new("4D Object")
        # mesh.fourD_param.is4D = True
        # mesh.fourD_param.modelling_space = self.modelling_space

        # The bmesh to work with
        bm = bmesh.new()
        id4D = get_4D_layer(bm)

        if uv_ref is not None:
            uv_layer = bm.loops.layers.uv.new()

            # ----- Loop on the slices
        for isl in range(slices):
            w = w_min + (w_max - w_min) / (slices - 1) * isl
            scale = 1.

            # Copy the vertices at w
            for vx in bm_ref.verts:
                v = scale * Vector(vx.co)
                bm.verts.new(v)

            # Create the faces
            bm.verts.ensure_lookup_table()
            for face in bm_ref.faces:
                new_f = [bm.verts[isl * count + vx.index] for vx in face.verts]
                bm.faces.new(new_f)

            # Copy the uv map
            if uv_ref is not None:
                bm.faces.ensure_lookup_table()

                for iface, face_ref in enumerate(bm_ref.faces):
                    face = bm.faces[isl * faces_count + iface]

                    for iloop, loop in enumerate(face.loops):
                        loop_ref = face_ref.loops[iloop]
                        loop[uv_layer].uv = Vector(loop_ref[uv_ref].uv)

            # 4D Layer
            for i in range(count):
                bm.verts[isl * count + i][id4D] = w

        # ----- Create the object
        bm_ref.free()

        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.object
        obj.name = "4D Object"

        obj.fourD_param.is4D = True
        obj.fourD_param.modelling_space = self.modelling_space

        bm.to_mesh(obj.data)
        obj.data.update()

        bm.free()


# ======================================================================================================================================================
# Display the 4th coordinate panel

class VIEW3D_PT_4d_coord(bpy.types.Panel):
    """4D extension"""

    bl_label = "4D vertex"
    bl_idname = "FOURD_PT_fourth_dimension"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '4D'
    bl_order = 20

    @classmethod
    def poll(cls, context):
        # Only allow in edit mode for a selected mesh.
        return context.mode == "EDIT_MESH" and context.object is not None and context.object.type == "MESH"

    def draw(self, context):

        obj = context.object
        prm = obj.fourD_param

        layout = self.layout

        n = prm.selected_count()
        col = layout.column()
        if n == 0:
            col.label(text="No selection")
        else:
            txt = "Vertex" if n == 1 else "Median"
            col.prop(prm, "vertex_4D_location", text=txt)
            if n == 1:
                v = prm.vertex_4D_location
                col.label(text="Length: {}".format(v.length))


# ======================================================================================================================================================
# 4D Object param panel

class FourDParamPanel(bpy.types.Panel):
    """4D extension"""

    bl_label = "4D parameters"
    bl_idname = "FOURD_PT_Settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '4D'
    bl_order = 10

    @classmethod
    def poll(cls, context):
        return True

        # Panel is always displayed with limited content in order to avoid unwanted visual effects
        obj = context.active_object
        if obj is None:
            return False
        return obj.type == 'MESH'

    def draw(self, context):

        layout = self.layout

        # Not a mesh: display a simple information
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            layout.label(text="4D Object parameters")
            return

        # 4D Parameters
        param = obj.fourD_param

        # Is the 4D projection currently active
        active = param.display_4D

        if active:
            box = layout.box()
            box.label(text="4D display is active")
            return

        # 4D projection is not active, we can change de parameters

        # 4D switch
        row = layout.row(align=True)
        row.prop(param, "is4D")

        # Not a 4D object, let's stop here
        if not param.is4D:
            return

        # 4D Object type
        layout.label(text="4D type")
        row = layout.row(align=True)
        row.prop_enum(param, "type", 'POINT')
        row.prop_enum(param, "type", 'AXIS')
        row.prop_enum(param, "type", 'SURFACE')

        # Modelling space
        layout.label(text="Modelling space")
        row = layout.row(align=True)
        row.prop_enum(param, "modelling_space", 'XYZ')
        row.prop_enum(param, "modelling_space", 'XYW')
        row.prop_enum(param, "modelling_space", 'WYZ')
        row.prop_enum(param, "modelling_space", 'XWZ')

        col = layout.column()
        col.prop(param, "location_4D")

        # Rotation
        if param.type == 'SURFACE':
            split = layout.split()
            col = split.column()
            col.prop(param, "rotation0")
            col = split.column()
            col.prop(param, "rotation1")


# ======================================================================================================================================================
# Dessin du view port en 4D

viewport_shader = gpu.shader.from_builtin('3D_SMOOTH_COLOR')


# -----------------------------------------------------------------------------------------------------------------------------
# Drawing hook

def draw_callback_3d(self, context):
    """Draw a grid for a plane in the 4D space"""

    # Axis colors

    axis_color = {'X': (1., 0., 0., 1.), 'Y': (0., 1., 0., 1.), 'Z': (0., 0., 1., 1.), 'W': (1., 1., 1., 1.)}

    # Return the color of an axis
    def get_color(axis, alpha=1.):
        col = axis_color[axis]
        return [c * alpha for c in col]

    # Projection matrix
    settings = context.scene.fourD_settings
    overlay = context.space_data.overlay
    M = settings.M_proj()

    # Lines & colors of the grid
    # Lines are couples of 4D vertices

    def grid_verts(plane='XY', space=1., count=10):
        if count % 2 == 1:
            count += 1

        def vert(i, j):
            x = space * (i - count // 2)
            y = space * (j - count // 2)
            if plane == 'XY':
                v = (x, y, 0., 0.)
            elif plane == 'XZ':
                v = (x, 0., y, 0.)
            elif plane == 'XW':
                v = (x, 0., 0., y)
            elif plane == 'YZ':
                v = (0., x, y, 0.)
            elif plane == 'YW':
                v = (0., x, 0., y)
            elif plane == 'ZW':
                v = (0., 0., x, y)
            else:
                v = (x, y, 0., 0.)

            return M @ Vector(v)

        verts = []
        cols = []
        col = get_color(plane[1], 0.5)
        for i in range(count + 1):
            verts.extend([vert(i, 0), vert(i, count)])
            cols.extend([col, col])

        col = get_color(plane[0], 0.5)
        for j in range(count + 1):
            verts.extend([vert(0, j), vert(count, j)])
            cols.extend([col, col])

        col = {'XY': (0., 0., .5, 1.),
               'XZ': (0., .5, 0., 1.),
               'XW': (.5, 0., 0., 1.),
               'YZ': (0., .5, .5, 1.),
               'YW': (0., .5, .5, 1.),
               'ZW': (.5, .5, 0., 1.)}

        # Add the axis

        return verts, cols

    # All the grids in the batch

    verts = []
    cols = []

    space = overlay.grid_scale
    count = overlay.grid_lines

    if settings.show_grid_XY:
        v, c = grid_verts('XY', space=space, count=count)
        verts.extend(v)
        cols.extend(c)

    if settings.show_grid_YZ:
        v, c = grid_verts('YZ', space=space, count=count)
        verts.extend(v)
        cols.extend(c)

    if settings.show_grid_ZW:
        v, c = grid_verts('ZW', space=space, count=count)
        verts.extend(v)
        cols.extend(c)

    if settings.show_grid_WX:
        v, c = grid_verts('XW', space=space, count=count)
        verts.extend(v)
        cols.extend(c)

    # Add the axis

    axis_length = 1000.
    for axis in ['X', 'Y', 'Z', 'W']:
        P1 = axis_length * space * get_axis(axis)
        P0 = -P1
        verts.extend([M @ P0, M @ P1])

        col = get_color(axis)
        cols.extend([col, col])

    if len(verts) == 0:
        return

    # Draw

    batch = batch_for_shader(viewport_shader, 'LINES', {"pos": verts, "color": cols})

    viewport_shader.bind()
    batch.draw(viewport_shader)

    return


# -----------------------------------------------------------------------------------------------------------------------------
# Defines button for enable/disable the 4D display

class ShowGridButton(bpy.types.Operator):
    bl_idname = "spaceview3d.showgridbutton"
    bl_label = "4D grids"
    bl_description = "Show 4D grids and axis"

    _handle = None  # keep function handler

    # ----------------------------------
    # Enable gl drawing adding handler
    # ----------------------------------

    @staticmethod
    def handle_add(self, context):
        if ShowGridButton._handle is None:
            ShowGridButton._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_3d, (self, context), 'WINDOW',
                                                                            'POST_VIEW')
            # context.scene.fourD_settings.show_4D_grids = True

    # ------------------------------------
    # Disable gl drawing removing handler
    # ------------------------------------

    @staticmethod
    def handle_remove(self, context):
        if ShowGridButton._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(ShowGridButton._handle, 'WINDOW')
        ShowGridButton._handle = None
        # context.scene.fourD_settings.show_4D_grids = False

    # ------------------------------
    # Execute button action
    # ------------------------------

    def execute(self, context):
        if context.area.type == 'VIEW_3D':
            overlay = context.space_data.overlay
            settings = context.scene.fourD_settings

            if settings.fourD_display and settings.show_4D_grids:

                # Save 3D display options
                settings.show_floor_3D = overlay.show_floor
                settings.show_axis_x_3D = overlay.show_axis_x
                settings.show_axis_y_3D = overlay.show_axis_y
                settings.show_axis_z_3D = overlay.show_axis_z

                # Hide the standard grids
                overlay.show_floor = False
                overlay.show_axis_x = False
                overlay.show_axis_y = False
                overlay.show_axis_z = False

                self.handle_add(self, context)
                # self.handle_remove(self, context)
                context.area.tag_redraw()

            else:
                self.handle_remove(self, context)

                # Restore 3D display options
                overlay.show_floor = settings.show_floor_3D
                overlay.show_axis_x = settings.show_axis_x_3D
                overlay.show_axis_y = settings.show_axis_y_3D
                overlay.show_axis_z = settings.show_axis_z_3D

                context.area.tag_redraw()

            return {'FINISHED'}
        else:
            self.report({'WARNING'},
                        "View3D not found, cannot run operator")

        return {'CANCELLED'}


# ======================================================================================================================================================
# Add a new 4D object


#    """4D extension"""
#
#    bl_label        = "4D parameters"
#    bl_idname       = "FOURD_PT_Settings"
#    bl_space_type   = 'VIEW_3D'
#    bl_region_type  = 'UI'
#    bl_category     = '4D'
#    bl_order        = 10


class Object4DAdd(bpy.types.Operator):
    """Add 4D object from a 3D model"""

    bl_idname = "mesh.object_4d_add"
    bl_label = "4D object"
    bl_options = {'REGISTER', 'UNDO'}

    object_name: StringProperty(name="3D Model", default="")
    slices: IntProperty(name="Slices", default=5, min=2, max=100)
    size: FloatProperty(name="Size", default=1., min=0., max=math.inf)
    scale: FloatProperty(name="Scale", default=1., min=0.01, max=math.inf)
    symetrical: BoolProperty(name="Both sides", default=False)
    profile: EnumProperty(items=[
        ('CYLINDER', "Cylinder", "Cylinder slices"),
        ('CONE', "Cone", "Conic slices"),
        ('SPHERE', "Sphere", "Spheric slices"),
        ('HYPERBOLA1', "Hyperbola 1", "Hyperboloic slices - one sheet"),
        ('HYPERBOLA2', "Hyperbola 2", "Hyperboloic slices - two sheets"),
    ],
        default='CYLINDER',
        name="Profile",
    )

    # ----------------------------------------------------------------------
    # Profiles that can be symetrized

    def symetric_profiles(self):
        return {'CYLINDER', 'SPHERE', 'HYPERBOLA1'}

    # ----------------------------------------------------------------------
    # Draw the layout

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop_search(self, "object_name", scene, "objects")
        layout.prop(self, "profile")
        layout.prop(self, "slices")
        layout.prop(self, "size")
        layout.prop(self, "scale")
        if self.profile in self.symetric_profiles():
            layout.prop(self, "symetrical")

    # ----------------------------------------------------------------------
    # Crée une nouvelle surface

    def execute(self, context):

        scene = context.scene

        # Get the model if defined

        model = bpy.data.objects.get(self.object_name)
        if model is None:
            return {'FINISHED'}

        # ----- TEMPORATY IMPLEMENTATION

        model.fourD_param.extrude_4D(w_min=-1., w_max=1., slices=self.slices)

        # Update 4D display

        if scene.fourD_settings.fourD_display:
            scene.fourD_settings.update_projection()

        return {'FINISHED'}

        # --------------------------------------------------
        # Slice creation by duplicating a source object
        # Apply an offset along the slice dimension
        # Apply a scale factor on the 3D shape

        def new_slice(s=None, scale=1.0):
            slice = model.copy()
            slice.data = model.data.copy()
            slice.animation_data_clear()
            scene.objects.link(slice)

            prm = slice.fourD_param
            prm.is4D = True
            prm.type = 'SURFACE'

            if s is not None:
                bms = slice.fourD_param.bmesh_begin()
                id4Ds = get_4D_layer(bms)

                for v in bms.verts:
                    v[id4Ds] += s

                slice.fourD_param.bmesh_end(bms)

            slice.scale *= scale * self.scale

            context.scene.objects.active = slice
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

            return slice

        # --------------------------------------------------
        # Add a slide to the base

        def add_slice(base, s, scale, count=0):

            # New slice

            slice = new_slice(s, scale)

            # Join to the base

            for o in scene.objects:
                o.select = False

            slice.select = True
            base.select = True
            scene.objects.active = base

            bpy.ops.object.join()

        # --------------------------------------------------
        # 3D scale from the slice coordinate

        def s_scale(i, s, ds):
            scale = 1.

            if self.profile == 'CONE':
                scale *= s
            elif self.profile == 'SPHERE':
                scale = math.cos(math.asin(s / self.size))
            elif self.profile == 'HYPERBOLA1':
                scale = math.sqrt(s * s + 1.)
            elif self.profile == 'HYPERBOLA2':
                scale = math.sqrt(s ** 2 - 1.)

            return scale

        # --------------------------------------------------
        # Main

        # Nothing if only one slice

        if self.slices <= 1:
            return {'FINISHED'}

        # Get the distance between each slice

        ds = self.size / self.slices
        if self.profile in {'SPHERE', 'CONE', 'HYPERBOLA2'}:
            ds = self.size / (self.slices + 1)

        s0 = 0.
        if not self.profile in self.symetric_profiles():
            s0 = ds
        if self.profile == 'HYPERBOLA2':
            s0 += 1.

        # Create the base slice from the model
        # Base slice

        base = new_slice(s0, s_scale(0, s0, ds))
        base.name = model.name + " 4D"
        scene.objects.active = base

        # Slices creation loop
        # Slice number 0 is the base slide

        for i in range(1, self.slices + 1):

            s = s0 + ds * i

            # Add the slice to the base

            add_slice(base, s, s_scale(i, s, ds), count=i)

            # Symetrical

            if self.symetrical and (self.profile in self.symetric_profiles()):
                add_slice(base, -s, s_scale(i, -s, ds), count=i)

        # Update 4D display

        if scene.fourD_settings.fourD_display:
            scene.fourD_settings.update_projection()

        return {'FINISHED'}


# Draw a button in the SpaceView3D Display panel

def view3D_Draw_Extension(self, context):
    self.layout.operator("spaceview3d.showgridbutton")


# ======================================================================================================================================================
# The 4D main panel : a dedicated tab in the view3D panel

class FourDPanel(bpy.types.Panel):
    bl_label = "4D Panel"
    bl_idname = "FOURD_PT_Config"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '4D'

    @classmethod
    def poll(cls, context):
        return True

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="4D Panel")

    def draw(self, context):
        layout = self.layout
        layout.label(text="The test")


# bpy.utils.register_class(FourDPanel)

# ======================================================================================================================================================
# 4D builder

class Builder():
    def __init__(self):
        self.verts = []  # The vertices
        self.faces = []  # The faces : tuples of vertices

    def vert(self, v):
        self.verts.append(Vector(v))
        return len(self.verts) - 1

    def face(self, f):
        if len(f) == 3:
            i0 = f.index(min(f))
            i2 = f.index(max(f))
            i1 = 0 if 0 not in (i0, i2) else 1 if 1 not in (i0, i2) else 2
            ordered = (f[i0], f[i1], f[i2])
            try:
                return self.faces.index(ordered)
            except:
                self.faces.append(ordered)
        else:
            self.faces.append(tuple(f))
        return len(self.faces) - 1

    def vol(self, vl):
        self.vols.append(vl)

    @staticmethod
    def ordered_edge(edge):
        return (min(edge), max(edge))

    @staticmethod
    def face_edges(face):
        return [Builder.ordered_edge((face[i], face[(i + 1) % len(face)])) for i in range(len(face))]

    def edges(self):
        edges = []
        for face in self.faces:
            face_edges = Builder.face_edges(face)
            for edge in face_edges:
                if not edge in edges:
                    edges.append(edge)
        return edges

    def clone(self, clone_faces=True):
        builder = Builder()
        builder.verts = [Vector(v) for v in self.verts]
        if clone_faces:
            builder.faces = [tuple(face) for face in self.faces]
        return builder

    def create_object(self, name="4D object"):

        bm = bmesh.new()  # create an empty BMesh
        id4D = get_4D_layer(bm)

        # ----- Vertices creations

        for i, v4 in enumerate(self.verts):
            v3 = (v4[0], v4[1], v4[2])
            vx = bm.verts.new(v3)
            vx[id4D] = v4[3]
        bm.verts.ensure_lookup_table()

        # ----- Faces creation

        bm.verts.ensure_lookup_table()
        for i in range(len(self.faces)):
            face = tuple(bm.verts[j] for j in self.faces[i])
            bm.faces.new(face)
            # bmface.material_index = materials[self.mats[i]]

        bm.faces.ensure_lookup_table()

        # Create the mesh, apply

        mesh = bpy.data.meshes.new(name=name)
        bm.to_mesh(mesh)
        bm.free()
        bpy.context.view_layer.update()

        # Create Object whose Object Data is our new mesh
        obj = bpy.data.objects.new(name, mesh)

        # Add *Object* to the scene, not the mesh
        scene = bpy.context.scene
        scene.collection.objects.link(obj)

        # Select the new object and make it active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # 4D !
        obj.fourD_param.is4D = True

        return obj


# ======================================================================================================================================================
# Hyper icosphere

def hyper_icosphere(radius=1., divisions=1):
    builder = Builder()

    # ------ Build the initial 4D pyramid
    factor = sqrt(3) / 2

    # Dim 1
    builder.vert((-1., 0., 0., 0.))
    builder.vert((+1., 0., 0., 0.))

    # Dim 2
    for i in range(len(builder.verts)):
        builder.verts[i] *= factor
        builder.verts[i].y = -0.5
    builder.vert((0., 1., 0., 0.))
    builder.face((0, 1, 2))

    # Dim 3
    for i in range(len(builder.verts)):
        builder.verts[i] *= factor
        builder.verts[i].z = -0.5
    builder.vert((0., 0., 1., 0.))

    edges = builder.edges()
    for e in edges:
        builder.face((e[0], e[1], 3))

    """
    # Dim 4
    for i in range(len(builder.verts)):
        builder.verts[i]   *= factor
        builder.verts[i].w = -0.5
    builder.vert((0., 0., 0., 1.))

    edges = builder.edges()
    for e in edges:
        builder.face((e[0], e[1], 4))
    """

    # ----- Divisions
    #
    # An icosphere is defined by a surface made of equi triangles
    # The initial icosphere (1 division) is an equi pyramid made of 4 equi triangles:
    #    the initial triangle plus one triangle per edge
    # A division for the icosphere consists in splitting each triangle in smaller triangles
    #
    # In a triangle, each point is linked to the 2 points of an edge
    # By splitting theses edges, a smaller triangle is built between the point and the splitting points
    # ---> With 3 points these gives birth to 3 smaller triangles
    # 3 new points are created. These points are used to build an fourth triangle
    #
    # ---
    #
    # An h-icosphere is defined by a voument made of equi pyramids
    # The initial 4D icosphere (1 division) is an h-pyramid made of 5 pyramids:
    #    the initial pyramid plus one per face
    # A division for the h-icosphere consists in splitting each pyramid in smaller pyramids
    #
    # In a pyramid, each point is linked to the 3 points of a triangle
    # By splitting these edges, a smaller pyramid is built between the point and the splitting points
    # ---> With 4 points these gives birth to 4 smaller pyramids
    # 6 new points are created (one per edge). These points are used to build other pyramids
    # The 6 points form 4 triangles used in each of the built pyramids
    # 4 more triangles are created in the middle of each initial face
    # A new point is created at the center to build 8 new pyramid with each of the 8 triangles

    for division in range(divisions):

        edges = builder.edges()  # The list of edges
        middles = [builder.vert((builder.verts[edge[0]] + builder.verts[edge[1]]).normalized()) for edge in
                   edges]  # New points created in the middle of the edges
        faces = []

        # Loop on the faces
        for face in builder.faces:

            # 3 new faces at the corners
            for i in range(len(face)):
                edge0 = Builder.ordered_edge((face[i], face[(i + 1) % len(face)]))
                edge1 = Builder.ordered_edge((face[i], face[(i - 1) % len(face)]))
                i0 = edges.index(edge0)
                i1 = edges.index(edge1)
                faces.append((middles[i0], face[i], middles[i1]))

            # Supplementory face
            face_edges = Builder.face_edges(face)
            faces.append([middles[edges.index(edge)] for edge in face_edges])

        builder.faces = faces

        print(faces)

    """

    # ----- Split an edge

    splitted = []
    new_verts = []

    def split_edge(edge):
        if edge in splitted:
            return new_verts[splitted.index(edge)]

        v = (builder.verts[edge[0]] + builder.verts[edge[1]])/2.
        v.normalize()
        idx = builder.vert(v)
        splitted.append(edge)
        new_verts.append(idx)
        return idx

    # ----- Split a pyramid (4 points, 4 faces and 6 edges)

    def split_pyramid(pyra):
        edges = []
        for i in range(3):
            for j in range(i+1, 4):
                e = (pyra[i], pyra[j])
                edges.append((min(e), max(e)))

        # Edges are ordered as
        # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        # Split the edges

        centers = [split_edge(edge) for edge in edges]

        # The 4 sub pyramids on existing summits
        subpyrs = [
                (pyra[0], centers[0], centers[1], centers[2]),
                (pyra[1], centers[0], centers[3], centers[4]),
                (pyra[2], centers[1], centers[3], centers[5]),
                (pyra[3], centers[2], centers[4], centers[5])
                ]

        return subpyrs

    # HACK

    subpyrs = split_pyramid((0, 1, 2, 3))

    builder.faces = []

    for pyr in subpyrs:
        builder.face((pyr[0], pyr[1], pyr[2]))
        builder.face((pyr[0], pyr[1], pyr[3]))
        builder.face((pyr[0], pyr[2], pyr[3]))
        builder.face((pyr[1], pyr[2], pyr[3]))

    """

    # ------ Create the object
    obj = builder.create_object(name="Hyper icosphere")
    return obj


class HypershereAdd(bpy.types.Operator):
    """Add an hypersphere"""

    bl_idname = "mesh.hypersphere_add"
    bl_label = "Hypersphere"
    bl_options = {'REGISTER', 'UNDO'}

    divisions: IntProperty(name="Divisions", default=1)

    # ----------------------------------------------------------------------
    # Draw the layout

    def draw(self, context):
        layout = self.layout
        layout.label(text="Hypersphere")
        layout.prop(self, "divisions")

    def execute(self, context):
        scene = context.scene

        hyper_icosphere(divisions=self.divisions)

        # Update 4D display
        if scene.fourD_settings.fourD_display:
            scene.fourD_settings.update_projection()

        return {'FINISHED'}


def menu_func(self, context):
    # self.layout.operator(Object4DAdd.bl_idname,   icon='MESH_CUBE')
    self.layout.operator(HypershereAdd.bl_idname, icon='MESH_CUBE')


# ======================================================================================================================================================
# Module register

def register():
    print("\n4D Registering 4D addon")

    # Extension classes
    bpy.utils.register_class(FourD_settings)
    bpy.utils.register_class(FourD_Param)

    # Operators
    bpy.utils.register_class(ShowGridButton)
    bpy.utils.register_class(FourDProjectionOperator)

    # Type extensions
    bpy.types.Scene.fourD_settings = PointerProperty(type=FourD_settings)
    bpy.types.Object.fourD_param = PointerProperty(type=FourD_Param)

    # Panels
    bpy.utils.register_class(VIEW3D_PT_4d_coord)
    bpy.utils.register_class(FourDProjectionPanel)
    bpy.utils.register_class(FourDParamPanel)

    # OLD

    # OLD bpy.types.VIEW3D_PT_view3d_display.append(view3D_Draw_Extension)
    bpy.types.VIEW3D_PT_view3d_properties.append(view3D_Draw_Extension)

    # Add menu
    bpy.utils.register_class(Object4DAdd)
    bpy.utils.register_class(HypershereAdd)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    bpy.utils.unregister_class(Object4DAdd)
    bpy.utils.unregister_class(FourDProjectionOperator)
    bpy.utils.unregister_class(ShowGridButton)
    bpy.utils.unregister_class(FourD_settings)
    bpy.utils.unregister_class(FourDProjectionPanel)
    bpy.utils.unregister_class(FourD_Param)
    bpy.utils.unregister_class(FourDParamPanel)

    bpy.types.VIEW3D_PT_view3d_properties.remove(view3D_Draw_Extension)
    bpy.utils.unregister_class(VIEW3D_PT_4d_coord)

    bpy.utils.unregister_class(Object4DAdd)
    bpy.utils.unregister_class(HypershereAdd)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)


if __name__ == "__main__":
    register()

# register()