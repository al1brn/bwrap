"""
4D management

Version 1.0
Date: May 31st 2020
Author: Alain Bernard
"""

import numpy as np

import bpy

from bpy.props import BoolProperty, FloatProperty, EnumProperty, FloatVectorProperty, StringProperty, IntProperty, \
    PointerProperty, IntVectorProperty

from mathutils import Vector, Matrix, Euler, Quaternion

from .bezier import from_points

# ---- Evaluating the expression

from math import * # CAUTION : keep it for custom function eval
import sys

# ----- Writing the grids

import gpu
from gpu_extras.batch import batch_for_shader

ZERO = 0.00001
ei4 = np.array((1., 0., 0., 0.))
ej4 = np.array((0., 1., 0., 0.))
ek4 = np.array((0., 0., 1., 0.))
el4 = np.array((0., 0., 0., 1.))


def find_3dview_space():
    # Find 3D_View window and its scren space
    area = None
    for a in bpy.data.window_managers[0].windows[0].screen.areas:
        if a.type == 'VIEW_3D':
            area = a
            break

    if area:
        space = area.spaces[0]
    else:
        space = bpy.context.space_data

    return space

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
# Rotation within a plane

def plane_rotation(plane, alpha):

    c = np.cos(alpha)
    s = np.sin(alpha)

    if plane in ['XY', 'YX']:
        if plane == 'YX':
            s = -s
        return np.array(((c, -s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    elif plane in ['XZ', 'ZX']:
        if plane == 'XZ':
            s = -s
        return np.array(((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1)))
    elif plane in ['YZ', 'ZY']:
        if plane == 'ZY':
            s = -s
        return np.array(((1, 0, 0, 0), (0, c, s, 0), (0, -s, c, 0), (0, 0, 0, 1)))

    elif plane in ['XW', 'WX']:
        if plane == 'XW':
            s = -s
        return np.array(((c, 0, 0, s), (0, 1, 0, 0), (0, 0, 1, 0), (-s, 0, 0, c)))
    elif plane in ['YW', 'WY']:
        if plane == 'WY':
            s = -s
        return np.array(((1, 0, 0, 0), (0, c, 0, -s), (0, 0, 1, 0), (0, s, 0, c)))
    elif plane in ['ZW', 'WZ']:
        if plane == 'ZW':
            s = -s
        return np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, c, s), (0, 0, -s, c)))
    else:
        raise RuntimeError(f"plane_rotation matrix: undefined plane '{plane}'")

# --------------------------------------------------------------------------------------------------
# coding of the six planes

six_planes = ['YZ', 'ZX', 'XY', 'XW', 'YW', 'ZW']

# ----------------------------------------------------------------------------------------------------
# Return plane index in euler6 and orientation of the plane

def plane_index(plane):
    try:
        index = six_planes.index(plane)
        return index, 1
    except:
        rev_plane = plane[1] + plane[0]
        return six_planes.index(rev_plane), -1

# --------------------------------------------------------------------------------------------------
# Rotation matrix from 4D euler

def euler6_to_matrix(euler6):
    m = np.identity(4, np.float)
    for i, plane in enumerate(six_planes):
        #m = m.dot(plane_rotation(plane, euler6[i]))
        m = plane_rotation(plane, euler6[i]).dot(m)
    return m

# --------------------------------------------------------------------------------------------------
# Euler order must be adapted to the mapping
# - XYZ = (YZ, ZX, XY) = (1, 2, 3) --> XYZ / + + +
# - WYZ = (YZ, ZW, WY) = (1, 6, 5) --> XZY / + + -
# - XWZ = (WZ, ZX, XW) = (6, 2, 4) --> YZX / - + +
# - XYW = (YW, WX, XY) = (5, 4, 3) --> ZYX / + - +
#
# CAUTION : In addition, there is need to change WYX signs from (+ + -) to (- + +) !!!

class EulerMapping():
    def __init__(self, planes, i_planes, mode, signs):
        self.planes   = planes
        self.i_planes = i_planes
        self.mode     = mode
        self.signs    = signs

euler_mapping = {
    'XYZ': EulerMapping(('YZ', 'ZX', 'XY'), (0, 1, 2), 'XYZ', ( 1,  1,  1)),
    'WYZ': EulerMapping(('YZ', 'ZW', 'WY'), (0, 5, 4), 'XZY', (-1,  1, -1)),
    'XWZ': EulerMapping(('WZ', 'ZX', 'XW'), (5, 1, 3), 'YZX', (-1,  1,  1)),
    'XYW': EulerMapping(('YW', 'WX', 'XY'), (4, 3, 2), 'ZYX', ( 1, -1,  1)),
}

# --------------------------------------------------------------------------------------------------
# Compute a projection matrix along a 3D axis and an angle

def matrix_4D_axis3(V, w):
    sph = cart_to_sph3(V)
    return matrix_4D_uvw(sph[1], sph[2], w)

# =============================================================================================================================
# 4D Object extension

layer_prefix = "4D "

class D4Param(bpy.types.PropertyGroup):

    # ====================================================================================================
    # Global attributes
    # - is4D  (True/False)  : The object is a 4D object
    # - vmode ('3D' / '4D') : Currently in one of the subspace or is projectec
    # - mapping ('XYZ'...)  : 3 letters among X Y Z W indicating the 4D axis mapped by Blender axis
    # - type                : How the object is projected
    #         . POINT   : Location only
    #         . AXIS    : Location and rotation
    #         . SURFACE : Location, rotation and vertices (only for meshes)
    #

    # ----------------------------------------------------------------------------------------------------
    # Bool: The object is a 4D object

    is4D_: BoolProperty(default=False)

    def is4D_get(self):
        return self.is4D_

    def is4D_set(self, value):
        self.is4D_ = value
        if self.mesh is not None:
            self.create_layers(init=True)
            self.object_type_ = 2 # Shape

    is4D: BoolProperty(
        name        = "4D object",
        description = "This is a 4D object",
        get         = is4D_get,
        set         = is4D_set,
    )

    # ----------------------------------------------------------------------------------------------------
    # Visu mode
    # - 3D : the mesh is mapped from 4D space to Blender XYZ according the mapping property
    # - 4D : the mesh is projected from 4D space to Blender

    vmode_: StringProperty(default='3D')

    def vmode_get(self):
        return self.vmode_

    def vmode_set(self, value):
        if value == self.vmode_:
            return

        self.lock_update()

        # --> Pass in projection mode

        if self.vmode_ == '3D':
            self.object_to_space4D()
            self.vmode_ = value

            # Euler rotation
            self.id_data.rotation_euler = (0., 0., 0.)

        # --> Pass in 3D design mode

        else:
            self.vmode_ = value

            # Euler rotation
            self.id_data.rotation_mode = euler_mapping[self.mapping].mode

            # Apply 4D to 3D
            self.space4D_to_object()
            self.rotation4D_changed()

        self.unlock_update()

    vmode: StringProperty(
        name        = "vmode",
        description = "Visualization mode: 3D or 4D projection",
        get         = vmode_get,
        set         = vmode_set)

    # ----------------------------------------------------------------------------------------------------
    # Mapping
    # Three of the four XYZW axis are mapped on the XYZ blender axis using a triplet such as 'XYW'

    mapping_: StringProperty(default = "XYZ")

    def mapping_get(self):
        return self.mapping_

    def mapping_set(self, value):
        if self.mapping_ == value:
            return

        self.lock_update()

        self.object_to_space4D()
        self.mapping_ = value

        # Euler rotation
        self.id_data.rotation_mode = euler_mapping[value].mode

        # 4D to 3D
        self.space4D_to_object()

        self.unlock_update()

    mapping: StringProperty(
        name        = "Mapping",
        description = "Mapping of the 3D axis XYZ in the 4D space axis XYZW",
        get         = mapping_get,
        set         = mapping_set)

    # ----------------------------------------------------------------------------------------------------
    # Object type

    object_type_: IntProperty(default = 0)

    def object_type_get(self):
        return self.object_type_

    def object_type_set(self, value):

        self.lock_update()

        self.object_type_ = value

        self.unlock_update()


    object_type: EnumProperty(
        items = [
            ('POINT',   "Point", "The shape of the object is not deformed by the 4D projection, just its location"),
            ('AXIS',    "Axis",  "The vertical axis, with its Blender 3D rotation, is rotated by the 4D projection without deforming the shape"),
            ('SHAPE',   "Shape", "The shape is deformed by the 4D projection"),
                ],
        name    = "Type",
        get     = object_type_get,
        set     = object_type_set,
    )

    # ====================================================================================================
    # Axis object parameters

    axis_source: EnumProperty(
        items = [
            ('X', 'X', "3D axis X"),
            ('Y', 'Y', "3D axis Y"),
            ('Z', 'Z', "3D axis Z"),
            ],
        name    = "3D Axis",
        default = 'Z',
    )

    # Target as letter mode
    axis_target: EnumProperty(
        items = [
            ('X', 'X', "4D axis X"),
            ('Y', 'Y', "4D axis Y"),
            ('Z', 'Z', "4D axis Z"),
            ('W', 'W', "4D axis W"),
            ],
        name    = "4D Axis",
        default = 'W',
    )

    # Target as 4D vector : NOT USED (more complex)
    axis_target4D: FloatVectorProperty(
        name    = "4D Axis",
        size    = 4,
        description = "Axis in 4D space",
        default = (0., 0., 0., 1.),
    )

    # ====================================================================================================
    # 4D location management
    # Location 4D is stored in the location4D attribute
    # - GET
    #       In 3D mode, it reads the Blender true location before returning the value
    #       In 4D mode, it returns the value without updating from blender
    # - SET
    #       In 3D mode, Blender location is updated
    #       In 4D mode, projection is updated

    # ----------------------------------------------------------------------------------------------------
    # Location 4D cache

    location4D_: FloatVectorProperty(
        size        = 4,
        default     = (0., 0., 0., 0.)
    )

    # ----------------------------------------------------------------------------------------------------
    # Getter

    def location4D_get(self):

        if self.vmode == '3D':
            map = self.mapping
            for i in range(3):
                self.location4D_['XYZW'.index(map[i])] = self.id_data.location[i]

        return self.location4D_

    # ----------------------------------------------------------------------------------------------------
    # Getter

    def location4D_set(self, value):

        self.location4D_ = value

        if self.vmode == '3D':
            map = self.mapping
            self.id_data.location = [self.location4D_['XYZW'.index(map[i])] for i in range(3)]
        else:
            self.projection()

    # ----------------------------------------------------------------------------------------------------
    # The resulting location 4D

    location4D: FloatVectorProperty(
        name        = "4D location",
        description = "4D location",
        subtype     = 'XYZ_LENGTH',
        size        = 4,
        get         = location4D_get,
        set         = location4D_set
    )

    # ====================================================================================================
    # Rotation management
    # In 3D, Euler is a 3-vector, each angle representing a rotation around one of the 3 axis X, Y
    # In 4D, Euler is a 6-vector, each angle representing a rotation within one of the 6 planes YZ, ZX, XY, XW, YZ, YW, ZW
    # - 3 first planes: YZ, ZX, XY map the Euler rotations around the axis
    # - 3 last planes:  XW, YW, ZW rotation of one 3D axis towards W
    # For a given mapping triplet, we need to map the 3 angles Euler3 in 3 of the Euler6

    # ----------------------------------------------------------------------------------------------------
    # Return the plane code of a given index

    def euler6_plane_from_3D_axis(self, index):
        # index is 0, 1 or 2
        mapping = self.mapping
        if index == 0:
            plane = mapping[1:]
        elif index == 1:
            plane = mapping[2] + mapping[0]
        else:
            plane = mapping[:2]

        return plane

    # ----------------------------------------------------------------------------------------------------
    # To ensure the consistency of the algorithm, write once
    # direction can be 3_TO_6 or 6_TO_3

    def euler3_euler6_move(self, direction='3_TO_6'):
        mapping = self.mapping
        emap    = euler_mapping[mapping]
        for i in range(3):
            i6 = emap.i_planes[i]
            if direction == '3_TO_6':
                self.rotation4D_[i6] = self.id_data.rotation_euler[i] * emap.signs[i]
            else:
                self.id_data.rotation_euler[i] = self.rotation4D_[i6] * emap.signs[i]

    # ----------------------------------------------------------------------------------------------------
    # Rotation 4D cache

    rotation4D_: FloatVectorProperty(
        size        = 6,
        default     = (0., 0., 0., 0., 0., 0.)
    )

    # ----------------------------------------------------------------------------------------------------
    # Update rotation4D

    def update_rotation4D(self):
        if self.vmode == '3D':
            self.euler3_euler6_move(direction='3_TO_6')

    def rotation4D_changed(self):
        if self.vmode == '3D':
            self.euler3_euler6_move(direction='6_TO_3')
        else:
            self.projection()

    # ----------------------------------------------------------------------------------------------------
    # Rotation 4D get

    def rotation4D_get(self):
        self.update_rotation4D()
        return self.rotation4D_

    # ----------------------------------------------------------------------------------------------------
    # Rotation 4D set

    def rotation4D_set(self, value):
        self.rotation4D_ = value
        self.rotation4D_changed()

    # ----------------------------------------------------------------------------------------------------
    # The resulting rotation4D attribute

    rotation4D: FloatVectorProperty(
        name        = "4D rotation",
        description = "4D rotation",
        subtype     = 'EULER',
        size        = 6,
        get         = rotation4D_get,
        set         = rotation4D_set
    )

    # ----------------------------------------------------------------------------------------------------
    # The 6 angles are displayed in two euler3

    def euler6_XYZ_get(self):
        self.update_rotation4D()
        return self.rotation4D_[:3]

    def euler6_XYZ_set(self, value):
        self.rotation4D_[:3] = value
        self.rotation4D_changed()

    def euler6_W_get(self):
        self.update_rotation4D()
        return self.rotation4D_[3:]

    def euler6_W_set(self, value):
        self.rotation4D_[3:] = value
        self.rotation4D_changed()

    euler6_XYZ: FloatVectorProperty(
        name        = "Rotation XYZ",
        description = "Euler rotation in 3D XYZ space",
        size        = 3,
        subtype     = 'EULER',
        get         = euler6_XYZ_get,
        set         = euler6_XYZ_set,
    )

    euler6_W: FloatVectorProperty(
        name        = "Rotation W",
        description = "Euler rotation in a plane including axis W",
        size        = 3,
        subtype     = 'EULER',
        get         = euler6_W_get,
        set         = euler6_W_set,
    )

    # ====================================================================================================
    # Mesh methods and properties
    # 4D coordinates of vertices are stores in 4 layers

    # ----------------------------------------------------------------------------------------------------
    # Access to mesh
    # return None if the object is not a MESH

    @property
    def mesh(self):
        data = self.id_data.data
        if data is None:
            return None
        if type(data).__name__ != 'Mesh':
            return None

        return data

    # ----------------------------------------------------------------------------------------------------
    # Vertices count

    @property
    def verts_count(self):
        mesh = self.mesh
        if mesh is None:
            return 0
        else:
            return len(mesh.vertices)

    # ----------------------------------------------------------------------------------------------------
    # Access to mesh vertices

    @property
    def verts3(self):
        mesh = self.mesh
        if mesh is None:
            return None

        count  = self.verts_count
        verts3 = np.empty(count*3, np.float)
        mesh.vertices.foreach_get("co", verts3)

        return verts3.reshape((count, 3))

    @verts3.setter
    def verts3(self, value):
        mesh = self.mesh
        if mesh is None:
            return

        count  = self.verts_count
        verts3 = np.array(value).reshape(count*3)
        mesh.vertices.foreach_set("co", verts3)

        self.id_data.update_tag()

    # ----------------------------------------------------------------------------------------------------
    # 4D vertices are stored in 4 vertex layers

    def create_layers(self, init=True):
        mesh = self.mesh
        if mesh is None:
            return None

        for axis in ['X', 'Y', 'Z', 'W']:
            mesh.vertex_layers_float.new(name=layer_prefix + axis)

        if init:
            self.mesh_to_layers()

    # ----------------------------------------------------------------------------------------------------
    # Get a float layer

    def get_layer(self, axis):
        mesh = self.mesh
        if mesh is None:
            return None

        name = layer_prefix + axis
        layer = mesh.vertex_layers_float.get(name)

        return layer

    # ----------------------------------------------------------------------------------------------------
    # Access to vertex layers floats
    # Axis : X, Y, Z or W

    def get_floats(self, axis):
        layer = self.get_layer(axis)

        count = len(layer.data)
        vals  = np.empty(count, np.float)
        layer.data.foreach_get("value", vals)

        return vals

    def set_floats(self, axis, value):
        layer = self.get_layer(axis)

        count = len(layer.data)
        vals = np.array(value).reshape(count)

        layer.data.foreach_set("value", vals)

    # ----------------------------------------------------------------------------------------------------
    # mesh <--> layers

    def mesh_to_layers(self):
        mapping = self.mapping
        verts3  = self.verts3
        if verts3 is None:
            return

        for i in range(3):
            self.set_floats(mapping[i], verts3[:, i])

    def layers_to_mesh(self):
        mapping = self.mapping
        count   = self.verts_count
        if count == 0:
            return

        verts3  = np.zeros((count, 3), np.float)
        for i in range(3):
            verts3[:, i] = self.get_floats(mapping[i])
        self.verts3 = verts3

    # ----------------------------------------------------------------------------------------------------
    # Access to the 4D vertices

    @property
    def verts4(self):
        count = self.verts_count
        if count == 0:
            return

        verts = np.zeros((count, 4), np.float)
        if self.vmode == "3D":
            self.mesh_to_layers()

        verts[:, 0]  = self.get_floats('X')
        verts[:, 1]  = self.get_floats('Y')
        verts[:, 2]  = self.get_floats('Z')
        verts[:, 3]  = self.get_floats('W')

        return verts

    @verts4.setter
    def verts4(self, value):
        count = self.verts_count
        if count == 0:
            return

        verts = np.array(value).reshape((count, 4))

        self.set_floats('X', verts[:, 0])
        self.set_floats('Y', verts[:, 1])
        self.set_floats('Z', verts[:, 2])
        self.set_floats('W', verts[:, 3])

        if self.vmode == "3D":
            self.layers_to_mesh()

    # ----------------------------------------------------------------------------------------------------
    # Fourth layer can be addressed directly

    @property
    def fourth_axis(self):
        if self.vmode != '3D':
            return
        mesh = self.mesh
        if mesh is None:
            return

        # Ok, let's go
        map = self.mapping
        for i in range(4):
            axis = 'XYZW'[i]
            if not axis in map:
                return axis

        raise RuntimeError("No fourth axis ????!!!!!, map=", map)

    @property
    def fourth_floats(self):
        return self.get_floats(self.fourth_axis)

    @fourth_floats.setter
    def fourth_floats(self, value):
        self.set_floats(self.fourth_axis, np.array(value).reshape(np.size(value)))


    # ====================================================================================================
    # Curve methods and property
    #
    # A curve is not edited in Blender, it is computed
    # - Either by an expression
    # - Or externally by setting the curve_verts4 array

    # ----------------------------------------------------------------------------------------------------
    # How the vertices are defined
    # return None if the object is not a CURVE

    curve_source_: IntProperty(default=0)

    def curve_source_get(self):
        return self.curve_source_

    def curve_source_set(self, value):
        self.curve_source_ = value
        self.curve_update()

    curve_source: EnumProperty(
        name        = "Source",
        items       = [
            ('EXPRESSION', "Expression", "Curve is computed with the provided python expression."),
            ('EXTERN',     "Extern",     "4D vertices are set externally by setting the property object.d4.curve_verts4"),
        ],
        description = "Define how the curve points are set.",
        get         = curve_source_get,
        set         = curve_source_set,
        )

    # ----------------------------------------------------------------------------------------------------
    # Access to curve
    # return None if the object is not a CURVE

    @property
    def curve(self):
        data = self.id_data.data
        if data is None:
            return None
        if type(data).__name__ != 'Curve':
            return None

        return data

    # ----------------------------------------------------------------------------------------------------
    # The function to compute the points

    curve_function: StringProperty(
        name        = "Function",
        default     = "(cos(2*pi*t), sin(2*pi*t), 1, 2*t)",
        description = "A valid python expression returning 4 floats from the parameter t",
    )

    # ----------------------------------------------------------------------------------------------------
    # Function expression is valid when error is None

    @property
    def curve_function_error(self):

        def f(t):
            frame = bpy.context.scene.frame_current
            return np.array(eval(self.curve_function))

        try:
            v = f(0.)
            if v.size != 4:
                return f"Must return 4 floats, not '{np.array(v).shape}'"

        except:
            return str(sys.exc_info()[1])

        return None

    # ----------------------------------------------------------------------------------------------------
    # Function parameters: t min, t max and number of points

    curve_t0: FloatProperty(
        name        = "t min",
        description = "Starting value for t parameter",
        default     = 0.,
    )
    curve_t1: FloatProperty(
        name        = "t max",
        description = "Ending value for t parameter",
        default     = 1.,
    )

    curve_count_: IntProperty(default = 100)

    def curve_count_get(self):
        if self.curve_source == 'EXPRESSION':
            return self.curve_count_
        else:
            return len(self.get_sk(self.sk_names[0]).data)

    def curve_count_set(self, value):
        self.curve_count_ = value

    curve_count: IntProperty(
        name        = "Points",
        description = "Number of points on the curve",
        min         = 2,
        max         = 10000,
        get         = curve_count_get,
        set         = curve_count_set,
    )

    # ----------------------------------------------------------------------------------------------------
    # The 4D coordinates of the vertices are stored in two shape key
    #
    # If the source is EXPRESSION, use of shape keys is optional

    # ------ Define is shape keys are used

    use_shape_keys_: BoolProperty(default=True)

    def use_shape_keys_get(self):
        if self.curve_source == 'EXTERN':
            return True
        else:
            return self.use_shape_keys_

    def use_shape_keys_set(self, value):
        self.use_shape_keys_ = value
        if not self.use_shape_keys_:
            self.id_data.shape_key_clear()

    use_shape_keys: BoolProperty(
        name        = "Use SK",
        description = "Use shape keys as cache for 4D vertices.",
        get         = use_shape_keys_get,
        set         = use_shape_keys_set,
    )

    # ------ The 3 necessary names

    sk_names = ["4D Curve", "4D XYZ", "4D W"]

    # ------ The shape keys are available

    @property
    def sk_ok(self):
        sks = self.curve.shape_keys
        if sks is None:
            return False

        for name in self.sk_names:
            if self.curve.shape_keys.key_blocks.get(name) is None:
                return False

        return True

    # ----- Clear the shape keys when the number of points change

    def sk_clear(self):
        self.id_data.shape_key_clear()
        return

        sks = self.curve.shape_keys
        if sks is None:
            return

        for name in self.sk_names:
            sk = self.curve.shape_keys.key_blocks.get(name)
            if sk is not None:
                self.id_data.shape_key_remove(sk)

    # ----------------------------------------------------------------------------------------------------
    # Adjust the number of vertices
    # To be done when having the verts4 available

    def curve_adjust_count(self, count):

        spline = self.curve.splines[0]
        bezier = spline.type == 'BEZIER'

        points = spline.bezier_points if bezier else spline.points

        # The current count
        cur = len(points)

        # Ok: nothing to do
        if cur == count:
            return

        # Adjust the number of points
        curve = self.curve
        if cur > self.curve_count:
            curve.splines.new('BEZIER' if bezier else 'NURBS')
            curve.splines.remove(spline)
            spline = curve.splines[0]
            points = spline.bezier_points if bezier else spline.points
            cur = len(points)

        if cur < self.curve_count:
            points.add(self.curve_count - cur)

        # Invalidate the shape keys
        self.sk_clear()

    # ----------------------------------------------------------------------------------------------------
    # Get a shape key by its name

    def get_sk(self, name):
        curve = self.curve
        if curve is None:
            return None

        sks = curve.shape_keys
        if sks is None:
            for nm in self.sk_names:
                sk = self.id_data.shape_key_add(name=nm)
                sk.mute = nm != self.sk_names[0]

        curve.shape_keys.use_relative = False

        sk = curve.shape_keys.key_blocks.get(name)
        if sk is None:
            sks.shape_key.add(name)
            sk = curve.shape_keys.key_blocks.get(name=name)

        return sk

    # ----------------------------------------------------------------------------------------------------
    # Get / set the vertices from the shapes keys

    def get_sk_verts3(self, name):
        sk = self.get_sk(name)
        count = len(sk.data)
        verts = np.empty(count*3, np.float)
        sk.data.foreach_get("co", verts)
        return verts.reshape(count, 3)

    def set_sk_verts3(self, name, verts):
        sk = self.get_sk(name)
        count = len(sk.data)

        # When setting the display curve, need do compute the handles for bezier curvers

        if (name == self.sk_names[0]) and (self.curve.splines[0].type == 'BEZIER'):
            vs, ls, rs = from_points(count, verts)
            sk.data.foreach_set("co", vs.reshape(count*3))
            sk.data.foreach_set("handle_left", ls.reshape(count*3))
            sk.data.foreach_set("handle_right", rs.reshape(count*3))
        else:
            sk.data.foreach_set("co", verts.reshape(count*3))

    # ----------------------------------------------------------------------------------------------------
    # Get the 4D vertices from the shape keys

    def get_sk_verts4(self):
        vertsw        = self.get_sk_verts3(self.sk_names[2])
        count         = len(vertsw)
        verts4        = np.zeros((count, 4), np.float)
        verts4[:, 3]  = vertsw[:, 0]
        verts4[:, :3] = self.get_sk_verts3(self.sk_names[1])
        return verts4

    # ----------------------------------------------------------------------------------------------------
    # Set the 4D vertices to the shape keys

    def set_sk_verts4(self, verts4):

        count  = len(verts4)

        # Ensure the number of vertices is ok
        self.curve_count = count
        self.curve_adjust_count(count)

        # Set the vertices in the layers
        self.set_sk_verts3(self.sk_names[1], verts4[:, :3]) # 4D XYZ

        vertsw       = np.zeros((count, 3), np.float)
        vertsw[:, 0] = verts4[:, 3]
        self.set_sk_verts3(self.sk_names[2], vertsw)  # 4D W

    # ----------------------------------------------------------------------------------------------------
    # Get the 4D vertices
    # If the shape_keys are ok, read from the vertices

    @property
    def curve_verts4(self):

        # ----- Read from cache in shape keys
        if self.use_shape_keys:
            if self.sk_ok:
                return self.get_sk_verts4()

        # ----- Function error
        if self.curve_function_error is not None:
            return np.ones((self.curve_count, 4), np.float)/self.curve_count

        # ----- Compute the vertices
        def f(t):
            # 'frame' is a possible value in the expression
            frame = bpy.context.scene.frame_current
            return np.array(eval(self.curve_function))

        verts4 = np.empty((self.curve_count, 4), np.float)
        ts = np.linspace(self.curve_t0, self.curve_t1, self.curve_count)
        for i in range(self.curve_count):
            verts4[i] = f(ts[i])

        # ----- Write to cache
        if self.use_shape_keys:
            self.set_sk_verts4(verts4)

        # ----- return the vertices
        return verts4

    # ----------------------------------------------------------------------------------------------------
    # Set the 4D vertices
    # Can be used externally to compute the vertices externally

    @curve_verts4.setter
    def curve_verts4(self, value):
        verts4 = np.array(value)
        count  = verts4.size // 4
        self.set_sk_verts4(np.array(verts4).reshape(count, 4))
        self.object_update()

    # ----------------------------------------------------------------------------------------------------
    # set curve 3D vertices

    def set_curve_verts(self, verts3):

        # ---- If shape keys, simply use the curve method

        if self.use_shape_keys:
            self.set_sk_verts3(self.sk_names[0], verts3)

        # ---- Otherwise write the splines points

        else:
            curve = self.curve
            if curve is None:
                return

            spline = curve.splines[0]
            bezier = spline.type == 'BEZIER'

            points = spline.bezier_points if bezier else spline.points

            # The number of points must be adjusted before calling !!
            count = len(verts3)

            if bezier:
                vs, ls, rs = from_points(count, verts3)
                spline.bezier_points.foreach_set("co", vs.reshape(count*3))
                spline.bezier_points.foreach_set("handle_left", ls.reshape(count*3))
                spline.bezier_points.foreach_set("handle_right", rs.reshape(count*3))
            else:
                vs = np.resize(verts3.transpose(), (4, count)).transpose()
                vs[:, 3] = 1.
                spline.points.foreach_set("co", vs.reshape(count * 4))

        # ----- Mark update

        self.id_data.update_tag()

    # ----------------------------------------------------------------------------------------------------
    # Update the curve after changes

    def curve_update(self, force_compute=False):
        if self.curve is None:
            return

        # Force the computation of the vertices
        if force_compute and (self.curve_source == 'EXPRESSION'):
            self.sk_clear()

        if self.vmode == '3D':
            verts4 = self.curve_verts4
            verts3 = np.empty((len(verts4), 3), np.float)
            map    = self.mapping

            for i in range(3):
                i4 = 'XYZW'.index(map[i])
                verts3[:, i] = verts4[:, i4]

            self.set_curve_verts(verts3)
        else:
            self.projection()

    # ====================================================================================================
    # Update

    # ----------------------------------------------------------------------------------------------------
    # object <--> space4

    def object_to_space4D(self):

        # Read loc and rot to force the update
        loc = self.location4D
        rot = self.rotation4D

        # Mesh to layers
        self.mesh_to_layers()

    def space4D_to_object(self):

        self.lock_update()

        # Force loc and rot
        self.location4D = self.location4D_
        self.rotation4D = self.rotation4D_

        self.unlock_update()

    # ----------------------------------------------------------------------------------------------------
    # Need an update
    # After 3D and 4D are synchronized

    update_stack = 0

    def lock_update(self):
        self.update_stack += 1

    def unlock_update(self):
        self.update_stack -= 1
        if self.update_stack < 0:
            raise RuntimeError("Severe programmation error when managing updates!!!! >:(E(")

        if self.update_stack == 0:
            self.object_update()

    def object_update(self):

        if self.update_stack != 0:
            return

        if self.vmode == '3D':
            # Mesh update
            self.layers_to_mesh()

            # Curve update
            self.curve_update()

        else:
            self.projection()

    # ====================================================================================================
    # Projection is given by the scene management
    # A copy is kept in order to be able to perform updates when location or rotation are changed

    # ----------------------------------------------------------------------------------------------------
    # Copy of the last projection matrix

    last_projection: FloatVectorProperty(
        size    = 12,
        default = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0],
        )

    # ----------------------------------------------------------------------------------------------------
    # Projection

    def projection(self, mproj=None):

        if self.update_stack != 0:
            return

        # Pass in 4D mode
        self.vmode = "4D"

        # ----- Get the projection matrix from scene if None
        if mproj is None:
            mproj = bpy.context.scene.d4.projection_matrix

        # ----- Location projection

        loc4D = self.location4D
        loc3D = mproj.dot(loc4D)
        self.id_data.location = loc3D

        # ----- AXIS

        if self.object_type == 'AXIS':
            axis = np.array({'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[self.axis_source], np.float)
            vector_mode = False
            if vector_mode:
                targ4 = np.array(self.axis_target4D)
            else:
                targ4 = np.array({'X':(1,0,0,0),'Y':(0,1,0,0),'Z':(0,0,1,0), 'W':(0,0,0,1)}[self.axis_target], np.float)

            if np.linalg.norm(targ4) < 0.001:
                targ4 = np.array((0, 0, 0, 1), np.float)
            targ3 = mproj.dot(targ4)
            targ3 = targ3 / np.linalg.norm(targ3)

            v0 = Vector(axis)
            v1 = Vector(targ3)
            ag = v0.angle(v1)
            if abs(ag) < 0.001:
                euler = Euler((0, 0, 0))
            else:
                quat = Quaternion(v0.cross(v1), ag)
                euler = quat.to_euler('XYZ')

            self.id_data.rotation_euler = euler

        # ----- Mesh projection

        if self.object_type == 'SHAPE':

            # Rotation matrix to be applied before projection
            rot4D = euler6_to_matrix(self.rotation4D)  # 4x4
            mproj = mproj.dot(rot4D)                   # 3x4

            # A mesh
            if self.mesh is not None:
                # Vertices projection
                verts4 = self.verts4 - loc4D
                self.verts3 = mproj.dot(verts4.transpose()).transpose() - loc3D

            # A curve
            if self.curve is not None:

                # Vertices projection
                verts4 = self.curve_verts4 - loc4D
                verts3 = mproj.dot(verts4.transpose()).transpose() - loc3D

                self.set_curve_verts(verts3)


# ======================================================================================================================================================
# 4D Scene extension settings

class D4Scene(bpy.types.PropertyGroup):

    # ----------------------------------------------------------------------------------------------------
    # Toggle display 4D / 3D

    d4_projection_: BoolProperty(default=False) = False

    def d4_projection_get(self):
        return self.d4_projection_

    def d4_projection_set(self, value):
        self.d4_projection_ = value
        self.update_objects()
        # Grids
        bpy.ops.spaceview3d.showgridbutton()

    d4_projection: BoolProperty(
        name        = "Display 4D",
        description = "Display 4D projection or 3D slice",
        set         = d4_projection_set,
        get         = d4_projection_get,
    )

    # ----------------------------------------------------------------------------------------------------
    # Mapping
    # Three of the four XYZW axis are mapped on the XYZ blender axis using a triplet such as 'XYW'

    mapping_: IntProperty(default=0)

    def mapping_get(self):
        return self.mapping_

    def mapping_set(self, value):
        self.mapping_ = value
        self.update_objects()

    mapping: EnumProperty(
        items=[
            ('XYZ', "XYZ", "W as hidden axis"),
            ('XYW', "XYW", "Z as hidden axis"),
            ('XWZ', "XWZ", "Y as hidden axis"),
            ('WYZ', "WYZ", "X as hidden axis"),
        ],
        set=mapping_set,
        get=mapping_get,
    )

    # ----------------------------------------------------------------------------------------------------
    # Update the objects of the scene

    def update_objects(self, objects=None):
        scene = self.id_data
        if objects is None:
            objects = scene.objects

        if self.d4_projection:
            mproj = self.projection_matrix

        for obj in objects:
            if obj.d4.is4D:
                if self.d4_projection:
                    obj.d4.projection(mproj)
                    obj.d4.mapping = self.mapping
                else:
                    obj.d4.mapping = self.mapping
                    obj.d4.vmode = '3D'

    # ----------------------------------------------------------------------------------------------------
    # Projection mode

    projection_mode_: IntProperty(default=0) = 0

    def projection_mode_get(self):
        return self.projection_mode_

    def projection_mode_set(self, value):
        self.projection_mode_ = value
        self.update_projection()

    projection_mode: EnumProperty(
        items=[
            ('ANGLES', "Angles", "Define 3 angles"),
            ('VECTOR', "Vector", "Along a vector"),
        ],
        set=projection_mode_set,
        get=projection_mode_get,
    )

    # ----------------------------------------------------------------------------------------------------
    # Vector mode : the projection vector

    projection_vector_: FloatVectorProperty(
        size    = 4,
        default = (10.0, 10.0, 0.0, 10.0),
    )

    def projection_vector_get(self):
        return self.projection_vector_

    def projection_vector_set(self, value):
        self.projection_vector_ = value
        self.update_projection()

    projection_vector: FloatVectorProperty(
        name        = "Vector",
        description = "Projection axis",
        subtype     = 'XYZ',
        size        = 4,
        get         = projection_vector_get,
        set         = projection_vector_set,
    )

    # ----------------------------------------------------------------------------------------------------
    # Angles mode : the rotation angles

    projection_angles_: FloatVectorProperty(
        size    = 3,
        default = (0.0, 0.0, 0.0),
    )

    def projection_angles_get(self):
        return self.projection_angles_

    def projection_angles_set(self, value):
        self.projection_angles_ = value
        self.update_projection()

    projection_angles: FloatVectorProperty(
        name        = "Angles",
        description = "Projection angles along XY, YZ, ZW",
        subtype     = 'EULER',
        size        = 3,
        step        = 100,
        get         = projection_angles_get,
        set         = projection_angles_set,
    )

    # ----------------------------------------------------------------------------------------------------
    # 4D --> 3D projection matrix

    @property
    def projection_matrix(self):

        # ----- VECTOR projection

        if self.projection_mode == 'VECTOR':
            return matrix_4D_axis4(self.projection_vector)

        # ---------------------------------------------------------------------------
        # Projection according angles

        elif self.projection_mode == 'ANGLES':
            angles = self.projection_angles
            return matrix_4D_uvw(angles[0], angles[1], angles[2])

        else:
            print("Unknown projection code : ", self.projection_mode)
            return None

    # ----------------------------------------------------------------------------------------------------
    # Something changed in the projection parameters --> need to update the projection

    def update_projection(self):
        if not self.d4_projection:
            return
        self.update_objects()

    # ====================================================================================================
    # Grid management

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

    show_floor_3D:  BoolProperty(default=True)
    show_axis_x_3D: BoolProperty(default=True)
    show_axis_y_3D: BoolProperty(default=True)
    show_axis_z_3D: BoolProperty(default=False)

    show_4D_grids_: BoolProperty(default=False)

    def show_4D_grids_get(self):
        return self.show_4D_grids_

    def show_4D_grids_set(self, value):
        self.show_4D_grids_ = value
        bpy.ops.spaceview3d.showgridbutton()

    show_4D_grids: BoolProperty(
        name="4D grids",
        get=show_4D_grids_get,
        set=show_4D_grids_set,
        description="Show / hide the 4D grids"
    )

# ======================================================================================================================================================
# 4D settings scene panel

class D4ScenePanel(bpy.types.Panel):
    """4D extension"""

    bl_label        = "4D projection"
    bl_idname       = "D4_PT_SCENE"
    bl_space_type   = 'VIEW_3D'
    bl_region_type  = 'UI'
    bl_category     = '4D'
    bl_order        = 1

    def draw(self, context):

        layout = self.layout

        obj = context.active_object
        if (obj is not None) and (obj.mode == 'EDIT'):
            layout.label(text="Edit mode")
            return

        settings = context.scene.d4

        # ----- 4D Display switch

        # 4D Display switch
        mbox = layout.box()
        row = mbox.row(align=True)
        row.prop(settings, "d4_projection", toggle=True)

        # ----- 4D projection parameters

        if settings.d4_projection:

            # Projection parameters
            mbox.label(text="Projection parameters")
            row = mbox.row(align=True)
            row.prop_enum(settings, "projection_mode", 'VECTOR')
            row.prop_enum(settings, "projection_mode", 'ANGLES')

            if settings.projection_mode == 'VECTOR':
                row = mbox.column()
                row.prop(settings, "projection_vector")

            if settings.projection_mode == 'ANGLES':
                row = mbox.column()
                row.prop(settings, "projection_angles")

        # ----- Mapping

        else:
            box = layout.box()
            box.label(text="4D axis mapping")
            row = box.row(align=True)
            row.prop_enum(settings, "mapping", 'XYZ')
            row.prop_enum(settings, "mapping", 'XYW')
            row.prop_enum(settings, "mapping", 'XWZ')
            row.prop_enum(settings, "mapping", 'WYZ')

        # ----- 4D grids

        # show 4D grids switch
        # The button is not used directly but though the toggle boolean

        if settings.d4_projection:
            mbox = layout.box()

            row = mbox.row(align=True)
            row.prop(settings, "show_4D_grids", toggle=True)

            mbox.label(text="Display grids")
            row = mbox.row(align=True)
            row.prop(settings, "show_grid_XY", toggle=True)
            row.prop(settings, "show_grid_YZ", toggle=True)
            row.prop(settings, "show_grid_ZW", toggle=True)
            row.prop(settings, "show_grid_WX", toggle=True)

# ============================================================================================================================
# 4D settings object panel

# -----------------------------------------------------------------------------------------------------------------------------
# Button to refresh the 4D curve vertices

class CurveComputeOperator(bpy.types.Operator):
    """Compute the 4D vertices of curve"""
    bl_idname = "object.curve_compute_operator"
    bl_label = "Compute the vertices of the 4D curve"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        obj = context.active_object
        if obj is not None:
            obj.d4.curve_update(True)

        return {'FINISHED'}


# -----------------------------------------------------------------------------------------------------------------------------
# Drawing 4D location in Item > Transform panel

def draw_location4D(self, context):
    obj = context.object
    if obj is None:
        return

    if obj.mode != 'EDIT':
        layout = self.layout
        layout.prop(obj.d4, "is4D")

        if not obj.d4.is4D:
            return

        # Object type
        box = layout.box()
        row = box.row(align=True)
        row.prop(obj.d4, "object_type", expand=True)

        if (obj.d4.vmode == '3D') and (obj.d4.object_type == 'AXIS'):
            #box.label(text="Axis orientation")
            row = box.row()
            row.prop(obj.d4, "axis_source", expand=True)

            row = box.row()
            row.prop(obj.d4, "axis_target", expand=True)

            #box.prop(obj.d4, "axis_target4D")

        # ----- Location

        col = layout.column()
        col.prop(obj.d4, "location4D")

        # ----- Rotation
        if obj.d4.object_type == 'SHAPE':
            box = layout.box()
            box.label(text="Rotation 4D")
            col = box.column()
            col.prop(obj.d4, "euler6_XYZ", text="3D")
            col.prop(obj.d4, "euler6_W", text="W_")

# -----------------------------------------------------------------------------------------------------------------------------
# Dedicated Object 4D panel

class D4ObjectPanel(bpy.types.Panel):
    """Creates a 4D Panel in the Object properties window"""
    bl_label       = "4D parameters"
    bl_idname      = "OBJECT_PT_D4"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "object"

    def draw(self, context):
        draw_location4D(self, context)

        layout = self.layout
        obj = context.object

        if (obj.d4.curve is not None) and (obj.d4.object_type == 'SHAPE'):
            row = layout.row()
            row.prop(obj.d4, "curve_source", expand=True)

            if obj.d4.curve_source == 'EXTERN':
                box = layout.box()

                row = box.row(align=True)
                row.alignment = 'EXPAND'
                row.label(text="Use python to set the vertices:")

                row = box.row(align=True)
                row.alignment = 'EXPAND'
                row.label(text="object.d4.curve_verts4 = array of shape (n, 4)")

            else:
                row = layout.row()
                row.label(text="Curve expression")
                row = layout.row()
                row.prop(obj.d4, "curve_function")
                msg = obj.d4.curve_function_error
                if msg is not None:
                    box = layout.box()
                    box.label(text="Expression not valid:")
                    box.label(text=msg)
                else:
                    layout.prop(obj.d4, "curve_count")
                    layout.prop(obj.d4, "curve_t0")
                    layout.prop(obj.d4, "curve_t1")
                    layout.prop(obj.d4, "use_shape_keys")
                    layout.operator(CurveComputeOperator.bl_idname, text="Compute")

# ======================================================================================================================================================
# Dessin du view port en 4D

viewport_shader = gpu.shader.from_builtin('3D_SMOOTH_COLOR')

# -----------------------------------------------------------------------------------------------------------------------------
# Drawing hook

def draw_callback_3d(self, context):
    """Draw a grid for a plane in the 4D space"""

    # Could be optimzed

    # Axis colors

    axis_color = {'X': (1., 0., 0., 1.), 'Y': (0., 1., 0., 1.), 'Z': (0., 0., 1., 1.), 'W': (1., 1., 1., 1.)}

    # Return the color of an axis
    def get_color(axis, alpha=1.):
        col = axis_color[axis]
        return [c * alpha for c in col]

    # Projection matrix
    settings = context.scene.d4
    overlay  = context.space_data.overlay
    mproj    = settings.projection_matrix

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

            return Vector(mproj.dot(v))

        verts = []
        cols  = []
        col   = get_color(plane[1], 0.5)
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
        verts.extend([Vector(mproj.dot(P0)), Vector(mproj.dot(P1))])

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
            ShowGridButton._handle = bpy.types.SpaceView3D.draw_handler_add(
                draw_callback_3d, (self, context), 'WINDOW', 'POST_VIEW')
            # context.scene.d4_settings.show_4D_grids = True

    # ------------------------------------
    # Disable gl drawing removing handler
    # ------------------------------------

    @staticmethod
    def handle_remove(self, context):
        if ShowGridButton._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(ShowGridButton._handle, 'WINDOW')
        ShowGridButton._handle = None
        # context.scene.d4_settings.show_4D_grids = False

    # ------------------------------
    # Execute button action
    # ------------------------------

    def execute(self, context):
        if context.area.type == 'VIEW_3D':
            overlay  = context.space_data.overlay
            settings = context.scene.d4

            if settings.d4_projection and settings.show_4D_grids:

                # Save 3D display options
                settings.show_floor_3D  = overlay.show_floor
                settings.show_axis_x_3D = overlay.show_axis_x
                settings.show_axis_y_3D = overlay.show_axis_y
                settings.show_axis_z_3D = overlay.show_axis_z

                # Hide the standard grids
                overlay.show_floor  = False
                overlay.show_axis_x = False
                overlay.show_axis_y = False
                overlay.show_axis_z = False

                self.handle_add(self, context)
                # self.handle_remove(self, context)
                context.area.tag_redraw()

            else:
                self.handle_remove(self, context)

                # Restore 3D display options
                overlay.show_floor  = settings.show_floor_3D
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
# Enable 4D

def d4_enabled():
    return hasattr(bpy.context.scene, "d4")

def enable_4D(force=False):

    if d4_enabled() and (not force):
        return

    # Extension classes
    bpy.utils.register_class(D4Param)
    bpy.utils.register_class(D4Scene)

    # Extend Scene and Object types
    bpy.types.Scene.d4  = PointerProperty(type=D4Scene)
    bpy.types.Object.d4 = PointerProperty(type=D4Param)

    # 4D grids drawing
    bpy.utils.register_class(ShowGridButton)

    # Panels
    bpy.utils.register_class(CurveComputeOperator)

    bpy.utils.register_class(D4ScenePanel)
    bpy.utils.register_class(D4ObjectPanel)

    # Properties panel extension with Object 4D params
    bpy.types.VIEW3D_PT_context_properties.append(draw_location4D)

    return

    # Menus extension
    bpy.utils.register_class(Object4DAdd)
    bpy.utils.register_class(HypershereAdd)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)

def disable_4D():

    if not d4_enabled():
        return

    # Extension classes
    bpy.utils.unregister_class(D4Param)
    bpy.utils.unregister_class(D4Scene)

    # Extend Scene and Object types
    #bpy.types.Scene.d4  = PointerProperty(type=D4Scene)
    #bpy.types.Object.d4 = PointerProperty(type=D4Param)

    # 4D grids drawing
    bpy.utils.unregister_class(ShowGridButton)

    # Panels
    bpy.utils.unregister_class(CurveComputeOperator)

    bpy.utils.unregister_class(D4ScenePanel)
    bpy.utils.unregister_class(D4ObjectPanel)

    # Properties panel extension with Object 4D params
    bpy.types.VIEW3D_PT_context_properties.remove(draw_location4D)


