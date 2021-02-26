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
from mathutils import Vector, Matrix, Euler, Quaternion
import collections
import math
from math import cos, sin, asin, atan2, radians, degrees, pi, sqrt

from .wrappers import WObject

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
        return np.array(((c, s, 0, 0), (-s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    elif plane in ['XZ', 'ZX']:
        if plane == 'XZ':
            s = -s
        return np.array(((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1)))
    elif plane in ['YZ', 'ZY']:
        if plane == 'ZY':
            s = -s
        return np.array(((1, 0, 0, 0), (0, c, s, 0), (0, -s, c, 0), (0, 0, 0, 1)))

    elif plane in ['XW', 'WX']:
        if plane == 'WX':
            s = -s
        return np.array(((c, 0, 0, s), (0, 1, 0, 0), (0, 0, 1, 0), (-s, 0, 0, c)))
    elif plane in ['YW', 'WY']:
        if plane == 'WY':
            s = -s
        return np.array(((1, 0, 0, 0), (0, c, 0, -s), (0, 0, 1, 0), (0, s, 0, c)))
    elif plane in ['ZW', 'WZ']:
        if plane == 'WZ':
            s = -s
        return np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, c, s), (0, 0, -s, c)))
    else:
        raise RuntimeError(f"plane_rotation matrix: undefined plane '{plane}'")

# --------------------------------------------------------------------------------------------------
# Rotation matrix from 4D euler

def euler6_to_matrix(euler6):
    m = np.identity(4, np.float)
    for i, plane in enumerate(['YZ', 'ZX', 'XY', 'XW', 'YW', 'ZW']):
        m = m.dot(plane_rotation(plane, euler6[i]))
    return m

# --------------------------------------------------------------------------------------------------
# Compute a projection matrix along a 3D axis and an angle

def matrix_4D_axis3(V, w):
    sph = cart_to_sph3(V)
    return matrix_4D_uvw(sph[1], sph[2], w)

# =============================================================================================================================
# 4D Object extension

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
            self.object_type_ = 2 # Surface

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

        # --> Pass in projection mode

        if self.vmode_ == '3D':
            self.object_to_space4D()
            self.vmode_ = value
            self.id_data.rotation_euler = (0., 0., 0.)

        # --> Pass in design mode

        else:
            self.vmode_ = value
            self.space4D_to_object()
            self.rotation4D_changed()

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
        self.object_to_space4D()
        self.mapping_ = value
        self.space4D_to_object()

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
        self.object_type_ = value
        if self.vmode == '4D':
            self.projection(None)

    object_type: EnumProperty(
        items = [
            ('POINT',   "Point", "The shape of the object is not deformed by the 4D projection, just its location"),
            ('AXIS',    "Axis",  "The vertical axis, with its Blender 3D rotation, is rotated by the 4D projection without deforming the shape"),
            ('SURFACE', "Surf",  "The shape is deformed by the 4D projection"),
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

    # Target as 4D vector (more complex)
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
            self.projection(None)

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
    # The 6 planes array

    @property
    def euler6_planes(self):
        return ['YZ', 'ZX', 'XY', 'XW', 'YW', 'ZW']

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
    # Return plane index in euler6 and orientation of the plane

    def euler6_index(self, plane):
        planes = self.euler6_planes
        try:
            index = planes.index(plane)
            return index, 1
        except:
            pass

        rev_plane = plane[1] + plane[0]
        return planes.index(rev_plane), -1

    # ----------------------------------------------------------------------------------------------------
    # To ensure the consistency of the algorithm, write once
    # direction can be 3_TO_6 or 6_TO_3

    def euler3_euler6_move(self, direction='3_TO_6'):
        mapping = self.mapping
        for i in range(3):
            plane   = self.euler6_plane_from_3D_axis(i)
            i6, one = self.euler6_index(plane)
            if direction == '3_TO_6':
                self.rotation4D_[i6] = one * self.id_data.rotation_euler[i]
            else:
                self.id_data.rotation_euler[i] = one * self.rotation4D_[i6]

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
            self.projection(None)

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
            mesh.vertex_layers_float.new(name="4D " + axis)

        if init:
            self.mesh_to_layers()

    # ----------------------------------------------------------------------------------------------------
    # Get a float layer

    def get_layer(self, axis):
        mesh = self.mesh
        if mesh is None:
            return None

        name = "4D " + axis
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
    # object <--> space4

    def object_to_space4D(self):

        # Read loc and rot to force the update
        loc = self.location4D
        rot = self.rotation4D

        # Mesh to layers
        self.mesh_to_layers()

    def space4D_to_object(self):

        # Force loc and rot
        self.location4D = self.location4D_
        self.rotation4D = self.rotation4D_

        # Layers to mesh
        self.layers_to_mesh()

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

    def projection(self, mproj):

        # Pass in 4D mode
        self.vmode = "4D"

        # ----- Read last projection matrix or save the passed matrix

        if mproj is None:
            mproj = np.array(self.last_projection).reshape(3, 4)
        else:
            self.last_projection = np.array(mproj).reshape(12)

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

        if self.object_type == 'SURFACE':

            # Rotation matrix to be applied before projection

            rot4D = euler6_to_matrix(self.rotation4D)  # 4x4
            mproj = mproj.dot(rot4D)                   # 3x4

            # Vertices projection

            verts4 = self.verts4 - loc4D
            self.verts3 = mproj.dot(verts4.transpose()).transpose() - loc3D


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
        #bpy.ops.spaceview3d.showgridbutton()

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

    def update_objects(self):
        scene = self.id_data

        if self.d4_projection:
            mproj = self.projection_matrix

        for obj in scene.objects:
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

            # Update the projection
            #if settings.d4_display:
            #    mbox.operator("scene.d4_projection")

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
        if obj.d4.object_type == 'SURFACE':
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

def enable_4D():
    bpy.utils.register_class(D4Param)
    bpy.utils.register_class(D4Scene)

    bpy.types.Scene.d4  = PointerProperty(type=D4Scene)
    bpy.types.Object.d4 = PointerProperty(type=D4Param)

    bpy.utils.register_class(ShowGridButton)

   # Panels
    bpy.utils.register_class(D4ScenePanel)
    bpy.utils.register_class(D4ObjectPanel)

    try:
        bpy.types.VIEW3D_PT_context_properties.remove(draw_location4D)
    except:
        pass
    bpy.types.VIEW3D_PT_context_properties.append(draw_location4D)



# ======================================================================================================================================================
# Add a new 4D object


#    """4D extension"""
#
#    bl_label        = "4D parameters"
#    bl_idname       = "d4_PT_Settings"
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

        model.d4_param.extrude_4D(w_min=-1., w_max=1., slices=self.slices)

        # Update 4D display

        if scene.d4_settings.d4_display:
            scene.d4_settings.update_projection()

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

            prm = slice.d4_param
            prm.is4D = True
            prm.type = 'SURFACE'

            if s is not None:
                bms = slice.d4_param.bmesh_begin()
                id4Ds = get_4D_layer(bms)

                for v in bms.verts:
                    v[id4Ds] += s

                slice.d4_param.bmesh_end(bms)

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

        if scene.d4_settings.d4_display:
            scene.d4_settings.update_projection()

        return {'FINISHED'}


# Draw a button in the SpaceView3D Display panel

def view3D_Draw_Extension(self, context):
    self.layout.operator("spaceview3d.showgridbutton")


# ======================================================================================================================================================
# The 4D main panel : a dedicated tab in the view3D panel

class FourDPanel(bpy.types.Panel):
    bl_label = "4D Panel"
    bl_idname = "d4_PT_Config"
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
        obj.d4_param.is4D = True

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
        if scene.d4_settings.d4_display:
            scene.d4_settings.update_projection()

        return {'FINISHED'}


def menu_func(self, context):
    # self.layout.operator(Object4DAdd.bl_idname,   icon='MESH_CUBE')
    self.layout.operator(HypershereAdd.bl_idname, icon='MESH_CUBE')


# ======================================================================================================================================================
# Module register

def register():
    print("\n4D Registering 4D addon")

    # Extension classes
    bpy.utils.register_class(d4_settings)
    bpy.utils.register_class(d4_Param)

    # Operators
    bpy.utils.register_class(ShowGridButton)
    bpy.utils.register_class(FourDProjectionOperator)

    # Type extensions
    bpy.types.Scene.d4_settings = PointerProperty(type=d4_settings)
    bpy.types.Object.d4_param = PointerProperty(type=d4_Param)

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
    bpy.utils.unregister_class(d4_settings)
    bpy.utils.unregister_class(FourDProjectionPanel)
    bpy.utils.unregister_class(d4_Param)
    bpy.utils.unregister_class(FourDParamPanel)

    bpy.types.VIEW3D_PT_view3d_properties.remove(view3D_Draw_Extension)
    bpy.utils.unregister_class(VIEW3D_PT_4d_coord)

    bpy.utils.unregister_class(Object4DAdd)
    bpy.utils.unregister_class(HypershereAdd)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)


if __name__ == "__main__":
    register()

# register()