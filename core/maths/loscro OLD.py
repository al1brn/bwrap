#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:33:50 2022

@author: alain.bernard@loreal.com
"""

"""Matrix, vectors, quaternions and eulers managed in arrays
Implement computations and transformations with arrays of vectors, eulers...
Created: Jul 2020
"""

__author__     = "Alain Bernard"
__copyright__  = "Copyright 2020, Alain Bernard"
__credits__    = ["Alain Bernard"]

__license__    = "GPL"
__version__    = "1.0"
__maintainer__ = "Alain Bernard"
__email__      = "wrapanime@ligloo.net"
__status__     = "Production"


import traceback

from math import pi
import numpy as np
    
# Zero
zero = 1e-6

# ====================================================================================================
# Root for coordinates
#
# Implements:
# - Vectors     : cartesian
# - Spheric     : spherical
# - Cylindric   : cylindircal
# - Quaternions : quaternion

class Coordinates:
    pass


# ====================================================================================================
# Root for array of geometry items
#
# Children must override item_shape if different from (3,)

class GeoArray:
    def __init__(self, values=0., shape=None):
        
        item_shape = self.item_shape()
        
        values = self.other_a(values)
        
        if shape is None:
            if np.shape(values) == ():
                self.a = np.empty((1,) + item_shape, float)
                self.a[:] = values
            else:
                self.a = values
        else:
            self.a = np.empty(self.as_shape(shape) + item_shape, float)
            if self.shape != (0,):
                self.a[:] = values
            
        if np.shape(self.a) == self.item_shape():
            self.reshape((1,))
            
        # ----- Check
        
        if np.shape(self.a)[-len(item_shape):] != item_shape:
            self.error(f"The 'values' argument is shapped {np.shape(values)} wihich is not an array of shapes {item_shape}. a is shapped {np.shape(self.a)}.")

    # ---------------------------------------------------------------------------
    # Item shape. To be overriden if different of (3,)

    @staticmethod
    def item_shape():
        return (3,)

    # ---------------------------------------------------------------------------
    # A parameter can be a GeoArray or something else
    # return the something else or the a property of tge GeoArray
            
    @staticmethod
    def other_a(other):
        return other.a if issubclass(type(other), GeoArray) else other

    # ---------------------------------------------------------------------------
    # Get the shape of a parameter
    
    @classmethod
    def other_shape(cls, other):
        if other is None:
            return None
        elif issubclass(type(other), GeoArray):
            return other.shape
        else:
            shape = np.shape(other)
            n = len(cls.item_shape())
            if len(shape) == n:
                return (1,)
            else:
                return np.shape(other)[:-n]
    
    # ---------------------------------------------------------------------------
    # Return a GeoArray of same type
    
    @classmethod
    def same(cls, other):
        if issubclass(type(other), cls):
            return other
        else:
            return cls(cls.other_a(other))
            
    # ---------------------------------------------------------------------------
    # Shaping and sizing
    
    @staticmethod
    def as_shape(value):
        if type(value) is int:
            return (value,)
        else:
            return value

    @property
    def shape(self):
        return np.shape(self.a)[:-len(self.item_shape())]

    def reshape(self, shape):
        self.a = np.reshape(self.a, shape + self.item_shape())
        if np.shape(self.a) == self.item_shape():
            self.reshape((1,))
        return self
    
    @property
    def size(self):
        return np.product(self.shape).astype(int)
    
    def resize(self, shape):
        self.a = np.resize(self.a, self.as_shape(shape) + self.item_shape())
        if np.shape(self.a) == self.item_shape():
            self.reshape((1,))
        return self
        
    # ---------------------------------------------------------------------------
    # As an array
    
    def __len__(self):
        return len(self.a)
        
    def __getitem__(self, index):
        return type(self)(self.a[index])
    
    # ---------------------------------------------------------------------------
    # Representation
    
    def str_item(self, item):
        return f"{item}"
    
    def str_items(self, indices=None):
        
        if indices is None:
            items = self
        else:
            items = self[indices]
     
        if items.shape == (1,):
            return self.str_item(items.a[0])
        
        s = f" {items.shape} [\n"
        for i in range(len(items)):
            if i < 5 or i >= len(items)-5:
                lines = items.str_items(i).split("\n")
                for line in lines:
                    s += "  " + line + "\n"
            elif i == 5:
                s += f"  ... {items.shape}\n"
        
        return s + " ]"
    
    def __repr__(self):
        
        shape = self.shape
        
        s = f"<{type(self).__name__} {shape}"
        if len(self)== 0:
            pass
        
        elif len(self) == 1:
            if len(self.item_shape()) > 1:
                s += ":\n" + self.str_item(self.a[0])
            else:
                s += ": " + self.str_item(self.a[0])
        else:
            s += self.str_items()

        return s + ">"
    
    # ---------------------------------------------------------------------------
    # Distance
    
    def difference_with(self, other):
        return np.linalg.norm(self.a - self.other_a(other))
    
    # ---------------------------------------------------------------------------
    # Append
    
    def append(self, other):
        
        a = self.other_a(other)

        if np.shape(a) == self.item_shape():
            self.a = np.append(self.a, np.reshape(a, (1,) + self.item_shape()), axis=0)
        else:
            self.a = np.append(self.a, a, axis=0)

        return self
    
    # ---------------------------------------------------------------------------
    # Raise an error
    
    def error(self, message, other=None, **kwargs):
        
        stack = traceback.extract_stack()[-2]
        print('-'*80)
        print("ERROR in npGeometry:", message)
        print()
        print(f"class:      {type(self).__name__}, shape: {self.shape}")
        print(f"item_shape: {self.item_shape()}")
        print(f"Method:     {stack.name} at line {stack.lineno}")
        #print("file:", stack.filename)
        if other is not None:
            try:
                shape = other.shape
            except:
                shape = "None"
                
            print(f"Other:  {type(other).__name__}, shape: {shape}")
        
        if len(kwargs) > 0:
            print()
            for k, v in kwargs:
                print(f"{k:10s}: {v}")
            
        print('-'*80)
        
        raise RuntimeError(message)

    # ---------------------------------------------------------------------------
    # OLD: to be checked
    
    @staticmethod
    def broadcasted_args(a, a_item_shape, b, b_item_shape, a_type=float, b_type=float):
        
        a_shape = np.shape(a)[:len(a_item_shape)]
        b_shape = np.shape(b)[:len(b_item_shape)]
        shape   = np.broadcast_shapes(a_shape, b_shape)
    
        if a_shape == shape:
            ra = a
        else:
            ra = np.empty(shape + a_item_shape, a_type)
            ra[:] = a
            
        if b_shape == shape:
            rb = b
        else:
            rb = np.empty(shape + b_item_shape, b_type)
            rb[:] = b
    
        return ra, rb
    
# ====================================================================================================
# Common to vectors of any dimension

class NVectors(GeoArray):
    
    # ---------------------------------------------------------------------------
    # Vector dimension
    
    @property
    def vdim(self):
        return self.item_shape()[0]

    # ---------------------------------------------------------------------------
    # Display a vector
        
    def str_item(self, item):
        s = ""
        sep = "["
        for v in item:
            s += sep + f"{v:8.3f}"
            sep = " "
        return s + f"], norm: {np.linalg.norm(item):8.3f}"

    # ---------------------------------------------------------------------------
    # Unary operators
    
    def __pos__(self):
        return self.same(self.a)
    
    def __neg__(self):
        return self.same(-self.a)
    
    def __abs__(self):
        return self.same(np.abs(self.a))

    # ---------------------------------------------------------------------------
    # Binary operators
    
    def __add__(self, other):
        return self.same(self.a + self.other_a(other))
        
    def __sub__(self, other):
        return self.same(self.a + self.other_a(other))
        
    def __mul__(self, other):
        return self.same(self.a * self.other_a(other))
        
    def __truediv__(self, other):
        return self.same(self.a / self.other_a(other))
    
    # ---------------------------------------------------------------------------
    # Reflected binary operators
    
    def __radd__(self, other):
        return self.same(self.a + self.other_a(other))
    
    def __rsub__(self, other):
        return self.same(self.other_a(other) - self.a)
    
    def __rmul__(self, other):
        return self.same(self.other_a(other) * self.a)
    
    def __rtruediv__(self, other):
        return self.same(self.other_a(other) / self.a)
    
    # ---------------------------------------------------------------------------
    # Augmented assignment
    
    def __iadd__(self, other):
        self.a += self.other_a(other)
        return self
        
    def __isub__(self, other):
        self.a -= self.other_a(other)
        return self
        
    def __imul__(self, other):
        self.a *= self.other_a(other)
        return self
        
    def __itruediv__(self, other):
        self.a /= self.other_a(other)
        return self

    # ---------------------------------------------------------------------------
    # Norm
        
    def norm(self):
        return np.linalg.norm(self.a, axis=-1)
    
    def norm2(self):
        return np.einsum('...i, ...i', self.a, self.a)

    # ---------------------------------------------------------------------------
    # Normalize
    
    def normalize(self, null_replace=None):
        
        vn = np.linalg.norm(self.a, axis=-1)
        nulls = vn < zero
        vn[nulls] = 1.
        
        if null_replace is not None:
            
            # null_replace can be eithera single vector or an array of vectors
            
            rep = Vectors.same(null_replace)
            if len(rep) == 1:
                self.a[nulls] = rep.a[0]
            else:
                self.a[nulls] = rep.a[nulls]
            
        self.a /= np.expand_dims(vn, axis=-1)
        
        return self

    # ---------------------------------------------------------------------------
    # Normalized copy
        
    def normalized(self, null_replace=None):
        return self.same(self.a).normalize(null_replace)

    # ---------------------------------------------------------------------------
    # dot product
    
    def dot(self, other):
        return np.einsum('...i,...i', *self.broadcasted_args(self.a, (3,), self.same(other).a, (3,)))
    
        
# ====================================================================================================
# 3-Vectors

class Vectors(NVectors, Coordinates):
    
    # ---------------------------------------------------------------------------
    # Coordinates interface
    
    def vectors(self):
        return self
    
    def spherics(self):
        sph = Spherics(shape=self.shape)
        
        sph.rho   = np.linalg.norm(self.a)
        sph.theta = np.arctan2(self.y, self.x)
        sph.phi   = np.arctan2(self.z, np.linalg.norm(self.a[..., :2]))
                             
        return sph
    
    def cylindrics(self):
        cyl = Cylindrics(shape=self.shape)
        
        cyl.rho   = np.linalg.norm(self.a[..., :2])
        cyl.theta = np.arctan2(self.y, self.x)
        cyl.z     = self.z
        return cyl
    
    def quaternions(self):
        return Quaternions(self.wvectors(w=0).a)
        
    # ---------------------------------------------------------------------------
    # Access to coordinates (x, y, z)
    
    @property
    def x(self):
        return self.a[..., 0]
    
    @x.setter
    def x(self, v):
        self.a[..., 0] = v
    
    @property
    def y(self):
        return self.a[..., 1]
    
    @y.setter
    def y(self, v):
        self.a[..., 1] = v
    
    @property
    def z(self):
        return self.a[..., 2]
    
    @z.setter
    def z(self, v):
        self.a[..., 2] = v

    # ---------------------------------------------------------------------------
    # Initialize unitary vectors
    
    @classmethod
    def Axis(cls, axis, shape=None):
        
        if type(axis) is str:
            
            upper = axis.upper()
            
            if upper in ['X', '+X', 'POS_X', 'I', '+I']:
                v = np.array((1., 0., 0.))
            elif upper in ['Y', '+Y', 'POS_Y', 'J', '+J']:
                v = np.array((0., 1., 0.))
            elif upper in ['Z', '+Z', 'POS_Z', 'K', '+K']:
                v = np.array((0., 0., 1.))
    
            elif upper in ['-X', 'NEG_X', '-I']:
                v = np.array((-1., 0., 0.))
            elif upper in ['-Y', 'NEG_Y', '-J']:
                v = np.array((0., -1., 0.))
            elif upper in ['-Z', 'NEG_Z', '-K']:
                v = np.array((0., 0., -1.))
            else:
                raise RuntimeError(f"Unknwon axis spec: '{axis}' in Vectors.Axis")
                
            return cls(v, shape=shape)
        
        else:
            return cls(axis, shape=shape).normalize()
        
    # ---------------------------------------------------------------------------
    # Initialize with random vectors
    
    @classmethod
    def Random(cls, shape=(1,), bounds=[0, 5]):
        
        shape = cls.as_shape(shape)
        
        z = np.random.uniform(-1., 1., shape)
        r = np.sqrt(1 - z*z)
        a = np.random.uniform(0., 2*np.pi, shape)
        
        R = bounds[0] + (1 - np.random.uniform(0, 1, shape)**3)*(bounds[1] - bounds[0])
        return cls(np.stack((r*np.cos(a), r*np.sin(a), z), axis=-1)*np.expand_dims(R, axis=-1))

    # ---------------------------------------------------------------------------
    # cross product
        
    def cross(self, other):
        return Vectors(np.cross(self.a, self.same(other).a))
    
    # ----- Capture the ^ operator
    
    def __xor__(self, other):
        return self.cross(other)
    
    # ---------------------------------------------------------------------------
    # Get one perpendicular vector
        
    def one_perp_vector(self):
        return Vectors(np.cross(self.a, (0, 0, 1))).normalize(null_replace=(0, 1, 0))

    # ---------------------------------------------------------------------------
    # Axis and angle perpendicular with another vector
    
    def axis_angles_with(self, other, are_normalized=False):
        
        other = self.same(other)
        
        if are_normalized:
            nv = self
            nw = other
        else:
            nv = self.normalized()
            nw = other.normalized()
            
        angle = np.arccos(np.clip(nv.dot(nw), -1, 1))
        axis  = nv.cross(nw)
        nrms  = axis.norm()
        
        nulls = nrms < zero
        axis.a[nulls] = Vectors(self.a, shape=axis.shape)[nulls].one_perp_vector().a
        nrms[nulls] = 1.
        
        axis.a /= np.expand_dims(nrms, axis=-1)
        
        return AxisAngles(axis, angles=angle)

    # ---------------------------------------------------------------------------
    # Angle with another vector
        
    def angle_with(self, other):
        return np.arccos(np.clip(self.normalized().dot(self.same(other).normalized()), -1, 1))

    # ---------------------------------------------------------------------------
    # Axis angle with a plane
    # The angle is the angle with the plane defined by a normalized perpendicular vector
    # The axis is a vector within the plane which can be used to rotate the vector into
    # the plane
        
    def axis_angles_with_plane(self, plane):
        axg = self.axis_angles_with(plane)
        axg.angles = -axg.angles
        return axg


    # ---------------------------------------------------------------------------
    # Perpendicular with another vector
    
    def perp_with(self, other, null_replace=None):
        return self.cross(other).normalize(null_replace=self.one_perp_vector() if null_replace is None else null_replace)

    # ---------------------------------------------------------------------------
    # Projection on a plane
    # The plane is defined by its perpendicular vector
    
    def plane_projection(self, plane):

        plane = self.same(plane)
        
        try:
            shape = np.broadcast_shapes(self.shape, plane.shape)
        except:
            raise self.error(f"Impossible to compute plane projection with shapes {self.shape} and {plane.shape}")

        vs = Vectors(self.a, shape)
        ps = Vectors.Axis(plane, shape)
        
        return Vectors(vs.a - ps.a*np.expand_dims(vs.dot(ps).a, axis=-1))
    
    # ---------------------------------------------------------------------------
    # Vector 4D extension
    
    def wvectors(self, w=0):
        return WVectors.FromVectors(self, w=w)
    
    def vectorst(self, t=0):
        return VectorsT.FromVectors(self, t=t)
    

# ====================================================================================================
# 4D vectors W X Y Z

class WVectors(NVectors):

    # ---------------------------------------------------------------------------
    # Item shape
    
    @staticmethod
    def item_shape():
        return (4,)
    
    # ---------------------------------------------------------------------------
    # Access to coordinates (w, x, y, z)
    
    @property
    def w(self):
        return self.a[..., 0]
    
    @w.setter
    def w(self, v):
        self.a[..., 0] = v
    
    @property
    def x(self):
        return self.a[..., 1]
    
    @x.setter
    def x(self, v):
        self.a[..., 1] = v
    
    @property
    def y(self):
        return self.a[..., 2]
    
    @y.setter
    def y(self, v):
        self.a[..., 2] = v
    
    @property
    def z(self):
        return self.a[..., 3]
    
    @z.setter
    def z(self, v):
        self.a[..., 3] = v
        
    # ---------------------------------------------------------------------------
    # The vectors part
    
    @property
    def vectors3(self):
        return Vectors(self.a[..., 1:])
    
    @vectors3.setter
    def vectors3(self, value):
        self.a[..., 1:] = self.other_a(value)
        
    # ---------------------------------------------------------------------------
    # From vectors 3D
    
    @classmethod
    def FromVectors(cls, vectors, w=0):
        return cls(np.insert(cls.other_a(vectors), 0, w, axis=-1))

# ====================================================================================================
# 4D vectors X Y Z T

class VectorsT(NVectors):

    # ---------------------------------------------------------------------------
    # Item shape
    
    @staticmethod
    def item_shape():
        return (4,)
    
    # ---------------------------------------------------------------------------
    # Access to coordinates (x, y, z, t)
    
    @property
    def x(self):
        return self.a[..., 0]
    
    @x.setter
    def x(self, v):
        self.a[..., 0] = v
    
    @property
    def y(self):
        return self.a[..., 1]
    
    @y.setter
    def y(self, v):
        self.a[..., 1] = v
    
    @property
    def z(self):
        return self.a[..., 2]
    
    @z.setter
    def z(self, v):
        self.a[..., 2] = v
        
    @property
    def t(self):
        return self.a[..., 3]
    
    @t.setter
    def t(self, v):
        self.a[..., 3] = v
        
    # ---------------------------------------------------------------------------
    # The vectors part
    
    @property
    def vectors3(self):
        return Vectors(self.a[..., :3])
    
    @vectors3.setter
    def vectors3(self, value):
        self.a[..., :3] = self.other_a(value)
        
    # ---------------------------------------------------------------------------
    # From vectors 3D
    
    @classmethod
    def FromVectors(cls, vectors, t=1):
        return cls(np.insert(cls.other_a(vectors), 3, t, axis=-1)) 
    
    
# ====================================================================================================
# Spherical coordinates

class Spherics(NVectors, Coordinates):
    
    def str_item(self, item):
        return f"[{item[0]:8.3f}, {np.degrees(item[1]):6.1}° {np.degrees(item[2]):6.1}°]"
    
    # ---------------------------------------------------------------------------
    # Coordinates interface
    
    def vectors(self):
        
        v = Vectors(shape=self.shape)
        
        r = self.rho * np.cos(self.phi)
        
        v.x = r * np.cos(self.theta)
        v.y = r * np.sin(self.theta)
        v.z = self.rho * np.sin(self.phi)
        
        return v
    
    def spherics (self):
        return self
    
    def cylindrics(self):
        
        cyl = Cylindrics(shape=self.shape)
        
        cyl.rho   = self.rho * np.cos(phi)
        cyl.theta = self.theta
        cyl.z     = self.rho * np.sin(phi)

        return cyl
    
    def quaternions(self):
        return self.vectors().quaternions()
        
    # ---------------------------------------------------------------------------
    # Access to coordinates rho, theta, phi
    
    @property
    def rho(self):
        return self.a[..., 0]
    
    @rho.setter
    def rho(self, v):
        self.a[..., 0] = v
    
    @property
    def theta(self):
        return self.a[..., 1]
    
    @theta.setter
    def theta(self, v):
        self.a[..., 1] = v
    
    @property
    def phi(self):
        return self.a[..., 2]
    
    @phi.setter
    def phi(self, v):
        self.a[..., 2] = v

# ====================================================================================================
# Cylindrical coordinates

class Cylindrics(NVectors, Coordinates):
    
    def str_item(self, item):
        return f"[{item[0]:8.3f}, {np.degrees(item[1]):6.1}° {item[2]:8.3}]"
    
    # ---------------------------------------------------------------------------
    # Coordinates interface
    
    def vectors(self):
        
        v = Vectors(shape=self.shape)
        
        v.x = self.r * np.cos(self.theta)
        v.y = self.r * np.sin(self.theta)
        v.z = self;z
        
        return v
    
    def spherics (self):
    
        sph = Spherics(shape=self.shape)
        
        sph.rho   = np.sqrt(np.square(self.rho) + np.square(self.z))
        sph.theta = self.theta
        sph.phi   = np.arctan2(self.z, self.rho)
        
        return sph
    
    def cylindrics(self):
        return self
    
    def quaternions(self):
        return self.vectors().quaternions()
        
    # ---------------------------------------------------------------------------
    # Access to coordinates rho, theta, phi
    
    @property
    def rho(self):
        return self.a[..., 0]
    
    @rho.setter
    def rho(self, v):
        self.a[..., 0] = v
    
    @property
    def theta(self):
        return self.a[..., 1]
    
    @theta.setter
    def theta(self, v):
        self.a[..., 1] = v
    
    @property
    def z(self):
        return self.a[..., 2]
    
    @z.setter
    def z(self, v):
        self.a[..., 2] = v
    
    
# ====================================================================================================
# Rotation: Matrices, Eulers, Quaternions are rotations
# Must implement
# - inverse: opposite rotation
# - rotate:  rotation of vectors
# - compose: compose with another rotation
#
# and transformations
# - matrices
# - eulers
# - quaternions
# - axis_angles 
#
# The Rotation class implements operators based on these methods
# - -Rotation = inverse rotation
# - a*b       = compose two rotations
# - a/b       = compose with the invert


class Rotation:
    
    # ---------------------------------------------------------------------------
    # Unary operators
    
    def __invert__(self):
        return self.inverse()
    
    # ---------------------------------------------------------------------------
    # Rotation
    
    def __matmul__(self, other):
        if issubclass(type(other), Rotation):
            return self.compose(other)
        else:
            return self.rotate(other)
        
    # ---------------------------------------------------------------------------
    # Braodcasting rotation
    
    def __pow__(self, other):
        if issubclass(type(other), Rotation):
            return self.compose_explode(other)
        else:
            return self.rotate_explode(other)
        
    # ---------------------------------------------------------------------------
    # Composition and rotation can be done either with compatible shapes
    # or by exploding the argument with the shape of the matrices
    #
    # - 1 to 1 : (2, 3) @ (2, 3) -> (2, 3)
    #            (2, 3) @ (10)   -> error
    # - explode: (2, 3) ** (2, 3) -> (2, 3, 2, 3)
    #            (2, 3) ** (10)   -> (2, 3, 10)
    #
    # To explode, the rotation object is reshaped by appending 1s to its shape
    
    def compose_explode(self, other):
        other_shape = cls.same(other).shape
        old_shape   = self.shape
        
        self.reshape(old_shape + (1,)*len(other_shape))
        
        res = self.compose(other)
        
        self.reshape(old_shape)
        
        return res

    def rotate_explode(self, other):
        other_shape = Vectors.same(other).shape
        old_shape   = self.shape
        
        self.reshape(old_shape + (1,)*len(other_shape))
        
        res = self.rotate(other)
        
        self.reshape(old_shape)
        
        return res
    
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Tests for dev
    #
    # To be run to check that the rotations give the same results and that
    # all conversions work properly
    
    @staticmethod
    def eulers_test(order='XYZ'):
        eulers = Eulers(shape=(0,))
        ags = np.linspace(-np.pi, np.pi, 13, endpoint=True)
        for x in ags:
            for y in ags:
                for z in ags:
                    eulers.append((x, y, z))
        eulers.order = order
        return eulers
    
    @staticmethod
    def test(stop=False):
        
        def compare(title, v0, rots, vs, explode=False, raise_error=False):
            
            if explode:
                v1 = rots ** vs
            else:
                v1 = rots @ vs
                
            
            d = v0.difference_with(v1)
            ok = d < 1e-10
            if ok:
                res = "ok"
            else:
                res = f"{d:.10f}"
            print(f"{title:15s}: {res}")
            
            if ok:
                return True
            
            if raise_error:
                print(v0)
                print(rots)
                print(vs)
                print(rots @ vs)
                raise RuntimeError("KO")
            
            if stop:
                print(v0.shape)
                print(v0.size)
                print(v0.item_shape())
                v0.reshape((v0.size,))
                vs.reshape((vs.size,))
                rots.reshape((rots.size,))
                for rot in rots:
                    for v0_, vs_ in zip(v0, vs):
                        compare("Check", v0_, rot, vs_, explode=False, raise_error=True)
            
            
        def test_rot(title, rots, vs, order='XYZ'):
            
            print(f"---------- {title}")
            print()
            
            print("Check that the computed rotations give the same result")
            
            vref = rots ** vs
            
            compare("    Matrices",    vref, rots.matrices(),    vs, explode=True)
            compare("    Eulers",      vref, rots.eulers(order), vs, explode=True)
            compare("    Quaternions", vref, rots.quaternions(), vs, explode=True)
            compare("    Axis angles", vref, rots.axis_angles(), vs, explode=True)
            print()
            
            print("Check that the inverted rotation comes back to initial vectors")
            
            inv_rots = ~rots
            inv_rots.reshape(inv_rots.shape + (1,) * (len(vref.shape)-1))
            
            back_vs = inv_rots @ vref
            
            inv_mats   = ~(rots.matrices())
            inv_eulers = ~(rots.eulers(order))
            inv_quats  = ~(rots.quaternions())
            inv_aags   = ~(rots.axis_angles())
            
            inv_mats.reshape(inv_rots.shape)
            inv_eulers.reshape(inv_rots.shape)
            inv_quats.reshape(inv_rots.shape)
            inv_aags.reshape(inv_rots.shape)
            
            compare("    Matrices",    back_vs, inv_mats,   vref)
            compare("    Eulers",      back_vs, inv_eulers, vref)
            compare("    Quaternions", back_vs, inv_quats,  vref)
            compare("    Axis angles", back_vs, inv_aags,   vref)

            print()
        
        # ---------------------------------------------------------------------------
        # Generate specific and random vectors
        # Shape is (2, 125), first block for specific vectors and second for random ones
        
        vs = Vectors(shape=(0,))
        
        coords = (-10, -1, 0, 1, 10)
        for x in coords:
            for y in coords:
                for z in coords:
                    vs.append((x, y, z))
                    
        count = len(vs)
                    
        vs.append(Vectors.Random(shape=count))
        vs.reshape((2, count))
        
        # ---------------------------------------------------------------------------
        # Generate specific and random eulers as a base
        # Shape is 13**3  2197
        
        eulers = Eulers(shape=(0,))
        ags = np.linspace(-np.pi, np.pi, 13, endpoint=True)
        for x in ags:
            for y in ags:
                for z in ags:
                    eulers.append((x, y, z))
                    
        # ---------------------------------------------------------------------------
        # Check with transformations
        
        print('-'*80)
        print("Test rotations")
        print()
        print("Rotations shape:      ", eulers.shape)
        print("Vectors shape :       ", vs.shape)
        print("Rotated vectors shape:", (eulers ** vs).shape)
        print()
        
        for order in Eulers.ORDERS:
            print("="*80)
            print("Euler order", order)
            print()
            
            eulers.order = order
            
            test_rot("Eulers",      eulers,               vs, order)
            test_rot("Matrices",    eulers.matrices(),    vs, order)
            test_rot("Quaternions", eulers.quaternions(), vs, order)
            test_rot("Axis angles", eulers.axis_angles(), vs, order)
            
        
    
# ====================================================================================================
# Matrices

class Matrices(GeoArray, Rotation):

    @staticmethod
    def item_shape():
        return (3, 3)
        
    def str_item(self, item):
        s  = f"[[{item[0, 0]:8.3f} {item[0, 1]:8.3f} {item[0, 2]:8.3f}]\n"
        s += f" [{item[1, 0]:8.3f} {item[1, 1]:8.3f} {item[1, 2]:8.3f}]\n"
        s += f" [{item[2, 0]:8.3f} {item[2, 1]:8.3f} {item[2, 2]:8.3f}]\n"
        s += "]"
        
        return s
    
    # ---------------------------------------------------------------------------
    # Rotation interface
    
    @classmethod
    def Identity(cls, shape=None):
        return cls(np.identity(3), shape=shape)
    
    def matrices(self):
        return self
    
    def inverse(self):
        return Matrices(np.linalg.inv(self.a))
    
    def rotate(self, coords):
        
        if issubclass(type(coords), Coordinates):
            v = coords.vectors()
        else:
            v = Vectors.same(vectors)
        
        try:
            shape = np.broadcast_shapes(self.shape, v.shape)
        except:
            self.error(f"Impossible to rotate vectors of shape {v.shape} with matrices of shape {self.shape}")
            
        return Vectors(np.einsum('...jk,...j', self.a, v.a))
    
    def compose(self, other):
        m = self.other.matrices().a
        return Matrices(np.matmul(self.a, m))
    
    # ---------------------------------------------------------------------------
    # Matrices to eulers

    def eulers(self, order='XYZ'):
        
        count = self.size
        
        ms = np.reshape(self.a, (count, 3, 3))
    
        # ---------------------------------------------------------------------------
        # Indices in the array to compute the angles
    
        if order == 'XYZ':
    
            # cz.cy              | cz.sy.sx - sz.cx   | cz.sy.cx + sz.sx
            # sz.cy              | sz.sy.sx + cz.cx   | sz.sy.cx - cz.sx
            # -sy                | cy.sx              | cy.cx
    
            xyz = [1, 0, 2]
    
            ls0, cs0, sgn = (2, 0, -1)          # sy
            ls1, cs1, lc1, cc1 = (2, 1, 2, 2)   # cy.sx cy.cx
            ls2, cs2, lc2, cc2 = (1, 0, 0, 0)   # cy.sz cy.cz
    
            ls3, cs3, lc3, cc3 = (0, 1, 1, 1)   
    
        elif order == 'XZY':
    
            # cy.cz              | -cy.sz.cx + sy.sx  | cy.sz.sx + sy.cx
            # sz                 | cz.cx              | -cz.sx
            # -sy.cz             | sy.sz.cx + cy.sx   | -sy.sz.sx + cy.cx
    
            xyz = [2, 0, 1]
    
            ls0, cs0, sgn = (1, 0, +1)
            ls1, cs1, lc1, cc1 = (1, 2, 1, 1)
            ls2, cs2, lc2, cc2 = (2, 0, 0, 0)
    
            ls3, cs3, lc3, cc3 = (0, 2, 2, 2)
    
        elif order == 'YXZ':
    
            # cz.cy - sz.sx.sy   | -sz.cx             | cz.sy + sz.sx.cy
            # sz.cy + cz.sx.sy   | cz.cx              | sz.sy - cz.sx.cy
            # -cx.sy             | sx                 | cx.cy
    
            xyz = [1, 0, 2]
            xyz = [2, 0, 1]
            xyz = [0, 1, 2]
    
            ls0, cs0, sgn = (2, 1, +1)
            ls1, cs1, lc1, cc1 = (2, 0, 2, 2)
            ls2, cs2, lc2, cc2 = (0, 1, 1, 1)
    
            ls3, cs3, lc3, cc3 = (1, 0, 0, 0)
    
        elif order == 'YZX':
    
            # cz.cy              | -sz                | cz.sy
            # cx.sz.cy + sx.sy   | cx.cz              | cx.sz.sy - sx.cy
            # sx.sz.cy - cx.sy   | sx.cz              | sx.sz.sy + cx.cy
    
            xyz = [2, 1, 0]
            
    
            ls0, cs0, sgn = (0, 1, -1)
            ls1, cs1, lc1, cc1 = (0, 2, 0, 0)
            ls2, cs2, lc2, cc2 = (2, 1, 1, 1)
    
            ls3, cs3, lc3, cc3 = (1, 2, 2, 2)
    
        elif order == 'ZXY':
    
            # cy.cz + sy.sx.sz   | -cy.sz + sy.sx.cz  | sy.cx
            # cx.sz              | cx.cz              | -sx
            # -sy.cz + cy.sx.sz  | sy.sz + cy.sx.cz   | cy.cx
    
            xyz = [0, 2, 1]
    
            ls0, cs0, sgn = (1, 2, -1)
            ls1, cs1, lc1, cc1 = (1, 0, 1, 1)
            ls2, cs2, lc2, cc2 = (0, 2, 2, 2)
    
            ls3, cs3, lc3, cc3 = (2, 0, 0, 0)
    
        elif order == 'ZYX':
    
            # cy.cz              | -cy.sz             | sy
            # cx.sz + sx.sy.cz   | cx.cz - sx.sy.sz   | -sx.cy
            # sx.sz - cx.sy.cz   | sx.cz + cx.sy.sz   | cx.cy
    
            xyz = [1, 2, 0]
    
            ls0, cs0, sgn = (0, 2, +1)
            ls1, cs1, lc1, cc1 = (0, 1, 0, 0)
            ls2, cs2, lc2, cc2 = (1, 2, 2, 2)
    
            ls3, cs3, lc3, cc3 = (2, 1, 1, 1)
    
        else:
            raise self.error(f"Conversion to eulers error: '{order}' is not a valid euler order")
            
        # ---------------------------------------------------------------------------
        # Compute the euler angles
    
        angles = np.zeros((len(ms), 3), float)   # Place holder for the angles in the order of their computation
        
        # Computation depends upon sin(angle 0) == ±1
    
        neg_1  = np.where(np.abs(ms[:, cs0, ls0] + 1) < zero)[0] # sin(angle 0) = -1
        pos_1  = np.where(np.abs(ms[:, cs0, ls0] - 1) < zero)[0] # sin(angle 0) = +1
        rem    = np.delete(np.arange(len(ms)), np.concatenate((neg_1, pos_1)))
        
        if len(neg_1) > 0:
            angles[neg_1, xyz[0]] = -pi/2 * sgn
            angles[neg_1, xyz[1]] = 0
            angles[neg_1, xyz[2]] = np.arctan2(sgn * ms[neg_1, cs3, ls3], ms[neg_1, cc3, lc3])
    
        if len(pos_1) > 0:
            angles[pos_1, xyz[0]] = pi/2 * sgn
            angles[pos_1, xyz[1]] = 0
            angles[pos_1, xyz[2]] = np.arctan2(sgn * ms[pos_1, cs3, ls3], ms[pos_1, cc3, lc3])
    
        if len(rem) > 0:
            angles[rem, xyz[0]] = sgn * np.arcsin(ms[rem, cs0, ls0])
            angles[rem, xyz[1]] = np.arctan2(-sgn * ms[rem, cs1, ls1], ms[rem, cc1, lc1])
            angles[rem, xyz[2]] = np.arctan2(-sgn * ms[rem, cs2, ls2], ms[rem, cc2, lc2])
            
        # ---------------------------------------------------------------------------
        # At this stage, the result could be two 180 angles and a value ag
        # This is equivalent to two 0 values and 180-ag
        # Let's correct this
        
        # -180° --> 180°
            
        angles[abs(angles+np.pi) < zero] = np.pi
        
        # Let's change where we have two 180 angles
        
        idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < zero, abs(angles[:, 1]-np.pi) < zero))[0]
        angles[idx, 0] = 0
        angles[idx, 1] = 0
        angles[idx, 2] = np.pi - angles[idx, 2]
        
        idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < zero, abs(angles[:, 2]-np.pi) < zero))[0]
        angles[idx, 0] = 0
        angles[idx, 2] = 0
        angles[idx, 1] = np.pi - angles[idx, 1]
        
        idx = np.where(np.logical_and(abs(angles[:, 1]-np.pi) < zero, abs(angles[:, 2]-np.pi) < zero))[0]
        angles[idx, 1] = 0
        angles[idx, 2] = 0
        angles[idx, 0] = np.pi - angles[idx, 0]
        
        # ---------------------------------------------------------------------------
        # Returns the result
        
        return Eulers(np.reshape(angles, self.shape + (3,)), order=order)
    
    
    # ---------------------------------------------------------------------------
    # Matrices to quaternions
    
    def quaternions(self):
        
        # ----- The result
        
        quat = Quaternions(shape=self.shape)

        q = quat.a
        m = self.a
        
        # ----- Let's go
        
        v = 1 + np.trace(m, axis1=-2, axis2=-1)
        
        # ----- Null values, need a specific computation
        
        nulls = v <= zero
        m_nulls = Matrices(m[nulls])
        
        if len(m_nulls) > 0:
            eulers = m_nulls.eulers()
            q[nulls] = m_nulls.eulers().quaternions().a
            oks = np.logical_not(nulls)
        else:
            oks = slice(None, None, None)
        
        # ----- Not null computation
        
        q[oks, 0] = np.sqrt(v[oks]) / 2
        
        q[oks, 1] = (m[oks, 1, 2] - m[oks, 2, 1])
        q[oks, 2] = (m[oks, 2, 0] - m[oks, 0, 2]) 
        q[oks, 3] = (m[oks, 0, 1] - m[oks, 1, 0])
        q[oks, 1:] *= np.expand_dims(.25/q[oks, 0], axis=-1)
        
        return quat
    
    # ---------------------------------------------------------------------------
    # Axis angles
    
    def axis_angles(self):
        return self.quaternions().axis_angles()
    
    
# ====================================================================================================
# Eulers

class Eulers(Vectors, Rotation):

    ORDERS = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
    ORDERS_INDS = {
            'XYZ': (0, 1, 2),
            'XZY': (0, 2, 1),
            'YXZ': (1, 0, 2),
            'YZX': (1, 2, 0),
            'ZXY': (2, 0, 1),
            'ZYX': (2, 1, 0),
        }
    ORDERS_REV_INDS = {
            'XYZ': [0, 1, 2],
            'XZY': [0, 2, 1],
            'YXZ': [1, 0, 2],
            'YZX': [2, 0, 1],
            'ZXY': [1, 2, 0],
            'ZYX': [2, 1, 0],   
        }
    
    def __init__(self, values=0, shape=None, order='XYZ'):
        super().__init__(values, shape)
        self.order = self.check_order(order)
    
    # ---------------------------------------------------------------------------
    # Display item

    def str_item(self, item):
        euler = np.degrees(item)
        return f"[{euler[0]:6.1f}° {euler[1]:6.1f}° {euler[2]:6.1f}°]"
    
    # ---------------------------------------------------------------------------
    # Euler order management
    
    @classmethod
    def check_order(cls, order):
        if order in cls.ORDERS:
            return order
        else:
            raise RuntimeError(f"Euler order '{order}' is not valid.")
            
    @property
    def order(self):
        try:
            return self.order_
        except:
            return 'XYZ'
        
    @order.setter
    def order(self, value):
        self.order_ = self.check_order(value)
        
    # ---------------------------------------------------------------------------
    # Random eulers
    
    @classmethod
    def Random(cls, shape, bounds=[-180, 180]):
        return cls(np.radians(np.random.randint(bounds[0]/10, bounds[1]/10 + 1, shape + (3,))*10))
    
    # ---------------------------------------------------------------------------
    # All squares rotations
    
    @classmethod
    def AllSquares(cls):
        angles = [0, 90, 180, 270, 360]
        a = np.empty((len(angles)**3, 3), int)
        
        i = 0
        for ai in angles:
            for aj in angles:
                for ak in angles:
                    a[i] = (ai, aj, ak)
                    i += 1
                    
        return cls(np.radians(a))
    
    # ---------------------------------------------------------------------------
    # Rotation interface
    
    @classmethod
    def Identity(cls, shape=None):
        return cls(shape=shape)
    
    def eulers(self, order='XYZ'):
        if order == self.order:
            return self
        else:
            return self.matrices().eulers(order)
    
    def inverse(self):
        return self.matrices().inverse().eulers(self.order)
    
    def rotate(self, coords):
        return self.matrices().rotate(coords)
    
    def compose(self, other):
        return self.matrices().compose(other.matrices()).eulers(self.order)

    # ---------------------------------------------------------------------------
    # Eulers to matrices
    
    def matrices(self):
        
        order = self.order
        count = self.size
        es    = np.reshape(self.a, (count, 3))
        
        # ----- Let's go
    
        m = np.zeros((count, 3, 3), float)
    
        cx = np.cos(es[:, 0])
        sx = np.sin(es[:, 0])
        cy = np.cos(es[:, 1])
        sy = np.sin(es[:, 1])
        cz = np.cos(es[:, 2])
        sz = np.sin(es[:, 2])
        
        if order == 'XYZ':
            m[:, 0, 0] = cz*cy
            m[:, 1, 0] = cz*sy*sx - sz*cx
            m[:, 2, 0] = cz*sy*cx + sz*sx
            m[:, 0, 1] = sz*cy
            m[:, 1, 1] = sz*sy*sx + cz*cx
            m[:, 2, 1] = sz*sy*cx - cz*sx
            m[:, 0, 2] = -sy
            m[:, 1, 2] = cy*sx
            m[:, 2, 2] = cy*cx
    
        elif order == 'XZY':
            m[:, 0, 0] = cy*cz
            m[:, 1, 0] = -cy*sz*cx + sy*sx
            m[:, 2, 0] = cy*sz*sx + sy*cx
            m[:, 0, 1] = sz
            m[:, 1, 1] = cz*cx
            m[:, 2, 1] = -cz*sx
            m[:, 0, 2] = -sy*cz
            m[:, 1, 2] = sy*sz*cx + cy*sx
            m[:, 2, 2] = -sy*sz*sx + cy*cx
    
        elif order == 'YXZ':
            m[:, 0, 0] = cz*cy - sz*sx*sy
            m[:, 1, 0] = -sz*cx
            m[:, 2, 0] = cz*sy + sz*sx*cy
            m[:, 0, 1] = sz*cy + cz*sx*sy
            m[:, 1, 1] = cz*cx
            m[:, 2, 1] = sz*sy - cz*sx*cy
            m[:, 0, 2] = -cx*sy
            m[:, 1, 2] = sx
            m[:, 2, 2] = cx*cy
    
        elif order == 'YZX':
            m[:, 0, 0] = cz*cy
            m[:, 1, 0] = -sz
            m[:, 2, 0] = cz*sy
            m[:, 0, 1] = cx*sz*cy + sx*sy
            m[:, 1, 1] = cx*cz
            m[:, 2, 1] = cx*sz*sy - sx*cy
            m[:, 0, 2] = sx*sz*cy - cx*sy
            m[:, 1, 2] = sx*cz
            m[:, 2, 2] = sx*sz*sy + cx*cy
    
        elif order == 'ZXY':
            m[:, 0, 0] = cy*cz + sy*sx*sz
            m[:, 1, 0] = -cy*sz + sy*sx*cz
            m[:, 2, 0] = sy*cx
            m[:, 0, 1] = cx*sz
            m[:, 1, 1] = cx*cz
            m[:, 2, 1] = -sx
            m[:, 0, 2] = -sy*cz + cy*sx*sz
            m[:, 1, 2] = sy*sz + cy*sx*cz
            m[:, 2, 2] = cy*cx
    
        elif order == 'ZYX':
            m[:, 0, 0] = cy*cz
            m[:, 1, 0] = -cy*sz
            m[:, 2, 0] = sy
            m[:, 0, 1] = cx*sz + sx*sy*cz
            m[:, 1, 1] = cx*cz - sx*sy*sz
            m[:, 2, 1] = -sx*cy
            m[:, 0, 2] = sx*sz - cx*sy*cz
            m[:, 1, 2] = sx*cz + cx*sy*sz
            m[:, 2, 2] = cx*cy
            
        return Matrices(m.reshape(self.shape + (3, 3)))
    
    # ---------------------------------------------------------------------------
    # Eulers to quaternions
    
    def quaternions(self):
        
        
        # ---------------------------------------------------------------------------
        # Naive algorithm, clearer but slower
        
        if False:
        
            qx = AxisAngles((1, 0, 0), angles=self.x).quaternions()
            qy = AxisAngles((0, 1, 0), angles=self.y).quaternions()
            qz = AxisAngles((0, 0, 1), angles=self.z).quaternions()
            
            if self.order == 'XYZ':
                return qz @ qy @ qx
            elif self.order == 'XZY':
                return qy @ qz @ qx
            elif self.order == 'YXZ':
                return qz @ qx @ qy
            elif self.order == 'YZX':
                return qx @ qz @ qy
            elif self.order == 'ZXY':
                return qy @ qx @ qz
            elif self.order == 'ZYX':
                return qx @ qy @ qz
            
        # ---------------------------------------------------------------------------
        # Less elegant algorithm but 5x quicker
        # Source code automatically generated with formal coputation
            
        else:
            
            cx = np.cos(self.a[..., 0]/2)
            sx = np.sin(self.a[..., 0]/2)
            cy = np.cos(self.a[..., 1]/2)
            sy = np.sin(self.a[..., 1]/2)
            cz = np.cos(self.a[..., 2]/2)
            sz = np.sin(self.a[..., 2]/2)
            
            q = Quaternions(shape=self.shape)
            
            # Order XYZ: qz @ qy @ qx
            
            if self.order == 'XYZ':
                q.a[..., 0] =  cz*cy*cx + sz*sy*sx
                q.a[..., 1] =  cz*cy*sx - sz*sy*cx
                q.a[..., 2] =  cz*sy*cx + sz*cy*sx
                q.a[..., 3] = -cz*sy*sx + sz*cy*cx
                
    
            # Order XZY: qy @ qz @ qx
    
            elif self.order == 'XZY':
                q.a[..., 0] =  cy*cz*cx - sy*sz*sx
                q.a[..., 1] =  cy*cz*sx + sy*sz*cx
                q.a[..., 2] =  sy*cz*cx + cy*sz*sx
                q.a[..., 3] = -sy*cz*sx + cy*sz*cx
                
    
            # Order YXZ: qz @ qx @ qy
    
            elif self.order == 'YXZ':
                q.a[..., 0] =  cz*cx*cy - sz*sx*sy
                q.a[..., 1] =  cz*sx*cy - sz*cx*sy
                q.a[..., 2] =  cz*cx*sy + sz*sx*cy
                q.a[..., 3] =  cz*sx*sy + sz*cx*cy
                
    
            # Order YZX: qx @ qz @ qy
    
            elif self.order == 'YZX':
                q.a[..., 0] =  cx*cz*cy + sx*sz*sy
                q.a[..., 1] =  sx*cz*cy - cx*sz*sy
                q.a[..., 2] =  cx*cz*sy - sx*sz*cy
                q.a[..., 3] =  sx*cz*sy + cx*sz*cy
                
    
            # Order ZXY: qy @ qx @ qz
    
            elif self.order == 'ZXY':
                q.a[..., 0] =  cy*cx*cz + sy*sx*sz
                q.a[..., 1] =  cy*sx*cz + sy*cx*sz
                q.a[..., 2] = -cy*sx*sz + sy*cx*cz
                q.a[..., 3] =  cy*cx*sz - sy*sx*cz
                
    
            # Order ZYX: qx @ qy @ qz
    
            elif self.order == 'ZYX':
                q.a[..., 0] =  cx*cy*cz - sx*sy*sz
                q.a[..., 1] =  sx*cy*cz + cx*sy*sz
                q.a[..., 2] = -sx*cy*sz + cx*sy*cz
                q.a[..., 3] =  cx*cy*sz + sx*sy*cz
                            
            return q
        
    # ---------------------------------------------------------------------------
    # Axis angles
    
    def axis_angles(self):
        return self.quaternions().axis_angles()

    
# ====================================================================================================
# Quaternions
#
# Quaternions implements both Coordinates and Rotation interface
# A quaternion is a rotation only if its norm is 1. The user must control the context
# into which quaternions are used.

class Quaternions(WVectors, Coordinates, Rotation):
    
    # ---------------------------------------------------------------------------
    # From Axis Angles
    
    @classmethod
    def FromAxisAngles(cls, axis, angles, shape=None):
        return cls(AxisAngles(axis, angles=angles, shape=shape).quaternions().a)

    # ---------------------------------------------------------------------------
    # Item shape
        
    @staticmethod
    def item_shape():
        return (4,)

    # ---------------------------------------------------------------------------
    # Item to string
        
    def str_item(self, item):
        axg = Quaternions(item).axis_angles()
        v = axg.axis.a[0]
        sax = f"[{v[0]:8.3f} {v[1]:8.3f} {v[2]:8.3f}]"
        return f"[{item[0]:8.3f} {item[1]:8.3f} {item[2]:8.3f} {item[3]:8.3f}] ax: {sax} ag: {np.degrees(axg.angles[0]):6.1f}°"

    # ---------------------------------------------------------------------------
    # Random quaternions
    
    @classmethod
    def Random(cls, shape=(1,)):
        q = Vectors.Random(shape, bounds=[1, 1])
        ags = (np.radians(np.random.randint(-18, 19, shape)*10)/2)
        q.a *= np.expand_dims(np.sin(ags), axis=-1)
        return cls(np.insert(q.a, 0, np.cos(ags), axis=-1))
    
    # ---------------------------------------------------------------------------
    # Coordinates interface
    
    # quaternions will be implemented with the Rotation interface
    
    def vectors(self):
        return self.vectors3

    def spherics(self):
        return self.vectors().spherics()
    
    def cylindrics(self):
        return self.vectors().cylindrics()
    
    # ---------------------------------------------------------------------------
    # Conjugate
    
    def conjugate(self):
        q = Quaternions(np.array(self.a))
        q.a[..., 1:] *= -1
        return q

    # ---------------------------------------------------------------------------
    # Unaty multiplication
    
    @staticmethod
    def q_mult(qa, qb):
        a = qa[0]
        b = qa[1]
        c = qa[2]
        d = qa[3]
    
        e = qb[0]
        f = qb[1]
        g = qb[2]
        h = qb[3]
    
        return np.array((
            a*e - b*f - c*g - d*h,
            a*f + b*e + c*h - d*g,
            a*g - b*h + c*e + d*f,
            a*h + b*g - c*f + d*e))
    
    # ---------------------------------------------------------------------------
    # Rotation interface
    
    @classmethod
    def Identity(cls, shape=None):
        return cls((1, 0, 0, 0), shape=shape)
    
    def quaternions(self):
        return self
    
    def inverse(self):
        return self.conjugate()
    
    def rotate(self, coords):
        
        # ---------------------------------------------------------------------------
        # Non optimized algorithm
        
        if False:
            
            return_quat = False
            
            if issubclass(type(coords), Coordinates):
                q = coords.quaternions()
                return_quat = issubclass(type(coords), Quaternions)
            else:
                q = Quaternions.FromVectors(coords, w=0)

            r = self @ q @ self.inverse()
            
            if return_quat:
                return r
            else:
                return r.vectors()
            
        # ---------------------------------------------------------------------------
        # Direct computation
        
        else:
            if issubclass(type(coords), Coordinates):
                return_quat = coords.vdim == 4
                v = coords.a
            else:
                return_quat = np.shape(coords)[-1] == 4
                v = coords
                
            if return_quat:
                x = v[..., 1]
                y = v[..., 2]
                z = v[..., 3]
            else:
                x = v[..., 0]
                y = v[..., 1]
                z = v[..., 2]

            qw = self.a[..., 0]
            qx = self.a[..., 1]
            qy = self.a[..., 2]
            qz = self.a[..., 3]
            
            qw2 = qw*qw
            qx2 = qx*qx
            qy2 = qy*qy
            qz2 = qz*qz
            
            qwx = qw*qx
            qwy = qw*qy
            qwz = qw*qz
            
            qxy = qx*qy
            qxz = qx*qz
            qyz = qy*qz
            
            rx = x*(qw2 + qx2 - qy2 - qz2) + 2*( y*(qxy - qwz) + z*(qxz + qwy) )
            ry = y*(qw2 - qx2 + qy2 - qz2) + 2*( x*(qxy + qwz) + z*(qyz - qwx) )
            rz = z*(qw2 - qx2 - qy2 + qz2) + 2*( x*(qxz - qwy) + y*(qyz + qwx) )
            
            del qw2, qx2, qy2, qz2
            del qwx, qwy, qwz, qxy, qxz, qyz
            
            if return_quat:
                a = np.zeros(np.shape(rx) + (4,))
                a[..., 1] = rx
                a[..., 2] = ry
                a[..., 3] = rz
                
                return Quaternions(a)
            
            else:
                a = np.empty(np.shape(rx) + (3,))
                a[..., 0] = rx
                a[..., 1] = ry
                a[..., 2] = rz
                
                return Vectors(a)
            
    # ---------------------------------------------------------------------------
    # Rotation interface
    
    def compose(self, other):
        
        qa = self.a
        qb = other.quaternions().a
        
        try:
            shape = np.broadcast_shapes(self.shape, np.shape(qb)[:-1])
        except:
            raise self.error(f"Impossible to multiply quaternions with shapes {self.shape} and {np.shape(qb)[:-1]}.")
            
        qas = np.empty(shape + (4,), float)
        qbs = np.empty(shape + (4,), float)
        
        qas[:] = qa
        qbs[:] = qb
        
        # ---------------------------------------------------------------------------
        # a = s*t - sum(p*q)
    
        w = qas[..., 0] * qbs[..., 0] - np.sum(qas[..., 1:] * qbs[..., 1:], axis=-1)
    
        # v = s*q + t*p + np.cross(p,q)
        v  = qbs[..., 1:] * np.expand_dims(qas[..., 0], -1) + \
             qas[..., 1:] * np.expand_dims(qbs[..., 0], -1) + \
             np.cross(qas[..., 1:], qbs[..., 1:])
             
        return Quaternions(np.insert(v, 0, w, axis=-1).reshape(shape + (4,)))

    # ---------------------------------------------------------------------------
    # Axis angle in radians
    
    def axis_angles(self):

        sn  = np.linalg.norm(self.a[..., 1:], axis=-1)
        ags = 2*np.arctan2(sn, self.a[..., 0])
        
        return AxisAngles(self.vectors3, angles=ags)
    
    # ---------------------------------------------------------------------------
    # Quaternions to matrices
    
    def matrices(self):
        
        count = self.size
        
        qs = np.reshape(self.a, (count, 4))
    
        # m1
        # +w	 +z -y +x
        # -z +w +x +y
        # +y	 -x +w +z
        # -x -y -z +w
    
        # m2
        # +w	 +z -y -x
        # -z +w +x -y
        # +y	 -x +w -z
        # +x +y +z +w
    
        m1 = np.stack((
            qs[:, [0, 3, 2, 1]]*(+1, +1, -1, +1),
            qs[:, [3, 0, 1, 2]]*(-1, +1, +1, +1),
            qs[:, [2, 1, 0, 3]]*(+1, -1, +1, +1),
            qs[:, [1, 2, 3, 0]]*(-1, -1, -1, +1)
            )).transpose((1, 0, 2))
    
        m2 = np.stack((
            qs[:, [0, 3, 2, 1]]*(+1, +1, -1, -1),
            qs[:, [3, 0, 1, 2]]*(-1, +1, +1, -1),
            qs[:, [2, 1, 0, 3]]*(+1, -1, +1, -1),
            qs[:, [1, 2, 3, 0]]*(+1, +1, +1, +1)
            )).transpose((1, 0, 2))
        
        return Matrices(np.matmul(m1, m2)[:, :3, :3].reshape(self.shape + (3, 3)))
    
    # ---------------------------------------------------------------------------
    # Quaternions to eulers
    
    def eulers(self, order='XYZ'):
        
        # ---------------------------------------------------------------------------
        # Indirect algorithm
        
        if False:
            return self.matrices().eulers(order)

        # ---------------------------------------------------------------------------
        # Direct computation : 2x more performant for big arrays
        # Source code automatically generated with formal coputation
        
        w = self.a[..., 0]
        x = self.a[..., 1]
        y = self.a[..., 2]
        z = self.a[..., 3]
        
        eulers = Eulers(shape = self.shape, order=order)
        
        
        if order == 'XYZ':
            
            y2   = y**2
            pole = 2*(w*y - x*z)
            
            eulers.a[..., 0] = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y2))
            eulers.a[..., 1] = np.arcsin(np.clip(pole, -1, 1))
            eulers.a[..., 2] = np.arctan2(2*(w*z + x*y), 1 - 2*(y2 + z**2))
            
            reg = abs(pole) >= 1 - 1e-6
            eulers.a[reg, 0] = 2 * np.arctan2(x[reg], w[reg])
            eulers.a[reg, 2] = 0
          
          
        elif order == 'XZY':
            
            z2   = z**2
            pole = 2*(w*z + x*y)
          
            eulers.a[..., 0] = np.arctan2(2*(w*x - y*z), 1 - 2*(x**2 + z2))
            eulers.a[..., 1] = np.arctan2(2*(w*y - x*z), 1 - 2*(y**2 + z2))
            eulers.a[..., 2] = np.arcsin(np.clip(pole, -1, 1))
            
            reg = abs(pole) >= 1 - 1e-6
            eulers.a[reg, 0] = 2 * np.arctan2(x[reg], w[reg])
            eulers.a[reg, 1] = 0
          
          
        elif order == 'YXZ':
            
            x2   = x**2
            pole = 2*(w*x + y*z)
          
            eulers.a[..., 0] = np.arcsin(np.clip(pole, -1, 1))
            eulers.a[..., 1] = np.arctan2(2*(w*y - x*z), 1 - 2*(x2 + y**2))
            eulers.a[..., 2] = np.arctan2(2*(w*z - x*y), 1 - 2*(x2 + z**2))
            
            reg = abs(pole) >= 1 - 1e-6
            eulers.a[reg, 1] = 0
            eulers.a[reg, 2] = 2 * np.arctan2(z[reg], w[reg])
          
          
        elif order == 'YZX':
            
            z2   = z**2
            pole = 2*(w*z - x*y)
          
            eulers.a[..., 0] = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + z2))
            eulers.a[..., 1] = np.arctan2(2*(w*y + x*z), 1 - 2*(y**2 + z2))
            eulers.a[..., 2] = np.arcsin(np.clip(pole, -1, 1))
            
            reg = abs(pole) >= 1 - 1e-6
            eulers.a[reg, 0] = 0
            eulers.a[reg, 1] = 2 * np.arctan2(y[reg], w[reg])
          
          
        elif order == 'ZXY':
            
            x2   = x**2
            pole = 2*(w*x - y*z)
          
            eulers.a[..., 0] = np.arcsin(np.clip(pole, -1, 1))
            eulers.a[..., 1] = np.arctan2(2*(w*y + x*z), 1 - 2*(x2 + y**2))
            eulers.a[..., 2] = np.arctan2(2*(w*z + x*y), 1 - 2*(x2 + z**2))
            
            reg = abs(pole) >= 1 - 1e-6
            eulers.a[reg, 1] = 2 * np.arctan2(y[reg], w[reg])
            eulers.a[reg, 2] = 0
          
          
        elif order == 'ZYX':
            
            y2   = y**2
            pole = 2*(w*y + x*z)
          
            eulers.a[..., 0] = np.arctan2(2*(w*x - y*z), 1 - 2*(x**2 + y2))
            eulers.a[..., 1] = np.arcsin(np.clip(pole, -1, 1))
            eulers.a[..., 2] = np.arctan2(2*(w*z - x*y), 1 - 2*(y2 + z**2))     
            
            reg = abs(pole) >= 1 - 1e-6
            eulers.a[reg, 0] = 2 * np.arctan2(x[reg], w[reg])
            eulers.a[reg, 2] = 0
          
          
        return eulers


    # -----------------------------------------------------------------------------------------------------------------------------
    # A tracker orients axis towards a target direction.
    # Another contraint is to have the up axis oriented towards the sky
    # The sky direction is the normally the Z
    #
    # - axis   : The axis to rotate toward the target axis
    # - target : Thetarget direction for the axis
    # - up     : The up direction wich must remain oriented towards the sky
    # - sky    : The up direction must be rotated in the plane (target, sky)
    
    @staticmethod
    def tracker(axis, target, up='Y', sky='Z', no_up=False):
        """Compute a quaternion which rotates an axis towards a target.
        
        The rotation is computed using a complementary axis named 'up' which
        must be oriented upwards.
        The upwards direction is Z by default and can be overriden by the argument 'sky'.
        
        After rotation:
            - 'axis' points towards 'target'.
            - 'up' points such as 'up' cross 'target' is perpendicular to vertical axis.
            - 'sky' is used to specify a sky direction different from Z
        
        Parameters
        ----------
        axis : vector
            The axis to orient.
        target : vector
            The direction the axis must be oriented towards.
        up : vector, optional
            The axis which must be oriented upwards. The default is 'Y'.
        sky : vector, optional
            The direction of the sky, i.e. the upwards direction. The default is 'Z'.
        no_up : bool, optional
            Don't rotate around the target axis. The default is True.
        Raises
        ------
        RuntimeError
            If array lengths are not compatible.
        Returns
        -------
        array of quaternions
            The quaternions that can be used to rotate the axis according the arguments.
        """
        
        uaxis = Vectors.Axis(axis)    # Vectors to rotate
        utarg = Vectors.Axis(target)  # The target direction after the rotation
        
        #shape = np.broadcast_shapes(uaxis.shape, utxs.shape)
        #uaxis.resize(shape)
        #utarg.resize(shape)
        
        # ===========================================================================
        # First rotation
        
        # ---------------------------------------------------------------------------
        # First rotation will be made around a vector perp to (axis, target)
        
        qrot = uaxis.axis_angles_with(utarg, are_normalized=True).quaternions()
        
        # ---------------------------------------------------------------------------
        # No up management (for cylinders for instance)
        
        if no_up:
            return qrot
        
        # ===========================================================================
        # Second rotation around the target axis
            
        # ---------------------------------------------------------------------------
        # The first rotation places the up axis in a certain direction
        # An additional rotation around the target is required
        # to put the up axis in the plane (target, up_direction)
    
        # The "sky" is normally the Z direction. Let's name it Z for clarity
        # If there are only one vector, the number of sky can give the returned shape
        
        Z   = Vectors.Axis(sky)     # Direction of the sky (let's name it Z for clarity)
        uup = Vectors.Axis(up)      # Up axis of the object
        
        # Since with must rotate 'up vector' in the plane (Z, target),
        # let's compute a normalized vector perpendicular to this plane
        
        N = Z.perp_with(utarg)
        
        # Let's compute where is now the up axis
        # Note that 'up axis' is supposed to be perpendicular to the axis.
        # Hence, the rotated 'up' direction is perpendicular to the plane (Z, target)
        
        rotated_up = qrot @ uup
        
        # Let's compute the angle between the 'rotated up' and the plane (Z, target)
        
        qrot2 = rotated_up.axis_angles_with_plane(N).quaternions()
        
        # Let's combine teh two rotations
        
        return qrot2 @ qrot
    
    
# ====================================================================================================
# Axis angles

class AxisAngles(VectorsT, Rotation):
    
    def __init__(self, axis, shape=None, angles=None):
        if angles is None:
            super().__init__(axis, shape=shape)
        else:
            v3 = Vectors(axis, shape)
            shape = np.broadcast_shapes(v3.shape, np.shape(angles))
            a = np.empty(shape + (4,), float)
            a[..., :3] = v3.a
            a[..., 3]  = angles
            super().__init__(a)

    # ---------------------------------------------------------------------------
    # Item shape
        
    @staticmethod
    def item_shape():
        return (4,)    
    
    # ---------------------------------------------------------------------------
    # Item to string
        
    def str_item(self, item):
        return f"[{item[0]:8.3f} {item[1]:8.3f} {item[2]:8.3f}], ag: {np.degrees(item[3]):6.1f}"
    
    # ---------------------------------------------------------------------------
    # Axis and angles
    
    @property
    def axis(self):
        return Vectors(self.a[..., :3])
    
    @axis.setter
    def axis(self, value):
        self.a[..., :3] = self.other_a(value)
    
    @property
    def angles(self):
        return self.a[..., 3]
    
    @angles.setter
    def angles(self, value):
        self.a[..., 3] = value
        
    @property
    def anglesd(self):
        return np.degrees(self.a[..., 3])
    
    @anglesd.setter
    def anglesd(self, value):
        self.a[..., 3] = np.radians(value)
    
    # ---------------------------------------------------------------------------
    # Rotation interface
    
    @classmethod
    def Identity(cls, shape=None):
        return cls((1, 0, 0, 0), shape=shape)
    
    def axis_angles(self):
        return self
    
    def quaternions(self):
        quat = Quaternions(shape = self.shape)
        quat.a[..., 0]  = np.cos(self.angles/2)
        quat.a[..., 1:] = self.axis.normalized().a * np.expand_dims(np.sin(self.angles/2), axis=-1)

        return quat        
    
    def inverse(self):
        axg = AxisAngles(np.array(self.a))
        axg.angles *= -1
        return axg
    
    def rotate(self, coords):
        return self.quaternions().rotate(coords)
    
    def compose(self, other):
        return self.quaternions().compose(self.same(other).quaternions()).axis_angles()
    
    # ---------------------------------------------------------------------------
    # Marices
    
    def matrices(self):
        return self.eulers().matrices()

    # ---------------------------------------------------------------------------
    # Eulers angles
    
    def eulers(self, order='XYZ'):
        return self.quaternions().eulers(order)
    
# ====================================================================================================
# Transformations matrices

class TMatrices(GeoArray):
    
    def __init__(self, tmatrices=None, shape=None, locations=None, matrices=None, scales=None):
        
        # ----- Target shape
        
        shape = np.broadcast_shapes(TMatrices.other_shape(tmatrices), shape,
                    Vectors.other_shape(locations), Matrices.other_shape(matrices), Vectors.other_shape(scales))

        # ----- Nothing is passed: a 0-length array
        
        if shape == ():
            return super().__init__(shape=(0,))
        
        # ----- Initialize with identity of the matrices
        
        if tmatrices is None:
            empty = True
            super().__init__(np.identity(4), shape=shape)
        else:
            empty = False
            super().__init__(tmatrices, shape=shape)
            
        # ----- Insert the locations
            
        if locations is not None:
            self.locations = locations

        # ----- Insert matrices and scales together
        
        if matrices is None:
            if scales is not None:
                if empty:
                    self.matrices_scales = np.identity(3), scales
                else:
                    self.scales = scales
        else:
            if scales is None:
                self.matrices = matrices
            else:
                self.matrices_scales = matrices, scales
                
    # ---------------------------------------------------------------------------
    # Item shape
        
    @staticmethod
    def item_shape():
        return (4, 4)

    # ---------------------------------------------------------------------------
    # Identity
    
    def str_item(self, item):
        return "TOTO"
    
        axg = Quaternions(item).axis_angles()
        v = axg.axis.a[0]
        sax = f"[{v[0]:8.3f} {v[1]:8.3f} {v[2]:8.3f}]"
        return f"[{item[0]:8.3f} {item[1]:8.3f} {item[2]:8.3f} {item[3]:8.3f}] ax: {sax} ag: {np.degrees(axg.angles[0]):6.1f}°"
    
    # ---------------------------------------------------------------------------
    # Identity
    
    @classmethod
    def Identity(cls, shape=None):
        return cls(np.identity(4), shape=shape)
    
    # ---------------------------------------------------------------------------
    # Scaled matrices
    
    @property
    def scaled_matrices(self):
        return Matrices(self.a[..., :3, :3])
    
    @scaled_matrices.setter
    def scaled_matrices(self, value):
        self.a[..., :3, :3] = Matrices.other_a(value)
        
    # ---------------------------------------------------------------------------
    # Scales and matrices
    
    @property
    def matrices_scales(self):
        
        mats   = self.scaled_matrices
        scales = Vectors(np.linalg.norm(mats.a, axis=-1))
        
        # ----- Ensure no null scale
        nulls = np.where(scales.a < zero)
        scales.a[nulls] = 1.
        
        # ----- Normalize the matrices
        
        mats.a /= np.expand_dims(scales.a, -1)
        
        # Restore the nulls
        scales.a[nulls] = 0.
        
        # ----- Let's return the results
    
        return mats, scales    
    
    @matrices_scales.setter
    def matrices_scales(self, value):
        
        mats   = Matrices.same(value[0])
        scales = Vectors.same(value[1])
        
        if mats.shape != self.shape:
            mats = Matrices(value[0], self.shape)
        
        if scales.shape != self.shape:
            scales = Vectors(value[1])
        
        mats.a[..., 0, :3] *= np.expand_dims(scales.x, axis=-1)
        mats.a[..., 1, :3] *= np.expand_dims(scales.y, axis=-1)
        mats.a[..., 2, :3] *= np.expand_dims(scales.z, axis=-1)
        
        self.scaled_matrices = mats
        
    # ---------------------------------------------------------------------------
    # Locations
    
    @property
    def locations(self):
        return Vectors(self.a[..., 3, :3])
    
    @locations.setter
    def locations(self, value):
        self.a[..., 3, :3] = Vectors.other_a(value)

    # ---------------------------------------------------------------------------
    # Scales
    
    @property
    def scales(self):
        return Vectors(np.linalg.norm(self.scaled_matrices.a, axis=-1))
    
    @scales.setter
    def scales(self, value):
        mats, scales = self.matrices_scales
        self.matrices_scales = mats, value

    # ---------------------------------------------------------------------------
    # Matrices
        
    @property
    def matrices(self):
        mats, scales = self.matrices_scales
        return mats
    
    @matrices.setter
    def matrices(self, value):
        mats, scales = self.matrices_scales
        self.matrices_scales = value, scales
        
    # ---------------------------------------------------------------------------
    # Operators
    
    def __matmul__(self, other):
        if issubclass(type(other), TMatrices):
            return self.compose(other)
        else:
            return self.transform(other)
        
    def __pow__(self, other):
        if issubclass(type(other), TMatrices):
            return self.compose_explode(other)
        else:
            return self.transform_explode(other)
        
    # ---------------------------------------------------------------------------
    # Composition
    
    def compose(self, other):
        return TMatrices(np.matmul(self.a, self.other_a(other)))
    
    # ---------------------------------------------------------------------------
    # Transformation
    
    def transform(self, vectors):
        
        if issubclass(type(vectors), GeoArray):
            ok_v4 = vectors.vdim == 4
            if ok_v4:
                vs = vectors
            else:
                vs = VectorsT.FromVectors(vectors, t=1)
        else:
            ok_v4 = True
            vs = VectorsT(vectors)
            
        res = np.matmul(vs.a, self.a)
        
        if ok_v4:
            return VectorsT(res)
        else:
            return Vectors(np.array(res[..., :3]))

    # ---------------------------------------------------------------------------
    # Transformation with explosion
    
    def transform_explode(self, vectors):
        
        return None
        
        if issubclass(type(vectors), GeoArray):
            ok_v4 = vectors.vdim == 4
            if ok_v4:
                vs = vectors
            else:
                vs = VectorsT.FromVectors(vectors, t=1)
        else:
            ok_v4 = True
            vs = VectorsT(vectors)
            
        res = np.matmul(vs.a, self.a)
        
        if not ok_v4:
            return res.vectors3
        else:
            return res

        







