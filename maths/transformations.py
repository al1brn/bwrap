#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:58:47 2021

@author: alain
"""

import numpy as np

if True:
    from .shapes import get_main_shape, get_full_shape
    from .geometry import get_axis, scalar_mult, signed_axis_index, q_tracker, axis_angle, quaternion
    from .geometry import tmat_transform4, tmat_transform43, tmat_transform, tmat_compose
    from .geometry import tmat_inv_transform, tmat_vect_transform, tmat_vect_inv_transform
    from .geometry import tmatrix, mat_from_tmat, scale_from_tmat, decompose_tmat, mat_scale_from_tmat
    from .geometry import m_to_euler, e_to_matrix, q_to_matrix, m_to_quat
    
    from ..core.commons import WError
else:
    from shapes import get_main_shape, get_full_shape
    from geometry import get_axis, scalar_mult, signed_axis_index, q_tracker, axis_angle, quaternion
    from geometry import tmat_transform4, tmat_transform43, tmat_transform, tmat_compose
    from geometry import tmat_inv_transform, tmat_vect_transform, tmat_vect_inv_transform
    from geometry import tmatrix, mat_from_tmat, scale_from_tmat, decompose_tmat, mat_scale_from_tmat
    from geometry import m_to_euler, e_to_matrix, q_to_matrix, m_to_quat
    
    WError = RuntimeError
    
# =============================================================================================================================
# A class which implement location, rotation and scale can be transformed: oriented, scaled, translated, rotated...
# Such a class is named a Frame. 

class Transformations():
    """Manages an array of transformations matrices.
    
    This class can be used as is to manage transformation matrices.
    
    It can be overriden to managed 3D-objects for each of them a transformation
    matrix is required. When overriden, the apply method is used to apply the transforamtions
    to the target.
    
    A lock / unlock mechanism is implemented to lock transformations while updating
    the matrices. Typical use is:
        trans.lock()
        # Operations on matrices
        ...
        trans.unlock()
        
    the lock_apply method calls apply method only when the transformation is unlocked.
    """
    
    def __init__(self, location=0., matrix=np.identity(3, np.float), scale=1, count=None):
        self.locked = 0
        self.tmat_ = tmatrix(location, matrix, scale, count)
        
    # ---------------------------------------------------------------------------
    # Basics
        
    def __repr__(self):
        np.set_printoptions(precision=3)
        s = f"<{self.__class__.__name__}: size={self.size} shape={self.shape}>"
        np.set_printoptions(suppress=True)
        return s
    
    # ===========================================================================
    # From transformation matrix
    
    @staticmethod
    def FromTMat(tmat):
        transfo = Transformations(count=get_main_shape(np.shape(tmat), (4, 4)))
        transfo.tmat_ = np.array(tmat)
        return transfo
    
    def compose(self, other, center=0.):
        return Transformations.FromTMat(tmat_compose(self.tmat, other.tmat, center=center))
    
    def compose_with(self, other, center=0.):
        self.tmat = self.compose(other, center=center).tmat
    
    # ===========================================================================
    # Can be overriden

    # ---------------------------------------------------------------------------
    # Euler order and orientation axis

    @property
    def euler_order(self):
        """Order to use when handling eulers.
        
        Can be overriden if the default value is not XYZ.

        Returns
        -------
        str
            The euler order.
        """
        
        return 'XYZ'
    
    @property
    def track_axis(self):
        return 'Z'
    
    @property
    def up_axis(self):
        return 'Y'
    
    # ---------------------------------------------------------------------------
    # Transformation matrices
    
    @property
    def tmat(self):
        return self.tmat_
    
    @tmat.setter
    def tmat(self, value):
        self.tmat_[:] = value
        self.lock_apply()
        
    def set_tmat(self, tmat):
        self.tmat_ = np.array(tmat)
        self.lock_apply()

    # -----------------------------------------------------------------------------------------------------------------------------
    # Override if transformation matrices has to be applied to real objects
    
    def apply(self):
        """Apply the transformations to the target objects.
        
        Normally, there is no need to directly call this method. This method is part of
        the optimization mechanism based on lock and unlock methods.
        
        Lock (unlock) increments (decrements) a counter. When the counter reaches 0 at unlock time,
        apply is called.
        
        If lock / unlock optimization mechanism is not used, apply is called each time transformation
        matrices are modified. A modification to the transformation matrices calls lock_apply method which
        calls apply only if it isn't locked."
        
        Doesn't do nothing. Must be overriden. typical implementation is:
            def apply(self):
                # verts are read somewhere
                self.transform_verts43(verts)

        Returns
        -------
        None.
        """
    
        pass        
    
    # ===========================================================================
    # Lock / unlock
    
    def lock(self):
        self.locked += 1
        
    def unlock(self):
        self.locked -= 1
        if self.locked < 0:
            raise WError("Algorithm error: lock / unlock unbalanced. Ensure to call 'lock' before calling 'unlock'.")
            
        if self.locked == 0:
            self.apply()
            
    def lock_apply(self):
        if self.locked == 0:
            self.apply()
        
    # ===========================================================================
    # Shape management
        
    # ---------------------------------------------------------------------------
    # The shape of matrices : allow to manage objects in a multi-dimensional
    # array
    
    @property
    def shape(self):
        return get_main_shape(self.tmat.shape, (4, 4))

    # ---------------------------------------------------------------------------
    # Change the oganization of the matrices
    
    def reshape(self, new_shape):
        if np.product(new_shape) != np.product(self.shape):
            raise WError("Impossible to reshape the transformations matrices:",
                    Class = "RootTransformations",
                    Method = "reshape",
                    new_shape = new_shape,
                    current_shape = self.shape)
            
        self.tmat_ = self.tmat_.reshape(get_full_shape(new_shape, (4, 4)))
        
    # ---------------------------------------------------------------------------
    # Total number of matrices        
        
    @property
    def size(self):
        return np.product(self.shape)
    
    # ---------------------------------------------------------------------------
    # Number of dimensions       
    
    @property
    def ndim(self):
        try:
            return len(self.shape)
        except:
            return 1
        
    # ===========================================================================
    # As an array of transformation matrices
    # Note that:
    # - getitem returns a class TransformationSlicer
    # - setitem needs a compatible array of matrices
        
    def __len__(self):
        
        shape = self.shape
        
        if hasattr(shape, '__len__'):
            if len(shape) == 0:
                return 1
            else:
                return shape[len(shape)-1]
        else:
            return shape
        
    
    def __getitem__(self, index):
        return TransformationsSlicer(self, index)

    def __setitem__(self, index, value):
        self.tmat[index] = value
        self.lock_apply()
        
    # ===========================================================================
    # Implement geomtry from transformation matrices
    
    # ---------------------------------------------------------------------------
    # Location
        
    @property
    def location(self):
        return np.array(self.tmat[..., 3, :3])
    
    @location.setter
    def location(self, value):
        self.tmat[..., 3, :3] = value
        self.lock_apply()
        
    @property 
    def x(self):
        return np.array(self.tmat[..., 3, 0])
    
    @x.setter
    def x(self, value):
        self.tmat[..., 3, 0] = value 
        self.lock_apply()
        
    @property 
    def y(self):
        return np.array(self.tmat[..., 3, 1])
    
    @y.setter
    def y(self, value):
        self.tmat[..., 3, 1] = value 
        self.lock_apply()
        
    @property 
    def z(self):
        return np.array(self.tmat[..., 3, 2])
    
    @z.setter
    def z(self, value):
        self.tmat[..., 3, 2] = value 
        self.lock_apply()
    
    # ---------------------------------------------------------------------------
    # Scales
    
    @property
    def scale(self):
        return scale_from_tmat(self.tmat)
    
    @scale.setter
    def scale(self, value):
        self.lock()
        m, s = mat_scale_from_tmat(self.tmat)
        self.tmat = tmatrix(self.location, m, value)
        self.unlock()
    
    @property
    def sx(self):
        return self.scale[..., 0]
    
    @sx.setter
    def sx(self, value):
        s = self.scale
        s[..., 0] = value
        self.scale = s
    
    @property
    def sy(self):
        return self.scale[..., 1]
    
    @sy.setter
    def sy(self, value):
        s = self.scale
        s[..., 1] = value
        self.scale = s
    
    @property
    def sz(self):
        return self.scale[..., 2]

    @sz.setter
    def sz(self, value):
        s = self.scale
        s[..., 2] = value
        self.scale = s

    # ---------------------------------------------------------------------------
    # Rotation matrices
    
    @property
    def matrix(self):
        return mat_from_tmat(self.tmat)
    
    @matrix.setter
    def matrix(self, value):
        self.lock()
        l, m, s = decompose_tmat(self.tmat)
        self.tmat = tmatrix(l, value, s)
        self.unlock()
        
    # ---------------------------------------------------------------------------
    # Euler
    
    @property
    def euler(self):
        return m_to_euler(self.matrix, self.euler_order)
    
    @euler.setter
    def euler(self, value):
        self.matrix = e_to_matrix(value, self.euler_order)
        
    @property
    def eulerd(self):
        return np.degrees(self.euler)
    
    @eulerd.setter
    def eulerd(self, value):
        self.euler = np.radians(value)
        
    @property
    def rx(self):
        return self.euler[..., 0]
    
    @rx.setter
    def rx(self, value):
        euler = self.euler
        euler[..., 0] = value
        self.euler = euler
        
    @property
    def ry(self):
        return self.euler[..., 1]
    
    @ry.setter
    def ry(self, value):
        euler = self.euler
        euler[..., 1] = value
        self.euler = euler
        
    @property
    def rz(self):
        return self.euler[..., 2]
    
    @rz.setter
    def rz(self, value):
        euler = self.euler
        euler[..., 2] = value
        self.euler = euler
        
    # ---------------------------------------------------------------------------
    # Degrees version
    
    @property
    def rxd(self):
        return np.degrees(self.rx)
    
    @rxd.setter
    def rxd(self, value):
        self.rx = np.radians(value)
        
    @property
    def ryd(self):
        return np.degrees(self.ry)
    
    @ryd.setter
    def ryd(self, value):
        self.ry = np.radians(value)
        
    @property
    def rzd(self):
        return np.degrees(self.rz)
    
    @rzd.setter
    def rzd(self, value):
        self.rz = np.radians(value)
        
    # ---------------------------------------------------------------------------
    # Quaternions
    
    @property
    def quaternion(self):
        return m_to_quat(self.matrix)
    
    @quaternion.setter
    def quaternion(self, value):
        self.matrix = q_to_matrix(value)   
            
    # ---------------------------------------------------------------------------
    # Axis angle
    
    @property
    def axis_angle(self):
        return axis_angle(self.quaternion)
    
    @axis_angle.setter
    def axis_angle(self, value):
        self.quaternion = quaternion(value[0], value[1])
    
    @property
    def axis_angled(self):
        ax, ag = self.axis_angle
        return ax, np.degrees(ag)
    
    @axis_angled.setter
    def axis_angled(self, value):
        self.axis_angle = (value[0], np.radians(value[1]))
        
    # ===========================================================================
    # Transform
    # If mats is the number of matrices, ie mats=len(tmat), the shape of verts
    # can be of two shapes:
    # - (n, 4)       
    # - (mats, n, 4)
    # In both cases, it returns n*mats vertices reshaped in (n*mats, 4)
    # n is the number of vertices of the mesh to transform

    def transform_verts4(self, verts4, one_one=False):
        return tmat_transform4(self.tmat, verts4, one_one=one_one)

    def transform_verts43(self, verts4, one_one=False):
        return tmat_transform43(self.tmat, verts4, one_one=one_one)

    def transform(self, verts3, one_one=False):
        return tmat_transform(self.tmat, verts3, one_one=one_one)

    def inv_transform(self, verts3, one_one=False):
        return tmat_inv_transform(self.tmat, verts3, one_one=one_one)
        
    # ---------------------------------------------------------------------------
    # Vector transformation
        
    def vector_transform(self, vectors, one_one=False):
        return tmat_vect_transform(self.tmat, vectors)

    def vector_inv_transform(self, vectors, one_one=False):
        return tmat_vect_inv_transform(self.tmat, vectors, one_one=one_one)
    
    # ===========================================================================
    # Geometric utilities
    

    # ---------------------------------------------------------------------------
    # Spherical coordinates
    
    def get_axis_distance(self, center=0):
        """Get the directions and distances to a given center.

        Parameters
        ----------
        center : vector or array of vectors, optional
            The locations to compute the directions to and the distances to. The default is (0., 0., 0.).

        Returns
        -------
        array of vectors
            The normalized axis between the items and the center.
        array of floats
            The distances between the items and the center
        """
        
        loc = self.location - center
        
        dist = np.linalg.norm(loc, axis=-1)
        return get_axis(loc), dist
    
    # ---------------------------------------------------------------------------
    # Set the spherical coordinates
    
    def set_axis_distance(self, axis=None, distance=None, center=0):
        """Change the locations using spherical axis and distances to given center.

        Parameters
        ----------
        axis : array of vectors, optional
            The directions between the items and the locations. The default is None.
        distances : array of floats, optional
            The distances between the items and the locations. The default is None.
        center : vector or array of vectors, optional
            The locations from which to locate the items. The default is (0., 0., 0.).

        Returns
        -------
        None.
        """
        
        # ----- Default values
        if (axis is None) or (distance is None):
            a, d = self.get_axis_distance(center)
            if axis is None:
                axis = a
            if distance is None:
                distance = d
                
        # ----- Change the locations
        self.location = center + scalar_mult(distance, axis)
        
    # ---------------------------------------------------------------------------
    # Distances
    
    def distance(self, center=0):
        """Distances to given centers.

        Parameters
        ----------
        center : vector or array of vectors, optional
            The location(s) to compte the distance to. The default is (0, 0, 0).

        Returns
        -------
        array of floats
            The distances.
        """
        
        return np.linalg.norm(self.location - center, axis=-1)
    
    def proximity(self, center=0, bounds=(0., 1.)):
        """A normalized factor between 0 and 1 giving the proximity to locations.
        
        The result is 0 if the distance is less than the lower bound, 1 if the distance
        is greater than the upper bound, and between 0 and 1 for intermediary values.

        Parameters
        ----------
        center : vector or array of vectors, optional
            The locations to compute the proximity to. The default is (0, 0, 0).
        bounds : couple of floats, optional
            The bounds where to map the interval [0, 1] of the result. The default is (0., 1.).

        Returns
        -------
        array of floats
            The factors.
        """
        
        return np.clip((self.distance(center) - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    
    def direction(self, center=0):
        """The directions towards given center.
        
        Return an array of normalized vectors.

        Parameters
        ----------
        center : vector or array of vectors, optional
            The locations to compute the directions to. The default is (0., 0., 0.).

        Returns
        -------
        array of vectors
            The normalized vectors towards the center.
        """
        
        return get_axis(self.location - center)
        
    @property
    def radius(self):
        return self.distance(center=(0., 0., 0.))
    
    @radius.setter
    def radius(self, value):
        self.set_axis_distance(distance=value, center=(0., 0., 0.))
        
    # ---------------------------------------------------------------------------
    # Normals orientation
    
    @property
    def normal(self):
        iup, sup = signed_axis_index(self.up_axis)
        return self.matrix[..., iup, :]*sup

    @normal.setter
    def normal(self, value):
        
        # Relative:      normals are rotated towards the target
        # Not relative : transformation rotation is changed
        
        relative = True
        if relative:
            
            # ----- The rotation from current normals towards target
            q = q_tracker(self.normal, get_axis(value), no_up=True)
            
            # ----- The axis is transformed, must invert it
            axis, angle = axis_angle(q)
            inv = self.vector_inv_transform(axis, one_one=True)
            
            # ----- Rotation
            self.rotate_quat(quaternion(inv, angle))
            
        else:
            self.quaternion = q_tracker(get_axis(self.up_axis), np.resize(get_axis(value), (len(self), 3)), no_up=True)

    # ---------------------------------------------------------------------------
    # Orientation
    
    def orient(self, target, axis=None, up=None, sky='Z', no_up=False):
        if axis is None:
            axis = self.track_axis

        if up is None:
            up = self.up_axis
            
        self.quaternion = q_tracker(axis, target=target, up=up, sky=sky, no_up=no_up)
        
    def track_to(self, location, axis=None, up=None, sky='Z', no_up=False):
        self.orient(location - self.location, axis=axis, up=up, sky=sky, no_up=no_up)
        
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Override relative transformations
    
    def rotate_mat(self, mat, center=0):
        self.lock()
        
        self.tmat = tmat_compose(self.tmat, tmatrix(location=-center, matrix=mat))
        self.location += center
        
        self.unlock()
    
    def rotate(self, axis, angle, center=0):
        self.rotate_mat(q_to_matrix(quaternion(axis, angle)), center=center)
        
    def rotated(self, axis, angle, center=0):
        self.rotate(axis, np.radians(angle), center=center)
        
    def rotate_euler(self, euler, center=0):
        self.rotate_mat(e_to_matrix(euler, self.euler_order), center=center)

    def rotate_eulerd(self, euler, center=0):
        self.rotate_euler(np.radians(euler), center=center)
        
    def rotate_quat(self, quat, center=0):
        self.rotate_mat(q_to_matrix(quat), center=center)
        
    def rotate_vector(self, axis, target, center=0):
        self.rotate_mat(q_to_matrix(q_tracker(axis, target, no_up=True)), center=center)

    # ---------------------------------------------------------------------------
    # Randomization
    
    def random(self, noise=lambda a, b, shape: np.random.uniform(a, b, shape), location=None, euler=None, scale=None):
        self.lock()
        
        if location is not None:
            self.location = noise(location[0], location[1], get_full_shape(self.shape, 3))
        if scale is not None:
            self.scale = noise(scale[0], scale[1], get_full_shape(self.shape, 3))
        if euler is not None:
            self.euler = noise(euler[0], euler[1], get_full_shape(self.shape, 3))
        
        self.unlock()
        
# =============================================================================================================================
# A slave Transformations (apply )

class SlaveTransformations(Transformations):
    
    def __init__(self, transformations):
        super().__init__(count=transformations.shape)
        self.mother = transformations
        
    def __repr__(self):
        return f"<Slave transformations {self.shape}>"
    
    def apply(self):
        self.mother.lock_apply()
    
    def lock_apply(self):
        self.mother.lock_apply()
    
    def lock(self):
        self.mother.lock()
        
    def unlock(self):
        self.mother.unlock()
        

# =============================================================================================================================
# A slicer on to a Transformations

class TransformationsSlicer(Transformations):
    
    def __init__(self, transformations, index):
        super(TransformationsSlicer, self).__init__()
        self.mother = transformations
        self.slice  = index
        
    def __repr__(self):
        return f"Slicer transformations of slice {self.slice} on:\n{self.mother}"
    
    @property
    def tmat(self):
        return self.mother.tmat[self.slice]
    
    @tmat.setter
    def tmat(self, value):
        self.mother.tmat[self.slice] = value
        self.mother.lock_apply()
    
    def apply(self):
        self.mother.apply()
    
    def lock_apply(self):
        self.mother.lock_apply()
    
    def lock(self):
        self.mother.lock()
        
    def unlock(self):
        self.mother.unlock()
        
# =============================================================================================================================
# A single transformation matrix to be used in a multiple inheritance with Object Wrapper

class ObjectTransformations(Transformations):
    
    def __init__(self, world=False):
        
        Transformations.__init__(self, count=())
        self.world_ = world

        if self.world:
            self.tmat_ = np.transpose(self.wrapped.matrix_world)
        else:
            self.tmat_ = np.transpose(self.wrapped.matrix_basis)
    
    @property
    def world(self):
        return self.world_
    
    @world.setter
    def world(self, value):
        self.apply()
        self.world_ = value
        if self.world_:
            self.tmat_ = np.transpose(self.wrapped.matrix_world)
        else:
            self.tmat_ = np.transpose(self.wrapped.matrix_basis)
    
    def apply(self):
        # CAUTION: No transpose when back to the object
        if self.world_:
            self.wrapped.matrix_world = self.tmat_
        else:
            self.wrapped.matrix_basis = self.tmat_
            
    @property
    def tmat(self):
        if self.world:
            self.tmat_ = np.transpose(self.wrapped.matrix_world)
        else:
            self.tmat_ = np.transpose(self.wrapped.matrix_basis)
        return self.tmat_
    
    @tmat.setter
    def tmat(self, value):
        self.tmat_ = value
        self.apply()
            
    
    @property
    def euler_order(self):
        return self.wrapped.rotation_euler.order
    
    @property
    def track_axis(self):
        return self.wrapped.track_axis

    @property
    def up_axis(self):
        return self.wrapped.up_axis
    
# =============================================================================================================================
# Mulitple transformation matrices to be used with the objects of a collection

class ObjectsTransformations(Transformations):
    
    def __init__(self, world=False, shape=None, owner=True):
        
        self.shape_ = self.check_shape(shape)
        
        Transformations.__init__(self, shape=self.shape_)
        self.world_ = world
        self.owner  = owner
            
    def check_shape(self, shape):
        count = len(self.wrapped.objects)
        if shape is None:
            return count
        if np.product(shape) != count:
            raise WError(f"Shape {shape} not compatible with the number {count} of objects.",
                    Class = "ObjectsTransformations")
        return shape
    
    @property
    def shape(self):
        return self.shape_
    
    def reshape(self, shape):
        self.shape_ = self.check_shape(shape)
        
    def read_tmat(self):
        objs = self.wrapper.objects
        count = len(objs)
        a = np.empty(count*16, np.float)
        if self.world_:
            objs.foreach_get('matrix_world', a)
        else:
            objs.foreach_get('matrix_local', a)
        return a.reshape(get_full_shape(self.shape, (4, 4)))
                         
    def write_tmat(self, tmat):
        objs = self.wrapper.objects
        count = len(objs)
        if self.world_:
            objs.foreach_set('matrix_world', tmat.reshape(count*16))
        else:
            objs.foreach_set('matrix_local', tmat.reshape(count*16))
        
    @property
    def tmat(self):
        if self.owner:
            if self.tmat_ is None:
                self.tmat_ = self.read_tmat()
            return self.tmat_
        else:
            self.tmat_ = None
            return self.read_tmat()
        
    @tmat.setter
    def tmat(self, value):
        if self.owner:
            if self.tmat_ is None:
                self.tmat_ = self.read_tmat()
            self.tmat_[:] = value
            self.lock_apply()
        else:
            self.tmat_ = None
            tmat = self.read_tmat()
            tmat[:] = value
            self.write_tmat(tmat)
            
    @property
    def world(self):
        return self.world_
    
    @world.setter
    def world(self, value):
        self.apply()
        self.world_ = value
        if self.world_:
            self.tmat_ = np.transpose(self.wrapped.matrix_world)
        else:
            self.tmat_ = np.transpose(self.wrapped.matrix_basis)
    
    def apply(self):
        if self.owner and (self.tmat_ is not None):
            self.write_tmat(self.tmat_)
    
    @property
    def euler_order(self):
        objs = self.wrapper.objects
        count = len(objs)
        if count == 0:
            return 'XYZ'
        else:
            return objs[0].rotation_euler.order
    
    @property
    def track_axis(self):
        objs = self.wrapper.objects
        count = len(objs)
        if count == 0:
            return 'X'
        else:
            return objs[0].track_axis

        return self.wrapped.track_axis

    @property
    def up_axis(self):
        objs = self.wrapper.objects
        count = len(objs)
        if count == 0:
            return 'Z'
        else:
            return objs[0].up_axis
        
        