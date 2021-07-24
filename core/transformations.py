#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:58:47 2021

@author: alain
"""

import numpy as np
try:
    from .geometry import *
except:
    from geometry import *
    

# =============================================================================================================================
# Root transformations

class RootTransformations():
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
    
    def __init__(self):
        self.locked = 0
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # To override
    
    @property
    def tmat(self):
        return None
    
    @tmat.setter
    def tmat(self, value):
        pass
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # Can be overriden
    
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
    # Basics
        
    def __repr__(self):
        n = len(self.tmat)
        np.set_printoptions(precision=3)
        s = f"<{self.__class__.__name__}: len={n} shape={self.shape}>"
        np.set_printoptions(suppress=True)
        return s

    # ---------------------------------------------------------------------------
    # A copy
    
    def clone(self):
        return Transformations.FromTMat(np.array(self.tmat))
        
    # ---------------------------------------------------------------------------
    # Matrices access
    # CAUTION:
    # - getitem return a Slicer Transformations on the matrices
    # - setitem accept directly an array of matrices
    # Caller must check the compatibility of the shapes

    @property
    def shape(self):
        return sub_shape(self.tmat.shape, 2)
    
    @property
    def size(self):
        return np.product(self.shape)
    
    @property
    def ndim(self):
        try:
            return len(self.shape)
        except:
            return 1
        
    def __len__(self):
        return len(self.tmat)
    
    def __getitem__(self, index):
        return TransformationsSlicer(self, index)

    def __setitem__(self, index, value):
        self.tmat[index] = value
        self.lock_apply()

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Lock management
    
    def lock_apply(self):
        if self.locked == 0:
            self.apply()
    
    def lock(self):
        self.locked += 1
        
    def unlock(self):
        self.locked -= 1
        if self.locked < 0:
            raise RuntimeError("Transformations error: lock/unlock use inconsistently: locked is negative")
        if self.locked == 0:
            self.apply()
    
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # Inverse transformations
    
    def inverse(self):
        return Transformations.FromTMat(np.linalg.inv(self.tmat))
    
    # ---------------------------------------------------------------------------
    # Vectors rotation
    
    def vector_rotation(self, vectors, one_one=False):
        return Transformations(matrices=self.matrices).transform(vectors, one_one=one_one)

    def vector_inv_rotation(self, vectors, one_one=False):
        n = self.ndim
        tr = [i for i in range(n)]
        tr.extend([n+1, n])
        return Transformations(matrices=self.matrices.transpose(tr)).transform(vectors, one_one=one_one)
    
    # ---------------------------------------------------------------------------
    # Transform other transformation
    
    def compose_after(self, other):
        self.lock()
        self.tmat = np.matmul(other.tmat, self.tmat)
        self.unlock()
    
    def compose_before(self, other):
        self.lock()
        locs = self.locations
        self.locations = 0
        self.tmat = np.matmul(self.tmat, other.tmat)
        self.locations += locs
        self.unlock()

    # ---------------------------------------------------------------------------
    # Locations
        
    @property
    def locations(self):
        return np.array(self.tmat[..., 3, :3])
    
    @locations.setter
    def locations(self, value):
        self.tmat[..., 3, :3] = np.resize(value, build_shape(self.shape, 3))
        self.lock_apply()
        
    @property 
    def xs(self):
        return np.array(self.tmat[..., 3, 0])
    
    @xs.setter
    def xs(self, value):
        self.tmat[..., 3, 0] = np.resize(value, self.shape)
        self.lock_apply()
        
    @property 
    def ys(self):
        return np.array(self.tmat[..., 3, 1])
    
    @ys.setter
    def ys(self, value):
        self.tmat[..., 3, 1] = np.resize(value, self.shape)
        self.lock_apply()
        
    @property 
    def zs(self):
        return np.array(self.tmat[..., 3, 2])
    
    @zs.setter
    def zs(self, value):
        self.tmat[..., 3, 2] = np.resize(value, self.shape)
        self.lock_apply()
        
    # ---------------------------------------------------------------------------
    # Matrices and scales
    
    @property
    def mat_scales(self):
        return mat_scale_from_tmat(self.tmat)
    
    # ---------------------------------------------------------------------------
    # Scales
    
    @property
    def scales(self):
        return mat_scale_from_tmat(self.tmat)[1]
    
    @scales.setter
    def scales(self, value):
        self.lock()
        m, s = mat_scale_from_tmat(self.tmat)
        self.tmat = tmatrix(self.locations, m, value, count=self.shape)
        self.unlock()
    
    @property
    def sx(self):
        return np.linalg.norm(self.tmat[..., 0, :3], axis=-1)
    
    @sx.setter
    def sx(self, value):
        m, s = self.mat_scales
        s[..., 0] = np.resize(value, self.shape)
        self.scales = s
    
    @property
    def sy(self):
        return np.linalg.norm(self.tmat[..., 1, :3], axis=-1)
    
    @sy.setter
    def sy(self, value):
        m, s = self.mat_scales
        s[..., 1] = np.resize(value, self.shape)
        self.scales = s
    
    @property
    def sz(self):
        return np.linalg.norm(self.tmat[..., 2, :3], axis=-1)

    @sz.setter
    def sz(self, value):
        m, s = self.mat_scales
        s[..., 2] = np.resize(value, self.shape)
        self.scales = s

    # ---------------------------------------------------------------------------
    # Rotation matrices
    
    @property
    def matrices(self):
        return mat_scale_from_tmat(self.tmat)[0]
    
    @matrices.setter
    def matrices(self, value):
        self.lock()
        m, s = self.mat_scales
        t = self.locations
        self.tmat = tmatrix(t, value, s, count=self.shape)
        self.unlock()
        
    # ---------------------------------------------------------------------------
    # Euler
    
    @property
    def eulers(self):
        return m_to_euler(self.matrices, self.euler_order)
    
    @eulers.setter
    def eulers(self, value):
        self.matrices = e_to_matrix(value, self.euler_order)
        
    @property
    def eulersd(self):
        return np.degrees(self.eulers)
    
    @eulersd.setter
    def eulersd(self, value):
        self.eulers = np.radians(value)
        
    @property
    def rx(self):
        return self.eulers[..., 0]
    
    @rx.setter
    def rx(self, value):
        eulers = self.eulers
        eulers[..., 0] = np.resize(value, self.shape)
        self.eulers = eulers
        
    @property
    def ry(self):
        return self.eulers[..., 1]
    
    @ry.setter
    def ry(self, value):
        eulers = self.eulers
        eulers[..., 1] = np.resize(value, self.shape)
        self.eulers = eulers
        
    @property
    def rz(self):
        return self.eulers[..., 2]
    
    @rz.setter
    def rz(self, value):
        eulers = self.eulers
        eulers[..., 2] = np.resize(value, self.shape)
        self.eulers = eulers
        
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
    def quaternions(self):
        return m_to_quat(self.matrices)
    
    @quaternions.setter
    def quaternions(self, value):
        self.matrices = q_to_matrix(value)
        
    # ---------------------------------------------------------------------------
    # Spherical coordinates
    
    def get_axis_distances(self, center=(0., 0., 0.)):
        """Get the directions and distances to given locations.

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
        
        locs = self.locations - center
        
        dists = np.linalg.norm(locs, axis=-1)
        return get_axis(locs), dists
    
    # ---------------------------------------------------------------------------
    # Set the spherical coordinates
    
    def set_axis_distances(self, axis=None, distances=None, center=(0., 0., 0.)):
        """Change the locations using spherical axis and distances to given locations.

        Parameters
        ----------
        axis : array of vectors, optional
            The directions between the items and the locations. The default is None.
        distances : array of floats, optional
            The distances between the items and the locations. The default is None.
        locations : vector or array of vectors, optional
            The locations from which to locate the items. The default is (0., 0., 0.).

        Returns
        -------
        None.
        """
        
        # ----- Default values
        if axis is None or distances is None:
            a, d = self.get_axis_distances(center)
            if axis is None:
                axis = a
            if distances is None:
                distances = d
                
        # ----- Change the locations
        self.locations = center + scalar_mult(distances, axis)
        
    # ---------------------------------------------------------------------------
    # Distances
    
    def distances(self, center=(0, 0, 0)):
        """Distances to given locations

        Parameters
        ----------
        locations : vector or array of vectors, optional
            The location(s) to compte the distance to. The default is (0, 0, 0).

        Returns
        -------
        array of floats
            The distances.
        """
        
        return np.linalg.norm(self.locations - center, axis=-1)
    
    def proximities(self, center=(0, 0, 0), bounds=(0., 1.)):
        """A normalized factor between 0 and 1 giving the proximity to locations.
        
        The result is 0 if the distance is less than the lower bound, 1 if the distance
        is greater than the upper bound, and between 0 and 1 for intermediary values.

        Parameters
        ----------
        locations : vector or array of vectors, optional
            The locations to compute the proximity to. The default is (0, 0, 0).
        bounds : couple of floats, optional
            The bounds where to map the interval [0, 1] of the result. The default is (0., 1.).

        Returns
        -------
        array of floats
            The factors.
        """
        
        return np.clip((self.distances(center) - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    
    def directions(self, center=(0., 0., 0.)):
        """The directions towards given locations.
        
        Return an array of normalized vectors.

        Parameters
        ----------
        locations : vector or array of vectors, optional
            The locations to compute the directions to. The default is (0., 0., 0.).

        Returns
        -------
        array of vectors
            The normalized vectors towards the locations.
        """
        
        return get_axis(self.locations - center)
        
    @property
    def radius(self):
        return distances()
    
    @radius.setter
    def radius(self, value):
        self.set_axis_distances(distances=value)
        
    # ---------------------------------------------------------------------------
    # Normals orientation
    
    @property
    def normals(self):
        iup, sup = signed_axis_index(self.up_axis)
        #return self.matrices[:, :, iup]*sup
        return self.matrices[..., iup, :]*sup

    @normals.setter
    def normals(self, value):
        
        # Relative:      normals are rotated towards the target
        # Not relative : transformation rotation is changed
        
        relative = True
        if relative:
            
            # ----- The rotation from current normals towards target
            q = q_tracker(self.normals, get_axis(value), no_up=True)
            
            # ----- The axis is transformed, must invert it
            axis, angle = axis_angle(q)
            inv = self.vector_inv_rotation(axis, one_one=True)
            
            # ----- Rotation
            self.rotate_quat(quaternion(inv, angle))
            
        else:
            self.quaternions = q_tracker(get_axis(self.up_axis), np.resize(get_axis(value), (len(self), 3)), no_up=True)

    # ---------------------------------------------------------------------------
    # Orientation
    
    def orient(self, target, axis=None, up=None, sky='Z', no_up=False):
        if axis is None:
            axis = self.track_axis

        if up is None:
            up = self.up_axis
            
        #self.matrices = m_tracker(axis, target, up)
        self.quaternions = q_tracker(axis, target=target, up=up, sky=sky, no_up=no_up)
        
    def track_to(self, location, axis=None, up=None, sky='Z', no_up=False):
        self.orient(location - self.locations, axis=axis, up=up, sky=sky, no_up=no_up)
        
    # ---------------------------------------------------------------------------
    # Relative
    
    def rotate(self, axis, angle):
        tf = Transformations(count=self.shape)
        tf.quaternions = quaternion(axis, angle)
        self.compose_after(tf)
        
    def rotated(self, axis, angle):
        tf = Transformations(count=self.shape)
        tf.quaternions = quaternion(axis, np.radians(angle))
        self.compose_after(tf)
        
    def rotate_euler(self, eulers):
        tf = Transformations(count=self.shape)
        tf.matrices = e_to_matrix(eulers, self.euler_order)
        self.compose_after(tf)

    def rotate_eulers(self, eulers):
        tf = Transformations(count=self.shape)
        tf.eulers = np.radians(eulers)
        self.compose_after(tf)
        
    def rotate_mat(self, mat):
        tf = Transformations(count=self.shape)
        tf.matrices = mat
        self.compose_after(tf)
        
    def rotate_quat(self, quat):
        tf = Transformations(count=self.shape)
        tf.quaternions = quat
        self.compose_after(tf)
        
    def rotate_vector(self, axis, target):
        tf = Transformations(count=self.shape)
        tf.quaternions = q_tracker(axis, target, no_up=True)
        self.compose_after(tf)
        
    # ---------------------------------------------------------------------------
    # Randomization
    
    def random(self, noise=lambda a, b, shape: np.random.uniform(a, b, shape), locations=None, eulers=None, scales=None):
        self.lock()
        if locations is not None:
            self.locations = noise(locations[0], locations[1], build_shape(self.shape, 3))
        if scales is not None:
            self.scales = noise(scales[0], scales[1], build_shape(self.shape, 3))
        if eulers is not None:
            self.eulers = noise(eulers[0], eulers[1], build_shape(self.shape, 3))
        self.unlock()
        
        
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# A slicer on to a Transformations

class TransformationsSlicer(RootTransformations):
    
    def __init__(self, transf, index):
        super().__init__()
        self.mother = transf
        self.slice  = index
        
    def __repr__(self):
        return f"Slicer transformations of slice {self.slice} on:\n{super().__repr__()}"
    
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
            

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Transformation Matrices 4x4 as a class

class Transformations(RootTransformations):
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
    
    def __init__(self, locations=(0., 0., 0.), matrices=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), scales=(1., 1., 1.), count=None):
        super().__init__()
        self.tmat_ = tmatrix(locations, matrices, scales, count)
    
    @property
    def tmat(self):
        return self.tmat_

    @tmat.setter
    def tmat(self, value):
        self.tmat_ = value
        self.lock_apply()
        
    # ---------------------------------------------------------------------------
    # Matrices access
        
    def __setitem__(self, index, value):
        self.tmat_[index] = value
        self.lock_apply()

    # ---------------------------------------------------------------------------
    # Create from an array of matrices
        
    @staticmethod
    def FromTMat(tmat):
        trans = Transformations(count=sub_shape(tmat, 2))
        trans.tmat_ = tmat
        return trans

    # ---------------------------------------------------------------------------
    # Can be reshaped and resized
    # Not sur it is usefull
    
    def reshape(self, shape):
        self.tmat_ = np.reshape(self.tmat_, build_shape(shape, (4, 4)))
    
    def resize(self, shape):
        self.tmat_ = np.resize(self.tmat_, build_shape(shape, (4, 4)))
        
        
        
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# A single transformation matrix for a Blender object

class ObjectTransformations(Transformations):
    
    def __init__(self, obj, world=True):
        super().__init__(count=1)
        self.obj   = obj
        self.world = world
        if self.world:
            self.tmat_ = (np.array(obj.matrix_world).transpose()).reshape(1, 4, 4)
        else:
            self.tmat_ = (np.array(obj.matrix_basis).transpose()).reshape(1, 4, 4)
            
    def apply(self):
        if self.world:
            self.obj.matrix_world = self.tmat_[0]
        else:
            self.obj.matrix_basis = self.tmat_[0]
    
    @property
    def euler_order(self):
        return self.obj.rotation_euler.order
    
    @property
    def track_axis(self):
        return self.obj.track_axis

    @property
    def up_axis(self):
        return self.obj.up_axis

