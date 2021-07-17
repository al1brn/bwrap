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

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Transformation Matrices 4x4 as a class

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
    
    def __init__(self, translation=(0., 0., 0.), mat=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), scale=(1., 1., 1.), count=None):        
        self.tmat_ = tmatrix(translation, mat, scale, count)
        self.locked = 0
        
    @classmethod
    def FromTMat(cls, tmat):
        trans = cls()
        trans.tmat_ = tmat
        return trans
    
    def clone(self):
        return type(self).FromTMat(np.array(self.tmat))
        
    def __len__(self):
        return len(self.tmat)
    
    def __getitem__(self, index):
        return self.tmat[index]
    
    # ---------------------------------------------------------------------------
    # To be overriden
    
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
    
    # ---------------------------------------------------------------------------
    # Redime
    
    def set_length(self, new_length):
        if len(self.tmat_) != new_length:
            self.tmat_ = np.resize(self.tmat, (new_length, 4, 4))

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
    # Transformation matrices
    
    @property
    def tmat(self):
        return self.tmat_
    
    @tmat.setter
    def tmat(self, value):
        self.tmat_ = value
        self.lock_apply()
    
    # ---------------------------------------------------------------------------
    # Transform
    # If mats is the number of matrices, ie mats=len(tmat), the shape of verts
    # can be of two shapes:
    # - (n, 4)       
    # - (mats, n, 4)
    # In both cases, it returns n*mats vertices reshaped in (n*mats, 4)
    # n is the number fo vertices of the mesh to transform

    def transform_verts4(self, verts):
        count = verts.shape[-2]*len(self.tmat)
        return np.matmul(verts, self.tmat).reshape(count, 4)
    
        # ----- OLD: tmat is resized to have one matrix per vertex
        return np.einsum('...jk,...k', 
            np.resize(self.tmat, (len(verts), 4, 4)),
            verts)

    def transform_verts3(self, verts):
        return self.transform_verts4(
            np.column_stack(verts, np.ones(len(verts))))[:, :3]
    
    def transform_verts43(self, verts):
        return self.transform_verts4(verts)[:, :3]
    
    # ---------------------------------------------------------------------------
    # Transform other transformation
    
    def compose_after(self, other):
        self.tmat = np.matmul(other.tmat, self.tmat)
        self.lock_apply()
    
    def compose_before(self, other):
        self.tmat = np.matmul(self.tmat, other.tmat)
        self.lock_apply()
    
    # ---------------------------------------------------------------------------
    # Locations
        
    @property
    def locations(self):
        return np.array(self.tmat[:, 3, :3])
    
    @locations.setter
    def locations(self, value):
        self.tmat[:, 3, :3] = np.resize(value, (len(self), 3))
        self.lock_apply()
        
    @property 
    def xs(self):
        return np.array(self.tmat[:, 3, 0])
    
    @xs.setter
    def xs(self, value):
        self.tmat[:, 3, 0] = np.resize(value, len(self))
        self.lock_apply()
        
    @property 
    def ys(self):
        return np.array(self.tmat[:, 3, 1])
    
    @ys.setter
    def ys(self, value):
        self.tmat[:, 3, 1] = np.resize(value, len(self))
        self.lock_apply()
        
    @property 
    def zs(self):
        return np.array(self.tmat[:, 3, 2])
    
    @zs.setter
    def zs(self, value):
        self.tmat[:, 3, 2] = np.resize(value, len(self))
        self.lock_apply()
        
    # ---------------------------------------------------------------------------
    # Matrices and scales
    
    @property
    def mat_scales(self):
        
        return mat_scale_from_tmat(self.tmat)
        
        scales = self.scales

        mat = np.array(self.tmat[:, :3, :3])
        
        invs = 1/scales
        scale_mat = np.resize(np.identity(3), (mat.shape[0], 3, 3))
        scale_mat[:, 0, 0] = invs[:, 0]
        scale_mat[:, 1, 1] = invs[:, 1]
        scale_mat[:, 2, 2] = invs[:, 2]
        
        return np.matmul(mat, scale_mat), scales
    
    # ---------------------------------------------------------------------------
    # Scales
    
    @property
    def scales(self):
        return np.stack((
            np.linalg.norm(self.tmat[:, 0, :3], axis=1),
            np.linalg.norm(self.tmat[:, 1, :3], axis=1),
            np.linalg.norm(self.tmat[:, 2, :3], axis=1))).transpose()
    
    @scales.setter
    def scales(self, value):
        m, s = self.mat_scales
        self.tmat = tmatrix(self.locations, m, value)
        self.lock_apply()
    
    @property
    def xscales(self):
        return np.linalg.norm(self.tmat[:, 0, :3], axis=1)
    
    @xscales.setter
    def xscales(self, value):
        m, s = self.mat_scales
        s[:, 0] = np.resize(value, len(self))
        self.tmat = tmatrix(self.locations, m, s)
        self.lock_apply()
    
    @property
    def yscales(self):
        return np.linalg.norm(self.tmat[:, 1, :3], axis=1)
    
    @yscales.setter
    def yscales(self, value):
        m, s = self.mat_scales
        s[:, 1] = np.resize(value, len(self))
        self.tmat = tmatrix(self.locations, m, s)
        self.lock_apply()
    
    @property
    def zscales(self):
        return np.linalg.norm(self.tmat[:, 2, :3], axis=1)

    @zscales.setter
    def zscales(self, value):
        m, s = self.mat_scales
        s[:, 2] = np.resize(value, len(self))
        self.tmat = tmatrix(self.locations, m, s)
        self.lock_apply()

    # ---------------------------------------------------------------------------
    # Rotation matrices
    
    @property
    def matrices(self):
        return self.mat_scales[0]
    
    @matrices.setter
    def matrices(self, value):
        m, s = self.mat_scales
        t = self.locations
        self.tmat = tmatrix(t, value, s)
        self.lock_apply()
        
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
    def xeulers(self):
        return self.eulers[:, 0]
    
    @xeulers.setter
    def xeulers(self, value):
        eulers = self.eulers
        eulers[:, 0] = np.resize(value, len(self))
        self.eulers = eulers
        
    @property
    def yeulers(self):
        return self.eulers[:, 1]
    
    @yeulers.setter
    def yeulers(self, value):
        eulers = self.eulers
        eulers[:, 1] = np.resize(value, len(self))
        self.eulers = eulers
        
    @property
    def zeulers(self):
        return self.eulers[:, 2]
    
    @zeulers.setter
    def zeulers(self, value):
        eulers = self.eulers
        eulers[:, 2] = np.resize(value, len(self))
        self.eulers = eulers
        
    # ---------------------------------------------------------------------------
    # Degrees version
    
    @property
    def xeulersd(self):
        return np.degrees(self.xeulers)
    
    @xeulersd.setter
    def xeulersd(self, value):
        self.xeulers = np.radians(value)
        
    @property
    def yeulersd(self):
        return np.degrees(self.zeulers)
    
    @yeulersd.setter
    def yeulersd(self, value):
        self.yeulers = np.radians(value)
        
    @property
    def zeulersd(self):
        return np.degrees(self.zeulers)
    
    @zeulersd.setter
    def zeulersd(self, value):
        self.zeulers = np.radians(value)
        
    # ---------------------------------------------------------------------------
    # Quaternions
    
    @property
    def quaternions(self):
        return m_to_quat(self.matrices)
    
    @quaternions.setter
    def quaternions(self, value):
        self.matrices = q_to_matrix(value)
        
    # ---------------------------------------------------------------------------
    # Distances
    
    def distances(self, locations=(0, 0, 0)):
        return np.linalg.norm(self.locations - locations, axis=1)
    
    def proximities(self, locations=(0, 0, 0), bounds=(0., 1.)):
        return np.clip((self.distances(locations) - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    
    @property
    def directions(self):
        locs = self.locations
        return locs/(np.resize(np.linalg.norm(locs, axis=1), (3, len(self))).transpose())

    @directions.setter
    def directions(self, value):
        ds = np.resize(value, (len(self), 3))
        ns = self.radius/np.linalg.norm(ds, axis=1)
        self.locations = ds*(np.resize(ns, (3, len(self))).transpose())
        
    @property
    def radius(self):
        return np.linalg.norm(self.locations, axis=1)
    
    @radius.setter
    def radius(self, d):
        self.locations = self.directions*(np.resize(d, (3, len(self))).transpose())
        
    # ---------------------------------------------------------------------------
    # Orientation
    
    def orient(self, target, axis=None, up=None):
        if axis is None:
            try:
                axis = self.track_axis
            except:
                axis = 'Z'
                
        if up is None:
            try:
                up = self.up_axis
            except:
                up = 'Y'
            
        self.quaternions = q_tracker(axis, target=target, up=up)
        
    def track_to(self, location, axis=None, up=None):
        self.orient(location - self.locations, axis=axis, up=up)
        






