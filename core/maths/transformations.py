#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:28:13 2022

@author: alain.bernard@loreal.com
"""

import numpy as np
from .geoerrays import Vectors, Matrices, Quaternions, TMatrices, AxisAngles

class Transformations:
    
    def __init__(self, shape=(1,), owner=None, indices=None):
        
        self.owner = owner
            
        # ------------------------------------------------------------
        # Owns its own matrices
        
        if self.owner is None:
            self.locked = 0
            
            if type(shape) is int:
                shape = (shape,)
            else:
                shape = tuple(shape)
                
            self.location_ = Vectors(0, shape)
            self.scale_    = Vectors(1, shape)
            
            self.euler_    = Eulers(shape)
            self.quat_     = None
            self.mat_      = None
            self.ax_ag_    = None
            
            self.euler_order_ = 'XYZ'
        
        # ------------------------------------------------------------
        # Uses the matrices of its owner
        
        else:
            self.indices = indices
            
    # ---------------------------------------------------------------------------
    # Representation
    
    def __repr__(self):
        
        s = f"<{type(self).__name__}: len: {len(self)}, shape: {self.shape}"
        
        if self.owner is None:
            s += ", no owner,"

            if self.euler_ is not None: s += " [euler]"
            if self.quat_  is not None: s += " [quat]"
            if self.mat_   is not None: s += " [mat]"
            if self.ax_ag_ is not None: s += " [axisag]"
            
        else:
            so = f"{self.owner}".split("\n")
            s += f", indices {self.indices} on:\n"
            for line in so:
                s += "   " + line + "\n"
            
        return s + ">"
        
    # ---------------------------------------------------------------------------
    # As an array
    
    def __len__(self):
        return len(self.location)
    
    def __getitem__(self, index):
        return Transformations(owner=self, indices=index)
    
    # ---------------------------------------------------------------------------
    # Shape
    
    @property
    def shape(self):
        return self.location.shape
    
    # ---------------------------------------------------------------------------
    # Euler order
    
    @property
    def euler_order(self):
        if self.owner is None:
            if self.euler_ is None:
                return self.euler_order_
            else:
                return self.euler_.order
        else:
            return self.owner.euler_order
    
    @euler_order.setter
    def euler_order(self, value):
        if self.owner is None:
            self.euler_order_ = value
            if self.euler_ is not None:
                self.euler_.order = value
        else:
            self.owner.euler_order = value
    
    # ---------------------------------------------------------------------------
    # Lock / unlock mechanism
    
    def lock(self):
        if self.owner is None:
            self.locked += 1
        else:
            self.owner.lock()
        
    def unlock(self):
        if self.owner is None:
            self.locked -= 1
            if self.locked == 0:
                self.update()
        else:
            self.owner.unlock()
            
    def update(self, force=False):
        if self.owner is None:
            if (self.locked == 0) or force:
                self.do_update()
        else:
            self.owner.update(force=force)
            
    def do_update(self):
        if self.owner is not None:
            self.owner.do_update()
    
    # ---------------------------------------------------------------------------
    # Location
    
    @property
    def location(self):
        if self.owner is None:
            return self.location_
        else:
            return self.owner.location[self.indices]
    
    @location.setter
    def location(self, value):
        if self.owner is None:
            self.location_[:] = value
        else:
            val = self.owner.location
            val[self.indices] = value
            self.owner.location = val
    
    @property
    def x(self):
        return self.location[..., 0]
    
    @x.setter
    def x(self, value):
        if self.owner is None:
            self.location_[..., 0] = value
        else:
            val = self.location
            val[..., 0] = value
            self.location = val
        
    @property
    def y(self):
        return self.location[..., 1]
    
    @y.setter
    def y(self, value):
        if self.owner is None:
            self.location_[..., 1] = value
        else:
            val = self.location
            val[..., 1] = value
            self.location = val
        
    @property
    def z(self):
        return self.location[..., 2]
    
    @z.setter
    def z(self, value):
        if self.owner is None:
            self.location_[..., 2] = value
        else:
            val = self.location
            val[..., 2] = value
            self.location = val
        
    # ---------------------------------------------------------------------------
    # Scale
    
    @property
    def scale(self):
        if self.owner is None:
            return self.scale_
        else:
            return self.owner.scale[self.indices]
    
    @scale.setter
    def scale(self, value):
        if self.owner is None:
            self.scale_[:] = value
        else:
            val = self.owner.scale
            val[self.indices] = value
            self.owner.scale = val
    
    @property
    def sx(self):
        return self.scale[..., 0]
    
    @sx.setter
    def sx(self, value):
        if self.owner is None:
            self.scale_[..., 0] = value
        else:
            val = self.scale
            val[..., 0] = value
            self.scale = val
        
    @property
    def sy(self):
        return self.scale[..., 1]
    
    @sy.setter
    def sy(self, value):
        if self.owner is None:
            self.scale_[..., 1] = value
        else:
            val = self.scale
            val[..., 1] = value
            self.scale = val
        
    @property
    def sz(self):
        return self.scale[..., 2]
    
    @sz.setter
    def sz(self, value):
        if self.owner is None:
            self.scale_[..., 2] = value
        else:
            val = self.scale
            val[..., 2] = value
            self.scale = val
            
    # ---------------------------------------------------------------------------
    # The rotation
    
    @property
    def rotation(self):
        if self.mat_ is not None:
            return self.mat_
        elif self.quat_ is not None:
            return self.quat_
        elif self.euler_ is not None:
            return self.euler_
        elif self.ax_ag_ is not None:
            return self.ax_ag_

        return Npne
        
    # ---------------------------------------------------------------------------
    # Euler in radians
    
    @property
    def euler(self):
        if self.owner is None:
            if self.euler_ is None:
                self.euler_ = self.rotation.eulers(self.euler_order)
            return self.euler_
        else:
            return self.owner.euler[self.indices]
    
    @euler.setter
    def euler(self, value):
        if self.owner is None:
            self.euler_  = Eulers(value, shape=self.shape, order=self.euler_order)
            self.quat_   = None
            self.mat_    = None
            self.ax_ag_  = None
        else:
            val = self.owner.euler
            val[self.indices] = value
            self.owner.euler = val
    
    @property
    def rx(self):
        return self.euler[..., 0]
    
    @rx.setter
    def rx(self, value):
        if self.owner is None:
            self.euler[..., 0] = value
        else:
            val = self.euler
            val[..., 0] = value
            self.euler = val
        
    @property
    def ry(self):
        return self.euler[..., 1]
    
    @ry.setter
    def ry(self, value):
        if self.owner is None:
            self.euler[..., 1] = value
        else:
            val = self.euler
            val[..., 1] = value
            self.euler = val
        
    @property
    def rz(self):
        return self.euler[..., 2]
    
    @rz.setter
    def rz(self, value):
        if self.owner is None:
            self.euler[..., 2] = value
        else:
            val = self.euler
            val[..., 2] = value
            self.euler = val
        
    # ---------------------------------------------------------------------------
    # Rotation euler in degrees
    
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
    # Quaternion
    
    @property
    def quat(self):
        if self.owner is None:
            if self.quat_ is None:
                
                # ----- Euler to quat
                if self.euler_ is not None:
                    self.quat_ = e_to_quat(self.euler_, self.euler_order)
                
                # ----- Matrix to quat
                elif self.mat_ is not None:
                    self.quat_ = m_to_quat(self.mat_)
                
                # ----- Error
                else:
                    pass
                
            return self.quat_
        else:
            return self.owner.quat[self.indices]
    
    @quat.setter
    def quat(self, value):
        if self.owner is None:
            if self.quat_ is None:
                self.quat_ = np.empty((self.shape) + (4,), float)
            
            self.quat_[:] = value
            
            self.euler_  = None
            self.mat_    = None
            
        else:
            val = self.owner.quat
            val[self.indices] = value
            self.owner.quat = val   
            
    # ---------------------------------------------------------------------------
    # Axis angle
    
    @property
    def axis_angle(self):
        return axis_angle(self.quat)
    
    @axis_angle.setter
    def axis_angle(self, value):
        self.quat = quaternion(value[0], value[1])
            
    # ---------------------------------------------------------------------------
    # Matrix
    
    @property
    def mat(self):
        if self.owner is None:
            if self.mat_ is None:
                
                # ----- Euler to matrix
                if self.euler_ is not None:
                    self.mat_ = e_to_matrix(self.euler_, self.euler_order)
                
                # ----- Quat to matrix
                elif self.quat_ is not None:
                    self.mat_ = q_to_matrix(self.quat_)
                
                # ----- Error
                else:
                    pass
                
            return self.mat_
        else:
            return self.owner.mat[self.indices]
    
    @mat.setter
    def mat(self, value):
        if self.owner is None:
            if self.mat_ is None:
                self.mat_ = np.empty((self.shape) + (3, 3), float)
            
            self.mat_[:] = value
            
            self.euler_  = None
            self.quat_   = None
            
        else:
            val = self.owner.mat
            val[self.indices] = value
            self.owner.mat = val   
              
    # ---------------------------------------------------------------------------
    # Transformation matrix
    
    @property
    def tmat(self):
        return tmatrix(location=self.location, matrix=self.mat, scale=self.scale)
    



