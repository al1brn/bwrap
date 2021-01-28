#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:49:17 2021

@author: alain
"""

import numpy as np

import bpy

from .blender import get_object, wrap_collection, duplicate_object, delete_object

from .plural import to_shape, getattrs, setattrs
from .wrappers import wrap
from .geometry import mul_tmatrices, axis_angle, q_to_euler, e_to_quat, m_to_quat, m_to_euler, q_tracker
        
# =============================================================================================================================
# Objects collection

class Duplicator():

    DUPLIS = {}

    def __init__(self, model, length=None, linked=True, modifiers=False):
        
        # The model to replicate must exist
        mdl = get_object(model, mandatory=True)
        
        self.model         = mdl
        self.model_name    = mdl.name
        self.base_name     = f"Z_{self.model_name}"
        
            
        # Let's create the collection to host the duplicates
        
        coll_name  = self.model_name + "s"
        self.collection = wrap_collection(coll_name)
        
        self.linked        = linked
        self.modifiers     = modifiers
        
        if length is not None:
            self.set_length(length)
            
    # -----------------------------------------------------------------------------------------------------------------------------
    # Adjust the number of objects in the collection
    
    def set_length(self, length):
        
        count = length - len(self)
        
        if count > 0:
            for i in range(count):
                new_obj = duplicate_object(self.model, self.collection, self.linked, self.modifiers)
                if not self.linked:
                    new_obj.animation_data_clear()
                    
        elif count < 0:
            for i in range(-count):
                obj = self.collection.objects[-1]
                delete_object(obj)
                
    def __len__(self):
        return len(self.collection.objects)
    
    def __getitem__(self, index):
        return wrap(self.collection.objects[index])
    
    def mark_update(self):
        for obj in self.collection.objects:
            obj.update_tag()
        bpy.context.view_layer.update()
        
    @property
    def as_array(self):
        return np.array([obj for obj in self.collection.objects])
                
    # -----------------------------------------------------------------------------------------------------------------------------
    # The objects are supposed to all have the same parameters
    
    @property
    def rotation_mode(self):
        if len(self) > 0:
            return self[0].rotation_mode
        else:
            return 'XYZ'
        
    @rotation_mode.setter
    def rotation_mode(self, value):
        for obj in self.collection:
            obj.rotation_modes = value
        
    @property
    def euler_order(self):
        if len(self) > 0:
            return self[0].rotation_euler.order
        else:
            return 'XYZ'
        
    @euler_order.setter
    def euler_order(self, value):
        for obj in self.collection:
            obj.rotation_euler.order = value
        
    @property
    def track_axis(self):
        if len(self) > 0:
            return self[0].track_axis
        else:
            return 'POS_Y'
        
    @track_axis.setter
    def track_axis(self, value):
        for obj in self.collection:
            obj.track_axis = value
        
    @property
    def up_axis(self):
        if len(self) > 0:
            return self[0].up_axis
        else:
            return 'Z'
        
    @up_axis.setter
    def up_axis(self, value):
        for obj in self.collection:
            obj.up_axis = value
            
    # -----------------------------------------------------------------------------------------------------------------------------
    # Basics
    
    @property
    def locations(self):
        return getattrs(self.collection.objects, "location", 3, np.float)
        
    @locations.setter
    def locations(self, value):
        setattrs(self.collection.objects, "location", value, 3)

    @property
    def scales(self):
        return getattrs(self.collection.objects, "scale", 3, np.float)
        
    @scales.setter
    def scales(self, value):
        setattrs(self.collection.objects, "scale", value, 3)
            
    @property
    def rotation_eulers(self):
        return getattrs(self.collection.objects, "rotation_euler", 3, np.float)
        
    @rotation_eulers.setter
    def rotation_eulers(self, value):
        setattrs(self.collection.objects, "rotation_euler", value, 3)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Individual getters
        
    @property
    def xs(self):
        return self.locations[:, 0]
    
    @property
    def ys(self):
        return self.locations[:, 1]
    
    @property
    def zs(self):
        return self.locations[:, 2]
    
    @property
    def rxs(self):
        return self.rotation_eulers[:, 0]
    
    @property
    def rys(self):
        return self.rotation_eulers[:, 1]
    
    @property
    def rzs(self):
        return self.rotation_eulers[:, 2]
    
    @property
    def sxs(self):
        return self.scales[:, 0]
    
    @property
    def sys(self):
        return self.scales[:, 1]
    
    @property
    def szs(self):
        return self.scales[:, 2]
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Individual setters
    
    @xs.setter
    def xs(self, value):
        a = self.locations
        a[:, 0] = to_shape(value, len(a))
        self.locations = a

    @ys.setter
    def ys(self, value):
        a = self.locations
        a[:, 1] = to_shape(value, len(a))
        self.locations = a

    @zs.setter
    def zs(self, value):
        a = self.locations
        a[:, 2] = to_shape(value, len(a))
        self.locations = a

    @rxs.setter
    def rxs(self, value):
        a = self.rotation_eulers
        a[:, 0] = to_shape(value, len(a))
        self.rotation_eulers = a

    @rys.setter
    def rys(self, value):
        a = self.rotation_eulers
        a[:, 1] = to_shape(value, len(a))
        self.rotation_eulers = a

    @rzs.setter
    def rzs(self, value):
        a = self.rotation_eulers
        a[:, 2] = to_shape(value, len(a))
        self.locations = a

    @sxs.setter
    def sxs(self, value):
        a = self.scales
        a[:, 0] = to_shape(value, len(a))
        self.scales = a

    @sys.setter
    def sys(self, value):
        a = self.scales
        a[:, 1] = to_shape(value, len(a))
        self.scales = a

    @szs.setter
    def szs(self, value):
        a = self.scales
        a[:, 2] = to_shape(value, len(a))
        self.scales = a
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with a quaternion
    
    @property
    def matrix_locals(self):
        return getattrs(self.collection.objects, "matrix_local", (4, 4), np.float)
        
    @matrix_locals.setter
    def matrix_locals(self, value):
        setattrs(self.collection.objects, "matrix_local", value, (4, 4))
        
    def transform(self, tmat):
        mls  = self.matrix_locals
        new_mls =  mul_tmatrices(mls, tmat)
        self.matrix_locals =new_mls
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with a quaternion
    
    def quat_orient(self, quat):
        if self.rotation_mode == 'QUATERNION':
            setattrs(self.collection.objects, "rotation_quaternion", quat, 4)
            #self.rotation_quaternions = quat
        elif self.rotation_mode == 'AXIS_ANGLE':
            setattrs(self.collection.objects, "rotation_axis_angle", axis_angle(quat, True), 4)
            #self.rotation_axis_angles = wgeo.axis_angle(quat, True)
        else:
            setattrs(self.collection.objects, "rotation_euler", q_to_euler(quat, self.euler_order), 3)
            #self.rotation_eulers = wgeo.q_to_euler(quat, self.euler_order)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with euler
    
    def euler_orient(self, euler):
        if self.rotation_mode == 'QUATERNION':
            setattrs(self.collection.objects, "rotation_quaternion", e_to_quat(euler, self.euler_order), 4)
            #self.rotation_quaternions = wgeo.e_to_quat(euler, self.euler_order)
        elif self.rotation_mode == 'AXIS_ANGLE':
            setattrs(self.collection.objects, "rotation_axis_angle", axis_angle(e_to_quat(euler, self.euler_order)), 4)
            #self.rotation_axis_angles = wgeo.axis_angle(wgeo.e_to_quat(euler, self.euler_order), True)
        else:
            setattrs(self.collection.objects, "rotation_euler", euler, 3)
            #self.rotation_eulers = euler
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with matrix
    
    def matrix_orient(self, matrix):
        if self.rotation_mode in ['QUATERNION', 'AXIS_ANGLE']:
            self.quat_orient(m_to_quat(matrix))
        else:
            self.euler_orient(m_to_euler(matrix, self.euler_order))
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Track to a target location
    
    def track_to(self, location):
        locs = np.array(location) - self.locations
        q    = q_tracker(self.track_axis, locs, up=self.up_axis)
        self.quat_orient(q)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient along a given axis
    
    def orient(self, axis):
        q    = q_tracker(self.track_axis, axis, up=self.up_axis)
        self.quat_orient(q)
        
    # ---------------------------------------------------------------------------
    # Snapshot
    
    def snapshots(self, key="Wrap"):
        for wo in self:
            wo.snapshot(key)
        
    def to_snapshots(self, key="Wrap", mandatory=False):
        for wo in self:
            wo.to_snapshot(key, mandatory)
               
