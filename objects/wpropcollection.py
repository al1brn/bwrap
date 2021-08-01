#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:20:40 2021

@author: alain
"""

import numpy as np

from ..maths.geometry import get_full_shape, axis_angle, quaternion

from ..wrappers.wrap_function import wrap

from ..core.commons import WError

# ===========================================================================
# Blender prop_collection wrapper
#
# The PropCollection keeps a reference to a wrapper which must implement
# the objects property

class WPropCollection():
    
    def __init__(self, wrapper):
        self.wrapper = wrapper
    
    def find(self, key):
        return wrap(self.wrapper.objects.find(key))
    
    def get(self, key):
        return wrap(self.wrapper.objects.get(key))

    # ---------------------------------------------------------------------------
    # As an array of objects
    
    def __len__(self):
        return len(self.wrapper.objects)
    
    def __getitem__(self, index):
        return wrap(self.wrapper.objects[index])
    
    def keys(self):
        return self.wrapper.objects.keys()
    
    def values(self):
        return [wrap(o) for o in self.wrapper.objects.values()]
    
    def items(self):
        return [(k, wrap(o)) for k, o in self.wrapper.objects.items()]
    
    # ---------------------------------------------------------------------------
    # Get float attributes    
    
    def get_float_attrs(self, name, shape=1):
        
        objs = self.wrapper.objects
        
        a = np.empty(len(objs)*np.product(shape), np.float)
        objs.foreach_get(name, a)
        
        if shape == 1:
            return a
        else:
            return a.reshape(get_full_shape(len(objs), shape))
        
    # ---------------------------------------------------------------------------
    # Set float attributes    
        
    def set_float_attrs(self, name, array, shape=1):

        objs = self.wrapper.objects
        
        target_shape = get_full_shape(len(objs), shape)
        size = len(objs) * np.product(shape)
        
        # ----- Array is already at the correct shape
        # A linearisation is just need
        
        if (np.shape(array) == target_shape) or (np.shape(array) == (size, )):
            
            objs.foreach_set(name, np.reshape(array, size))
            
        # ----- Broadcasting
        
        else:

            a = np.empty(target_shape, np.float)
            
            try:
                # Broadcasting could fail
                a[:] = array
                objs.foreach_set(name, np.reshape(a, size))
                return
            
            except:
                pass
            
            raise WError("Error when broadcasting a value in a Blender collection property.",
                Class = "WPropCollection",
                Method = "set_float_attrs",
                objects = objs,
                attr_name = name,
                shape = shape,
                array_shape = np.shape(array),
                target_shape = target_shape)
            
    # ---------------------------------------------------------------------------
    # Get int attributes    
        
    def get_int_attrs(self, name, shape=1):
        
        objs = self.wrapper.objects
        
        a = np.empty(len(objs)*np.product(shape), int)
        objs.foreach_get(name, a)
        
        if shape == 1:
            return a
        else:
            return a.reshape(get_full_shape(len(objs), shape))
        
    # ---------------------------------------------------------------------------
    # Set float attributes    
        
    def set_int_attrs(self, name, array, shape=1):

        objs = self.wrapper.objects

        target_shape = get_full_shape(len(objs), shape)
        size = len(objs) * np.product(shape)
        
        # ----- Array is already at the correct shape
        # A linearisation is just need
        
        if (np.shape(array) == target_shape) or (np.shape(array) == (size, )):
            
            objs.foreach_set(name, np.reshape(array, size))
            
        # ----- Broadcasting
        
        else:

            a = np.empty(target_shape, int)
            
            try:
                # Broadcasting could fail
                a[:] = array
                objs.foreach_set(name, np.reshape(a, size))
                return
            except:
                pass
            
            raise WError("Error when broadcasting a value in a Blender collection property.",
                Class = "WPropCollection",
                Method = "set_int_attrs",
                objects = objs,
                attr_name = name,
                shape = shape,
                array_shape = np.shape(array),
                target_shape = target_shape)
            
    # ---------------------------------------------------------------------------
    # Get float array item
    
    def get_float_array_item(self, name, index, shape=3):
        return self.get_float_attrs(name, shape=shape)[index]
        
    # ---------------------------------------------------------------------------
    # Set float attributes    
        
    def set_float_array_item(self, name, index, array, shape=3):
        
        a = self.get_float_attrs(name, shape=shape)
        
        try:
            a[:, index] = array
            return
        except:
            pass
        
        raise WError("Error when broadcasting a value in a Blender collection array property.",
                Class = "WPropCollection",
                Method = "set_float_array_item",
                attr_name = name,
                index = index,
                array_shape = np.shape(array),
                shape = shape)
     
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Geometry - Rotation mode
    
    @property
    def rotation_mode(self):
        return [o.rotation_mode for o in self.wrapper.objects]
    
    @rotation_mode.setter
    def rotation_mode(self, value):
        
        objects = self.wrapper.objects
        
        if type(value) is str:
            for o in objects:
                o.rotation_mode = value
                
        elif hasattr(value, '__len__'):
            
            if len(value) != len(objects):
                raise WError("The length of the array is not equal to the number of objects in the collecion.",
                    Class = "WPropCollection",
                    Method = "rotation_mode",
                    value = value,
                    objects_number = len(objects))

            for o, mode in zip(objects, value):
                o.rotation_mode = mode
                
        else:
            raise WError("The rotation_mode parameter must be a valide string or an array of valid strings.",
                Class = "WPropCollection",
                Method = "rotation_mode",
                value = value,
                objects_number = len(objects))
            
            
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Geometry - Location
    
    @property
    def location(self):
        return self.get_float_attrs('location', 3)
    
    @location.setter
    def location(self, value):
        self.set_float_attrs('location', value, 3)
        
    @property
    def x(self):
        return self.location[:, 0]
    
    @x.setter
    def x(self, value):
        self.set_float_array_item('location', 0, value, shape=3)
        
    @property
    def y(self):
        return self.location[:, 1]
    
    @y.setter
    def y(self, value):
        self.set_float_array_item('location', 1, value, shape=3)
        
    @property
    def z(self):
        return self.location[:, 2]
    
    @z.setter
    def z(self, value):
        self.set_float_array_item('location', 2, value, shape=3)
        
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Geometry - Scale

    @property
    def scale(self):
        return self.get_float_attrs('scale', 3)
    
    @scale.setter
    def scale(self, value):
        self.set_float_attrs('scale', value, 3)
        
    @property
    def sx(self):
        return self.scale[:, 0]
    
    @sx.setter
    def sx(self, value):
        self.set_float_array_item('scale', 0, value, shape=3)
        
    @property
    def sy(self):
        return self.scale[:, 1]
    
    @sy.setter
    def sy(self, value):
        self.set_float_array_item('scale', 1, value, shape=3)
        
    @property
    def sz(self):
        return self.scale[:, 2]
    
    @sz.setter
    def sz(self, value):
        self.set_float_array_item('scale', 2, value, shape=3)    
        
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Geometry - Euler in radians

    @property
    def euler(self):
        return self.get_float_attrs('rotation_euler', 3)
    
    @euler.setter
    def euler(self, value):
        self.set_float_attrs('rotation_euler', value, 3)
        
    @property
    def rx(self):
        return self.euler[:, 0]
    
    @rx.setter
    def rx(self, value):
        self.set_float_array_item('rotation_euler', 0, value, shape=3)
        
    @property
    def ry(self):
        return self.euler[:, 1]
    
    @ry.setter
    def ry(self, value):
        self.set_float_array_item('rotation_euler', 1, value, shape=3)
        
    @property
    def rz(self):
        return self.euler[:, 2]
    
    @rz.setter
    def rz(self, value):
        self.set_float_array_item('rotation_euler', 2, value, shape=3)    
        
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Geometry - Euler in degrees

    @property
    def eulerd(self):
        return np.degrees(self.get_float_attrs('rotation_euler', 3))
    
    @eulerd.setter
    def eulerd(self, value):
        self.set_float_attrs('rotation_euler', np.radians(value), 3)
        
    @property
    def rxd(self):
        return np.degrees(self.euler[:, 0])
    
    @rxd.setter
    def rxd(self, value):
        self.set_float_array_item('rotation_euler', 0, np.radians(value), shape=3)
        
    @property
    def ryd(self):
        return np.degrees(self.euler[:, 1])
    
    @ryd.setter
    def ryd(self, value):
        self.set_float_array_item('rotation_euler', 1, np.radians(value), shape=3)
        
    @property
    def rzd(self):
        return np.degrees(self.euler[:, 2])
    
    @rzd.setter
    def rzd(self, value):
        self.set_float_array_item('rotation_euler', 2, np.radians(value), shape=3)   
        
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Geometry - Quaternion
    
    @property
    def quaternion(self):
        return self.get_float_attrs('rotation_quaternion', 4)
    
    @quaternion.setter
    def quaternion(self, value):
        self.set_float_attrs('rotation_quaternion', value, 4)
        
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Geometry - Axis angle
    
    @property
    def axis_angle(self):
        return axis_angle(self.quaternion)
    
    @axis_angle.setter
    def axis_angle(self, value):
        self.quaternion = quaternion(value[0], value[1])
        
    @property
    def axis_angled(self):
        ax, ag = axis_angle(self.quaternion)
        return ax, np.degrees(ag)
    
    @axis_angled.setter
    def axis_angled(self, value):
        self.quaternion = quaternion(value[0], np.radians(value[1]))
        
        
        
    
    

        
        
        
        
    
        
        
    
    
        
        
        
        
        
            
