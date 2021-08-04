#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:05:57 2021

@author: alain
"""

import numpy as np
import bpy

from .wid import WID

from ..blender import blender
from ..blender import depsgraph 

from ..maths.transformations import Transformations
from ..objects.wpropcollection import WPropCollection
from ..maths.shapes import get_full_shape
from ..wrappers.wrap_function import wrap, unwrap


from ..core.commons import WError


class WCollection(WID, Transformations):
    
    def __init__(self, wrapped, world=False, owner=False):
        
        # ----- WID initialization

        super().__init__(wrapped)
        
        # ----- Transformations initialization
        
        Transformations.__init__(self)
        
        self.shape_ = len(self.wrapped.objects)
        self.world_ = world
        self.tmat_  = self.read_tmat()
        self.owner  = owner
        
    @property
    def wrapped(self):
        """The wrapped Blender instance.

        Returns
        -------
        Struct
            The wrapped object.
        """
        
        return depsgraph.get_collection(self.name)
        
        if self.wrapped_ is None:
            return bpy.data.collections[self.name_]
        else:
            return self.wrapped_
        
        
    # ---------------------------------------------------------------------------
    # Plural access to the objects
    
    @property
    def objects(self):
        return self.wrapped.objects

    @property
    def wobjects(self):
        return WPropCollection(self.wrapped.objects)
    
    def get_object(self, *args):
        
        if self.shape == ():
            return wrap(self.wrapped.objects[0])
        
        if len(args) != len(self.shape):
            raise WError(f"get_object needs as many indices as the length of the shape {self.shape}.",
                Class   = "WCollection",
                Method  = "get_object",
                shape   = self.shape,
                shape_len = len(self.shape),
                args = args,
                args_len = len(args)
                )
            
        index = 0
        mult  = 1
        for i in reversed(range(len(self.shape))):
            index += args[i] * mult
            mult *= self.shape[i]
            
        return wrap(self.wrapped.objects[index])

    @property
    def indices(self):
        return np.arange(0, self.size).reshape(self.shape)

    def composed_indices(self, indices=None):
        if indices is None:
            return None
        return np.unravel_index(np.reshape(indices, np.size(indices)), self.shape)
    
    def objects_array(self, indices=None):
        if indices is None:
            return np.array(self.wrapped.objects)
        else:
            return np.array(self.wrapped.objects)[self.composed_indices(indices)]
    
    def foreach(self, f, indices, *args, **kwargs):
        
        if indices is None:
            indices = self.indices
            
        for index in indices.reshape(np.size(indices)):
            cindex = np.unravel_index(index, self.shape)
            f(wrap(self.wrapped.objects[index]), cindex, *args, **kwargs)
            
        return 
            
            
        inds = self.composed_indices()

        a = np.array(self.wrapped.objects)
        
        if indices is None:
            it = np.nditer(a, flags=['multi_index'])
            for o in it:
                f(wrap(o), it.multi_index, *args, **kwargs)
        else:
            inds = np.arange(self.size).reshape(self.shape)
            it = np.nditer(a[indices], flags=['multi_index'])
            for o in it:
                idx = inds[indices][it.multi_index]
                f(wrap(o), np.unravel_index(idx, self.shape), *args, **kwargs)
    
    # ---------------------------------------------------------------------------
    # Check if a new shape is acceptable
    
    def check_shape(self, shape):
        count = len(self.wrapped.objects)
        if shape is None:
            return count
        if np.product(shape) != count:
            raise WError(f"Shape {shape} not compatible with the number {count} of objects.",
                    Class = "ObjectsTransformations")
        return shape
    
    # ---------------------------------------------------------------------------
    # Overrides Transformations methods
            
    @property
    def shape(self):
        return self.shape_
    
    def reshape(self, shape):
        super().reshape(self.check_shape(shape))
        
    def read_tmat(self):
        objs = self.wrapped.objects
        count = len(objs)
        a = np.empty(count*16, np.float)
        if self.world_:
            objs.foreach_get('matrix_world', a)
        else:
            objs.foreach_get('matrix_local', a)
        return a.reshape(get_full_shape(self.shape, (4, 4)))
                         
    def write_tmat(self, tmat):
        objs = self.wrapped.objects
        count = len(objs)
        if self.world_:
            objs.foreach_set('matrix_world', tmat.reshape(count*16))
        else:
            objs.foreach_set('matrix_local', tmat.reshape(count*16))
            
    def reload_tmat(self):
        self.tmat_ = self.read_tmat()
        
    @property
    def world(self):
        return self.world_
    
    @world.setter
    def world(self, value):
        self.apply()
        self.world_ = value
        if self.owner:
            self.tmat_ = self.read_tmat()
            
    # ---------------------------------------------------------------------------
    # Plural access to the objects
    def mark_update(self):
        """When vertices changed, tag the objects for update by Blender engine.

        Returns
        -------
        None.
        """
        if self.locked == 0:
            objs = self.wrapped.objects
            for o in objs:
                o.update_tag()
    
    def apply(self):
        self.write_tmat(self.tmat_)
        self.mark_update()
        
    def lock_apply(self):
        if self.owner:
            super().lock_apply()
        else:
            self.apply()
    
    @property
    def euler_order(self):
        objs = self.wrapped.objects
        count = len(objs)
        if count == 0:
            return 'XYZ'
        else:
            return objs[0].rotation_euler.order
        
    @euler_order.setter
    def euler_order(self, value):
        for obj in self.wrapped.objects:
            obj.rotation_euler.order = value
    
    @property
    def track_axis(self):
        objs = self.wrapped.objects
        count = len(objs)
        if count == 0:
            return 'X'
        else:
            return objs[0].track_axis
        
    @track_axis.setter
    def track_axis(self, value):
        for obj in self.wrapped.objects:
            obj.track_axis = value

    @property
    def up_axis(self):
        objs = self.wrapped.objects
        count = len(objs)
        if count == 0:
            return 'Z'
        else:
            return objs[0].up_axis    
        
    @up_axis.setter
    def up_axis(self, value):
        for obj in self.wrapped.objects:
            obj.up_axis = value
        
    # ---------------------------------------------------------------------------
    # Snapshot

    def snapshots(self, key="Wrap"):
        """Set a snapshot of the duplicates.

        Parameters
        ----------
        key : str, optional
            The snapshot key. The default is "Wrap".

        Returns
        -------
        None.
        """
        
        for wo in self.wobjects:
            wo.snapshot(key)

    def to_snapshots(self, key="Wrap", mandatory=False):
        """Restore a snapshot by its key.        

        Parameters
        ----------
        key : str, optional
            The key of a previously created snapshot. The default is "Wrap".
        mandatory : bool, optional
            Raise an error is the snapshot doesn't exist. The default is False.

        Returns
        -------
        None.
        """
        
        for wo in self.wobjects:
            wo.to_snapshot(key, mandatory)
            
    # -----------------------------------------------------------------------------------------------------------------------------
    # Evaluation time
    
    @property
    def eval_times(self):
        """The eval_time attribute of the objects.
        
        Plural access to the data.shapekeys.eval_time attribute.
        
        This attribute is only valid for meshes with linked=False.

        Returns
        -------
        array of float
            The eval_time attribute.

        """
        return np.array([o.data.shape_keys.eval_time for o in self.wrapped.objects])

    @eval_times.setter
    def eval_times(self, value):
        for o, evt in zip(self.wrapped.objects, np.resize(value, len(self))):
            o.data.shape_keys.eval_time = evt
            
    # ===========================================================================
    # Collections management
    
    @property
    def wchildren(self):
        return [WCollection(c, world=self.world, owner=False) for c in self.wrapped.children]
    
    def create_child(self, name):
        return WCollection(blender.create_collection(name, self.wrapped), world=self.world, owner=False)
    
    @staticmethod
    def New(name, world=False, owner=False):
        return WCollection(blender.create_collection(name), world=world, owner=owner)
    
    def link_object(self, obj, unlink=False):
        blender.put_object_in_collection(unwrap(obj), self.wrapped, unlink=unlink)
        
    def link_objects(self, objs, unlink=False):
        for obj in objs:
            blender.put_object_in_collection(unwrap(obj), self.wrapped, unlink=unlink)
            
    @staticmethod
    def WrapCollection(name=None):
        return blender.wrap_collection(name)
            
    # ===========================================================================
    # Wrap the properties and methods of Blender Collection
    
    @property
    def all_objects(self):
        return self.wrapped.all_objects
    
    @property
    def children(self):
        return self.wrapped.children
    
    @property
    def color_tag(self):
        return self.wrapped.color_tag
    
    @color_tag.setter
    def color_tag(self, value):
        self.wrapped.color_tag = value
        
        
    @property
    def hide_render(self):
        return self.wrapped.hide_render
    
    @hide_render.setter
    def hide_render(self, value):
        self.wrapped.hide_render = value
        
    @property
    def hide_select(self):
        return self.wrapped.hide_select
    
    @hide_select.setter
    def hide_select(self, value):
        self.wrapped.hide_select = value
        
    @property
    def hide_viewport(self):
        return self.wrapped.hide_viewport
    
    @hide_viewport.setter
    def hide_viewport(self, value):
        self.wrapped.hide_viewport = value
        

    @property
    def instance_offset(self):
        return self.wrapped.instance_offset
    
    @instance_offset.setter
    def instance_offset(self, value):
        self.wrapped.instance_offset = value
        
    @property
    def lineart_usage(self):
        return self.wrapped.lineart_usage
    
    @lineart_usage.setter
    def lineart_usage(self, value):
        self.wrapped.lineart_usage = value

    
    
    
    
                    
    
    