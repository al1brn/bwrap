#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:49:17 2021

@author: alain
"""

import numpy as np

import bpy

from ..maths.transformations import Transformations

from .wpropcollection import WPropCollection

from ..core.commons import WError

# =============================================================================================================================
# Objects collection

class WObjects(Transformations):
    """Manages a collection of objects through transformations matrices.
    
    In this root class, the objects collection is not initialized. It is up to
    the descending classes to implement the objects property.
    
    Note that 
    """
    
    def __init__(self, objects):
        
        super().__init__(count=len(objects))
        self.matrix_name = 'matrix_local'
        self.reload_matrices()
        
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # To be overriden
        
    @property
    def objects(self):
        return None
    
    # ---------------------------------------------------------------------------
    # Implement the PropCollection interface on the objects collection
    # Note that the initialization of WPropCollection needs a wrapper
    # as an argument, not the objects directly
    
    @property
    def wobjects(self):
        return WPropCollection(self)
    
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Overriding            
            
    def apply(self):
        """Apply the modified transformations matrices to the duplicates.
        
        Called after the transformation matrices have been modified.
        
        This method overrides the Transformations.apply default method.

        Returns
        -------
        None.
        """
        
        self.wobjects.set_float_attrs(self.matrix_name, self.tmat, shape=(4, 4))
        self.mark_update()    
    
    @property
    def euler_order(self):
        if len(self.objects) == 0:
            return 'XYZ'
        else:
            return self.objects[0].rotation_euler.order
        
    @euler_order.setter
    def euler_order(self, value):
        for obj in self.objects:
            obj.rotation_euler.order = value
    
    @property
    def track_axis(self):
        if len(self.objects) == 0:
            return 'X'
        else:
            return self.objects[0].track_axis
        
    @track_axis.setter
    def track_axis(self, value):
        for obj in self.objects:
            obj.track_axis = value

    @property
    def up_axis(self):
        if len(self.objects) == 0:
            return 'Y'
        else:
            return self.objects[0].up_axis
        
    @up_axis.setter
    def up_axis(self, value):
        for obj in self.objects:
            obj.up_axis = value

    # ---------------------------------------------------------------------------
    # Some utilities
    
    def reload_matrices(self):
        objects = self.objects
        
        a = np.empty(len(objects)*16, np.float)
        objects.foreach_get(self.matrix_name, a)
        self.tmat_ = a.reshape(len(objects), 4, 4)
    
    @property
    def world(self):
        return self.matrix_name == 'matrix_world'
    
    @world.setter
    def world(self, value):
        mn = self.matrix_name
        if value:
            self.matrix_name = 'matrix_world'
        else:
            self.matrix_name = 'matrix_local'
        if mn != self.matrix_name:
            self.reload_matrices()

    # ---------------------------------------------------------------------------
    # Overriding            

    def mark_update(self):
        """When vertices changed, tag the objects for update by Blender engine.

        Returns
        -------
        None.
        """
        
        for obj in self.objects:
            obj.update_tag()
        bpy.context.view_layer.update()
        
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
        return np.array([o.data.shape_keys.eval_time for o in self.objects])

    @eval_times.setter
    def eval_times(self, value):
        for o, evt in zip(self.objects, np.resize(value, len(self))):
            o.data.shape_keys.eval_time = evt
            
            
            
