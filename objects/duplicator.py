#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:49:17 2021

@author: alain
"""

import numpy as np

import bpy

from .blender import get_object, wrap_collection, duplicate_object, delete_object

from .plural import getattrs, setattrs
from .wrappers import wrap
from .transformations import Transformations

from ..core.commons import WError

# =============================================================================================================================
# Objects collection

class Duplicator(Transformations):
    """Duplicate objects in a dedicated collection.
    
    The duplicates can share or not the same data. The modifiers can be copied or not.
    When a lot of mesh duplicates are required, better use Crowd which also inherits from Transformations.
    
    Transformations manages one transformation matrix per duplicate. locations, rotations and matrices
    are gotten from Transformation, not from the duplicates. When modified, the transformation matrices
    are used to update the base matrices of the objects.
    
    The duplicates are put in a collection named after the model name (model --> Models). This collection
    is placed in a collection specific the Wrap addon.
    """

    def __init__(self, model, length=None, linked=True, modifiers=False):
        
        # The model to replicate must exist
        mdl = get_object(model, mandatory=True)
        
        # Collection
        coll_name  = mdl.name + "s"
        
        # Let's create the collection to host the duplicates
        coll = wrap_collection(coll_name)
        
        # We've got an idea of the number of duplicates to initialize the transformations
        if length is None:
            length = len(coll.objects)
        length = max(1, length)
        
        super().__init__(count=length)
        
        # Complementary initializations
        self.model         = mdl
        self.model_name    = mdl.name
        self.base_name     = f"Z_{self.model_name}"
        self.collection    = coll

        self.linked        = linked
        self.modifiers     = modifiers
        
        # Adjust the number of objects to the requested length
        if length != len(self.collection.objects):
            self.set_length(length)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Adjust the number of objects in the collection

    def set_length(self, length):
        """Set the number of duplicates.

        Parameters
        ----------
        length : int
            The number of duplicats.

        Returns
        -------
        None.
        """
        
        # Adjust the number of transforamtion matrices
        super().set_length(length)

        count = length - len(self.collection.objects)

        # Create missing objects
        if count > 0:
            for i in range(count):
                new_obj = duplicate_object(self.model, self.collection, self.linked, self.modifiers)
                if not self.linked:
                    new_obj.animation_data_clear()

        # Or delete supernumeraries objects
        elif count < 0:
            for i in range(-count):
                obj = self.collection.objects[-1]
                delete_object(obj)

    def __len__(self):
        length = len(self.collection.objects)
        if len(self.tmat) != length:
            raise WError(f"Error in Duplicator algorithm: len(tmat)={len(self.tmat)} doesn't match len(coll)={length}",
                        Class = "Duplicator",
                        Method = "__len__")
        return length

    def __getitem__(self, index):
        return wrap(self.collection.objects[index])

    def mark_update(self):
        """When vertices changed, tag the objects for update by Blender engine.

        Returns
        -------
        None.
        """
        
        for obj in self.collection.objects:
            obj.update_tag()
        bpy.context.view_layer.update()

    @property
    def as_array(self):
        """The objects in a numpy array.        

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        
        return np.array([wrap(obj) for obj in self.collection.objects])

    # -----------------------------------------------------------------------------------------------------------------------------
    # The objects are supposed to all have the same parameters

    @property
    def euler_order(self):
        """Euler order for rotations using the euler triplets.
        
        The order is read in the first object of the collection.

        Returns
        -------
        str
            Euler order.
        """
        
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
        """Track axis for the objects.
        
        This attribute is used in the orient and track_to methods.

        Returns
        -------
        str
            The tracking axis.
        """
        
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
        """The up axis.
        
        This attribute is used in the orient and track_to methods.

        Returns
        -------
        str
            The up axis.

        """
        if len(self) > 0:
            return self[0].up_axis
        else:
            return 'Z'

    @up_axis.setter
    def up_axis(self, value):
        for obj in self.collection:
            obj.up_axis = value

    # -----------------------------------------------------------------------------------------------------------------------------
    # Local matrices

    @property
    def matrix_locals(self):
        """The local matrices of the objects.

        Returns
        -------
        array of matrices (4x4)
            The local matrices.

        """
        return getattrs(self.collection.objects, "matrix_local", (4, 4), np.float)

    @matrix_locals.setter
    def matrix_locals(self, value):
        setattrs(self.collection.objects, "matrix_local", value, (4, 4))
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Apply the transformation
    
    def apply(self):
        """Apply the modified transformations matrices to the duplicates.
        
        Called after the transformation matrices have been modified.
        
        This method overrides the Transformations.apply default method.

        Returns
        -------
        None.
        """
        
        if False:
            setattrs(self.collection.objects, "matrix_local", self.tmat, (4, 4))
        else:
            self.collection.objects.foreach_set("matrix_local", self.tmat.reshape(len(self.tmat)*16))
        self.mark_update()    

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
        
        for wo in self:
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
        
        for wo in self:
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
        return np.array([o.data.shape_keys.eval_time for o in self.collection.objects])

    @eval_times.setter
    def eval_times(self, value):
        for o, evt in zip(self.collection.objects, np.resize(value, len(self))):
            o.data.shape_keys.eval_time = evt
            
            
