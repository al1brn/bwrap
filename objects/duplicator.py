#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:49:17 2021

@author: alain
"""

import numpy as np

#import bpy

from ..blender.blender import get_object, duplicate_object, delete_object
from ..wrappers.wcollection import WCollection

#from ..core.commons import WError

# =============================================================================================================================
# Objects collection

class Duplicator(WCollection):
    """Duplicate objects in a dedicated collection.
    
    The duplicates can share or not the same data. The modifiers can be copied or not.
    When a lot of mesh duplicates are required, better use Crowd which also inherits from Transformations.
    
    Transformations manages one transformation matrix per duplicate. locations, rotations and matrices
    are gotten from Transformation, not from the duplicates. When modified, the transformation matrices
    are used to update the base matrices of the objects.
    
    The duplicates are put in a collection named after the model name (model --> Models). This collection
    is placed in a collection specific the Wrap addon.
    """

    def __init__(self, model, count=None, linked=True, modifiers=False):
        
        # ----- The object to replicate
        mdl = get_object(model, mandatory=True)
        
        # ----- The collection name to create
        
        coll_name  = mdl.name + "s"
        coll = WCollection.WrapCollection(coll_name)
        
        super().__init__(coll.name, world=False, owner=True)
        
        # ----- Duplication parameters

        self.model         = mdl
        self.model_name    = mdl.name
        self.base_name     = f"Z_{self.model_name}"

        self.linked        = linked
        self.modifiers     = modifiers
        
        # ----- Ensure the size is ok
        
        self.resize(count)
        

    # -----------------------------------------------------------------------------------------------------------------------------
    # Adjust the number of objects in the collection

    def resize(self, shape=None):
        """Resize the number of duplicates.

        Parameters
        ----------
        shape : shape, optional
            The shape to use in the resize. The default is None.

        Returns
        -------
        None.
        """
        
        if shape is None:
            
            self.shape_ = len(self.wrapped.objects)
            
        else:
                
            count = 1 if shape == () else np.product(shape)
            
            # ---------------------------------------------------------------------------
            # Create or delete the objects to match the count
            
            diff = count - len(self.wrapped.objects)
    
            # Create missing objects
            if diff > 0:
                for i in range(diff):
                    new_obj = duplicate_object(self.model, self.wrapped, self.linked, self.modifiers)
                    if not self.linked:
                        new_obj.animation_data_clear()
    
            # Or delete supernumeraries objects
            elif diff < 0:
                for i in range(-diff):
                    obj = self.wrapped.objects[-1]
                    delete_object(obj)
                    
            
            self.shape_ = shape
                    
                
        # ---------------------------------------------------------------------------
        # New shape
        
        print(f"Maj the matrices, shape={self.shape}, len={len(self.wrapped.objects)}")
        self.tmat_ = self.read_tmat()
