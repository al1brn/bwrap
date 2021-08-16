#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:03:03 2021

@author: alain
"""

import numpy as np

from ..maths.key_shapes import KeyShapes
from ..core.commons import WError

# ---------------------------------------------------------------------------
# Blender shape keys are organized
#
# object
#      shape_keys (Key)
#           key_blocks (Prop Collection of ShapeKey)
#                data (Prop Collection of
#                     ShapeKeyPoint
#                     ShapeKeyBezierPoint
#                     ShapeKeyCurvePoint
#       
# cube.data.shape_keys.key_blocks[].data[].co




# ---------------------------------------------------------------------------
# WShapeKeys interface
# 
# Inherited by Objects supporting shape keys (Mesh and Curves)
# WShapeKeys relies on object properties

class WShapeKeys():
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Shape_key eval time
    
    @property
    def eval_time(self):
        return self.wrapped.data.shape_keys.eval_time
    
    @eval_time.setter
    def eval_time(self, value):
        self.wrapped.data.shape_keys.eval_time = value
    
    # ---------------------------------------------------------------------------
    # All the shape key names
    
    @property
    def all_sk_names(self):
        """Return all the key names sorted by frame.
        """
        
        if not self.has_sk:
            return []
        
        sks = {sk.name: sk.frame for sk in self.wrapped.data.shape_keys.key_blocks}

        return list(dict(sorted(sks.items(), key=lambda item: item[1])).keys())
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Indexed shape key name

    @staticmethod
    def sk_name(name, step=None):
        """Stepped shape key name. 

        Parameters
        ----------
        name : str
            Base name of the shape key name.
        step : int, optional
            The step number. The default is None.

        Returns
        -------
        str
            Full name of the shape key.
        """
        
        return name if step is None else f"{name} {step:3d}"

    @property
    def has_sk(self):
        """The object has or not shape keys.
        """
        
        return self.wrapped.data.shape_keys is not None

    @property
    def shape_keys(self):
        """The Blender shape_keys block.
        """
        
        return self.wrapped.data.shape_keys
    
    def create_shape_keys(self, basis="basis"):
        
        obj = self.wrapped
        if obj.data.shape_keys is None:
            obj.shape_key_add(name=basis)
            obj.data.shape_keys.use_relative = False
            
        return obj.data.shape_keys

    @property
    def sk_len(self):
        """Number of shape keys.
        """
        
        sks = self.shape_keys
        if sks is None:
            return 0
        return len(sks.key_blocks)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Get a shape key
    # Can create it if it doesn't exist

    def get_sk(self, name, step=None, create=True):
        """The a shape key by its name, and possible its step.
        
        Can create the shape key if it doesn't exist.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.
        create : bool, optional
            Create the shape key if it doesn't exist. The default is True.

        Returns
        -------
        WShapekey
            Wrapper of the shape or None if it doesn't exist.
        """
        
        fname = WShapeKeys.sk_name(name, step)
        obj   = self.wrapped
        data  = obj.data

        if data.shape_keys is None:
            if create:
                obj.shape_key_add(name=fname)
                obj.data.shape_keys.use_relative = False
            else:
                return None

        # Does the shape key exists?

        sk = data.shape_keys.key_blocks.get(fname)

        # No !

        if (sk is None) and create:

            eval_time = data.shape_keys.eval_time

            if step is not None:
                # Ensure the value is correct
                data.shape_keys.eval_time = step*10

            sk = obj.shape_key_add(name=fname)

            # Less impact as possible :-)
            obj.data.shape_keys.eval_time = eval_time

        # Depending upon the data type

        return sk

    # -----------------------------------------------------------------------------------------------------------------------------
    # Create a shape

    def create_sk(self, name, step=None):
        """Create a shape key.
        
        Equivalent to a call to get_sk with create=True.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Returns
        -------
        WShapekey
            Wrapper of the shape or None if it doesn't exist.
        """
        
        return self.get_sk(name, step, create=True)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Does a shape key exist?

    def sk_exists(self, name, step):
        """Looks if a shape key exists.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Returns
        -------
        bool
            True if the shape key exists.
        """
        
        return self.get_sk(name, step, create=False) is not None

    # -----------------------------------------------------------------------------------------------------------------------------
    # Set the eval_time property to the shape key

    def set_on_sk(self, name, step=None):
        """Set the evaluation time of the object on the specified shape key.
        
        The method raises an error if the shape key doesn't exist. Call sk_exists
        before for a safe call.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Raises
        ------
        RuntimeError
            If the shape key doesn't exist.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """

        sk = self.get_sk(name, step, create=False)
        if sk is None:
            raise WError(f"The shape key '{self.sk_name(name, step)}' doesn't exist in object '{self.name}'!",
                         Class = "WObject",
                         Method = "set_on_sk",
                         name = name,
                         step = step)

        self.wrapped.data.shape_keys.eval_time = sk.frame
        
        return self.wrapped.data.shape_keys.eval_time

    # -----------------------------------------------------------------------------------------------------------------------------
    # Delete a shape key

    def delete_sk(self, name=None, step=None):
        """Delete a shape key.
        
        If name is None, all the shape keys are deleted.

        Parameters
        ----------
        name : str, optional
            Base name of the shape key. The default is None.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Returns
        -------
        None.
        """

        if not self.has_sk:
            return

        if name is None:
            self.wrapped.shape_key_clear()
        else:
            sk = self.get_sk(name, step, create=False)
            if sk is not None:
                self.wrapped.shape_key_remove(sk)
                
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Access to the shapes values
    
    def get_key_shapes(self, key_names=None):
        
        if key_names is None:
            key_names = self.all_sk_names
        if len(key_names) == 0:
            return None
        
        return KeyShapes.Read(self.wrapped, key_names)
    
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Series management
    # A series can be managed through a base name plus a step number
    
    # ---------------------------------------------------------------------------
    # Get all the names shaping a series: 'name 001' ... 'name 099' ...
    
    def series_names(self, name):
        
        blocks = self.key_blocks
        if blocks is None:
            return []
        
        indices = []
        names   = []
        for sk in blocks:
            base = sk.name[:len(name)] 
            sind = sk.name[len(name):]
            if base == name:
                names.append(base)
            else:
                try:
                    indices.add(int(sind))
                except:
                    pass
                
        indices.sort()
        names.extend([self.sk_name(name, ind) for ind in indices])
        
        return names    
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Create a series
    
    def create_series(self, name, count):
        
        current = self.series_names(name)
        target = [WShapeKeys.sk_name(name, i) for i in range(count)]
        
        # ----- Delete excess keys

        for sn in current:
            if not sn in target:
                self.delete_sk(sn)
                
        # ----- Create missing keys
        
        for sn in target:
            if not sn in current:
                self.create_sk(sn)
                
        # ----- Weight
        
        for i in range(count):
            self.get_sk(name, i).frame = i*10
            
    def get_series_key_shapes(self, name):
        return self.get_key_shapes(self.series_names)
            
    # ---------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # Get animation from shape keys
    
    def get_sk_animation(self, eval_times):
        memo = self.eval_time
        
        verts = np.empty((len(eval_times), self.verts_count, 3), np.float)
        for i, evt in enumerate(eval_times):
            self.eval_time = evt
            verts[i] = self.evaluated.verts
        
        self.eval_time = memo
        
        return verts
    
