#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:03:03 2021

@author: alain
"""

import numpy as np

import bpy

from .wstruct import WStruct
from .wsplines import WSplines
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
# WShapeKeys
# Initializer is a data wrapper

class WShapeKeys(WStruct):
    
    def __init__(self, data):
        super().__init__(name=data.name)
        self.object_type = type(data).__name__
        
    @property
    def wrapped(self):
        if self.object_type == 'Mesh':
            return bpy.data.meshes[self.name]
        else:
            return bpy.data.curves[self.name]
        
    # ----------------------------------------------------------------------------------------------------
    # Blender object
    
    @property
    def blender_object(self):
        for obj in bpy.data.objects:
            if obj.data.name == self.name:
                return obj
            
    # ----------------------------------------------------------------------------------------------------
    # Mesh or curve
    
    @property
    def is_mesh(self):
        return self.object_type == 'Mesh'
        
    # ----------------------------------------------------------------------------------------------------
    # Do we have shape keys at least
    
    @property
    def shape_keys(self):
        """The Blender shape_keys block.
        """
        return self.wrapped.shape_keys

    # ----------------------------------------------------------------------------------------------------
    # Create the shape_keys structure
    
    @property
    def safe_shape_keys(self):
        sks = self.shape_keys
        if sks is not None:
            return sks
        
        obj = self.blender_object
        
        obj.shape_key_add()
        obj.data.shape_keys.use_relative = False
        
        return self.shape_keys
    
    # ====================================================================================================
    # Eval time
    
    # ----------------------------------------------------------------------------------------------------
    # Shape_key eval time
    
    @property
    def eval_time(self):
        return self.safe_shape_keys.eval_time
    
    @eval_time.setter
    def eval_time(self, value):
        self.safe_shape_keys.eval_time = value
        
    
    # ====================================================================================================
    
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Shape keys are managed:
    # - By an individual name
    # - By a name and a step --> name xxx is the single name of a shape key
    # - By an array of names
    #
    # The following utility method return:
    # - An array of key names
    # - A flag indicating if the input was or not a single name
    
    # ----------------------------------------------------------------------------------------------------
    # Build the stepped names
    
    @staticmethod
    def stepped_keys(name, step=None):
        """Stepped shape key name. 

        Parameters
        ----------
        name : str
            Base name of the shape key name.
        step : int or array of ints, optional
            The step number. The default is None.

        Returns
        -------
        str
            Array of full names of the shape key.
            bool : True if single name
        """
        
        if step is None:
            return [name], True
        
        if hasattr(step, '__len__'):
            return [f"{name} {i:3d}" for i in step], False
        
        return [f"{name} {step:3d}" for i in step], True
    
    # ----------------------------------------------------------------------------------------------------
    # Test if a key exists
    
    def key_exists(self, key):
        if len(self) == 0:
            return False
        return self.shape_keys.key_blocks.get(key) is not None
    
    # ----------------------------------------------------------------------------------------------------
    # If a name is not an existing name, it could be the whole series
    
    def name_is_series_name(self, name):
        
        if len(self) == 0:
            return False
        
        blocks = self.shape_keys.key_blocks
        
        for sk in blocks:
            base = sk.name[:len(name)] 
            sind = sk.name[len(name):]
            if base == name:
                try:
                    _  = int(sind)
                    return True
                except:
                    pass
                
        return False
    
    # ---------------------------------------------------------------------------
    # Get all the names shaping a series: 'name 001' ... 'name 099' ...
    
    def series_keys(self, name):
        
        if len(self) == 0:
            return []
        
        blocks = self.shape_keys.key_blocks
        
        indices = []
        keys    = []
        for sk in blocks:
            base = sk.name[:len(name)] 
            sind = sk.name[len(name):]
            if base == name:
                try:
                    indices.append(int(sind))
                    keys.append(sk.name)
                except:
                    pass
                
        return keys
        
    # ----------------------------------------------------------------------------------------------------
    # Utility to convert a couple (name, step) is an array of keys plus the single flag
    
    def keys_array(self, name, step, sort=False):
        
        # ----- Name is None: we want all the keys
        
        if name is None:
            keys   = [sk.name for sk in self.safe_shape_keys.key_blocks]
            single = False
        
        # ----- Name is a string, a single name or the stepped keys
        
        elif type(name) is str:
            if step is None:
                if self.key_exists(name):
                    return [name], True
                
                if self.name_is_series_name(name):
                    keys   = self.series_keys(name)
                    single = False
                else:
                    return [name], True
                
            else:
                keys, single = self.stepped_keys(name, step)

        # ----- Otherwise: name is an array of strings (let's hoep so !)
        
        else:
            keys   = name
            single = False
            
        # ---------------------------------------------------------------------------
        # Let's sort if requested
        
        if sort:
            blocks = self.safe_shape_keys.key_blocks
            sks = {}
            for key in keys:
                sk = blocks.get(key)
                if sk is not None:
                    sks[sk.name] = sk.frame
                    
            keys = list(dict(sorted(sks.items(), key=lambda item: item[1])).keys())

        # ---------------------------------------------------------------------------
        # We are good
        
        return keys, single    
    
    # ====================================================================================================
    # An array of shape keys
    #
    # If requested, a shape key can be created if it doesn't exist
    
    def get_create_shape_key(self, key, create=False):
        
        if create:
            sks = self.safe_shape_keys
        else:
            sks = self.shape_keys
            if sks is None:
                return None
            
        sk = sks.key_blocks.get(key)
        if (sk is not None) or (not create):
            return sk
        
        self.blender_object.shape_key_add(name=key)
        return sks.key_blocks[key]

    # ----------------------------------------------------------------------------------------------------
    # As an array of shape keys
    
    def __len__(self):
        return 0 if self.shape_keys is None else len(self.shape_keys.key_blocks)
    
        
    
    def index_to_indices(self, index):
        
        # ----- Key name is int : very simple
        if type(index) is int:
            return [index], True
        
        # ----- Key name is str, if it doesn't exist, could be the series
        if type(index) is str:
            if self.name_is_series_name(index):
                return self.series_keys(index), False
            else:
                return [index], True
        
        # ----- Can be a slice
        if isinstance(index, slice):
            inds = index.indices(len(self))
            return [i for i in range(inds[0], inds[1], inds[2])], False
        
        # ----- Can be a couple name, step
        if (type(index) is tuple) and (len(index) == 2):
            return self.keys_array(index[0], index[1], sort=False)
        
        else:
            return index, False # It should be an array 
        
        raise WError(f"Bad shape keys index: {index}. Use valid shape keys")
        
    
    def __getitem__(self, index):
        
        sks = self.shape_keys
        if sks is None:
            return None
        
        indices, single = self.index_to_indices(index)
        sks = [sks.key_blocks.get(i) for i in indices]
        
        if single:
            return sks[0]
        else:
            return sks
    
    # ====================================================================================================
    # All the keys
    
    @property
    def all_keys(self):
        if self.shape_keys is None:
            return []
        else:
            keys, _ = self.keys_array(None, None, sort=True)
            return keys
        
    # ====================================================================================================
    # Same as getitem but create shake keys which doesn't exist
    
    @property
    def safe_get(self, index):

        sks = self.safe_shape_keys
        if sks is None:
            return None
        
        indices, single = self.index_to_indices(index)
        sks = [self.get_create_shape_key(i, create=True) for i in indices]
        
        if single:
            return sks[0]
        else:
            return sks
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Set the eval_time property to the shape key

    def set_on_key(self, name, step=None):
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
        
        key, single = self.keys_array(name)
        if not single:
            raise WError(f"Impossible to set eval time on several shape keys: {key}.",
                         Class = "WShapeKeys", Method="set_on_key", name=name, step=step)

        sk = self[key]
        if key is None:
            raise WError(f"The shape key '{self.sk_name(name, step)}' doesn't exist in object '{self.name}'!",
                         Class = "WShapeKeys",
                         Method = "set_on_sk",
                         name = name,
                         step = step)
        
        self.eval_time = sk.frame
        return self.eval_time

    # -----------------------------------------------------------------------------------------------------------------------------
    # Delete a shape key

    def delete(self, name=None, step=None):
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
        
        if len(self) == 0:
            return
        
        obj = self.blender_object
        
        if name is None:

            obj.shape_key_clear()
            
        else:
            sks = self[name, step]
            for sk in sks:
                if sk is not None:
                    obj.shape_key_remove(sk)
                    
                
    # ====================================================================================================
    # Series management

    # ---------------------------------------------------------------------------
    # Create a series
    
    def create_series(self, name, steps):
        
        cur_keys = self.series_keys(name)
        target_keys, _ = self.keys_array(name, range(steps))
        
        dels = []
        for key in cur_keys:
            if  key not in target_keys:
                dels.append(key)
                
        self.delete(dels)
        
        for key in target_keys:
            if key in cur_keys:
                continue
            self.get_create_shape_key(key, create=True)
        
        return self[target_keys]

    # ---------------------------------------------------------------------------
    # The key shapes of a series
            
    def get_series(self, name):
        return self[self.series_keys(name)]
    

    # ===========================================================================
    # Get the mesh vertices
    
    def get_mesh_vertices(self, name=None, step=None):
        
        if len(self) == 0:
            return None
        
        keys, single = self.keys_array(name, step)
        sks = self[keys]
        if len(sks) == 0:
            return None

        n = len(sks[0].data)
        verts = np.zeros((len(sks), n, 3), np.float)
        
        a = np.zeros(n*3, np.float)

        for i_sk, sk in enumerate(sks):
            sk.data.foreach_get('co', a)
            verts[i_sk] = a.reshape(n, 3)
            
        if single:
            return verts[0]
        else:
            return verts
            
    # ===========================================================================
    # Get the curve vertices
    
    def get_curve_vertices(self, name=None, step=None):
        
        if len(self) == 0:
            return None
        
        keys, single = self.keys_array(name, step)
        sks = self[keys]
        if len(sks) == 0:
            return None
        
        # --------------------------------------------------
        # Need to read the alternance bezier / not bezier
        
        profile     = WSplines(self.wrapped).profile
        only_bezier = np.min(profile[:, 0]) == 3
        only_nurbs  = np.max(profile[:, 0]) == 1
        
        nverts      = np.sum(profile[:, 0] * profile[:, 1])
        
        if only_bezier:
            verts = np.zeros((len(sks), nverts, 3), np.float)
        else:
            verts = np.zeros((len(sks), nverts, 5), np.float) # plus radius & tilt
            
        # --------------------------------------------------
        # Loop on the shapes
        
        if only_bezier:
            co = np.zeros(nverts*3, np.float)
            le = np.zeros(nverts*3, np.float)
            ri = np.zeros(nverts*3, np.float)
        elif only_nurbs:
            co = np.zeros(nverts*3, np.float)
            ra = np.zeros(nverts,   np.float)
            ti = np.zeros(nverts,   np.float)
            

        for i_sk, sk in enumerate(sks):
            
            if sk is None:
                continue
            
            # --------------------------------------------------
            # Load bezier when only bezier curves
            
            if only_bezier:
                
                co = co.reshape(nverts * 3)
                le = co.reshape(nverts * 3)
                ri = ri.reshape(nverts * 3)

                sk.data.foreach_get('co',           co)
                sk.data.foreach_get('handle_left',  le)
                sk.data.foreach_get('handle_right', ri)
                
                co = co.reshape(nverts, 3)
                le = le.reshape(nverts, 3)
                ri = ri.reshape(nverts, 3)
                
                index = 0
                for t, n, st in profile:
                    verts[i_sk, 3*index       :3*index + n, ] = co[index]
                    verts[i_sk, 3*index + n   :3*index + 2*n] = le[index]
                    verts[i_sk, 3*index + 2*n :3*index + 3*n] = ri[index]
                    
                    index += n
                    
            # --------------------------------------------------
            # Load nurbs when no bezier curves
            
            elif only_nurbs:
                
                co = co.reshape(nverts*3)
                
                sk.data.foreach_get('co',     co)
                sk.data.foreach_get('radius', ra)
                sk.data.foreach_get('tilt',   ti)
                
                co = co.reshape(nverts, 3)

                index = 0
                for t, n, st in profile:
                    verts[i_sk, index :index + n, :3] = co[index]
                    verts[i_sk, index :index + n, 3]  = ra[index]
                    verts[i_sk, index :index + n, 4]  = ti[index]
                    
                    index += n
                    
            # --------------------------------------------------
            # We have to loop :-(
            
            i_data = 0
            index  = 0
            
            for t, n, st in profile:
                
                if t == 3:
                    
                    for i in range(n):
                        #print(f"Loop {(t, n)}: index= {index} / nverts= {nverts}, i_data= {i_data} -> {i_data+n} / {len(sk.data)}")
    
                        usk = sk.data[i_data + i]
    
                        verts[i_sk, index       + i, :3] = usk.co
                        verts[i_sk, index +   n + i, :3] = usk.handle_left
                        verts[i_sk, index + 2*n + i, :3] = usk.handle_right
                        
                    index += 3*n
                    
                else:
                
                    for i in range(n):
                        #print(f"Loop {(t, n)}: index= {index} / nverts= {nverts}, i_data= {i_data} -> {i_data+n}/ {len(sk.data)}")
    
                        usk = sk.data[i_data + i]
    
                        verts[i_sk, index + i, :3] = usk.co
                        verts[i_sk, index + i,  3] = usk.radius
                        verts[i_sk, index + i,  4] = usk.tilt
                        
                    index += n
                    
                i_data += n



        if single:
            return verts[0]
        else:
            return verts
        
    # ----------------------------------------------------------------------------------------------------
    # Get vertices
    
    def verts(self, name=None, step=None):
        
        if self.is_mesh:
            return self.get_mesh_vertices(name, step)
        else:
            return self.get_curve_vertices(name, step)
    

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
    
