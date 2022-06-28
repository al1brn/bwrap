#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:03:03 2021

@author: alain
"""

import numpy as np

import bpy

#from .wcurve import WCurve

from ..core.profile import Profile
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
#
# A key is either a string (name of the shape) or an int (index in the array)
# Series are managed through a key name and a number of steps

# ====================================================================================================
# WShapeKeys

class WShapeKeys():
    
    def __init__(self, name):
        
        if type(name) is str:
            self.name = name
        else:
            self.name = name.name
            
    def __repr__(self):
        s = f"<WShapeKeys of '{self.name}': {len(self)} shapes"
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # Access to the object

    @property
    def object(self):
        return bpy.data.objects[self.name]


    @property
    def object_type(self):
        return type(self.data).__name__
    
    # ---------------------------------------------------------------------------
    # Access to the data structure
    
    @property
    def data(self):
        return self.object.data
    
    # ----------------------------------------------------------------------------------------------------
    # Access to the shape_keys prperty
    # Create it if it doesn't exist.
    # Call data.shape_key to check it without create it
    
    def shape_keys(self, first_name="Basis"):
        
        obj = self.object
        shape_keys = obj.data.shape_keys

        if shape_keys is None:
            if first_name is None:
                return None
            obj.shape_key_add(name=first_name)
            shape_keys = obj.data.shape_keys
            
        return shape_keys
    
    # ----------------------------------------------------------------------------------------------------
    # The blocks
    
    def key_blocks(self, first_name="Basis"):
        return self.shape_keys(first_name).key_blocks
    
    # ====================================================================================================
    # Get a shape key or an array of shape keys
    #
    # Used by getitem with create=False
    
    def shape_key(self, key, create=False):
        
        # ----- Done if empty
        
        if len(self) == 0:
            if not create:
                raise WError(f"Shape key index error. The object '{self.name}' has no shape key",
                             Class    = "WShapeKeys",
                             Method   = "shape_key",
                             Object   = self.name,
                             key      = key,
                             create   = create)
        
        # ----- Integer index
        
        if type(key) is int:
            
            # ----- We don't create keys with int index
            
            if key >= len(self):
                raise WError(f"Shape key index '{key}' is greater than the number of shape keys {len(self)}.",
                             Class    = "WShapeKeys",
                             Method   = "shape_key",
                             Object   = self.name,
                             key      = key,
                             create   = create)
            
            return self.key_blocks()[key]
        
        # ----- Key name
        
        elif type(key) is str:
            
            block = self.key_blocks(key).get(key)
            if block is None:
                if not create:
                    raise WError(f"Shape key '{key}' doesn't exist.",
                             Class    = "WShapeKeys",
                             Method   = "shape_key",
                             Object   = self.name,
                             key      = key,
                             create   = create)
                    
                self.object.shape_key_add(name=key)
                return self.key_blocks()[key]
                
            return block
        
        # ----- An array of keys
        
        elif hasattr(key, "__len__"):
            return [self.shape_key(k, create) for k in key]
        
        # ----- What the hell is this key ?
        
        else:
            raise WError("Invalid shape key . The key ust be int, str or array of valid keys.",
                             Class    = "WShapeKeys",
                             Method   = "shape_key",
                             Object   = self.name,
                             key      = key,
                             create   = create)

    # ----------------------------------------------------------------------------------------------------
    # As an array of key_blocks
    
    def __len__(self):
        return 0 if self.data.shape_keys is None else len(self.key_blocks())
    
    def __getitem__(self, key):
        return self.shape_key(key, create=False)
        
    
    # ====================================================================================================
    # Eval time
    
    # ----------------------------------------------------------------------------------------------------
    # Use relative
    
    @property
    def use_relative(self):
        return self.shape_keys().use_relative
    
    @use_relative.setter
    def use_relative(self, value):
        self.shape_keys().use_relative = value
    
    # ----------------------------------------------------------------------------------------------------
    # Shape_key eval time
    
    @property
    def eval_time(self):
        return self.shape_keys().eval_time
    
    @eval_time.setter
    def eval_time(self, value):
        self.shape_keys().eval_time = value
        
    
    # ====================================================================================================
    # Keys utilities
    
    # ----------------------------------------------------------------------------------------------------
    # Test if a key exists
    
    def key_exists(self, key):
        if len(self) == 0:
            return False
        
        if type(key) is int:
            return key < len(self)
        
        return self.key_blocks().get(key) is not None
    
    # ----------------------------------------------------------------------------------------------------
    # All the keys
    
    @property
    def all_keys(self):
        if len(self) == 0:
            return []
        
        return [block.name for block in self.key_blocks()]
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Delete a shape key

    def delete(self, key):
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
        
        if key is None:
            self.object.shape_key_clear()
            
        elif type(key) is str:
            self.object.shape_key_remove(key)
            
        elif hasattr(key, '__len__'):
            for k in key:
                self.delete(k)
            
        else:
            raise WError("Invalid shape key for deletion: {key}.",
                    Class    = "WShapeKeys",
                    Method   = "delete",
                    Object   = self.name,
                    key      = key)
                
    # ====================================================================================================
    # Series management
    
    # ----------------------------------------------------------------------------------------------------
    # Build the stepped names
    # CAUTION: to match Blender behavior, keys are numbered from 1 not 0
    
    @staticmethod
    def series_key(name, step=None):
        """Stepped shape key name. 

        Parameters
        ----------
        name : str
            Base name of the shape key name.
        step : int or array of ints, optional
            The step number. The default is None.

        Returns
        -------
        str or array of str
        """
        
        if step is None:
            return name
        
        if hasattr(step, '__len__'):
            return [f"{name} {i+1}" for i in step]
        
        return f"{name} {step+1}"
    
    # ---------------------------------------------------------------------------
    # Is a series name
    
    def is_series_name(self, name):
        return self.key_exists(self.series_key(name, 0))
    
    # ---------------------------------------------------------------------------
    # Get all the names shaping a series: 'name 1' ... 'name 123' ...
    
    def series_keys(self, name):
        
        all_keys = self.all_keys
        
        keys = []
        for key_name in all_keys:
            base = key_name[:len(name)]
            sind = key_name[len(name)+1:]
            if (base == name) and (sind.isdigit()):
                keys.append(key_name)
                
        return keys

    # ---------------------------------------------------------------------------
    # Create a series
    
    def create_series(self, name, steps):
        
        cur_keys    = self.series_keys(name)
        target_keys = self.keys_array(name, range(steps))
        
        dels = []
        for key in cur_keys:
            if  key not in target_keys:
                dels.append(key)
                
        self.delete(dels)
        
        return self.shape_key(target_keys)

    # ---------------------------------------------------------------------------
    # The key shapes of a series
            
    def series(self, name):
        return self[self.series_keys(name)]
    
    # ---------------------------------------------------------------------------
    # Get the keys corresponding to a spec
    # - None:                  all the keys
    # - Existing key:          this key
    # - Base name of a series: all the series keys
    
    def get_keys(self, name):
        
        if name is None:
            return self.all_keys
        
        elif type(name) is str:
            if self.key_exists(name):
                return [name]
            else:
                return self.series_keys(name)
            
        elif hasattr(name, '__len__'):
            return name
        
        else:
            raise WError(f"Invalide name '{name}' to build a list of keys.",
                        Class = "WShapeKeys",
                        Method = "get_keys",
                        Object   = self.name,
                        name     = name)
    

    # ===========================================================================
    # Get the mesh vertices
    #
    # Verts are shape (shapes_count, verts_count, 3)
    
    def get_mesh_vertices(self, name=None):
        
        # --------------------------------------------------
        # Get the blocks to read
        
        keys = self.get_keys(name)
        if len(keys) == 0:
            return None
            
        blocks = self[keys]
        if len(blocks) == 0:
            return None

        # --------------------------------------------------
        # Let's read the blocks
        
        verts_count = len(blocks[0].data)
        verts = np.zeros((len(blocks), verts_count, 3), float)
        
        a = np.zeros(verts_count*3, float)

        for i_sk, block in enumerate(blocks):
            block.data.foreach_get('co', a)
            verts[i_sk] = a.reshape(verts_count, 3)
            
        return verts
        
    # ===========================================================================
    # Set the mesh vertices
    
    def set_mesh_vertices(self, verts, name="Key"):

        # ----- Check the validity of the vertices shape
        
        shape = np.shape(verts)
        if len(shape) == 2:
            count = 1
        elif len(shape) == 3:
            count = shape[0]
        else:
            raise WError(f"Impossible to set mesh vertices with shape {shape}. The vertices must be shaped either in two ot three dimensions.",
                        Class = "WShapeKeys",
                        Method = "set_mesh_vertices",
                        Object      = self.name,
                        verts_shape = np.shape(verts),
                        name        = name)
            
        # ----- The list of keys
            
        keys = self.series_key(name, range(count))
        
        # ----- Let's create the keys
        
        vs = np.reshape(verts, (count, shape[-2]*3))
        a = np.array(vs[0])
        for i, key in enumerate(keys):
            
            block = self.shape_key(key, create=True)
            
            a[:] = vs[i]
            block.data.foreach_set('co', a)
            
    # ===========================================================================
    # Get the curve vertices
    
    def get_curve_vertices(self, name=None):
        
        # --------------------------------------------------
        # Get the blocks to read
        
        keys = self.get_keys(name)
        if len(keys) == 0:
            return None
            
        blocks = self[keys]
        count = len(blocks)
        if count == 0:
            return None
        
        # --------------------------------------------------
        # Let's read the blocks
        
        # --------------------------------------------------
        # Need to read the alternance bezier / not bezier
        
        profile     = WCurve(self.data).profile
        
        only_bezier = profile.only_bezier
        only_nurbs  = profile.only_nurbs
        
        nverts      = profile.verts_count
        verts       = np.zeros((count, nverts, 3), float)
            
        # --------------------------------------------------
        # Loop on the shapes

        for i_sk, sk in enumerate(blocks):
            
            # --------------------------------------------------
            # Load bezier when only bezier curves
            
            if only_bezier:
                
                co = np.empty(nverts*3, float)
                le = np.empty(nverts*3, float)
                ri = np.empty(nverts*3, float)
                
                sk.data.foreach_get('co',           co)
                sk.data.foreach_get('handle_left',  le)
                sk.data.foreach_get('handle_right', ri)
                
                if True:
                    for _, index, n in profile.points_iter():
                        i3 = index*3
                        verts[i_sk, i3       :i3 + n  ] = co.reshape(nverts, 3)[index: index+n]
                        verts[i_sk, i3 + n   :i3 + 2*n] = le.reshape(nverts, 3)[index: index+n]
                        verts[i_sk, i3 + 2*n :i3 + 3*n] = ri.reshape(nverts, 3)[index: index+n]

                else:
                    index = 0
                    for t, n, st in profile:
                        i3 = index*3
                        verts[i_sk, i3       :i3 + n  ] = co.reshape(nverts, 3)[index: index+n]
                        verts[i_sk, i3 + n   :i3 + 2*n] = le.reshape(nverts, 3)[index: index+n]
                        verts[i_sk, i3 + 2*n :i3 + 3*n] = ri.reshape(nverts, 3)[index: index+n]
                        
                        index += n
                    
                    
            # --------------------------------------------------
            # Load nurbs when no bezier curves
            
            elif only_nurbs:
                
                co = np.empty(nverts*3, float)
                
                sk.data.foreach_get('co', co)
                
                if True:
                    for _, index, n in profile.points_iter():
                        verts[i_sk, index :index + n] = co.reshape(nverts, 3)[index: index+n]
                else:
                    index = 0
                    for t, n, st in profile:
                        verts[i_sk, index :index + n] = co.reshape(nverts, 3)[index: index+n]
                        
                        index += n
                    
            # --------------------------------------------------
            # We have to loop :-(
            
            else:
                if True:
                    index  = 0
                    for ctype, offset, n in profile.points_iter():
                        
                        if ctype == 0:
                            for i in range(n):
            
                                usk = sk.data[offset + i]
            
                                verts[i_sk, index       + i] = usk.co
                                verts[i_sk, index +   n + i] = usk.handle_left
                                verts[i_sk, index + 2*n + i] = usk.handle_right
                                
                            index += 3*n
                            
                        else:
                            for i in range(n):
            
                                usk = sk.data[offset + i]
                                verts[i_sk, index + i] = usk.co
                                
                            index += n
                            
                else:
                    i_data = 0
                    index  = 0
                    
                    for t, n, st in profile:
                        
                        if t == 3:
                            
                            for i in range(n):
            
                                usk = sk.data[i_data + i]
            
                                verts[i_sk, index       + i] = usk.co
                                verts[i_sk, index +   n + i] = usk.handle_left
                                verts[i_sk, index + 2*n + i] = usk.handle_right
                                
                            index += 3*n
                            
                        else:
                        
                            for i in range(n):
            
                                usk = sk.data[i_data + i]
            
                                verts[i_sk, index + i] = usk.co
                                
                            index += n
                            
                        i_data += n

        return verts

    # ===========================================================================
    # Get the curve vertices
    
    def set_curve_vertices(self, verts, name="Key"):
        
        # ----- Check the validity of the vertices shape
        
        shape = np.shape(verts)
        if len(shape) == 2:
            count = 1
        elif len(shape) == 3:
            count = shape[0]
        else:
            raise WError(f"Impossible to set curve vertices with shape {shape}. The vertices must be shaped either in two ot three dimensions.",
                        Class = "WShapeKeys",
                        Method = "set_curve_vertices",
                        Object      = self.name,
                        verts_shape = np.shape(verts),
                        name        = name)
            
        # --------------------------------------------------
        # Let's write the blocks
        
        # --------------------------------------------------
        # Need to read the alternance bezier / not bezier
        
        profile     = WCurve(self.data).profile
        only_bezier = profile.only_bezier
        only_nurbs  = profile.only_nurbs
        
        nverts      = profile.verts_count
        if nverts != shape[-2]:
            raise WError(f"Impossible to set curve vertices with shape {shape}. The number of vertices per shape must be {nverts}, not {shape[-2]}.",
                        Class = "WShapeKeys",
                        Method = "set_curve_vertices",
                        Object      = self.name,
                        verts_shape = np.shape(verts),
                        name        = name)
            
        # --------------------------------------------------
        # Loop on the shapes
        
        # ----- The list of keys
            
        keys = self.series_key(name, range(count))
        
        for i_sk, key in enumerate(keys):
            
            sk = self.shape_key(key, create=True)
            
            # --------------------------------------------------
            # Only bezier curves
            
            if only_bezier:
                
                co = np.empty((nverts, 3), float)
                le = np.empty((nverts, 3), float)
                ri = np.empty((nverts, 3), float)
                
                if True:
                    for _, index, n in profile.points_iter():
                        i3 = index*3
                        co[index:index + n] = verts[i_sk, i3       :i3 + n  ]
                        le[index:index + n] = verts[i_sk, i3 + n   :i3 + 2*n]
                        ri[index:index + n] = verts[i_sk, i3 + 2*n :i3 + 3*n]
                        
                else:
                    index = 0
                    for t, n, st in profile:
                        i3 = index*3
                        co[index:index + n] = verts[i_sk, i3       :i3 + n  ]
                        le[index:index + n] = verts[i_sk, i3 + n   :i3 + 2*n]
                        ri[index:index + n] = verts[i_sk, i3 + 2*n :i3 + 3*n]
                        
                        index += n
                        
                sk.data.foreach_set('co',           co.reshape(nverts * 3))
                sk.data.foreach_set('handle_left',  le.reshape(nverts * 3))
                sk.data.foreach_set('handle_right', ri.reshape(nverts * 3))
                    
            # --------------------------------------------------
            # No bezier curve at all
            
            elif only_nurbs:
                
                co = np.empty((nverts, 3), float)
                
                if True:
                    for _, index, n in profile.points_iter():
                        co[index: index+n] = verts[i_sk, index :index + n]
                else:
                    index = 0
                    for t, n, st in profile:
                        co[index: index+n] = verts[i_sk, index :index + n]
                        
                        index += n
                    
                sk.data.foreach_set('co', co.reshape(nverts * 3))

            # --------------------------------------------------
            # We have to loop :-(        
            
            else:
                if True:
                    index = 0
                    for ctype, offset, n in profile.points_iter():
                        if ctype == 3:
                            
                            for i in range(n):
            
                                usk = sk.data[offset + i]
            
                                usk.co           = verts[i_sk, index       + i]
                                usk.handle_left  = verts[i_sk, index +   n + i]
                                usk.handle_right = verts[i_sk, index + 2*n + i]
                                
                            index += 3*n
                            
                        else:
                        
                            for i in range(n):
            
                                usk = sk.data[offset + i]
            
                                usk.co = verts[i_sk, index + i]
                                
                            index += n
                        
                else:
                    i_data = 0
                    index  = 0
                    
                    for t, n, st in profile:
                        
                        if t == 3:
                            
                            for i in range(n):
            
                                usk = sk.data[i_data + i]
            
                                usk.co           = verts[i_sk, index       + i]
                                usk.handle_left  = verts[i_sk, index +   n + i]
                                usk.handle_right = verts[i_sk, index + 2*n + i]
                                
                            index += 3*n
                            
                        else:
                        
                            for i in range(n):
            
                                usk = sk.data[i_data + i]
            
                                usk.co = verts[i_sk, index + i]
                                
                            index += n
                            
                        i_data += n

    # ----------------------------------------------------------------------------------------------------
    # Get vertices
    
    def get_verts(self, name=None):
        
        if self.object_type == 'Mesh':
            return self.get_mesh_vertices(name)
        else:
            return self.get_curve_vertices(name)
    
    def set_verts(self, name="Key"):
        
        if self.object_type == 'Mesh':
            return self.set_mesh_vertices(name)
        else:
            return self.set_curve_vertices(name)
    

    # ---------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # Get animation from shape keys
    
    def get_sk_animation(self, eval_times):
        memo = self.eval_time
        
        verts = np.empty((len(eval_times), self.verts_count, 3), float)
        for i, evt in enumerate(eval_times):
            self.eval_time = evt
            verts[i] = self.evaluated.verts
        
        self.eval_time = memo
        
        return verts
    
