#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:03:03 2021

@author: alain
"""

import numpy as np

import bpy

from ..blender import depsgraph

from .wstruct import WStruct

from ..core.commons import WError

#from ..core.plural import to_shape

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


class WShapeKeys(WStruct):
    
    def __init__(self, object):
        if type(object) is str:
            name = object
        else:
            name = object.name
            
        super().__init__(name=name)
        
    @property
    def wrapped(self):
        return depsgraph.get_object(self.name).data.shape_keys
    
    @property
    def key_blocks(self):
        sks = depsgraph.get_object(self.name).data.shape_keys
        return None if sks is None else sks.key_blocks
    
    # ---------------------------------------------------------------------------
    # As an array of shape keys
    
    def __len__(self):
        blocks = self.key_blocks
        if blocks is None:
            return 0
        else:
            return len(blocks)
        
    def __getitem__(self, index):
        blocks = self.key_blocks
        if blocks is None:
            return None
        else:
            return blocks[index]
        
        
class WMeshShapeKeys(WShapeKeys):
        
    # ---------------------------------------------------------------------------
    # Vertices
    
    @property
    def verts_count(self):
        blocks = self.key_blocks
        if blocks is None:
            return 0
        if len(blocks) == 0:
            return 0
        return len(blocks[0].data)
    
    def get_verts(self, index=None):
        
        n = self.verts_count
        if n == 0:
            return None
        
        if index is None:
            verts = np.empty((len(self), n, 3), np.float)
            for i in range(len(self)):
                verts[i] = self.get_verts(i)
            return verts
        else:
            verts = np.empty(n*3, np.float)
            data = self[index].data
            data.foreach_get('co', verts)
            return verts.reshape(n, 3)
        
    def set_verts(self, value, index=None):
        
        n = self.verts_count
        shape = np.shape(value)
        
        if index is None:
            expected = (len(self), n, 3)
        else:
            expected = (n, 3)
            
        if shape != expected:
            raise WError("The vertices array shape doesn't match the number of vertices of the objects",
                    Object = self.name,
                    index = index,
                    shape_keys_count = len(self),
                    vertices_count = n,
                    expected_shape = expected,
                    vertices_shape = shape)
            
        if index is None:
            for i in range(len(self)):
                self.set_verts(value[i], index=i)
        else:
            data = self[index].data
            data.foreach_set('co', value.reshape(n*3))
        
        
        
                        
    
    
        
            
    
    
            



# wrapped = Shapekey (key_blocks item)


# ---------------------------------------------------------------------------
# Shape keys data blocks wrappers
# wrapped = Shapekey (key_blocks item)

class WShapekey_OLD(WStruct):
    """Wraps the key_blocks collection of a shapekey class
    """

    @staticmethod
    def sk_name(name, step=None):
        """Returns then name of a shape key within a series        

        Parameters
        ----------
        name : str
            Base name of tyhe shape key.
        step : int, optional
            The step number. The default is None.

        Returns
        -------
        str
            Full shape key name: "shapekey 999".
        """
        
        return name if step is None else f"{name} {step:3d}"

    def __len__(self):
        """The wrapper behaves as an array.        

        Returns
        -------
        int
            Le length of the collection.
        """
        
        return len(self.wrapped.data)

    def __getitem__(self, index):
        """The wrapper behaves as an array.

        Parameters
        ----------
        index : int
            Item index.

        Returns
        -------
        ShapeKey
            The indexed shape key.
        """
        
        return self.wrapped.data[index]

    def check_attr(self, name):
        """Check if an attribute exists.
        
        Return only if the attr exist, raise an error otherwise.
        Shape key is used for meshes and splines. This utility raises an error if the user
        makes a mistake with the type of keyed shape.

        Parameters
        ----------
        name : str
            Attribute name.

        Raises
        ------
        RuntimeError
            If the attr doesn't exist.

        Returns
        -------
        None.
        """
        
        if name in dir(self.wrapped.data[0]):
            return
        
        raise WError(f"The attribut '{name}' doesn't exist for this shape key '{self.name}'.")

    @property
    def verts(self):
        """The vertices of the shape key.

        Returns
        -------
        numpy array of shape (len, 3)
            The vertices.
        """
        
        data = self.wrapped.data
        count = len(data)
        a = np.empty(count*3, np.float)
        data.foreach_get("co", a)
        return a.reshape((count, 3))

    @verts.setter
    def verts(self, value):
        data = self.wrapped.data
        count = len(data)
        a = np.empty((count, 3), np.float)
        a[:] = value
        data.foreach_set("co", a.reshape(count*3))

    @property
    def lefts(self):
        """Left handles of spline.   

        Returns
        -------
        numpy array of shape (len, 3)
            The left handles.
        """
        
        self.check_attr("handle_left")
        data = self.wrapped.data
        count = len(data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_left", a)
        return a.reshape((count, 3))

    @lefts.setter
    def lefts(self, value):
        self.check_attr("handle_left")
        data = self.wrapped.data
        count = len(data)
        a = np.empty((count, 3), np.float)
        a[:] = value
        data.foreach_set("handle_left", a.reshape(count*3))

    @property
    def rights(self):
        """Right handles of spline.   

        Returns
        -------
        numpy array of shape (len, 3)
            The right handles.
        """
        
        self.check_attr("handle_right")
        data = self.wrapped.data
        count = len(data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_right", a)
        return a.reshape((count, 3))

    @rights.setter
    def rights(self, value):
        self.check_attr("handle_right")
        data = self.wrapped.data
        count = len(data)
        a = np.empty((count, 3), np.float)
        a[:] = value
        data.foreach_set("handle_right", a.reshape(count*3))

    @property
    def radius(self):
        """Radius of spline.   

        Returns
        -------
        numpy array of shape (len)
            The radius.
        """
        
        self.check_attr("radius")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count, np.float)
        data.foreach_get("radius", a)
        return a

    @radius.setter
    def radius(self, value):
        self.check_attr("radius")
        data = self.wrapped.data
        count = len(data)
        a = np.empty(count, np.float)
        a[:] = value
        data.foreach_set("radius", a)

    @property
    def tilts(self):
        """Tilts of spline.   

        Returns
        -------
        numpy array of shape (len)
            The tilts.
        """
        
        self.check_attr("tilt")
        data = self.wrapped.data
        count = len(data)
        a = np.empty(count, np.float)
        data.foreach_get("tilt", a)
        return a

    @tilts.setter
    def tilts(self, value):
        self.check_attr("tilt")
        data = self.wrapped.data
        count = len(data)
        a = np.empty(count, np.float)
        a[:] = value
        data.foreach_set("tilt", a)