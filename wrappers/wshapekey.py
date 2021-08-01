#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:03:03 2021

@author: alain
"""

import numpy as np

from .wstruct import WStruct

from ..core.plural import to_shape

# ---------------------------------------------------------------------------
# Shape keys data blocks wrappers
# wrapped = Shapekey (key_blocks item)

class WShapekey(WStruct):
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
        
        raise RuntimeError(
            error_title % "WShapekey" +
            f"The attribut '{name}' doesn't exist for this shape key '{self.name}'."
            )

    @property
    def verts(self):
        """The vertices of the shape key.

        Returns
        -------
        numpy array of shape (len, 3)
            The vertices.
        """
        
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("co", a)
        return a.reshape((count, 3))

    @verts.setter
    def verts(self, value):
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("co", a)

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
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_left", a)
        return a.reshape((count, 3))

    @lefts.setter
    def lefts(self, value):
        self.check_attr("handle_left")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("handle_left", a)

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
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_right", a)
        return a.reshape((count, 3))

    @rights.setter
    def rights(self, value):
        self.check_attr("handle_right")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("handle_right", a)

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
        count = len(self.data)
        a = to_shape(value, count)
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
        count = len(self.data)
        a = np.empty(count, np.float)
        data.foreach_get("tilt", a)
        return a

    @tilts.setter
    def tilts(self, value):
        self.check_attr("tilt")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count)
        data.foreach_set("tilt", a)