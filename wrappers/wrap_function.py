#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:43:24 2021

@author: alain
"""

import bpy

from .wstruct      import WStruct
from .wobject      import WObject
from .wmeshobject  import WMeshObject
from .wcurveobject import WCurveObject
from .wtextobject  import WTextObject
from .wcurve       import WCurve
from .wmesh        import WMesh
from .wtext        import WText

from ..blender.blender import create_object

from ..core.commons import WError

# ---------------------------------------------------------------------------
# Wrapper

def wrap(name, create=None, **kwargs):
    """Wrap an object.
    
    To wrap an object, use this function rather than the direct class instanciation:
        - use: wobj = wrap("Cube")
        - avoid: wobj = WObject("Cube")

    Parameters
    ----------
    name : str or object with name property.
        The Blender object to wrap.

    Raises
    ------
    RuntimeError
        If the object doesn't exist.

    Returns
    -------
    WID
        The wrapper of the object.
    """

    # Nothing to wrap    
    if name is None:
        return None
    
    # The name of an object is given rather than an object instance
    if type(name) is str:
        obj = bpy.data.objects.get(name)
        if (obj is None) and (create is not None):
            obj = create_object(name, what=create, **kwargs)
    else:
        obj = name
        
    # If None, it doesn't mean the object with the given name doesn't exist
    if obj is None:
        raise WError(f"Object named '{name}' not found",
                     Function = "wrap",
                     name = name,
                     create = create,
                     **kwargs)
        
    # The argument is already a wrapper
    if issubclass(type(obj), WStruct):
        return obj
    
    # Initialize with the proper wrapper depending on the type of the blender ID
    cname = obj.__class__.__name__
    
    # -------------------------
    # Wrap an object
    
    if cname == "Object":
        
        # Empty object
        if obj.data is None:
            return WObject(obj)
        
        # Not an empty
        data_class = obj.data.__class__.__name__

        if data_class == 'Mesh':
            return WMeshObject(obj)
        
        elif data_class in ['Curve', 'SurfaceCurve']:
            return WCurveObject(obj)
        
        elif data_class == 'TextCurve':
            return WTextObject(obj)
        
        else:
            return WObject(obj)
        
            # Doesn't raise an error 
        
            raise WError(f"The object '{obj.name}' has data of type '{data_class}' which is not yet supported!",
                         Function = "wrap",
                         name = name,
                         create = create,
                         **kwargs)
            
    # -------------------------
    # Wrap a data structure

    elif cname == "Curve":
        return WCurve(obj)
    
    elif cname == "Mesh":
        return WMesh(obj)
    
    elif cname == "TextCurve":
        return WText(obj)
    
    elif cname == 'Spline':
        return WCurve.spline_wrapper(obj)
    
    # -------------------------
    # Not supported

    else:
        raise WError(f"Blender class {cname} not yet wrapped !",
                     Function = "wrap",
                     name = name,
                     create = create,
                     **kwargs)


# ---------------------------------------------------------------------------
# Collection wrapper

class WColl():
    def __init__(self, name):
        self.collection = WColl.get_collection(name)
        
    @staticmethod
    def get_collection(name):
        
        # The name of an object is given rather than an object instance
        if type(name) is str:
            coll = bpy.data.collections.get(name)
            if coll is None:
                raise WError(f"Collection {name} doesn't exist !",
                             Class = "WColl",
                             Function = "get_collection",
                             name = name)
            return coll
        
        obj = name
        if hasattr(obj, 'name'):
            return WColl.get_collection(obj.name)
        
        raise WError(f"Object {obj} is not a collection !",
                Class = "WColl",
                Function = "get_collection",
                name = name)
        
    @property
    def name(self):
        return self.collection.name
    
    @name.setter
    def name(self, value):
        self.collection.name = value
        
    def __repr__(self):
        return f"Collection wrapper {self.collection.name}"
    
    def __len__(self):
        return len(self.collection.objects)
    
    def __getitem__(self, index):
        return wrap(self.collection.objects[index])
    
    def keys(self):
        return self.collection.objects.keys()
    
    def values(self):
        return [wrap(o) for o in self.collection.objects.values()]
    
    def items(self):
        return [(k, wrap(o)) for k, o in self.collection.objects.items()]
    

def wcollection(name):
    return WColl(name)

        
# ---------------------------------------------------------------------------
# Get the wrapped

def unwrap(thing):
    if hasattr(thing, 'wrapped'):
        return thing.wrapped
    elif type(thing) is str:
        return bpy.data.objects[thing]
    else:
        return thing
        
