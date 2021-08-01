#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 19:46:29 2021

@author: alain
"""

import bpy
import numpy as np

print('-'*100)

TAB = "    "

# ===========================================================================
# Source code generator for automatic wrapper classes enrichment
#
# The purpose is to have all properties and methods of the wrapped objects
# supported by tyhe wrapper.
#
# To avoid painfull documentation copy, this is automated with source code
# generator.
#
# They are two ways to enrich a wrapper class:
# - Dynamically at run time
# - By generating the source code for the class
#
# Both methods rely on the same two phases:
# 1) Properties and methods catching
# 2) Code generation
#
# For dynammic enrichment, a dictionnary of the properties and methods is generated
# At run time, when the module is loaded, one must call a method taking this
# dictionnary as parameter and which enrich the wrapper classes.
#
# For source code enrichment, the generated sources is simply copied in the
# the source file of the class
#
# ----- Properties and methods catching
#
# The generation must be done with Blender because and instance of the
# wrapped object if required.
#
# The generator list all the properties and method of the wrapped object and
# generate de source code when there is no name declared for the existing class.
#
# Exemple: Let's imagine we want to enrich the wrapper WObject of the
# Blender Object.
#
# One must create an object, a cube for instance.
# Then call the function:
#
# > pygen_properties_and_methods(Wrapper, object, code='DICT or 'CLASS')
#
# Then simply copy the code generated in the terminal into the source file.
#

# ===========================================================================
# GENERATOR FUNCTIONS

# ---------------------------------------------------------------------------
# Get the properties from an rna_type structure
# Allows to know if a property is read only (RO) or not (RW)
# Exluded is typically used to exclude the existing attributes in the
# wrapper class

def get_rna_properties(rna_type, excluded=[]):
    props = {}
    for p in rna_type.properties:
        name = p.identifier
        if not name in excluded:
            props[name] = 'RO' if p.is_readonly else 'RW'
    return props

# ---------------------------------------------------------------------------
# Dump the properties and methods of a Blender ID
# Properties and methodes are read from rna_type to get readonly status.
# Properties which are not in rna_type are implemented read_only

def get_properties_and_methods(obj, excluded=[]):
    
    # ----- rna properties
    pms = get_rna_properties(obj.rna_type, excluded=excluded)
    
    # ----- Python props and methods
    for k in dir(obj):
        if k[:2] != '__':
            if (k not in pms) and (k not in excluded):
                v = getattr(obj, k)
                if type(v).__name__ == 'bpy_func':
                    pms[k] = 'MT'
                else:
                    pms[k] = 'RO'
    return pms

# ---------------------------------------------------------------------------
# Generate source code for an array of properties and methods of a class
#
# - cls      : the class which will be enriched at run time
# - wrapped  : the wrapped object to copy props and methos in the class cls
# - code     : 'DICT' or 'CLASS'
# - f        : an open file. If not, output is done in the console
#
# 'DICT': the source code is a dictionary containing the names of the props
#         and methods. The generated dict is exactly the dict built with
#         the function 'get_properties_and_methods'.
#         The generated array can be used at run time to dynamically add
#         the properties and methods with the function 'enrich_class':
#
#               enrich_class(Wrapper, WRAPPER)
#
#         where Wrapper is the class to enrich and WRAPPER the generated dict.
#
# 'CLASS': the source code is the properties and methods python source code.
#          This code can be then copied in the source code of the class file.

def pygen_properties_and_methods(cls, wrapped, code='DICT', f=None):
    
    pms = get_properties_and_methods(wrapped, excluded=dir(cls))
    
    # ----- Dict lines generator
    
    def dict_gen():
        yield "# " + "="*75
        yield "# " + f"Generated source code for {cls.__name__} class"
        yield ""

        s = "%s = {" % cls.__name__.upper()
        sep = ""
        for k in pms:
            s += sep
            sep = ", "
            item = f"'{k}': '{pms[k]}'"
            if len(s + item) > 80:
                yield s
                s = TAB
            s += item
        yield s + "}"
    
    # ----- Dict lines generator
    
    def class_gen():
        blank = TAB
        yield blank + "# " + "="*75
        yield blank + "# " + f"Generated source code for {cls.__name__} class"
        yield ""
        
        for name, val in pms.items():
            if val in ['RO', 'RW']:
                yield blank + "@property"
                yield blank + f"def {name}(self):"
                yield blank + f"{TAB}return self.wrapped.{name}"
                yield ""
        
                if val == 'RW':
                    yield blank + f"@{name}.setter"
                    yield blank + f"def {name}(self, value):"
                    yield blank + f"{TAB}self.wrapped.{name} = value"
                    yield ""
            elif val == 'MT':
                yield blank + f"def {name}(self, *args, **kwargs):"
                yield blank + f"{TAB}return self.wrapped.{name}(*args, **kwargs)"
                yield ""
                
        yield blank + "# " + "End of generation"
        yield blank + "# " + "="*75
                
                
    if code == 'DICT':
        gen = dict_gen()
    elif code == 'CLASS':
        gen = class_gen()
    else:
        raise RuntimeError(f"Unknown code generator: {code}")
        
    for line in gen:
        if f is None:
            print(line)
        else:
            f.write(line + '\n')

        
# ===========================================================================
# Dynamic creation of the methods and properties
#
# Dynamic creation generates source code which is executed with eval.
# The attributes are implemented with lambda function, typically:
#   prop getter: lambda self: self.wrapped.__getattribute__('name')
#
# methods are implemented with no control on arguments:
#   lambda self, *args, **kwargs: self.wrapped.name(*args, **kwargs)
#
# Properties and methods are implemented with setattr:
#   prop:   setattr(class, 'name', property(getter, setter))
#   method: settattr(class, 'name', call)
#

# ---------------------------------------------------------------------------
# Property creation source code

def pygen_property_creation(class_name, name, read_only):
    
    fget = f"lambda self: self.wrapped.__getattribute__('{name}')"
    fset = f"lambda self, value: self.wrapped.__setattr__('{name}', value)"
    ro_prop = f"property({fget})"
    rw_prop = f"property({fget}, {fset})"
    template = f"setattr({class_name}, '{name}', %s)"
    
    if read_only:
        return template % ro_prop
    else:
        return template % rw_prop
    
# ---------------------------------------------------------------------------
# Method creation source code

def pygen_method_creation(class_name, name):
    return f"setattr({class_name}, '{name}', lambda self, *args, **kwargs: self.wrapped.{name}(*args, **kwargs))"

# ---------------------------------------------------------------------------
# Dynamically update methods and properties of a class
# Can be done at generation time to test with the result
# of get_properties_and_methods or at reun time with the generated dict.

def enrich_class(cls, props_meths):
    
    class_name = cls.__name__
    
    for name in props_meths:
        if not name in dir(cls):
            val = props_meths[name]
            if val == 'RO':
                s = pygen_property_creation(class_name, name, True)
            elif val == 'RW':
                s = pygen_property_creation(class_name, name, False)
            elif val == 'MT':
                s = pygen_method_creation(class_name, name)

            eval(s)
            
# ===========================================================================
# Some tests
# CAUTION: needs a Cube object       
            
# ---------------------------------------------------------------------------
# Wrapper mockup

class Wrapper():
    def __init__(self, object):
        self.wrapped = object
        
    @property
    def name(self):
        return "ORIGINAL: " + self.wrapped.name

    @property
    def wname(self):
        return self.wrapped.__getattribute__('name') 
    
    @wname.setter
    def wname(self, value):
        self.wrapped.__setattr__('name', value)
        
# ---------------------------------------------------------------------------
# Test if the Wrapper is enriched

def test_wrapper(title):
    cube = bpy.data.objects["Cube"]
    wcube = Wrapper(cube)
    
    print('-----', title)
    print("Name:", wcube.name)

    try:
        print("Location: ", wcube.location)
        wcube.location = np.random.randint(-3, 3, 3)
    except:
        print("No location property !")
        
# ---------------------------------------------------------------------------

def test_gen(code='DICT', f=None):

    cube = bpy.data.objects["Cube"]
    
    pygen_properties_and_methods(Wrapper, cube, code=code, f=f)
    

# ---------------------------------------------------------------------------

def test_enrich():

    cube = bpy.data.objects["Cube"]
    
    pms = get_properties_and_methods(cube, excluded=dir(Wrapper))
    
    test_wrapper("BEFORE")
    
    enrich_class(Wrapper, pms)
    
    test_wrapper("AFTER")
    


