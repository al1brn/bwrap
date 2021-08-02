#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 07:23:58 2021

@author: alain
"""

# ---------------------------------------------------------------------------
# Enhance the class by dinaically creating methods and properties

from inspect import signature

# ---------------------------------------------------------------------------
# Copy methods and properties from a class to another having a property of this class
#
# This allow to expose methods and properties of sub structure directly
# to the owner.
#
# For instances, methods and properties of WMesh are exposed directly to the
# class WMEshObject

def expose(target_class, source_class, prop_name, methods=[], properties={}):
    """Expose methods and properties of a class to another having a property of this type.

    Example:
        expose(WMeshObject, WMesh, "wmesh", ["verts"], {'verts_count': 'RO'})
        
    After this call, the class WMeshObject will expose the method verts and
    the readonly property verts_count calling the method and property of this
    names on the property wmesh.
    
    Parameters
    ----------
    target_class : class
        The target class for which to create methods ans properties.
    source_class : class
        The source class implementing the methods and properties.
    prop_name : str
        The name of the property of the target class which must be a class
        of type source_class.
    methods : array of str, optional
        List of the source class methodes to expose. The default is [].
    properties : Dictionary, optional
        Array of properties to expose. The keys are the names of the properties and
        the values are 'RO' for read only properties and 'RW' for read and write
        properties. The default is {}.
    """
    
    # ----- Methods
    
    for meth in methods:
        
        args = signature(getattr(source_class, meth, ''))

        s_args = ""
        s_prms = ""
        sepa   = ""
        
        for prm in args.parameters.values():
            if prm.name != 'self':
                s_args += sepa + f"{prm}"
                s_prms += sepa + prm.name
                
                sepa = ", "
            
        s_lambda = "lambda self, " + s_args + f": self.{prop_name}.{meth}(" + s_prms + ")"
        setattr(target_class, meth, eval(s_lambda))
        
    # ----- Properties
    
    for prop, ro in properties.items():

        s_get = f"lambda self: self.{prop_name}.{prop}"
        s_set = f"lambda self, value: setattr(self.{prop_name}, '{prop}', value)"

        if ro == 'RO':
            setattr(target_class, prop, property(eval(s_get)))
        else:
            setattr(target_class, prop, property(eval(s_get), eval(s_set)))
            
# ===========================================================================
# Some test materials
            
            
class Source():
    def __init__(self, s_param="init string", i_param=9, f_param=1., b_param=True):
        self.s_param = s_param
        self.i_param = i_param
        self.f_param = f_param
        self.b_param = b_param
        self.rw_     = None 
        
    def __repr__(self):
        return "<Class Source>"
        
    @property
    def ro_prop(self):
        return "The read only method"
    
    @property
    def rw_prop(self):
        return self.rw_
    
    @rw_prop.setter
    def rw_prop(self, value):
        self.rw_ = value
        
    def change(self, a, b, c="c", d=99, e=100., f=False):
        print("We change", a, b, c, d, e, f)
        print(self)
        self.s_param = c

class Target():
    def __init__(self):
        self.source = Source()
        
    def __repr__(self):
        return "<Class Target>"
        
    def test(self, change="The change"):
        print('-'*30)
        print("Attributes")
        for k in dir(self):
            if k[0] != '_':
                print(k, ':', getattr(self, k))
                try:
                    sig = signature(getattr(Target, k))
                    print(sig)
                except:
                    pass
        print()
        print("Go for test...")
        try:
            print(self.s_param)
            print(self.ro_prop)
            print(self.rw_prop)
            self.change(6, 7, c=change, f=True)
            print(self.s_param)
        except:
            print("Test failed")

def test():        
    t = Target()
    t.test("Shouldn't work")
    
    expose(Target, Source, "source", ["change"], {
        "s_param": 'RW', "i_param": 'RW', "f_param": 'RW', "b_param": 'RW', "rw_prop": 'RW', "ro_prop": 'RO'})
    
    t = Target()
    t.test("Should work!")



    