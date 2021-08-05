#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 08:57:02 2021

@author: alain
"""
import bpy

# ---------------------------------------------------------------------------
# depsgraph as a global var must be evaluated once per frame
# to avoid hard crash

DEPSGRAPH = None

def depsgraph():
    
    global DEPSGRAPH
    
    if DEPSGRAPH is None:
        DEPSGRAPH = bpy.context.evaluated_depsgraph_get()
    
    return DEPSGRAPH

def reset_depsgraph():
    global DEPSGRAPH
    
    DEPSGRAPH = None
    #bpy.context.view_layer.update()
    
def get_object(name, evaluated=False):
    
    # DEBUG
    #return bpy.data.objects[name]
    
    if evaluated:
        return bpy.data.objects[name].evaluated_get(depsgraph())        
    else:
        return bpy.data.objects[name]
    
def get_collection(name, evaluated=False):
    
    def find(coll):
        try:
            return coll.children[name]
        except:
            pass
        for c in coll.children: 
            ok = find(c)
            if ok is not None:
                return ok
        return None
    
    if evaluated:
        return find(depsgraph().scene_eval.collection)
    else:
        return bpy.data.collections[name]
    
def get_evaluated(thing):
    return thing.evaluated_get(depsgraph())

        
    

