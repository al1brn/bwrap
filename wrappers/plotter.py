#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 07:10:56 2021

@author: alain
"""

import bpy

import numpy as np
from .wrap_function import wrap

def get_material(name):
    
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        
    return mat

def get_col_material(num):
    mat = get_material(f"Plotter {num:03d}")
    
    coul = [0, 0, 0, 1]
    coul[0] = 1. if (num % 7) in [0, 3, 4, 6] else 0.1
    coul[1] = 1. if (num % 7) in [1, 3, 5, 6] else 0.1
    coul[2] = 1. if (num % 7) in [2, 4, 5, 6] else 0.1
    
    if num >= 7:
        coul[0] *= .6 + .4*np.sin(num*875.795)
        coul[1] *= .6 + .4*np.sin(num*123.67989)
        coul[2] *= .6 + .4*np.sin(num*768.678)
        
    mat.diffuse_color = coul
    
    return mat

def get_mat_index(object_name, material_name):
    obj = bpy.data.objects[object_name]
    mat = get_material(material_name)
    
    cur = obj.data.materials.get(mat.name)
    if cur is None:
        obj.data.materials.append(mat)
        
    for i in range(len(obj.data.materials)):
        if obj.data.materials[i].name == mat.name:
            return i
        
def get_col_index(object_name, num):
    return get_mat_index(object_name, get_col_material(num).name)
        


class Plotter2D():
    def __init__(self, name="Plotter 2D", type="Curve", size = (10, 10)):
        self.wobject = wrap(name, create='CUBE' if type == 'Mesh' else 'BEZIER')
        
        self.size  = np.array(size)
        self.xmin = 0
        self.ymin = 0
        self.xmax = 1
        self.ymax = 1
        
        self.curves = []
        
        self.refresh()
        
    @property
    def is_mesh(self):
        return self.wobject.object_type == 'Mesh'
    
    def add_curve(self, x, y):
        
        self.xmin = min(self.xmin, np.min(x))
        self.xmax = max(self.xmax, np.max(x))
        self.ymin = min(self.ymin, np.min(y))
        self.ymax = max(self.ymax, np.max(y))
        
        n = max(len(x), len(y))
        v = np.stack((np.resize(x, n), np.resize(y, n), np.zeros(n, float))).transpose()
        edges = np.stack((np.arange(n-1), np.arange(n-1)+1)).transpose()
        
        self.curves.append((v, edges))
        
        self.refresh()
        
    def add_function(self, f, x0=0, x1=1, count=100):
        x = np.linspace(x0, x1, count)
        self.add_curve(x, f(x))
        
        
    def refresh(self):
        
        corner = np.array((self.xmin, 0, self.ymin))
        ratio  = np.array((self.size[0]/(self.xmax - self.xmin), self.size[1]/(self.ymax - self.ymin), 1))
        
        v_count = 0
        e_count = 0
        profile = np.zeros((len(self.curves), 3), int)
        profile[:, 0] = 3
        profile[:, 2] = 0
        for i, (v, e) in enumerate(self.curves):
            v_count += len(v)
            e_count += len(e)
            profile[i, 1] = len(v) 

        verts = np.zeros((v_count, 3), float)
        edges = []
        v_index = 0
        
        for (v, e) in self.curves:
            verts[v_index:v_index+len(v)] = (v - corner)*ratio
            edges.extend([((edge + v_index)[0], (edge + v_index)[1]) for edge in e])
            v_index += len(v)
            
        if self.is_mesh:
            self.wobject.new_geometry(verts, edges=edges)
        else:
            self.wobject.wcurve.wsplines.profile = profile
            for i, (v, _) in enumerate(self.curves):
                self.wobject.wcurve.wsplines[i].set_vertices((v - corner)*ratio)
                self.wobject.wcurve.wsplines[i].material_index = get_col_index(self.wobject.name, i)
                
            
            
            
            
        
        
        
        
        
    
    
    
    
        
        
