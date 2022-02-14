#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 16:02:15 2022

@author: alain
"""

import numpy as np


# ---------------------------------------------------------------------------
# A mesh object as an array of group names : vertex_groups
# Each vertex has an array of couples (group index, weight)
# These values are linearized into an array of triplets:
# - group index
# - vertex index
# - 100000 * weight
    

class WVertexGroups():
    
    def __init__(self, object=None):
        if object is None:
            self.groups   = []
            self.triplets = np.zeros((0, 3), int)
        else:
            self.groups   = [group.name for group in object.vertex_groups]
            mesh = object.data
            
            self.triplets = WVertexGroups.read_triplets(mesh)
            
    # ---------------------------------------------------------------------------
    # Read the triplets from a mesh
                    
    @staticmethod
    def read_triplets(mesh):
        
        triplets = np.empty((0, 3), int)
        
        for index, vert in enumerate(mesh.vertices):
            n = len(vert.groups)
            if n > 0:
                ts = np.empty((n, 3), int)
                for i, g in enumerate(vert.groups):
                    ts[i] = (g.group, index, round(g.weight*100000))
                triplets = np.append(triplets, ts, axis=0)
                
        return triplets
    
    # ---------------------------------------------------------------------------
    # Clone the instance
    
    def clone(self):
        vg = WVertexGroups()
        vg.groups = [name for name in self.groups]
        vg.triplets = np.array(self.triplets)
        return vg
        
    # ---------------------------------------------------------------------------
    # Content
                    
    def __repr__(self):
        s = f"<WVertexGroups with {len(self.groups)} groups:\n"
        for i, name in enumerate(self.groups):
            s += f"   {name:20s}: {len(self.couples(i)):3d} {self.vertices(i)}\n"
        return s + ">"

    # ----------------------------------------------------------------------------------------------------
    # group name exists
    
    def group_exists(self, name):
        return name in self.groups

    # ----------------------------------------------------------------------------------------------------
    # Index of a group name
    
    def group_index(self, name):
        return self.groups.index(name)
    
    # ----------------------------------------------------------------------------------------------------
    # List of couples (vertex, weight) for a group
    
    def couples(self, index):
        if type(index) is str:
            idx = self.group_index(index)
        else:
            idx = index
            
        return np.array(self.triplets[self.triplets.T[0] == idx].T[1:].T)
    
    # ----------------------------------------------------------------------------------------------------
    # Indices of a group
    
    def vertices(self, index):
        if type(index) is str:
            idx = self.group_index(index)
        else:
            idx = index
            
        return np.array(self.triplets[self.triplets.T[0] == idx].T[1])
    
    
    # ----------------------------------------------------------------------------------------------------
    # Add a group
    
    def add(self, group_name, couples):
        
        index = None
        for i, name in enumerate(self.groups):
            if name == group_name:
                index = i
                break
            
        if index is None:
            index = len(self.groups)
            self.groups.append(group_name)
            
        n = len(couples)
        if n > 0:
            ts = np.empty((3, n), int)
            ts[0] = index
            ts[1:].T[:] = couples
            
            self.triplets = np.append(self.triplets, ts.T, axis=0)
    
    # ----------------------------------------------------------------------------------------------------
    # Join other vertex groups
    # To join another group, we need to change the indices:
    # - Group indices change in the new list
    # - Vertices are shifted with the number of vertices in the self mesh
    
    def join(self, other, verts_count):
        
        for index, name in enumerate(other.groups):
            couples = other.couples(index)
            couples.T[0] += verts_count
            self.add(name, couples)
            
    # ----------------------------------------------------------------------------------------------------
    # Array : duplicates the current triplets
    
    def array(self, count, verts_count):
        
        n = len(self.triplets)
        if n == 0:
            return
        
        triplets = np.empty((count, n, 3), int)
        triplets[:] = self.triplets
        triplets.T[1] += np.arange(count)*verts_count
        
        del self.triplets
        self.triplets = np.reshape(triplets, (count*n, 3))
        
    # ----------------------------------------------------------------------------------------------------
    # Set to an object
    
    def set_to(self, object):
        
        # ----- Reset the groups
        
        vertex_groups = object.vertex_groups
        vertex_groups.clear()
        
        for g_index, name in enumerate(self.groups):
            
            group = vertex_groups.new(name=name)
            
            couples = self.couples(g_index).T
            
            vertices = couples[0].tolist()
            
            group.add(vertices, 1.0, 'REPLACE')
            
            diff = 100000 - couples[1]
            n = len(diff[diff > 0])
            if n > 0:
                for i, w in enumerate(diff):
                    if w > 0:
                        group.add([vertices[i]], w/100000, 'SUBTRACT')

    

    
    

    
        

        
        

        
        
        
        
        