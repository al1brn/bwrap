#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:30:34 2021

@author: alain
"""

import types
import math

import numpy as np

# ===========================================================================
# Fractal structure basically manages:
# - verts:   An array fo vertices
# - indices: An array of arrays of indices called structures
#
# A structure can be a face or a pyramid, a cube or whatever
# - Sierpenski triangle : structures are made of 3 indices and are 1 face
# - Sierpenski pyrdamid : structures are made of 4 indices and are 4 faces
#
# The get_faces static method returns the faces from one structure
# It can be changed at initialization time
#
# The gen_f method can be overloaded or set at initialization time
#
# The gen_f function returns
# - The new vertices to create
# - An array of structures
#
# For instance, the sierpinski function return 3 more vertices and
# an array of 3 triplets of 3 indices in the array of vertices
#
# Lastly the depth of the recurring call is passed for information
#
# **kwargs arguments are possible


# ===========================================================================
# Fractal Generator

class Fractal():
    
    def __init__(self, verts, indices, shape, replace=True, gen_f=None, get_faces=None, get_edges=None):
        
        self.verts   = np.array(verts)
        
        self.shape   = shape if hasattr(shape, '__len__') else (shape,)
        self.indices = np.array(indices)
        if np.shape(self.indices) == self.shape:
            self.indices = np.reshape(self.indices, (1,) + self.shape)

        self.replace   = replace
        
        self.set_gen_f(gen_f)
        
        if get_faces is not None:
            self.get_faces = get_faces
        
        if get_edges is not None:
            self.get_edges = get_edges
            
        self.faces_ = []
        self.egdes_ = []

    def __repr__(self):
        return f"<Fractal shape:{self.shape}, verts: {len(self.verts)} indices: {self.indices.shape}>"
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        return self.indices[index]
    
    def set_gen_f(self, gen_f):
        if gen_f is None:
            return
        self.gen_f= types.MethodType(gen_f, self)
    
    # ---------------------------------------------------------------------------
    # Default conversion from structure to faces and edges
    
    @staticmethod
    def get_edges(indices):
        return [[indices[i], indices[i+1]] for i in range(len(indices)-1)]
    
    @staticmethod
    def get_closed_edges(indices):
        n = len(indices)
        return [[indices[i], indices[(i+1)%n]] for i in range(n)]
    
    @staticmethod
    def get_faces(indices):
        if len(indices) <= 2:
            return []
        else:
            return indices
    
    # ---------------------------------------------------------------------------
    # Generation step
    # By default implement the Koch fractal
    
    def gen_f(self, indices, depth, t=1):
        
        verts = self.verts[indices]
        
        vs = np.zeros((3, 3), float)
        vs[0] = verts[0]*2/3 + verts[1]/3
        vs[2] = verts[0]/3   + verts[1]*2/3
        
        side = verts[1] - verts[0]
        
        vp = np.array([-side[1], side[0], 0])
        
        vs[1] = verts[0] + side/2 + vp*np.cos(np.pi/6)/3
        
        base = len(self.verts)
        
        return vs, [
            (indices[0], base),
            (base,       base+1),
            (base+1,     base+2),
            (base+2,     indices[1])
            ]
    
    @property
    def edges(self):
        edges = []
        for struct in self.indices:
            e = self.get_edges(struct)
            if len(e) > 0:
                if len(np.shape(e)) == 1:
                    edges.append(e)
                else:
                    edges.extend(e)
                
        return edges
    
    @property
    def faces(self):
        faces = []
        for struct in self.indices:
            f = self.get_faces(struct)
            if len(f) > 0:
                if len(np.shape(f)) == 1:
                    faces.append(f)
                else:
                    faces.extend(f)
                
        return faces
    
    # ---------------------------------------------------------------------------
    # Add new vertices
    
    def add_verts(self, new_verts):
        index = len(self.verts)
        self.verts = np.append(self.verts, new_verts, axis=0)
        return np.arange(index, len(self.verts))
    
    # ===========================================================================
    # Koch flake
    
    @classmethod
    def Koch(cls, steps=0, flake=True):
        
        if flake:
            a = 2*math.cos(math.pi/6)
            verts = [(-1, -a/3, 0), (1, -a/3, 0), (0, 2*a/3, 0)]
            inds  = ((1, 0, 0), (0, 2, 0), (2, 1, 0))
        else:
            verts = [(-1, 0, 0), (1, 0, 0)]
            inds = (0, 1)
            
        frac = cls(verts, inds, 2)
        
        frac.compute(steps=steps)
        
        return frac
    
    # ===========================================================================
    # Siperpenski pyramid    
        
    @classmethod
    def Sierpenski(cls, steps=0, ndim=3):
        
        # ---------------------------------------------------------------------------
        # 2D generator
        
        def gen_f_2(self, indices, depth, t=1):
            
            vs = self.verts[indices]
            new_verts = np.array([(vs[i] + vs[(i+1)%3])/2 for i in range(3)])
            
            base = len(self.verts)
            return new_verts, [
                (indices[0], base+0, base+2), 
                (indices[1], base+1, base+0),
                (indices[2], base+2, base+1),
                ]
        
        # ---------------------------------------------------------------------------
        # 3D generator

        def gen_f_3(self, indices, depth, t=1):
            
            verts = self.verts[indices]
            
            new_verts = (
                (verts[0] + verts[1])/2, 
                (verts[0] + verts[2])/2,
                (verts[0] + verts[3])/2,
                
                (verts[1] + verts[2])/2,
                (verts[2] + verts[3])/2,
                (verts[3] + verts[1])/2
            )

            base = len(self.verts)
            return new_verts, [
                (indices[0], base+0, base+1, base+2),
                
                (base+0, indices[1], base+3, base+5),
                (base+1, indices[2], base+4, base+3),
                (base+2, indices[3], base+5, base+4)
                ]
        
        # ---------------------------------------------------------------------------
        # Initialization
        
        if ndim == 3:
            a = 1
            b = a/2/np.sqrt(6)
            
            v0 = ( 0, 0, a*np.sqrt(2/3))
            v1 = (-a/2, -b, 0)
            v2 = ( a/2, -b, 0)
            v3 = ( 0, -b + a*np.cos(np.pi/6), 0)
            
            verts   = [v0, v1, v2, v3]
            indices = (0, 1, 2, 3)
            
            get_faces = lambda inds: [
                [inds[0], inds[1], inds[2]],
                [inds[0], inds[2], inds[3]],
                [inds[0], inds[3], inds[1]],
                [inds[3], inds[2], inds[1]],
                ]
            
            frac = cls(verts, indices, 4, replace=True, gen_f=gen_f_3, get_faces=get_faces)
            
        else:
            verts   = np.array([(-1, 0, 0), (1, 0, 0), (0, 2*np.cos(np.pi/6), 0)], float)
            indices = (0, 1, 2)
            
            frac = cls(verts, indices, 3, replace=True, gen_f=gen_f_2)
        
        frac.compute(steps=steps)
        
        return frac
    
    # ===========================================================================
    # Pyramids
    
    @classmethod
    def Pyramids(cls, steps=0):
        
        def gen_f(self, indices, depth, t=1):
            
            verts = self.verts[indices]
            
            vs = np.zeros((4, 3), float)
            vs[0] = (verts[0] + verts[1])/2
            vs[1] = (verts[1] + verts[2])/2
            vs[2] = (verts[2] + verts[0])/2
            
            c = np.average(verts, axis=0)
            p = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            
            # Let's a be the length of the side
            # The target height is the height of a pyramid of length a/2, ie
            # t = a.sqrt(2/3)/2 = a.sqrt(1/6)
            # a cross a : h = a^2*sin(60) = a*a*sqrt(3/4)
            # t = h / a * sqrt(4/3) * sqrt(1/6) = h / q / sqrt(4 / 18)
            # t = h / a * sqrt(2) / 3
            
            a = np.linalg.norm(verts[1] - verts[0])
            
            vs[3] = c + p/a*np.sqrt(2)/3
            
            base = len(self.verts)
            return vs, [
                [indices[0], base+0, base+2],
                [indices[1], base+1, base+0],
                [indices[2], base+2, base+1],
                
                [base+3, base+2, base+0],
                [base+3, base+0, base+1],
                [base+3, base+1, base+2]
                ]
        
        a = 2*math.cos(math.pi/6)
        verts = [(-1, -a/3, 0), (1, -a/3, 0), (0, 2*a/3, 0)]
        inds  = (0, 1, 2)
        
        frac = cls(verts, inds, 3, gen_f=gen_f)
        
        frac.compute(steps)
        
        return frac
    
    
    # ===========================================================================
    # One step
    
    def one_step(self, depth, ratio=None, displacement=None, **kwargs):
        
        new_inds = np.empty((0,) + self.shape, int)
        
        # ---------------------------------------------------------------------------
        # If not replace, we can't select structures which have already be used
        
        if self.replace:
            valids = self.indices
        else:
            valids = self.indices[self.start_index:]
            
        # ---------------------------------------------------------------------------
        # If ratiois not None, let's select structues
            
        count  = len(valids)
        inds   = np.arange(count)
        remain = None
        
        if ratio is not None:
            a = np.arange(count)
            np.random.shuffle(a)

            count = min(count, max(1, (round(ratio*count))))

            inds = a[:count]
            remain = np.array(valids[a[count:]])
            
        # ---------------------------------------------------------------------------
        # Loop on the structure to explode
            
        for i, index in enumerate(inds):
            
            struct = valids[index]
            nv, ni = self.gen_f(struct, depth, **kwargs)
            
            if displacement is not None:
                ds = np.linalg.norm(np.max(nv, axis=0) - np.min(nv, axis=0))*displacement
                nv += np.random.normal(0, ds, (6, 3))
            
            self.add_verts(nv)
            new_inds = np.append(new_inds, ni, axis=0)
            
        # ---------------------------------------------------------------------------
        # Add the new structures
            
        if self.replace:
            del self.indices
            if remain is None:
                self.indices = new_inds
            else:
                self.indices = np.append(remain, new_inds, axis=0)
        else:
            self.start_index = len(self.indices)
            self.indices = np.append(self.indices, new_inds, axis=0)
            
    # ===========================================================================
    # Compute in depth
            
    def compute(self, steps, ratio=None, seed=0, **kwargs):

        if seed is not None:
            np.random.seed(seed)
            
        for i in range(steps):
            self.one_step(i, ratio=ratio, **kwargs)  
            
    # ===========================================================================
    # DEBUG / DEMO
            
    def plot(self, closed=True):
        
        try:
            import matplotlib.pyplot as plt
        except:
            return
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        for struct in self.indices:
            vs = self.verts[np.reshape(struct, struct.size)]
            if closed:
                vs = np.append(vs, np.expand_dims(vs[0], axis=0), axis=0)
            plt.plot(vs[:, 0], vs[:, 1], 'k')
            
        plt.show()
            
            
            
            
        
        
        


# ===========================================================================
# Fractal

class Fractal_OLD():
    def __init__(self, verts, indices, shape=None, replace=True):
        self.verts   = np.array(verts)
        
        if shape is None:
            self.shape = np.shape(indices)
        else:
            self.shape = shape if hasattr(shape, '__len__') else (shape,)
            
        count = np.size(indices) // np.product(self.shape)
            
        self.indices = np.array(indices).reshape((count,) + self.shape)
        self.replace = replace
        self.start_index = 0
        
        self.faces_ = []
        
    def __repr__(self):
        return f"<Fractal shape:{self.shape}, verts: {len(self.verts)} indices: {self.indices.shape}>"
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        return self.indices[index]
    
    # ===========================================================================
    # 
    
    def one_step(self, f, depth, ratio=None, displacement=None, **kwargs):
        
        new_inds = np.empty((0,) + self.shape, int)
        
        # ---------------------------------------------------------------------------
        # If not replace, we can't select structures which have already be used
        
        if self.replace:
            valids = self.indices
        else:
            keep   = np.array(self.indices[:self.start_index])
            valids = self.indices[self.start_index:]
            
        # ---------------------------------------------------------------------------
        # If ratiois not None, let's select structues
            
        count  = len(valids)
        inds   = np.arange(count)
        remain = None
        
        if ratio is not None:
            a = np.arange(count)
            np.random.shuffle(a)

            count = min(count, max(1, (round(ratio*count))))

            inds = a[:count]
            remain = np.array(valids[a[count:]])
            
        # ---------------------------------------------------------------------------
        # Loop on the structure to explode
            
        for i, index in enumerate(inds):
            
            struct = valids[index]
            nv, ni = f(self.verts[np.reshape(struct, struct.size)], struct, len(self.verts), depth, **kwargs)
            
            if displacement is not None:
                ds = np.linalg.norm(np.max(nv, axis=0) - np.min(nv, axis=0))*displacement
                nv += np.random.normal(0, ds, (6, 3))
            
            self.verts = np.append(self.verts, nv, axis=0)
            new_inds   = np.append(new_inds, ni, axis=0)
            
        if self.replace:
            del self.indices
            if remain is None:
                self.indices = new_inds
            else:
                self.indices = np.append(remain, new_inds, axis=0)
        else:
            self.indices = np.append(self.indices, new_inds, axis=0)
            
    def compute(self, f, steps, ratio=None, seed=0, **kwargs):

        if seed is not None:
            np.random.seed(seed)
            
        for i in range(steps):
            self.one_step(f, i, ratio=ratio, **kwargs)
            
    # ===========================================================================
    # Add a third dimension to 2D fractals
    
    def to_3D(self):
        if self.verts.shape[-1] == 3:
            return
        
        self.verts = np.insert(self.verts, 2, 0, axis=-1)
        
    # ===========================================================================
    # Faces
    
    def compute_faces(self, f=lambda x: [x]):
        self.faces_ = []
        for struct in self.indices:
            self.faces_.extend(f(struct))
    
    @property
    def faces(self):
        return self.faces_
        
        
    # ===========================================================================
    # edges
    
    @property
    def edges(self):
        faces = self.faces
        n = np.shape(faces)[-1]
        if n <= 2:
            return faces
        
        count = len(faces) * n
        edges = np.zeros((count, 2), int)
        index = 0
        for face in faces:
            for i, iv in enumerate(face):
                edges[index + i] = [iv, face[(i+1)%n]]
            index += n
            
        return edges
            
    # ===========================================================================
    # Sierpenski
    
    @classmethod
    def Sierpinski(cls, steps=4, ratio=None, displacement=None, seed=0):
        triangle = np.array([(-1, 0), (1, 0), (0, 2*np.cos(np.pi/6))], float)
        inds = (0, 1, 2)
                    
        frac = cls(triangle, inds, replace=True)
        frac.compute(sierpinski, steps, ratio=ratio, displacement=displacement, seed=seed)
        
        return frac

    # ===========================================================================
    # Sierpenski
    
    @classmethod
    def Koch(cls, flake=False, steps=4, ratio=None, displacement=None, seed=0):
        
        if flake:
            verts = np.array([(-1, 0), (1, 0), (0, 2*np.cos(np.pi/6))], float)
            inds  = ((1, 0), (0, 2), (2, 1))
        else:
            verts = np.array(((-1, 0), (1, 0)), float)
            inds = (0, 1)
        
        frac = cls(verts, inds, shape=2, replace=True)
        
        frac.compute(koch, steps, ratio=ratio, displacement=displacement, seed=seed)
        
        return frac
            
            
    # ===========================================================================
    # Sierpenski pyramid
    
    @classmethod
    def SierpenskiPyramid(cls, steps=4, ratio=None, displacement=None, seed=0):
        
        a = 1
        b = a/2/np.sqrt(6)
        
        v0 = ( 0, 0, a*np.sqrt(2/3))
        
        v1 = (-a/2, -b, 0)
        v2 = ( a/2, -b, 0)
        v3 = ( 0, -b + a*np.cos(np.pi/6), 0)
        
        pyramid = np.array([v0, v1, v2, v3], float)
        inds = (0, 1, 2, 3)
                    
        frac = cls(pyramid, inds, replace=True)
        frac.compute(sierpenski_pyramid, steps, ratio=ratio, displacement=displacement, seed=seed)
        
        def faces(inds):
            return [
                [inds[0], inds[1], inds[2]],
                [inds[0], inds[2], inds[3]],
                [inds[0], inds[3], inds[1]],
                [inds[3], inds[2], inds[1]],
                ]
        
        frac.compute_faces(faces)
        
        return frac
            
            
    # ===========================================================================
    # DEBUG / DEMO
            
    def plot(self, closed=True):
        
        try:
            import matplotlib.pyplot as plt
        except:
            return
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        for struct in self.indices:
            vs = self.verts[np.reshape(struct, struct.size)]
            if closed:
                vs = np.append(vs, np.expand_dims(vs[0], axis=0), axis=0)
            plt.plot(vs[:, 0], vs[:, 1], 'k')
            
        plt.show()
        

        
frac = Fractal.Sierpenski(4, ndim=2)
frac.plot(closed=True)
#frac = Fractal.Koch(flake=False)
#frac.plot(closed=False)
#frac = Fractal.Koch(flake=True)
#frac.plot(closed=False)

#frac = Fractal.SierpenskiPyramid(steps=2)
#frac.plot()






    