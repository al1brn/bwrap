#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:12:36 2021

@author: alain

"""

import numpy as np

def smoothstep(t):
    return t * t * (3. - 2. * t)

def lerp(t, a, b):
    return a + t * (b - a)

# ===========================================================================
# Perlin class

class Perlin():
    
    CONTRAST = [2.60, 1.50, 1.55, 1.70]
    
    def __init__(self, ndim, amp=1000, scale=1., contrast=1., seed=None):
        
        if seed is not None:
            np.random.seed(seed)
        
        self.ndim     = np.clip(ndim, 1, 4)
        self.amp      = np.clip(amp, 1, 1000000)
        self.contrast = contrast
        self.scale    = scale
        self.raw_     = False
        
        # ---------------------------------------------------------------------------
        # Generate random vectors
        
        if self.ndim == 1:
            self.vects = np.random.uniform(-1, 1, (self.amp, 1))
        
        elif self.ndim == 2:
            ratio = 4/np.pi
            
        elif self.ndim == 3:
            ratio = 6/np.pi
            
        else:
            ratio = 32/np.pi/np.pi
            
        if self.ndim > 1:
            
            ratio *= 1.2

            self.vects = np.empty((0, self.ndim), float)
            while len(self.vects) != self.amp:
                
                n = self.amp - len(self.vects)
                
                vs = np.random.uniform(-1, 1, (int(n*ratio), self.ndim))
                ns = np.linalg.norm(vs, axis=-1)
                sel = ns <= 1
                
                vs = (vs[sel] / np.expand_dims(ns[sel], axis=-1))[:n]
                self.vects = np.append(self.vects, vs, axis=0)
                
    # ===========================================================================
    # The grid vectors
    
    def grid_vectors(self, boxes):
        boxes_values = [1093, 1097, 1103, 1109][:self.ndim]
        coord_values = [7, 11, 13, 17][:self.ndim]
        return self.vects[
                    np.sum((79000177*np.sin(boxes*boxes_values)).astype(int)*coord_values, axis=-1) % self.amp      
                ]
    
    # ===========================================================================
    # Apply the contrats
    
    def apply_contrast(self, noise):
        cst = Perlin.CONTRAST[self.ndim-1] / max(0.00001, self.contrast)
        return np.clip((noise + cst) / (2*cst), 0, 1)
    
    # ===========================================================================
    # Compute the noise within a box
    # The box, and its assocated vectors, is computed by the caller
    # - points  : coordinates relative to the first corner
    # - vectors : vectors at the corners
    # NOT TESTED
        
    def box_noise_(self, points, vectors):
        
        # ---------------------------------------------------------------------------
        # The number of corners
        
        shape = np.shape(points)[:-1]
        
        # ---------------------------------------------------------------------------
        # Compute the dots
        
        dots = np.dot(points, vectors)
        
        # ---------------------------------------------------------------------------
        # Smooth step

        pts = smoothstep(points)
        
        # ---------------------------------------------------------------------------
        # Linear extrapolation

        if self.ndim >= 4:
            dots = dots[..., 0] + np.reshape(pts[..., 3], shape + (1, 1, 1))*(dots[..., 1] - dots[..., 0])
        
        if self.ndim >= 3:
            dots = dots[..., 0] + np.reshape(pts[..., 2], shape + (1, 1))*(dots[..., 1] - dots[..., 0])
        
        if self.ndim >= 2:
            dots = dots[..., 0] + np.reshape(pts[..., 1], shape + (1,))*(dots[..., 1] - dots[..., 0])
            
            
        dots = dots[..., 0] + pts[..., 0]*(dots[..., 1] - dots[..., 0])
        
        return dots
        

    # ===========================================================================
    # Compute the noise on any array of points
        
    def noise(self, points):
        
        # ---------------------------------------------------------------------------
        # Compute the shape of points
        
        shape = np.shape(points)
        
        if self.ndim == 1:
            if shape == ():
                return self.noise([points])[0]
        else:
            if len(shape) == 1:
                return self.noise([points])[0]
            shape = shape[:-1]
            
        # Work with a linear array of vectors of dimenion ndim
            
        count = np.product(shape)
        pts = np.reshape(points, (count, self.ndim))*self.scale
        
        # ---------------------------------------------------------------------------
        # The corners   
        # The points are places in cells, or boxes. The boxes have nc corners.
        # Each corner has ndim coordinates. These coordinates are used both:
        # - to get the random vector associated to it
        # - to compute the dot product with the points
        #
        # shape : (count, nc, ndim)
        # 1 : (count,  1, 1)
        # 2 : (count,  4, 2)
        # 3 : (count,  8, 3)
        # 4 : (count, 16, 4)

        nc = 2**self.ndim
        boxes   = np.zeros((count, nc, self.ndim), int)
        boxes[:] = np.floor(pts).astype(int).reshape(count, 1, self.ndim)
        
        if self.ndim == 1:
            boxes[:, 1, 0] += 1
            
        elif self.ndim == 2:
            boxes[:, [1, 3], 0] += 1
            boxes[:, [2, 3], 1] += 1
            
        elif self.ndim == 3:
            boxes[:, [1, 3, 5, 7], 0] += 1
            boxes[:, [2, 3, 6, 7], 1] += 1
            boxes[:, [4, 5, 6, 7], 2] += 1
            
        else:
            boxes[:, [1, 3,  5,  7,  9, 11, 13, 15], 0] += 1
            boxes[:, [2, 3,  6,  7, 10, 11, 14, 15], 1] += 1
            boxes[:, [4, 5,  6,  7, 12, 13, 14, 15], 2] += 1
            boxes[:, [8, 9, 10, 11, 12, 13, 14, 15], 3] += 1
            
        # ---------------------------------------------------------------------------
        # Compute the index on to the vectors array
        # The coordinates tuples are used to compute an index with the (0, amp)
        # This allows to get a big number of vectors without looking a dictionary
        #
        # The vectors shape is (count, nc, ndim) as for boxes
        
        if True:
            vectors = self.grid_vectors(boxes)
        else:
        
            indices = np.sum((79000177*np.sin(boxes*[1093, 1097, 1103, 1109][:self.ndim])).astype(int)*[7, 11, 13, 17][:self.ndim], axis=-1) % self.amp
            
            vectors = self.vects[indices]
            del indices
            
            
        
        # ---------------------------------------------------------------------------
        # The dots are computed using numpy function.
        # We have one scalar per corner
        # dots shape: (count, nc)
        # 1 : (count,  1)
        # 2 : (count,  4)
        # 3 : (count,  8)
        # 4 : (count, 16)
        
        dots = np.einsum('...i,...i', np.reshape(pts, (count, 1, self.ndim)) - boxes, vectors)
        
        # ---------------------------------------------------------------------------
        # Linear extrapolation along the dimensions
        # The dots shape is divided by two at each loop. 
        # Starting by the highest dimension allows to have simpler indices

        lerp_i = [
                [[0], [1]],
                [[0, 1], [2, 3]],
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
            ]
        
        # ------------------------------------------------------------------------------------------
        # We need to coordinates relative to the first corner of the box
        # Since it will be used to multiply vectors, let's reshape it once
        
        pts = np.reshape(smoothstep(pts-boxes[:, 0]), (count, 1, self.ndim))
        for d in reversed(range(self.ndim)):
            dots = dots[:, lerp_i[d][0]] + pts[..., d]*(dots[:, lerp_i[d][1]] - dots[:, lerp_i[d][0]])

        # ------------------------------------------------------------------------------------------
        # Return with contrast
        
        if self.raw_:
            return np.reshape(dots, shape)
        else:
            return np.reshape(self.apply_contrast(dots), shape)
    
    # ===========================================================================
    # Compute the contrats
    
    def get_amplitude(self, count=1000000):

        self.raw_ = True
        pts = np.random.uniform(-10.87, 10.87, (count, self.ndim))
        noise = self.noise(pts)
        self.raw_ = False
        
        vmin = np.min(noise)
        vmax = np.max(noise)
        vavg = np.average(noise)
        ratio = 1/max(1/np.min(noise), np.max(noise))
        
        print(f"Dimension {self.ndim}: min={vmin:.3f}, max={vmax:.3f}, avg={vavg:.3f}, ratio={ratio:.2f}")
        
        return vmin, vmax, vavg, ratio
    
    # ===========================================================================
    # Grid coordinates
    
    def grid(self, P0=0, P1=0, shape=(100, 100, 10, 10)):
        
        if self.ndim == 1:
            return np.linspace(P0, P1, shape[0] if hasattr(shape, '__len__') else shape)
        
        shape = shape[:self.ndim]
        
        if np.product(shape) > 100000000:
            if self.ndim == 2:
                shape = (10000, 10000)
            elif self.ndim == 3:
                shape = (1000, 1000, 100)
            else:
                shape = (1000, 1000, 10, 10)
                
        grid = np.empty(shape + (self.ndim,), float)
        
        v0 = np.zeros(self.ndim)
        v1 = np.zeros(self.ndim)
        v0[:] = P0
        v1[:] = P1
                
        xres = shape[0]
        yres = shape[1]
        if self.ndim > 2: zres = shape[2]
        if self.ndim > 3: wres = shape[3]
        
        if self.ndim == 2:
            grid[np.arange(xres), :, 0] = np.reshape(np.linspace(v0[0], v1[0], xres), (xres, 1))
            grid[np.arange(xres), :, 1] = np.linspace(v0[1], v1[1], yres)
            
        elif self.ndim == 3:
            grid[np.arange(xres), ..., 0] = np.reshape(np.linspace(v0[0], v1[0], xres), (xres, 1, 1))
            grid[np.arange(xres), ..., 1] = np.reshape(np.linspace(v0[1], v1[1], yres), (1, yres, 1))
            grid[np.arange(xres), ..., 2] = np.reshape(np.linspace(v0[2], v1[2], zres), (1, 1, zres))
            
        else:
            grid[np.arange(xres), ..., 0] = np.reshape(np.linspace(v0[0], v1[0], xres), (xres, 1, 1, 1))
            grid[np.arange(xres), ..., 1] = np.reshape(np.linspace(v0[1], v1[1], yres), (1, yres, 1, 1))
            grid[np.arange(xres), ..., 2] = np.reshape(np.linspace(v0[2], v1[2], zres), (1, 1, zres, 1))
            grid[np.arange(xres), ..., 3] = np.reshape(np.linspace(v0[3], v1[3], wres), (1, 1, 1, wres))
        
        return grid
    
    # ===========================================================================
    # Plot
    
    def noise_map(self, P0=0, P1=0, shape=(100, 100, 10, 10)):
        return self.noise(self.grid(P0, P1, shape))
    
    
    # ===========================================================================
    # Plot
    
    def plot(self, noise, z=0, w=0, title=None):
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        n = len(noise)
        img = np.zeros((n, n, 3), float)
        
        if self.ndim == 1:
            img[:] = np.reshape(noise, (n, 1))
            
        elif self.ndim == 2:
            img[:] = np.reshape(noise, (n, n, 1))
            
        elif self.ndim == 3:
            for i in range(3):
                img[..., i] = noise[..., min(z+i, np.shape(noise)[2]-1)]
            
        elif self.ndim == 4:
            for i in range(3):
                img[..., i] = noise[..., z, min(w+i, np.shape(noise)[3]-1)]


        ax.imshow(img)
        if title is None: title = f"dim {self.ndim} shape {np.shape(noise)}"
        plt.title(title)
        plt.show()  
        
    # ===========================================================================
    # Demo
    
    def demo(self, shape=None, scale=1):
        import time
        
        if shape is None:
            if self.ndim == 1:
                shape = (1024,)
            elif self.ndim == 2:
                shape = (1024, 1024)
            elif self.ndim == 3:
                shape = (1024, 1024, 3)
            else:
                shape = (512, 512, 3, 3)
                
        self.scale = scale
                
        t0 = time.time()
        noise = self.noise_map(0, 1, shape)
        t = time.time() - t0
        self.plot(noise, title=f"dim: {self.ndim}, shape: {shape}, time: {t:.2f}")
        
    @staticmethod
    def Demos(scale=6):
        Perlin(1).demo(scale=scale)
        Perlin(2).demo(scale=scale)
        Perlin(3).demo(scale=scale)
        Perlin(4).demo(scale=scale)


Perlin.Demos()            


        


