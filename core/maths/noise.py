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
    # Noisy curve
    
    @classmethod
    def Curve(self, amp=1000, scale=1., contrast=1., seed=None):
        perlin = Perlin(1, amp=amp, scale=scale, contrast=contrast, seed=seed)
        return lambda x: perlin.noise(x)
                
    # ===========================================================================
    # Noisy 2D map
    
    @classmethod
    def Map2D(self, amp=1000, scale=1., contrast=1., seed=None):

        perlin = Perlin(2, amp=amp, scale=scale, contrast=contrast, seed=seed)

        def map2d(x0=0., y0=0., x1=10., y1=10., shape=(100, 100)):
            return perlin.noise_map(P0=(x0, y0), P1=(x1, y1), shape=shape)

        return map2d
    
    # ===========================================================================
    # Noisy 3D map
    
    @classmethod
    def Map3D(self, amp=1000, scale=1., contrast=1., seed=None):

        perlin = Perlin(3, amp=amp, scale=scale, contrast=contrast, seed=seed)

        def map3d(x0=0., y0=0., x1=10., y1=10., z=0., shape=(100, 100)):
            return perlin.noise_map(P0=(x0, y0, z), P1=(x1, y1, z), shape=shape + (1,)).reshape(shape)

        return map3d
    
    # ===========================================================================
    # Noisy 4D map
    
    @classmethod
    def Map4D(self, amp=1000, scale=1., contrast=1., seed=None):

        perlin = Perlin(4, amp=amp, scale=scale, contrast=contrast, seed=seed)

        def map3d(x0=0., y0=0., x1=10., y1=10., z=0., w=0., shape=(100, 100)):
            return perlin.noise_map(P0=(x0, y0, z, w), P1=(x1, y1, z, w), shape=shape + (1, 1)).reshape(shape)

        return map3d
    
                
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
        
        vectors = self.grid_vectors(boxes)
        
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
            
            ax.plot(np.linspace(0, n, n), img[0, :, 0])
            
        elif self.ndim == 2:
            img[:] = np.reshape(noise, (n, n, 1))
            
        elif self.ndim == 3:
            for i in range(3):
                img[..., i] = noise[..., min(z+i, np.shape(noise)[2]-1)]
            
        elif self.ndim == 4:
            for i in range(3):
                img[..., i] = noise[..., z, min(w+i, np.shape(noise)[3]-1)]


        if self.ndim > 1:
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
    def Demos(scale=1, contrast=1):
        
        import matplotlib.pyplot as plt
        
        # ----- Curve 1D
        
        x = np.linspace(0, 10, 2000)
        noise = Perlin.Curve(scale=scale, contrast=contrast)
        y = noise(x)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.title("Curve")
        plt.show()
        
        # ----- Map 2D
        
        map2 = Perlin.Map2D(scale=scale, contrast=contrast)
        uv = map2()
        
        fig, ax = plt.subplots()
        ax.imshow(uv)
        plt.title("Map 2D")
        plt.show()
        
        # ----- Map 3D
        
        map3 = Perlin.Map3D(scale=scale, contrast=contrast)
        for z in range(10):
            fig, ax = plt.subplots()
            ax.imshow(map3(z=z/10))
            plt.title(f"Map 3D - Image {z}")
            plt.show()
        
        # ----- Map 4D
        
        map4 = Perlin.Map4D(scale=scale, contrast=contrast)
        for z in range(10):
            fig, ax = plt.subplots()
            ax.imshow(map4(z=z/10))
            plt.title(f"Map 4D - Image {z} z")
            plt.show()
        
        for z in range(10):
            fig, ax = plt.subplots()
            ax.imshow(map4(w=z/10))
            plt.title(f"Map 4D - Image {z} w")
            plt.show()
            
# ===========================================================================
# Voronoi class

class Voronoi():
    def __init__(self, scale=1., seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            
        self.scale = scale
        count = 10

        self.xs = np.random.uniform(0., 1., count)
        self.ys = np.random.uniform(0., 1., count)
        
        self.segments =[]
        
        # ---------------------------------------------------------------------------
        # Parabola between a focal and a leading line
        # The line is vertical at x = l
        # The distance to a point left to the line is : (l-x)
        # The point is at coordinate (x0, y0)
        # Distance to the point is sqrt((x-x0)^2 + (y-y0)^2)
        # A point M is equidistant when:
        # (x-x0)^2 + (y-y0)^2 = (l-x)^2
        # y^2 - 2(x0 - l)x - 2y0y + x0^2 + y0^2 - l^2 = 0 
        # x = (y^2 - 2y0y + K)/2/(x0-l)
        # with K = x0^2 + y0^2 - l^2)
        
        # ---------------------------------------------------------------------------
        # Parabola coefficients
 
        def p_abc(pt, ln):
            a = 1/(pt[0] - ln)
            b = -pt[1]*a
            a /= 2
            c = (pt[0]**2 + pt[1]**2 - ln**2)*a
            return (a, b, c)
        
        # ---------------------------------------------------------------------------
        # parabola value

        def parabola(abc, x):
            return x**2*abc[0] + x*abc[1] + abc[2]
        
        # ---------------------------------------------------------------------------
        # Intersection of two parabolas

        def intersect(abc0, abc1):
            
            k = 2/(abc1[1] - abc0[1]) # 1/b'

            a = (abc1[0] - abc0[0])*k
            c = (abc1[2] - abc0[2])*k
            
            D = 1 - a*c
            d = np.sqrt(D)
            
            y0 = (-1 - d)/a
            y1 = (-1 + d)/a
            
            return (parabola(abc0, y0), y0), (parabola(abc0, y1), y1)

        
        pt0 = (-1., 0)
        pt1 = (-3., -2)
        ln = 0.
        
        y = np.linspace(-5, 5, 100)
        
        abc0 = p_abc(pt0, ln)
        abc1 = p_abc(pt1, ln)
        abc0 = p_abc(pt0, ln)
        x0 = parabola(abc0, y)
        x1 = parabola(abc1, y)
        
        ip0, ip1 =  intersect(abc0, abc1)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        ax.plot([ln, ln], [-5, 5])
        ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'o')
        ax.plot(x0, y)
        ax.plot(x1, y)
        ax.plot([ip0[0], ip1[0]], [ip0[1], ip1[1]], 'o')
        
        plt.show()
        
    def distance(self, points):
        
        shape = np.shape(points)
        pts   = np.array(points, float)*self.scale
        
        count = len(self.xs)


Voronoi()    
            


#Perlin.Demos()

