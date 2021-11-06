#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 08:16:31 2021

@author: alain
"""

import numpy as np

if True:
    
    from ..maths.closed_faces import closed_faces
    from .textformat import CharFormat
    
    
else:
    
    path = "/Users/alain/Documents/blender/scripts/modules/bwrap/"
    import importlib
    
    closed_spec = importlib.util.spec_from_file_location("breader", path + "maths/closed_faces.py")
    closed  = importlib.util.module_from_spec(closed_spec)
    closed_spec.loader.exec_module(closed)
    
    closed_faces = closed.closed_faces


# ====================================================================================================
# Utility class used when computing contours to identify internal and external contours

class BBox():
    def __init__(self, contour=None, index=-1):
        
        self.top      = contour is None
        self.index    = index
        self.children = []
        self.owner    = None

        if not self.top:
            self.x0 = np.min(contour[:, 0, 0])
            self.x1 = np.max(contour[:, 0, 0])
            self.y0 = np.min(contour[:, 0, 1])
            self.y1 = np.max(contour[:, 0, 1])
            
    @property
    def depth(self):
        if self.owner is None:
            return 0
        else:
            return 1 + self.owner.depth
        
    def contains(self, other):
        if self.top:
            return True
        return self.x0 <= other.x0 and self.x1 >= other.x1 and self.y0 <= other.y0 and self.y1 >= other.y1
        
    def add(self, bbox):
        for c in self.children:
            b = c.add(bbox)
            if b is None:
                return None
            
        if self.contains(bbox):
            self.children.append(bbox)
            bbox.owner = self
            return None
        
        return bbox
    
    def set_int_ext(self, a):
        if not self.top:
            a[self.index] = self.depth % 2
        for c in self.children:
            c.set_int_ext(a)
            
            
def direction_code(direction):
    
    if direction[0] == 1:
        
        if direction[1] == 0:    # -> - 
            return 0
        elif direction[1] == 1:  # -> ^
            return 1
        else:                    # -> v
            return 7
        
    elif direction[0] == -1:
        
        if direction[1] == 0:    # <- - 
            return 4
        elif direction[1] == 1:  # <- ^
            return 3
        else:                    # <- v
            return 5
        
    else:
        
        if direction[1] == 1:    # - ^ 
            return 2
        else:                    # - v
            return 6
        
        
def vertical_shift(d0, d1):
    
    # up = 1
    # dn = -1
    # no = 0
    
    #         ->  /   |   \   <-  /   |   \
    shifts = [
            [ 1,  1,  1,  1,  0,  1,  1,  1],  # -> 
            [ 1,  1,  1,  0, -1,  1,  1,  1],  # /
            [ 1,  1,  0, -1, -1, -1,  0,  1],  # |
            [ 1,  0, -1, -1, -1, -1, -1, -1],  # \
            [ 0, -1, -1, -1, -1, -1, -1, -1],  # <-
            [ 1,  1, -1,  0, -1, -1, -1,  0],  # /
            [ 1,  1,  0, -1, -1, -1,  0, -1],  # |
            [ 1,  1,  1,  0, -1,  0,  1,  1],  # \
        ]
    
    c0 = direction_code(d0)
    c1 = direction_code(d1)
    
    return shifts[c0][c1]
            
# ====================================================================================================
# The contours are split in a series of left -> right and right -> left lines
# This will allow to compute bold by right and vertical shifts

class ZigZag():
    
    def __init__(self, start, end, n, direction):
        
        self.start     = start
        self.end       = end
        self.n         = n
        self.direction = direction
        
        self.fix_start = False
        self.fix_end   = False
        
        # ---------------------------------------------------------------------------
        # Horizontal line: 0 - > right
        
        if self.direction[1] == 0:
            
            # Forwards
            if self.direction[0] == 1:
                self.h_shift0  = 0
                self.h_shift1  = 1
                
                #self.fix_start = True
                #self.fix_end   = True
                
            # Backwards
            else:
                self.h_shift0  = 1
                self.h_shift1  = 0
                
                #self.fix_start = True
                #self.fix_end   = True
            
        # ---------------------------------------------------------------------------
        # Growing upwards : a left line
        
        elif self.direction[1] == 1:
            
            # Vertical line
            if self.direction[0] == 0:
                self.h_shift0 = 0
                self.h_shift1 = 0
                
                self.fix_start = True
                self.fix_end   = True
                
                
            # Forwards
            elif self.direction[0] == 1:
                self.h_shift0 = 0
                self.h_shift1 = 0
                
            # Backwards
            else:
                self.h_shift0 = 0
                self.h_shift1 = 0

        # ---------------------------------------------------------------------------
        # Growing downwards : a right line
        
        else:
            
            # Vertical line
            if self.direction[0] == 0:
                self.h_shift0 = 1
                self.h_shift1 = 1
                
                self.fix_start = True
                self.fix_end   = True
                
            # Forwards
            elif self.direction[0] == 1:
                self.h_shift0 = 1
                self.h_shift1 = 1
                
            # Backwards
            else:
                self.h_shift0 = 1
                self.h_shift1 = 1        
        
        
    def __repr__(self):
        sl = f" {self.direction[0]}" if self.direction[0] >= 0 else "-1"
        sr = f" {self.direction[1]}" if self.direction[1] >= 0 else "-1"
        sfs = "*" if self.fix_start else " "
        sfe = "*" if self.fix_end else " "
        
        return f"<ZigZag: {self.start%self.n:2d}[{self.h_shift0}{sfe}] --> {self.end%self.n:2d}[{self.h_shift1}{sfs}] dir: ({sl}, {sr}),  ({len(self):3d}): {self.indices} v0:{self.v0} v1:{self.v1}>"
        
    def __len__(self):
        if self.end < self.start:
            return self.n - self.start + self.end + 1
        else:
            return self.end - self.start + 1
        
    @property
    def indices(self):
        if self.end < self.start:
            return [i % self.n for i in range(self.start, self.end + self.n + 1)]
        else:
            return [i % self.n for i in range(self.start, self.end + 1)]
        
# ====================================================================================================
# The zig-zags of a contour        
        
class ZigZags():
    
    def __init__(self, contour, hrz = True):
        
        self.contour = contour
        self.n       = contour.shape[0]
        self.hrz     = hrz
        self.zigzags = []
        self.current = None
        
        self.build()

        
    def __repr__(self):
        s = f"<ZigZags({len(self)}) {'horizontal' if self.hrz else 'vertical'}:\n"
        for i, zz in enumerate(self):
            s += f"   {i:2d}: {zz}\n"
        return s
        
    def __len__(self):
        return len(self.zigzags)
    
    def __getitem__(self, index):
        return self.zigzags[index]
    
    def getzz(self, index):
        return self.zigzags[index % len(self.zigzags)]
    
    @staticmethod
    def direction(v):
        return (0 if v[0] == 0 else 1 if v[0]>0 else -1, 0 if v[1] == 0 else 1 if v[1]>0 else -1)
    
    def add(self, start, end, direction):
        zz = ZigZag(start, end, self.n, ZigZags.direction(direction))
        zz.v0 = self.point(start)
        zz.v1 = self.point(end)
        self.zigzags.append(zz)
        return zz
    
    # ----------------------------------------------------------------------------------------------------
    # Successive zigzags belonging to the same hrz line
    
    def hrz_line_of(self, index):
        
        index = index % len(self.zigzags)
        
        zz = self.getzz(index)
        d = zz.direction[0]
        
        # Vertical line
        if d == 0:
            return [index], 0
        
        # First 
        i_start = index
        for i in range(1, len(self.zigzags)):
            pz = self.getzz(index-i)
            if pz.direction[0] * d <= 0:
                i_start = index - i + 1
                break
            
        hrz = []
        for i in range(i_start, i_start + len(self.zigzags)):
            if self.getzz(i).direction[0] * d == 1:
                hrz.append(i % len(self.zigzags))
            else:
                break
            
        return hrz, d
    
    # ----------------------------------------------------------------------------------------------------
    # All the horizontal lines
    
    def hrz_lines(self):
        
        hrz, d = self.hrz_line_of(0)
        hrzs = [hrz]
        ds   = [d]
        
        while True:
            index = hrz[-1] + 1
            
            hrz, d = self.hrz_line_of(index)
            if hrz == hrzs[0]:
                break
            
            hrzs.append(hrz)
            ds.append(d)
            
        return hrzs, ds    
        
    # ----------------------------------------------------------------------------------------------------
    # Build the zigzags
        
    def build(self):
        
        # ---------------------------------------------------------------------------
        # Compute the delta x of segments
        # The entry i give the delta between points i and i+1
        
        dxs      = np.empty(self.n, int)
        dys      = np.empty(self.n, int)
        dxs[:-1] = self.contour[1:, 0, 0] - self.contour[:-1, 0, 0] 
        dys[:-1] = self.contour[1:, 0, 1] - self.contour[:-1, 0, 1] 
        dxs[-1]  = self.contour[0,  0, 0] - self.contour[-1, 0, 0]
        dys[-1]  = self.contour[0,  0, 1] - self.contour[-1, 0, 1]
        
        # The shifts
        
        self.shifts = np.zeros(self.contour.shape, float)
        
        # We start from a very left point
        
        i_left = np.argmin(self.contour[:, 0, 0])
        
        # ---------------------------------------------------------------------------
        # Zigzags can be:
        # - Horizontal line
        # - Vertical line
        # - A line with points progressing one of the 4 possible directions:
        #      . ( 1, 1) : right up
        #      . ( 1,-1) : right down
        #      . (-1, 1) : left up
        #      . (-1,-1) : left down
        
        ii = i_left
        for wd in range(self.n+1):
            
            i = ii % self.n
            
            # The delta of the point segment
            
            dx1 = dxs[i]
            dy1 = dys[i]
            
            # ----- Start of a vertical segment
            if dx1 == 0:
                self.add(ii, ii+1, (0, dy1))
                ii += 1
                
            # ----- Start of a horizontal segment
            
            elif dy1 == 0:
                self.add(ii, ii+1, (dx1, 0))
                ii += 1
                
            # ----- Continue while the direction is kept
            
            else:
                drc = ZigZags.direction((dx1, dy1))
                inds = []
                start = ii
                for jj in range(ii, ii+self.n):
                    j = jj % self.n
                    inds.append(j)

                    j_d = ZigZags.direction((dxs[j], dys[j]))
                    if drc[0]*j_d[0] <= 0 or drc[1]*j_d[1] <= 0:
                        ii = jj
                        break
                    
                self.add(start, ii, (dx1, dy1))
                    
            # Normally == should work but you never know !
                    
            if ii >= i_left + self.n:
                break
                
        # ---------------------------------------------------------------------------
        # We must ensure the end and start shift match
        
        for iz, zz in enumerate(self.zigzags):
            
            nz = self.zigzags[(iz+1)%len(self.zigzags)]
            if nz.h_shift0 != zz.h_shift1:
                if zz.fix_end:
                    nz.h_shift0 = zz.h_shift1
                elif nz.fix_start:
                    zz.h_shift1 = nz.h_shift0
                elif zz.direction[1] == 0:
                    zz.h_shift1 = nz.h_shift0
                else:
                    nz.h_shift0 = zz.h_shift1
                    
        # ---------------------------------------------------------------------------
        # Let's build the shifts
        
        for iz, zz in enumerate(self.zigzags):
                
            inds = zz.indices
            x0 = self.x(zz.start)
            ax = self.x(zz.end) - x0
            
            if ax == 0:
                self.shifts[inds, 0, 0]      = zz.h_shift0
                self.shifts[inds[:-1], 1, 0] = zz.h_shift0
                
            else:
                r = (zz.h_shift1 - zz.h_shift0)/ax
                for i, index in enumerate(inds):
                    x = self.x(index)
                    self.shifts[index, 0, 0] = zz.h_shift0 + r*(x - x0)

                    if i < len(inds)-1:
                        x = self.bx(index)
                        self.shifts[index, 1, 0] = zz.h_shift0 + r*(x - x0)
                        
        # ---------------------------------------------------------------------------
        # The vertical shifts
        
        ymin = np.min(self.contour[:, 0, 1])
        ymax = np.max(self.contour[:, 0, 1])
        
        p = .1
        self.v_shift_min = ((1-p)*self.contour[..., 1] + p*ymin).astype(int)
        self.v_shift_max = ((1-p)*self.contour[..., 1] + p*ymax).astype(int)
        
        for i_pt in range(len(self.contour)):
            pt = self.contour[i_pt, 0]
            b0 = self.contour[(i_pt-1)%self.n, 1]
            b1 = self.contour[i_pt, 1]
            
            d0 = pt - b0
            d1 = b1 - pt
            
            dir0 = (1 if d0[0]>0 else 0 if d0[0]==0 else -1, 1 if d0[1]>0 else 0 if d0[1]==0 else -1)
            dir1 = (1 if d1[0]>0 else 0 if d1[0]==0 else -1, 1 if d1[1]>0 else 0 if d1[1]==0 else -1)
            
            vs = vertical_shift(dir0, dir1)*.5
            self.shifts[i_pt, 0, 1] = vs
            if i_pt > 0:
                self.shifts[i_pt-1, 1, 1] = (self.shifts[i_pt-1, 0, 1] + vs)/2
            elif i_pt == len(self.contour)-1:
                self.shifts[i_pt, 1, 1] = (vs + self.shifts[0, 0, 1])/2
                

    # ----------------------------------------------------------------------------------------------------
    # A point
    
    def point(self, index):
        return self.contour[index%self.n, 0]
            
    def x(self, index):
        return self.contour[index%self.n, 0, 0]
            
    def y(self, index):
        return self.contour[index%self.n, 0, 1]
            
    def bx(self, index):
        return self.contour[index%self.n, 1, 0]
            
    def by(self, index):
        return self.contour[index%self.n, 1, 1]



# ====================================================================================================
# A glyphe builf from a a ttf font

class Glyphe():
    
    def __init__(self, ttf, glyf=None):
        
        self.ttf        = ttf
        self.glyf_index = None if glyf is None else glyf.glyf_index
        self.code       = 0 if glyf is None else glyf.code
        
        self.on_curve   = []    # On curve points
        self.points     = None  # Point to be built later
        self.ends       = []    # Ends as read in the ttf file
        self.xMin_      = 0     # bounding box
        self.yMin_      = 0
        self.xMax_      = 0
        self.yMax_      = 0
        
        self.contours   = None  # An array (n, 2, 2) of on and off curve points
        self.ext_int    = None  # 1 if contour is ext and 0 if contour is int  
        
        if glyf is not None:
            self.add(glyf)
            
    # ----------------------------------------------------------------------------------------------------
    # Some samples to debug
            
    @classmethod
    def Char(cls, c):
        
        class Glyf():
            pass
        
        glyf = Glyf()
        
        if c == "S":
            glyf.glyf_index = 54
            glyf.flags = [19, 55, 30, 30, 30, 51, 50, 54, 54, 53, 52, 38, 39, 38, 36, 39, 38, 38, 53, 52, 54, 54, 51, 50, 22, 22, 23, 7, 38, 38, 35, 34, 6, 21, 20, 23, 22, 4, 23, 22, 22, 21, 20, 6, 6, 35, 34, 36, 38]
            glyf.xCoordinates = [92, 183, 13, 95, 200, 125, 111, 170, 83, 0, 0, -80, -92, -59, -404, -81, -105, -103, 0, 0, 126, 242, 148, 163, 249, 134, 5, -186, -15, -173, -169, -176, -161, 0, 0, 57, 56, 473, 88, 128, 122, 0, 0, -134, -251, -157, -199, -269, -153]
            glyf.yCoordinates = [471, 16, -110, -141, -87, 0, 0, 66, 115, 68, 69, 103, 35, 23, 97, 43, 55, 163, 101, 111, 193, 100, 0, 0, -105, -204, -129, -14, 139, 142, 0, 0, -129, -91, -79, -51, -51, -107, -40, -59, -181, -118, -117, -207, -115, 0, 0, 116, 233]
            glyf.endPtsOfContours = [48]   
            
        elif c == "6":
            glyf.glyf_index = 25
            glyf.flags = [1, 7, 38, 39, 38, 35, 34, 7, 6, 6, 7, 54, 54, 51, 50, 18, 21, 20, 6, 6, 35, 34, 0, 17, 16, 55, 54, 51, 50, 22, 1, 20, 22, 22, 51, 50, 54, 53, 52, 38, 35, 34, 6]
            glyf.xCoordinates = [1019, -179, -24, -44, -73, -107, -86, -65, -85, -98, -2, 65, 188, 103, 180, 253, 0, 0, -119, -208, -132, -225, -284, 0, 0, 157, 137, 232, 173, 221, -713, 0, 79, 142, 78, 114, 164, 0, 0, -162, -123, -122, -170]
            glyf.yCoordinates = [1107, -14, 106, 48, 77, 0, 0, -48, -62, -238, -220, 99, 96, 0, 0, -265, -210, -138, -237, -126, 0, 0, 331, 380, 425, 193, 168, 0, 0, -194, -803, -93, -170, -89, 0, 0, 184, 158, 152, 175, 0, 0, -175]
            glyf.endPtsOfContours = [29, 42]    
            
        elif c == "m":
            glyf.glyf_index = 80
            glyf.flags = [51, 17, 51, 21, 54, 54, 51, 50, 22, 23, 54, 51, 50, 22, 21, 17, 35, 17, 52, 38, 38, 35, 34, 6, 21, 17, 35, 17, 52, 38, 35, 34, 6, 6, 21, 17]
            glyf.xCoordinates = [135, 0, 161, 0, 50, 166, 106, 118, 151, 31, 126, 202, 158, 170, 0, 0, -179, 0, 0, -35, -92, -62, -112, -148, 0, 0, -180, 0, 0, -88, -100, -76, -129, -58, 0, 0]
            glyf.yCoordinates = [0, 1062, 0, -149, 78, 95, 0, 0, -98, -88, 186, 0, 0, -175, -182, -729, 0, 669, 108, 95, 58, 0, 0, -149, -164, -617, 0, 690, 120, 120, 0, 0, -80, -154, -145, -551]
            glyf.endPtsOfContours = [35]            
            
        elif c == "u":
            glyf.glyf_index = 88
            glyf.flags = [33, 53, 6, 35, 34, 38, 38, 39, 38, 53, 17, 51, 17, 20, 23, 22, 22, 51, 50, 54, 54, 53, 17, 51, 17]
            glyf.xCoordinates = [831, 0, -124, -213, -94, -163, -79, -16, -11, 0, 0, 180, 0, 0, 11, 17, 110, 81, 81, 142, 59, 0, 0, 180, 0]
            glyf.yCoordinates = [0, 156, -180, 0, 0, 72, 109, 79, 53, 115, 658, 0, -589, -141, -49, -71, -81, 0, 0, 83, 143, 136, 569, 0, -1062]
            glyf.endPtsOfContours = [24]            
            
        elif c == "o":
            glyf.glyf_index = 82
            glyf.flags = [19, 16, 55, 54, 51, 50, 0, 21, 20, 6, 6, 35, 34, 0, 19, 20, 22, 51, 50, 54, 53, 52, 38, 35, 34, 6]
            glyf.xCoordinates = [68, 0, 164, 137, 197, 219, 278, 0, 0, -123, -235, -139, -223, -275, 185, 0, 178, 135, 134, 178, 0, 0, -179, -133, -135, -178]
            glyf.yCoordinates = [531, 295, 142, 118, 0, 0, -287, -253, -205, -235, -130, 0, 0, 286, 269, -204, -203, 0, 0, 204, 209, 197, 203, 0, 0, -202]
            glyf.endPtsOfContours = [13, 25]            
            
        elif c == "A":
            glyf.glyf_index = 36
            glyf.flags = [35, 1, 51, 1, 35, 3, 33, 3, 19, 33, 3, 38, 39, 6, 7]
            glyf.xCoordinates = [-3, 563, 209, 600, -221, -171, -613, -161, 217, 497, -153, -70, -34, -28, -51]
            glyf.yCoordinates = [0, 1466, 0, -1466, 0, 444, 0, -444, 602, 0, 406, 185, 119, -141, -139]
            glyf.endPtsOfContours = [7, 14]     
            
        elif c == "#":
            glyf.glyf_index = 6
            glyf.flags = [23, 19, 35, 53, 51, 19, 33, 53, 33, 19, 51, 3, 33, 19, 51, 3, 51, 21, 35, 3, 33, 21, 33, 3, 35, 19, 33, 3, 19, 33, 19, 33]
            glyf.xCoordinates = [103, 87, -169, 0, 199, 74, -273, 0, 303, 87, 150, -87, 315, 87, 151, -87, 173, 0, -203, -75, 278, 0, -308, -87, -150, 86, -314, -87, 117, 314, 75, -315]
            glyf.yCoordinates = [-25, 426, 0, 149, 0, 363, 0, 149, 0, 429, 0, -429, 0, 429, 0, -429, 0, -149, 0, -363, 0, -149, 0, -426, 0, 426, 0, -426, 575, 0, 363, 0]
            glyf.endPtsOfContours = [27, 31]
            
        elif c == "e":
            glyf.glyf_index = 72
            glyf.flags = [1, 23, 6, 6, 35, 34, 0, 17, 16, 0, 51, 50, 0, 17, 20, 7, 33, 22, 22, 51, 50, 54, 1, 33, 38, 39, 38, 35, 34, 6]
            glyf.xCoordinates = [862, 186, -44, -238, -185, -233, -273, 0, 0, 276, 220, 213, 270, 0, 0, -1, -792, 10, 178, 133, 99, 140, -550, 593, -12, -56, -86, -137, -124, -169]
            glyf.yCoordinates = [342, -23, -163, -180, 0, 0, 287, 259, 268, 296, 0, 0, -290, -263, -16, -32, 0, -175, -186, 0, 0, 104, 405, 0, 134, 67, 104, 0, 0, -166]
            glyf.endPtsOfContours = [21, 29]  
            
        elif c == "n":
            glyf.glyf_index = 81
            glyf.flags = [51, 17, 51, 21, 54, 51, 50, 22, 22, 23, 22, 21, 17, 35, 17, 52, 38, 38, 35, 34, 6, 21, 17]
            glyf.xCoordinates = [135, 0, 162, 0, 117, 221, 96, 161, 80, 16, 10, 0, 0, -180, 0, 0, -42, -107, -72, -115, -167, 0, 0]
            glyf.yCoordinates = [0, 1062, 0, -151, 175, 0, 0, -69, -112, -77, -50, -125, -653, 0, 646, 110, 109, 65, 0, 0, -146, -204, -580]
            glyf.endPtsOfContours = [22]            

        elif c == "E":
            glyf.glyf_index = 40
            glyf.flags = [51, 17, 33, 21, 33, 17, 33, 21, 33, 17, 33, 21]
            glyf.xCoordinates = [162, 0, 1060, 0, -866, 0, 811, 0, -811, 0, 900, 0]
            glyf.yCoordinates = [0, 1466, 0, -173, 0, -449, 0, -172, 0, -499, 0, -173]
            glyf.endPtsOfContours = [11]
            
        elif c == "8":
            glyf.glyf_index = 27
            glyf.flags = [1, 38, 38, 53, 52, 54, 51, 50, 22, 21, 20, 6, 7, 22, 22, 21, 20, 0, 35, 34, 0, 53, 52, 54, 19, 20, 22, 51, 50, 54, 53, 52, 38, 35, 34, 6, 3, 20, 22, 22, 51, 50, 54, 53, 52, 38, 35, 34, 6]
            glyf.xCoordinates = [362, -112, -108, 0, 0, 230, 191, 192, 234, 0, 0, -107, -109, 135, 141, 0, 0, -266, -217, -217, -266, 0, 0, 145, 98, 0, 134, 107, 104, 133, 0, 0, -137, -102, -103, -136, -58, 0, 73, 144, 83, 129, 168, 0, 0, -173, -130, -127, -167]
            glyf.yCoordinates = [795, 41, 152, 106, 160, 218, 0, 0, -223, -160, -102, -151, -41, -44, -196, -136, -188, -256, 0, 0, 257, 192, 143, 193, 340, -104, -132, 0, 0, 131, 95, 99, 135, 0, 0, -132, -769, -77, -144, -79, 0, 0, 166, 128, 130, 170, 0, 0, -168]
            glyf.endPtsOfContours = [23, 35, 48]            
            
        elif c == "%":
            glyf.glyf_index = 8
            glyf.flags = [19, 52, 54, 51, 50, 22, 21, 20, 6, 35, 34, 38, 1, 34, 6, 21, 20, 22, 51, 50, 54, 53, 52, 38, 3, 1, 51, 1, 1, 52, 54, 51, 50, 22, 21, 20, 6, 35, 34, 38, 1, 34, 6, 21, 20, 22, 51, 50, 54, 53, 52, 38]
            glyf.xCoordinates = [119, 0, 158, 150, 138, 181, 0, 0, -183, -134, -133, -177, 313, -67, -89, 0, 0, 90, 66, 68, 89, 0, 0, -90, -66, 802, 146, -799, 485, 0, 158, 151, 138, 181, 0, 0, -183, -135, -133, -177, 314, -68, -89, 0, 0, 90, 66, 69, 89, 0, 0, -90]
            glyf.yCoordinates = [1114, 157, 220, 0, 0, -197, -191, -186, -201, 0, 0, 198, 453, 0, -116, -155, -141, -115, 0, 0, 116, 154, 142, 115, -1421, 1545, 0, -1545, 398, 158, 219, 0, 0, -197, -191, -186, -201, 0, 0, 199, 452, 0, -116, -155, -140, -116, 0, 0, 116, 154, 142, 115]
            glyf.endPtsOfContours = [11, 23, 27, 39, 51]   
            
        else:
            glyf.glyf_index = 72
            glyf.flags = [1, 23, 6, 6, 35, 34, 0, 17, 16, 0, 51, 50, 0, 17, 20, 7, 33, 22, 22, 51, 50, 54, 1, 33, 38, 39, 38, 35, 34, 6]
            glyf.xCoordinates = [862, 186, -44, -238, -185, -233, -273, 0, 0, 276, 220, 213, 270, 0, 0, -1, -792, 10, 178, 133, 99, 140, -550, 593, -12, -56, -86, -137, -124, -169]
            glyf.yCoordinates = [342, -23, -163, -180, 0, 0, 287, 259, 268, 296, 0, 0, -290, -263, -16, -32, 0, -175, -186, 0, 0, 104, 405, 0, 134, 67, 104, 0, 0, -166]
            glyf.endPtsOfContours = [21, 29]                
        
        return Glyphe(None, glyf)

    # ---------------------------------------------------------------------------
    # Content
    
    @property
    def is_empty(self):
        return self.points is None
    
    @property
    def char(self):
        return "<?>" if self.code is None else chr(self.code)
    
    def __repr__(self):
 
        s = f"<Glyphe of '{self.char}'"
      
        if self.is_empty:
            return s + " empty>"
        
        s += f" made of {len(self.points)} points"
        if len(self) == 0:
            s += " empty glyphe"
        else:
            s += f" --> {len(self)} contours\n"
            for i, zzs in enumerate(self.contours_zigzags):
                s += f"  Contour {i}: {zzs}\n"
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # Clone the glyphe
    
    def clone(self):
        
        clone          = Glyphe(self.ttf)
        
        clone.on_curve = list(self.on_curve)
        clone.points   = np.array(self.points)
        clone.ends     = list(self.ends)
        clone.xMin_    = self.xMin_
        clone.xMax_    = self.xMax_
        clone.yMin_    = self.yMin_
        clone.yMax_    = self.yMax_
        
        return clone
    
    # ---------------------------------------------------------------------------
    # Add a glyf read from a ttf file
    
    def add(self, glyf):
        
        # Extend on curve points
        
        self.on_curve.extend([(flag & 0x01) > 0 for flag in glyf.flags])
        
        # Built the points from the glyf

        pts = np.stack((np.cumsum(glyf.xCoordinates), np.cumsum(glyf.yCoordinates)), axis=1)
        
        # Add to the current points
        
        if self.points is None:
            count = 0
            self.points = pts
        else:
            count = len(self.points)
            self.points = np.append(self.points, pts, axis=0)
            
        # Extend the ends
        
        self.ends.extend([count + i for i in glyf.endPtsOfContours])
        
        # Update bounding box
        
        self.xMin_ = min(self.points[:, 0])
        self.yMin_ = min(self.points[:, 1])
        self.xMax_ = max(self.points[:, 0])
        self.yMax_ = max(self.points[:, 1])
        
    # ---------------------------------------------------------------------------
    # Add a glyphe

    def add_glyphe(self, glyphe):
        
        # Add the on curve points
        
        self.on_curve.extend(glyphe.on_curve)
        
        # Extend the points array
        
        if self.points is None:
            count = 0
            self.points = np.array(glyphe.points)
        else:
            count = len(self.points)
            self.points = np.append(self.points, glyphe.points, axis=0)
            
        # Extend the ends
        
        self.ends.extend([count + i for i in glyphe.ends])
        
        # Update the bound box
        
        self.xMin_ = min(self.points[:, 0])
        self.yMin_ = min(self.points[:, 1])
        self.xMax_ = max(self.points[:, 0])
        self.yMax_ = max(self.points[:, 1])
        
    # ---------------------------------------------------------------------------
    # Compute the contours
    
    def compute_contours(self):
        
        if self.is_empty:
            return
        
        # Start with an empty list of contours
        
        self.contours = []
        
        # Current contour
        contour = None
        
        # Loop on the points
        
        for index in range(len(self.points)):
            
            # Current point
            pt  = np.array(self.points[index])
            
            # ----- First of the contour
            if contour is None:
                contour   = np.array([pt])
                last_OC   = self.on_curve[index]
                OC0       = last_OC
            
            else:
                # Add an intermediary point to altern on and off curve points
                if self.on_curve[index] == last_OC:
                    contour = np.append(contour, [(contour[-1] + pt)//2], axis=0)
                    
                # Add the point
                contour = np.append(contour, [pt], axis=0)
                last_OC = self.on_curve[index]
                
            # ----- Last point of the curve
            if index in self.ends:
                
                # Intermediary point with the first one 
                if last_OC == OC0:
                    contour = np.append(contour, [(contour[0] + pt)//2], axis=0)
                    
                # Ensure flag 0 is on curve
                if not OC0:
                    cont = np.array(contour)
                    contour[1:] = cont[:-1]
                    contour[-1] = cont[0]
                    
                # Let's start a new contour
                self.contours.append(contour.reshape(len(contour)>>1, 2, 2))
                contour = None
                
        # ---------------------------------------------------------------------------
        # Compute internal and external contours
                    
        bboxes = BBox()
        for index, contour in enumerate(self.contours):
            bboxes.add(BBox(contour, index))
        
        self.ext_int = np.zeros(len(self.contours), int)
        bboxes.set_int_ext(self.ext_int)
        
        # ---------------------------------------------------------------------------
        # Compute the zigzags of the contours
        
        self.contours_zigzags = [ZigZags(contour) for contour in self.contours]

    # ----------------------------------------------------------------------------------------------------
    # As an array of contours
                
    def __len__(self):
        if self.is_empty:
            return 0
        else:
            return len(self.contours)
                
    def __getitem__(self, index):
        return np.array(self.contours[index])
    
    # ----------------------------------------------------------------------------------------------------
    # Glyphe metrics
    
    def xMin(self, char_format=None):
        return self.xMin_
        
    def xMax(self, char_format=None):
        if char_format is None:
            return self.xMax_
        else:
            return self.xMax_ + char_format.bold_shift
            
        
    def yMin(self, char_format=None):
        return self.yMin_
        
    def yMax(self, char_format=None):
        return self.yMax_
    
    def xwidth(self, char_format=None):
        
        if self.glyf_index is None:
            xw = self.xMax(char_format) - self.xMin(char_format)
        else:
            xw = self.ttf.hmtx[self.glyf_index][0]
            
        if char_format is None:
            return xw
        else:
            return xw + char_format.bold_shift

    def width(self, char_format=None):
        return self.xMax(char_format) - self.xMin(char_format)
    
    def after(self, char_format=None):
        return self.xwidth(char_format) - self.width(char_format)

    def lsb(self, char_format=None):
        return self.xMin(char_format)
    
    def ascent(self, char_format=None):
        return self.yMax(char_format)
    
    def descent(self, char_format=None):
        return self.yMin(char_format)
    
    def height(self, char_format=None):
        return self.yMax(char_format) - self.yMin(char_format)
    
    # ===========================================================================
    # Align horizontally
    
    def x_align(self, contours, char_format):
        
        if char_format is None:
            return contours

        if char_format.x_base == 'LEFT':
            return contours
        
        if char_format.x_base == 'CENTER':
            xmin = self.xMin(char_format)
            dx = xmin + (self.xMax(char_format) - xmin) // 2
        else:
            dx = self.xMax(char_format)
            
        return [contour - np.array([dx, 0]) for contour in contours]
    
    
    # ===========================================================================
    # Bold contours
    
    def fmt_contours(self, char_format=None):
        
        if self.is_empty:
            return []
        
        if char_format is None:
            bold_shift = 0
            shear      = 0.
        else:
            bold_shift = char_format.bold_shift
            shear      = char_format.shear
            
        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------------------
        # Bold
        
        # ---------------------------------------------------------------------------
        # No bold : pretty simple :-)
        
        if bold_shift == 0:
            
            bold = [np.array(contour) for contour in self.contours]
            
        else:
            
            # ---------------------------------------------------------------------------
            # Loop on the contours to build the bold contours
        
            bold = [] # The recipient for the new contours
            
            for i_contour, contour in enumerate(self.contours):
                
                # ---------------------------------------------------------------------------
                # Interior contour :
                # - hrz: shift left and keep right
                # - vrt: squeeze vertically
                
                if self.ext_int[i_contour] == 0:
                    
                    pts = np.array(contour)
                    
                    x0 = np.min(pts[:, 0, 0])
                    x1 = np.max(pts[:, 0, 0])
                    ax = x1 - x0
                    
                    y0 = np.min(pts[:, 0, 1])
                    y1 = np.max(pts[:, 0, 1])
                    yc = (y0 + y1) / 2
                    ay = y1 - yc
                    
                    bs1 = bold_shift/ax
                    bs2 = bold_shift/ay/2
                    
                    
                    for i in range(len(pts)):
                        x = pts[i, 0, 0]
                        pts[i, 0, 0] += int((x1-x)*bs1)
                        x = pts[i, 1, 0]
                        pts[i, 1, 0] += int((x1-x)*bs1)
    
                        y = pts[i, 0, 1]
                        pts[i, 0, 1] -= int((y-yc)*bs2)
                        y = pts[i, 1, 1]
                        pts[i, 1, 1] -= int((y-yc)*bs2)
                        
                else:
                    
                    
                    # ---------------------------------------------------------------------------
                    # Compute the shifts with the zigzags
                    
                    zigzags = self.contours_zigzags[i_contour]
                    pts = contour + (zigzags.shifts * bold_shift).astype(int)
                    pts[..., 1] = np.clip(pts[..., 1], zigzags.v_shift_min, zigzags.v_shift_max)
                
                # ---------------------------------------------------------------------------
                # New bold contour
                
                bold.append(pts)
            
        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------------------
        # Shear
        
        if shear != 0.:
            for pts in bold:
                pts[..., 0] += (pts[..., 1]*shear).astype(int)

        return bold
    
    # ===========================================================================
    # Blender Bezier vertices
    # Return a list or array(n, 3, 3):
    # - n : vertices count
    # - 3 : verts, lefts an rights is this order
    # - 3 : 3D vectors
    
    def beziers(self, char_format=None, plane='XY'):
        
        if self.is_empty:
            return []
        
        scale = 1 if char_format is None else (char_format.xscale, char_format.yscale)
        
        contours = self.x_align(self.fmt_contours(char_format), char_format)
        beziers = []
        for pts in contours:
            
            n = len(pts)
            
            bz = np.zeros((n, 3, 2), np.float)

            bz[:,   0] = pts[:, 0]
            
            bz[1:,  1] = pts[1:  , 0]*.3333 + pts[:-1, 1]*0.6667
            bz[:-1, 2] = pts[ :-1, 0]*.3333 + pts[:-1, 1]*0.6667
            
            bz[ 0,  1] = pts[ 0, 0]*.3333 + pts[-1, 1]*0.6667
            bz[-1,  2] = pts[-1, 0]*.3333 + pts[-1, 1]*0.6667
            
            # Scale
            if scale != 1:
                bz *= scale
            
            # In the 3D right plane
            
            bz3 = np.zeros((n, 3, 3), np.float)
            if plane == 'XZ':
                bz3[..., 0] = bz[..., 0]
                bz3[..., 2] = bz[..., 1]
            elif plane == 'YZ':
                bz3[..., 1] = bz[..., 0]
                bz3[..., 2] = bz[..., 1]
            else:
                bz3[...,:2] = bz
            
            beziers.append(bz3)
                
        return beziers
    
    # ===========================================================================
    # Rasterization
    # Points and faces
    
    def raster(self, delta=10, lowest_geometry=True, char_format=None, plane='XY', return_faces = False):
        
        if self.is_empty:
            verts = np.zeros((0, 3), float)
            if return_faces:
                return verts, [], np.zeros((0, 2), float)
            else:
                return verts
        
        verts         = None
        closed_curves = []
        
        scale = 1 if char_format is None else char_format.scales(1., dim=2)
        
        # ---------------------------------------------------------------------------
        # The faces must be computed with unformatted contours
        
        if return_faces:
            
            contours = self.x_align(self.contours, char_format)
            
        else:
            
            contours = self.x_align(self.fmt_contours(char_format), char_format)
            
        # ---------------------------------------------------------------------------
        # The rasterization is computed on the unformatted contours
        
        base_contours = self.contours
        
        # ---------------------------------------------------------------------------
        # Compute the rasterized contours
        
        for contour, base_contour in zip(contours, base_contours):
            
            # Short for len(contour)
            n = len(contour)

            # ----- Compute intermediary points at index based on the precision

            def comp_points(index, prec):
                
                t = np.expand_dims(np.linspace(0, 1, max(2, int(round(prec)))), axis=1)
                c = t*t
                b = 2*t*(1-t)
                a = (1-t)*(1-t)
        
                return a*contour[index%n, 0] + b*contour[index%n, 1] + c*contour[(index+1)%n, 0]
            
            # ----- Compute the n curves to form the final closed curve
            
            closed_curve = []
            for index in range(n):

                # Estimate the number of points for the curve
                # The lesser the cross product, the lesser the precision
                
                v0 = base_contour[(index+1)%n, 0] - base_contour[index, 0]
                v1 = base_contour[index, 1] - base_contour[index, 0]
                prec = np.sqrt(abs(v0[0]*v1[1] - v0[1]*v1[0])) / delta
                
                # Not point on straigth lines
                # But we can need more geometry, to deform the char for instance
                
                if not lowest_geometry:
                    prec = max(prec, np.linalg.norm(base_contour[(index+1)%n, 0] - base_contour[index, 0])/delta)
                
                # Compute the points
                # No need of the last point which will be the first of the
                # following curve
                
                ras_verts = comp_points(index, prec)[:-1]
                
                # Add the computed points to the current series
                vert0 = 0 if verts is None else len(verts)
                
                if verts is None:
                    verts = ras_verts
                else:
                    verts = np.append(verts, ras_verts, axis=0)
                    
                # Extend the closed curve
                
                closed_curve.extend([vert0 + i for i in range(len(ras_verts))])
                
                
            closed_curves.append(closed_curve)
            
        # ---------------------------------------------------------------------------
        # Add the third component and apply the scale
        
        def to_3D(verts, scale):
        
            if plane == 'XZ':
                return np.insert(verts*scale, 1, 0, axis=1)
            elif plane == 'YZ':
                return np.insert(verts*scale, 0, 0, axis=1)
            else: # XY
                return np.insert(verts*scale, 2, 0, axis=1)
        
        # ---------------------------------------------------------------------------
        # Done if no faces to compute
        
        if not return_faces:
            return to_3D(verts, scale)
        
        # ---------------------------------------------------------------------------
        # Compute the faces
        
        faces = closed_faces(verts, closed_curves)
        
        # ---------------------------------------------------------------------------
        # Compute uv map
        
        ratio  = 1 / 2048
        center = (1024 - self.width()/2, 256)

        nuvs = 0
        for face in faces:
            nuvs += len(face)
        uvs = np.zeros((nuvs, 2), np.float)

        index = 0        
        for face in faces:
            uvs[index:index+len(face)] = (verts[face, :2] + center)*ratio 
            index += len(face)
        
        # ---------------------------------------------------------------------------
        # Faces were computed with the raw contours
        # Now vertices must be computed with the formatted contours
        
        if char_format is None:
            return to_3D(verts, scale), faces, uvs
        elif char_format.bold_shift == 0 and char_format.shear == 0.:
            return to_3D(verts, scale), faces, uvs
        else:
            return self.raster(delta=delta, lowest_geometry=lowest_geometry, char_format=char_format, plane=plane, return_faces=False), faces, uvs
    
    # ===========================================================================
    # DEBUG on matplotlib
    
    def get_plot_arrays(self, prec=12, with_points=False, char_format=None):
        
        contours = self.fmt_contours(char_format)
        
        t = np.expand_dims(np.linspace(0, 1, prec), axis=1)
        c = t*t
        b = 2*t*(1-t)
        a = (1-t)*(1-t)
        
        plots = []
        
        oc1 = np.zeros((0, 2), int)
        oc0 = np.zeros((0, 2), int)
        
        for pts in contours:
            
            oc0 = np.append(oc0, pts[:, 0, :], axis=0)
            oc1 = np.append(oc1, pts[:, 1, :], axis=0)
            
            points = None
            
            n = len(pts)
            for k in range(n):
                
                cs = a*pts[k%n, 0] + b*pts[k%n, 1] + c*pts[(k+1)%n, 0]
                
                if points is None:
                    points = cs
                else:
                    points = np.append(points, cs, axis=0)
                    
            plots.append(points)
            
        if with_points:
            return plots, oc0, oc1
        else:
            return plots, oc0, oc1
    
    def plot_contours(self, prec=12, with_points=False, char_format=CharFormat()):
        
        with_points = True
        
        base, base0, base1 = self.get_plot_arrays(prec, with_points=True, char_format=None)
        bold_shift = 0
        if char_format.bold_shift != 0:
            bold_shift = char_format.bold_shift
            bold, bold0, bold1 = self.get_plot_arrays(prec, with_points=True, char_format=char_format)
            
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        
        for i in range(len(base)):
            
            if bold_shift != 0:
                bo = bold[i]
                ax.plot(bo[:, 0], bo[:, 1], 'red')
            
            ba = base[i]
            ax.plot(ba[:, 0], ba[:, 1], 'k')
            
            if with_points:
                ax.plot(base0[:, 0], base0[:, 1], 'or')
                ax.plot(base1[:, 0], base1[:, 1], 'ob')
                if bold_shift != 0:
                    ax.plot(bold0[:, 0], bold0[:, 1], 'xr')
                    ax.plot(bold1[:, 0], bold1[:, 1], 'xb')
                
            
        plt.show()
    
    # ===========================================================================
    # DEBUG on matplotlib
    
    def plot_raster(self, delta=10, show_points = False, **kwargs):
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.set(**kwargs)
        ax.set_aspect(1.)
        ax.plot((0, 1000, 1000, 0), (0, 0, 1000, 1000), '.')
        
        verts, faces = self.raster(delta)
        
        line = '.-k' if show_points else '-k'
        
        for face in faces:
            x = verts[[edge[0] for edge in face], 0]
            y = verts[[edge[0] for edge in face], 1]
            ax.plot(x, y, line)
        
        plt.show()        
    
    def plot_bezier(self, curve=True, **kwargs):
        
        import matplotlib.pyplot as plt
        
        prec = 12
        t = np.expand_dims(np.linspace(0, 1, prec), axis=1)
        
        d = t*t*t
        c = 3*t*t*(1-t)
        b = 3*t*(1-t)*(1-t)
        a = (1-t)*(1-t)*(1-t)
        
        fig, ax = plt.subplots()
        ax.set(**kwargs)
        ax.set_aspect(1.)
        
        beziers = self.beziers
        
        for bz in beziers:
            
            verts  = bz[:, 0]
            lefts  = bz[:, 1]
            rights = bz[:, 2]
            
            if curve:

                points = None
                
                for i in range(len(verts)):
                    
                    cs = a*verts[i] + b*rights[i] + c*lefts[(i+1)%len(verts)] + d*verts[(i+1)%len(verts)]
                    
                    if points is None:
                        points = cs
                    else:
                        points = np.append(points, cs, axis=0)
                        
                if points is not None:
                    ax.plot(points[:, 0], points[:, 1], '-k')
                    
                
            
            else:
                ax.plot(verts[:, 0], verts[:, 1],  'ok')
                ax.plot(rights[:, 0], rights[:, 1], '.b')
                ax.plot(lefts[:, 0], lefts[:, 1],  '.r')
        
        plt.show()    
    
    def plot(self, curve=True, **kwargs):
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        
        ax.set(**kwargs)
        ax.set_aspect(1.)
        
        if curve:
            base = self.plot_points()
            for points in base:
                ax.plot(points[:, 0], points[:, 1], '-k')
                
        else:
            for pts in self:
                ax.plot(pts[:, 0, 0], pts[:, 0, 1], '-')
                ax.plot(pts[:, 1, 0], pts[:, 1, 1], '.')
        
        plt.show()  
        
    @staticmethod
    def test(char_format=CharFormat(100, .2)):

        s = "S6muoA#en8%"
        
        for c in s:
            print('-'*100)
            print(c)
            gl = Glyphe.Char(c)
            gl.compute_contours()
            gl.plot_contours(char_format=char_format)
            print(gl)
            
            
#Glyphe.test()
