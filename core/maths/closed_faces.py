#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 19:11:20 2021

@author: alain
"""

import numpy as np

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Compute the faces from closed faces with holes inside
# work in 2D only
# Contours is a list of faces 

# ---------------------------------------------------------------------------
# Debug utilities

def strv(v):
    return f"[{v[0]:5.2f} {v[1]:5.2f}]"

def strvs(vs):
    s = ""
    for v in vs:
        s += strv(v)+" "
        if len(s) > 50:
            s += f"... {len(vs)}"
            break
    return s

def strtab(s, tab="    "):
    lines = s.split("\n")
    s = ""
    for line in lines:
        s += tab + line + "\n"
    return s


# ----------------------------------------------------------------------------------------------------
# Utility class to manage the contours in a easier way

class Contour():
    def __init__(self, vertices, indices, in_inversed=False):
        
        self.vertices    = vertices
        self.indices     = indices
        self.in_inversed = False
        self.children    = []
        
        # By default, the contour is linked with its child
        # betwwen the closest points
        # can be changed to VERT or HORZ if several children were linked
        self.link_mode = 'CLOSEST'
        
        # If None, there is one face: the indices
        self.faces_ = None 
        
        # is a child
        self.owner = None
    
    def __repr__(self):
        s = f"<Contour bbox: [{self.left:.1f} {self.bot:.1f} {self.right:.1f} {self.top:.1f}]\n" 
        s += f"    indices: {self.indices}\n"
        s += f"    bbox indices: [{self.i_left} {self.i_bot} {self.i_top} {self.i_right}]\n" 
        
        s += f"       left : {self.left:5.2f} @ {[i for i in self.i_left]} --> {strvs(self.verts[self.i_left])}\n"
        s += f"       bot  : {self.bot:5.2f} @ {[i for i in self.i_bot]} --> {strvs(self.verts[self.i_bot])}\n"
        s += f"       right: {self.right:5.2f} @ {[i for i in self.i_right]} --> {strvs(self.verts[self.i_right])}\n"
        s += f"       top  : {self.top:5.2f} @ {[i for i in self.i_top]} --> {strvs(self.verts[self.i_top])}\n"
        
        s += f"    verts: {strvs(self.verts)}\n"
        n = len(self.children)
        if n == 0:
            s += "    no child!\n"
        else:
            s += f"    {n} child(ren):\n"
            for i in range(n):
                s += strtab(f"child {i}: {self.children[i]}\n")
                
        faces = self.faces
        s += f"    {len(faces)} face(s):\n"
        for i, face in enumerate(faces):
            s += f"       face {i}: {face}\n"
        
        return s + ">"
    
    @property
    def depth(self):
        if self.owner is None:
            return 0
        else:
            return 1 + self.owner.depth

    @property
    def even_depth(self):
        return self.depth % 2 == 0
        
        
    def add_child(self, child):
        self.children.append(child)
        child.owner = self
        
    @property
    def verts(self):
        return self.vertices[self.indices_]
    
    @property
    def verts3D(self):
        return np.insert(np.array(self.vertices), 2, 0, axis=-1)
    
    @property
    def faces(self):
        if self.faces_ is None:
            return [self.indices]
        else:
            return self.faces_ 
        
    @property
    def edges(self):
        edges = []
        for face in self.faces:
            for i in range(len(face)):
                edges.append((face[i], face[(i+1) % len(face)]))
        return edges
        
    @property
    def indices(self):
        return self.indices_
    
    @indices.setter
    def indices(self, value):
        
        self.indices_ = np.array(value)
        
        self.left    = np.min(self.verts[:, 0])
        self.bot     = np.min(self.verts[:, 1])
        self.right   = np.max(self.verts[:, 0])
        self.top     = np.max(self.verts[:, 1])
        
        zero = 0.0001
        
        self.i_left  = np.where(abs(self.verts[:, 0] - self.left) < zero)[0]
        self.i_bot   = np.where(abs(self.verts[:, 1] - self.bot) < zero)[0]
        self.i_right = np.where(abs(self.verts[:, 0] - self.right) < zero)[0]
        self.i_top   = np.where(abs(self.verts[:, 1] - self.top) < zero)[0]
        
    def rotated_indices(self, i_from):
        return np.append(self.indices_[i_from:], self.indices_[:i_from])
        
    @property
    def width(self):
        return self.right - self.left
    
    @property
    def height(self):
        return self.top - self.bot
    
    def is_in(self, other):
        return (self.left  > other.left and
                self.right < other.right and
                self.bot   > other.bot and
                self.top   < other.top)
                
    def is_out(self, other):
        return other.is_in(self)
                
    def is_left_to(self, other):
        return self.right <= other.left
    
    def is_right_to(self, other):
        return other.is_left_to(self)
    
    def is_above(self, other):
        return self.bot >= other.top
    
    def is_below(self, other):
        return other.is_above(self)
    
    def is_higher(self, other):
        return self.top >= other.top
    
    def is_lower(self, other):
        return other.is_higher(self)
    
    def is_lefter(self, other):
        return self.left <= other.left
    
    def is_righter(self, other):
        return other.is_lefter(self)
    
    # ---------------------------------------------------------------------------
    # Closest vertices indices
    
    def closest_indices(self, other):
        d_min = None
        for i, v_i in enumerate(self.indices):
            for j, v_j in enumerate(other.indices):
                v = self.vertices[v_i] - self.vertices[v_j]
                d = v[0]*v[0] + v[1]*v[1]
                if d_min is None or d < d_min:
                    d_min = d
                    inds = [i, j]
        return inds
    
    # ---------------------------------------------------------------------------
    # indirect closest
    
    def indirect_closest(self, other, self_inds, other_inds, condition=lambda self_v, other_v: True):
        d_min = None
        inds  = None
        for i in self_inds:
            sv = self.verts[i]
            for j in other_inds:
                ov = other.verts[j]
                if condition(sv, ov):
                    v = sv - ov
                    d = v[0]*v[0] + v[1]*v[1]
                    if d_min is None or d < d_min:
                        d_min = d
                        inds = [i, j]
        return inds, d_min
    
    # ---------------------------------------------------------------------------
    # Left right closest indices
    
    def left_right_closest_indices(self, other):
        inds, _ = self.indirect_closest(other, self.i_right, other.i_left)
        return inds

    # ---------------------------------------------------------------------------
    # Left right closest indices
    
    def bot_top_closest_indices(self, other):
        inds, _ = self.indirect_closest(other, self.i_top, other.i_bot)
        return inds
    
    # ---------------------------------------------------------------------------
    # Initialize the inclusion relationship if one countour includes the other
    
    def test_in_or_out(self, other):
        if self.is_in(other):
            other.add_child(self)
        elif other.is_in(self):
            self.add_child(other)
            
    # ---------------------------------------------------------------------------
    # Link the children to form one single child
    
    def link_children(self):
        
        # We need two children at least
        if len(self.children) < 2:
            return
        
        # More than 2 children. Should be rare
        # Not tested
        if len(self.children) > 2:
            first = [self.children[0], self.children[1]]
            memo  = list(self.children)
            self.children = first
            self.link_children()
            
            for i in range(2, len(memo)):
                self.children.append(memo[i])
                
            self.link_children()
            return
        
        # We have exactly two children
        left  = None
        right = None
        bot   = None
        top   = None
        ch0   = self.children[0]
        ch1   = self.children[1]
        closest = False
        
        if ch0.is_above(ch1):
            bot   = ch1
            top   = ch0
        elif ch0.is_below(ch1):
            bot   = ch0
            top   = ch1
        elif ch0.is_left_to(ch1):
            left  = ch0
            right = ch1
        elif ch1.is_left_to(ch0):
            left  = ch1
            right = ch0
        else:
            closest = True
            
        # ---------------------------------------------------------------------------
        # Strange topology : we link the closest vertices
        
        if closest:
            
            inds = ch0.closest_indices(ch1)
            
        # ---------------------------------------------------------------------------
        # Left / right connexion
        
        elif left is not None:
            ch0 = left
            ch1 = right
            inds = ch0.left_right_closest_indices(ch1)
            self.link_mode = 'HORZ'

        # ---------------------------------------------------------------------------
        # Top / bot connexion
        
        else:
            ch0 = bot
            ch1 = top
            inds = ch0.bot_top_closest_indices(ch1)
            self.link_mode = 'VERT'
            
            
        # ---------------------------------------------------------------------------
        # Let's connect at last
        
        indices = ch0.indices[:inds[0]+1]
        indices = np.append(indices, ch1.indices[inds[1]:])
        indices = np.append(indices, ch1.indices[:inds[1]+1])
        indices = np.append(indices, ch0.indices[inds[0]:])

        self.children = [Contour(self.vertices, indices)]
        
    # ---------------------------------------------------------------------------
    # Link with the inside child
    
    def link_with_inside(self):
        
        if len(self.children) == 0:
            return
        
        if len(self.children) > 1:
            self.link_children()
            
        # Ok: we have one child to connect
        # Let's find the connected indices
        
        child = self.children[0]
        
        if self.link_mode == 'CLOSEST':
            self.link_mode = 'HORZ'
        
        if self.link_mode == 'HORZ':
            
            inds0, _ = self.indirect_closest(child, np.arange(len(self.indices)), child.i_left,  condition = lambda sv, ov: sv[0] < ov[0])
            inds1, _ = self.indirect_closest(child, np.arange(len(self.indices)), child.i_right, condition = lambda sv, ov: sv[0] > ov[0])
                    
        else:
            
            inds0, _ = self.indirect_closest(child, np.arange(len(self.indices)), child.i_bot,  condition = lambda sv, ov: sv[1] < ov[1])
            inds1, _ = self.indirect_closest(child, np.arange(len(self.indices)), child.i_top,  condition = lambda sv, ov: sv[1] > ov[1])
            
        # Let's connect
        
        if inds0[0] > inds1[0]:
            inds  = inds0
            inds0 = inds1
            inds1 = inds
            
        def debug_inds(inds0, inds1, ch_inds):
            def si(inds, main):
                return f"({inds[0]}, {self.indices[inds[0]]})" if main else f"({inds[1]}, {ch_inds[inds[1]]})"
                
            print(f"{inds0}, {inds1}: Main{si(inds0, True)} --> Child{si(inds0, False)} and Main{si(inds1, True)} --> Child{si(inds1, False)}")
            
        #print("Links")
        #print(debug_inds(inds0, inds1, child.indices))
            
        # ----- Rotation and inversion
        # inds0[1] --> first of the inverted rotated series
        # let's note : inds0[1] = i0
        # - rotation  : i --> i + x % n
        # i0 --> n-1, hence x = n-1-i0 
        # - inversion : i --> n-1 - i
        # - both      : i --> n-1 - i-x
        # with x = n-1-i0
        # i --> n-1 - i - (n-1-i0)
        # i --> i0 - i %n
        
        n = len(child.indices)
        i0 = inds0[1]
        if self.in_inversed:
            ch_inds = np.array([child.indices[(i0 - i) % n] for i in range(n)])
            inds1[1] = (i0 - inds1[1]) % n
        else:
            #ch_inds = np.array([child.indices[(i + n-2-i0) % n] for i in range(n)])
            ch_inds = np.array([child.indices[(i0 + i) % n] for i in range(n)])
            inds1[1] = (inds1[1] - i0) % n
        
        inds0[1] = 0
        
        #print("AFTER.....")
        #print(debug_inds(inds0, inds1, ch_inds))
        #print()
        #print(np.array(np.arange(len(child.indices))))
        #print(child.indices)
        #print(ch_inds)
        
        
        # We are good
        # two faces are build with the order (inds0-->inds1) and (inds1-->inds0)
        # from main then the child and vide versa
        
        
        # face0 = [child(inds0 -> inds1), main(inds1 --> inds0)]
        # child(inds0) = 0
        # main(inds0) < main(inds1) ==> main(inds1 --> inds0) = main(inds1 --> n-1, 0 --> inds0)
        
        face0 = np.array(ch_inds[:inds1[1]+1])              # child(inds0 --> inds1])
        face0 = np.append(face0, self.indices[inds1[0]:])   # main(inds1 --> n-1)
        face0 = np.append(face0, self.indices[:inds0[0]+1]) # main(0 --> inds0)
        
        # face1 = [main(inds0 --> inds1), child(inds1 --> inds0=0]
        
        face1 = np.array(self.indices[inds0[0]:inds1[0]+1]) # main(inds0 --> inds1)
        face1 = np.append(face1, ch_inds[inds1[1]:])        # child(inds1 --> n-1)
        face1 = np.append(face1, ch_inds[0])                # child(inds0)
         
        self.faces_ = [face0, face1]
        
    # ---------------------------------------------------------------------------
    # Debug
    
    def plot_ax(self, ax):
        ax.plot(self.verts[:, 0], self.verts[:, 1], '.-')
        for i, vi in enumerate(self.indices):
            v = self.vertices[vi]
            ax.annotate(f"{i},{vi}", v)
    
    def plot(self, title="Contour"):
        
        if False:
            print("-"*30)
            print(f"Contour plot: vertices: {len(self.vertices)}, indices: {len(self.indices)}")
            print(f"    {self.indices}")
            print(self)
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        #ax.set_aspect(1.)
        
        self.plot_ax(ax)
            
        for child in self.children:
            child.plot_ax(ax)
            
        plt.show()
        
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Compute the closed faces
# - vertices : an array of 2-d vertices
# - curves : a list of list of valid vertices indices 

def closed_faces(vertices, curves, in_inversed=False):
    
    # ----- Build the utility instances
    
    contours = [Contour(vertices, curve, in_inversed) for curve in curves]
                
    # ---- Encapsulates the contours
    # As russian dolls, several levels of encapsulation are possible : ®
    
    for i, contour in enumerate(contours):
        owner = None
        for j, other in enumerate(contours):
            if i != j:
                if contour.is_in(other):
                    if owner is None:
                        owner = other
                    else:
                        if other.is_in(owner):
                            owner = other

        if owner is not None:
            owner.add_child(contour)

    # ---- Compute the faces
    
    faces = []
    for contour in contours:
        if contour.even_depth:
            contour.link_with_inside()
            faces.extend(contour.faces)
            
    # ----- Done
    
    return faces


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Some test    
    
def test():
    
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------------
    # Utility class to manage the contours in a easier way
    
    def calc_circle(count):
        ag = np.linspace(0, np.pi*2, count, False)
        return np.stack((np.cos(ag), np.sin(ag))).transpose()
        
    count = 16
    vertices = calc_circle(count) - [0, 2]
    
    diamond0 = np.array((0, 4, 8, 12))
    square0  = np.array((2, 6, 10, 14))
    circle0  = np.arange(count)
    
    vertices = np.append(vertices, calc_circle(count) + [0, 2], axis=0)
    diamond1 = diamond0 + count
    square1  = square0 + count
    circle1  = circle0 + count
    
    vertices = np.append(vertices, calc_circle(count) *10, axis=0)
    circle2  = circle1 + count
    diamond2 = diamond1 + count
    
    faces = closed_faces(vertices, [circle1, circle2, square0])
    

    fig, ax = plt.subplots()
    ax.set_aspect(1.)
    
    for face in faces:
        v = vertices[face]
        v = np.resize(v, (len(v)+1, 2))
        ax.plot(v[:, 0], v[:, 1])
        
    plt.show()
        
#test()

def debug():
    vertices = np.array([
        [420.0, 211.0],
        [406.4285714285715, 178.3469387755102],
        [392.8571428571429, 148.10204081632656],
        [379.2857142857143, 120.26530612244898],
        [365.7142857142857, 94.83673469387756],
        [352.14285714285717, 71.81632653061226],
        [338.57142857142856, 51.20408163265306],
        [325.0, 33.0],
        [311.4285714285714, 17.204081632653068],
        [297.8571428571429, 3.81632653061226],
        [284.28571428571433, -7.1632653061224385],
        [270.7142857142857, -15.734693877551017],
        [257.1428571428571, -21.897959183673468],
        [243.5714285714286, -25.653061224489793],
        [230.0, -27.0],
        [209.60330578512395, -26.18181818181818],
        [190.23140495867767, -23.36363636363636],
        [171.88429752066116, -18.545454545454547],
        [154.5619834710744, -11.727272727272728],
        [138.26446280991735, -2.9090909090909065],
        [122.9917355371901, 7.909090909090903],
        [108.74380165289256, 20.727272727272727],
        [95.52066115702479, 35.54545454545455],
        [83.32231404958678, 52.363636363636374],
        [72.1487603305785, 71.18181818181819],
        [62.0, 92.0],
        [51.10204081632654, 119.44897959183675],
        [41.83673469387755, 149.22448979591837],
        [34.20408163265306, 181.32653061224488],
        [28.204081632653065, 215.75510204081633],
        [23.836734693877553, 252.51020408163262],
        [21.102040816326532, 291.59183673469386],
        [20.0, 333.0],
        [20.46875, 374.53125],
        [22.375, 415.125],
        [25.71875, 454.78125],
        [30.5, 493.5],
        [36.71875, 531.28125],
        [44.375, 568.125],
        [53.46875, 604.03125],
        [64.0, 639.0],
        [73.7603305785124, 667.0247933884297],
        [84.13223140495867, 692.5537190082644],
        [95.11570247933885, 715.5867768595041],
        [106.7107438016529, 736.1239669421489],
        [118.91735537190084, 754.1652892561983],
        [131.73553719008265, 769.7107438016529],
        [145.16528925619835, 782.7603305785124],
        [159.20661157024793, 793.3140495867768],
        [173.85950413223142, 801.3719008264463],
        [189.12396694214877, 806.9338842975206],
        [205.0, 810.0],
        [230.0, 811.0],
        [252.9387755102041, 809.4897959183675],
        [274.3265306122449, 804.9591836734694],
        [294.1632653061224, 797.408163265306],
        [312.44897959183675, 786.8367346938776],
        [329.18367346938777, 773.2448979591836],
        [344.36734693877554, 756.6326530612245],
        [358.0, 737.0],
        [373.875, 704.875],
        [388.5, 664.5],
        [401.875, 615.875],
        [414.0, 559.0],
        [431.0, 499.0],
        [426.4444444444445, 478.6666666666667],
        [412.77777777777777, 461.66666666666663],
        [390.0, 448.0],
        [369.0, 444.0],
        [305.0, 463.0],
        [274.25, 486.0],
        [264.0, 511.0],
        [278.0, 550.0],
        [262.0, 633.5],
        [247.0, 664.125],
        [230.0, 677.0],
        [219.08000000000004, 674.8000000000001],
        [208.72, 662.6],
        [198.92000000000002, 640.4],
        [189.68, 608.2],
        [181.0, 566.0],
        [170.55555555555557, 498.44444444444457],
        [163.22222222222223, 426.44444444444446],
        [159.0, 350.0],
        [158.7777777777778, 292.11111111111114],
        [162.11111111111111, 237.11111111111114],
        [169.0, 185.0],
        [177.4375, 146.5625],
        [187.75, 118.25],
        [199.9375, 100.0625],
        [214.0, 92.0],
        [233.75, 101.75],
        [257.0, 137.0],
        [279.0, 185.0],
        [297.0, 241.0],
        [295.0, 260.0],
        [297.00000000000006, 278.02777777777777],
        [303.00000000000006, 292.7777777777778],
        [313.0, 304.25],
        [327.0, 312.44444444444446],
        [345.0, 317.36111111111114],
        [367.0, 319.0],
        [399.0, 316.0],
        [426.22222222222223, 306.11111111111114],
        [442.55555555555554, 289.77777777777777],
        [448.0, 267.0],
        [441.0, 237.0],
    ])
    curves = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 51, 52, 52, 53, 54, 55, 56, 57, 58, 59, 59, 60, 61, 62, 63, 63, 64, 64, 65, 66, 67, 67, 68, 68, 69, 69, 70, 71, 71, 72, 72, 73, 73, 74, 75, 75, 76, 77, 78, 79, 80, 80, 81, 82, 83, 83, 84, 85, 86, 86, 87, 88, 89, 90, 90, 91, 92, 92, 93, 94, 94, 95, 95, 96, 97, 98, 99, 100, 101, 101, 102, 102, 103, 104, 105, 105, 106, 0],
    ]
    
    # ----- Build the utility instances
    
    contours = [Contour(vertices, curve) for curve in curves]
    
    # ---- Encapsulates the contours
    # As russian dolls, several levels of encapsulation are possible : ®
    
    for i, contour in enumerate(contours):
        owner = None
        for j, other in enumerate(contours):
            if i != j:
                if contour.is_in(other):
                    if owner is None:
                        owner = other
                    else:
                        if other.is_in(owner):
                            owner = other

        if owner is not None:
            owner.add_child(contour)

    # ---- Compute the faces
    
    faces = []
    for contour in contours:
        if contour.even_depth:
            contour.link_with_inside()
            faces.extend(contour.faces)
            
    contours[0].plot()
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect(1.)
    
    for face in faces:
        v = vertices[face]
        v = np.resize(v, (len(v)+1, 2))
        ax.plot(v[:, 0], v[:, 1])
        
    plt.show()
    
    
#debug()
    
    
        
