#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 08:38:09 2021

@author: alain
"""

import numpy as np
from .geometry import e_rotate, q_rotate, quaternion, m_to_euler, e_to_matrix
from .bezier import from_points
from ..core.commons import WError

# ===========================================================================
# A chain is a series of segment. Each segment is made of:
# - an euler rotation and a distance
# - a next point
# The next point is on the x axis on the frame of the current point
# Each segment rotate all the following points. This allow to animate
# a curve
# The euler rotation needs only a y and z rotation inthe XYZ order
# An additional X rotation can be added to control the twist of the curve

class Chain():
    
    def __init__(self, count=10, length=1):
        
        self.eulers   = np.zeros((count, 3), np.float)
        self.twists   = np.zeros(count, np.float)
        
        self.verts_   = np.zeros((count, 3), np.float)
        self.verts_[:, 0] = np.linspace(0, length, count)

        self.location = np.zeros(3, np.float)
        self.segments = np.ones(count)*length/(count-1)
        
    def __repr__(self):
        
        mx = 5
        
        def sitem(i):
            return f"{self.segments[i]:.2f} ({np.degrees(self.eulers[i, 1]):.1f}° {np.degrees(self.eulers[i, 2]):.1f}°)"
        
        ss = ""
        sepa = ""
        
        for i in range(min(mx, len(self))):
            ss += sepa + sitem(i)
            sepa = " "
            
        if len(self) > mx:
            ss += " ... " + sitem(-1)
            
        s = f"<Chain of {len(self)} segments, length={np.sum(self.segments)}, location={self.location}\n segments: [{ss}]>"
        return s
        
    def __len__(self):
        return len(self.verts_)
    
    def clone(self):
        
        chain = Chain(count=len(self))
        
        chain.eulers   = np.array(self.eulers)
        chain.twists   = np.array(self.twists)
        chain.verts_   = np.array(self.verts_)
        
        chain.segments = np.array(self.segments)
            
        return chain
    
    # ---------------------------------------------------------------------------
    # Twist
    
    def twist(self, angle):
        self.twists = np.linspace(0, angle, len(self))
        
    # ---------------------------------------------------------------------------
    # Compute the euler rotation to rotation from a point on the x axis to
    # the location passed in argument
    
    @staticmethod
    def yz_euler(v):
        return np.array([0, -np.arctan2(v[2], np.linalg.norm(v[:2])), np.arctan2(v[1], v[0])])

    # ---------------------------------------------------------------------------
    # Initialize a chain from a series of vertices
    
    @classmethod
    def FromVertices(cls, verts):
        
        chain = cls(count=len(verts))
        chain.verts_ = np.array(verts)
        chain.location = np.array(chain.verts_[0])
        
        vs = np.array(verts)
        
        for i in range(1, len(chain)):
            pivot = vs[i-1]
            chain.eulers[i] = Chain.yz_euler(vs[i] - pivot)
            vs[i:] = pivot + e_rotate(-chain.eulers[i], vs[i:] - pivot, 'ZYX')
            
            chain.segments[i] = (vs[i] - pivot)[0]
            
        return chain
    
    # ---------------------------------------------------------------------------
    # Initialize a function

    @classmethod
    def FromFunction(cls, f, count, t0=0, t1=1):
        return cls.FromVertices(f(np.linspace(t0, t1, count)))
    
    # ---------------------------------------------------------------------------
    # Vertices are computed from 0 (bent vertices) to 1 (flat vertices)
    
    def verts(self, t):
        
        verts = np.array(self.verts_)
        for i in range(1, len(self)):
            
            pivot = verts[i-1]
            
            if i == 1:
                verts[:] = e_rotate(-self.eulers[i]*t, verts[:] - pivot, 'ZYX')
            else:
                verts[i:] = e_rotate(-self.eulers[i]*t, verts[i:] - pivot, 'ZYX')
            
            #verts[i:] = e_rotate(-self.eulers[i]*t, verts[i:] - pivot, 'ZYX')
            
            delta0 = np.array(verts[i])
            nrm = np.linalg.norm(delta0)
            if nrm > 0:
                delta1 = delta0 * (self.segments[i]*t + nrm*(1-t))/nrm
                verts[i:] += delta1 - delta0
                
            verts[i:] += pivot
            pivot = verts[i]
            
        #verts += (self.location - self.verts_[0])*t
        verts += self.verts_[0]*(1-t) + self.location*t
            
        return verts
    
    # ---------------------------------------------------------------------------
    # Invert
    # Change direction of transformation
    # By default:
    # - t = 0 --> bent vertices
    # - t = 1 --> flat vertices
    #
    # The Euler rotations, when combined, flatten the bent vertices:
    # - segment i is flattend once previous rotations are combined
    #
    # To go from flatten vertices to bent vertices,
    
    def invert(self):
        
        def smat(mats):
            s = ""
            for m in mats:
                s += f"{np.degrees(m_to_euler(m, 'ZYX')[2]):6.1f}° "
            return s
            
        verts = np.zeros((len(self), 3), np.float)
        verts[:, 0] = np.cumsum(self.segments)
        verts -= self.location
        
        if False:
        
            self.verts_ = verts
            
            for i, e in enumerate(self.eulers):
                self.eulers[i] = m_to_euler(e_to_matrix(-e, 'ZYX'), 'XYZ')
                
            return
        
        
        
        # ----- Conversion eulers to matrices
        
        mats = np.array([e_to_matrix(-e, 'ZYX') for e in self.eulers])
        
        print("Base")
        print(smat(mats))
        
        # ----- Cumulative rotations

        cumul = np.array(mats)
        for i in range(1, len(mats)-1):
            cumul[i+1:] = np.matmul(mats[i+1:], cumul[i])
            
        # ----- Invert to get the target rotations
        
        mats = np.linalg.inv(cumul)
        
        # ----- Back to the non cumulative rotations
        
        #mats = np.array(inv)
        for i in range(1, len(cumul)-1):
            mats[i+1:] = np.matmul(np.linalg.inv(mats[i]), mats[i+1:])
            
        print("res")
        print(smat(mats))
        
        self.verts_ = verts
        self.eulers = [m_to_euler(m, 'ZYX') for m in mats]
            
        return
            
        
        
        
        
        eulers = np.array(self.eulers)
        mats = np.zeros((len(self), 3, 3), np.float)
        mats[:] = np.identity(3)
        for i in range(1, len(self)):
            mats[i] = e_to_matrix(-self.eulers[i], 'ZYX')
            
        for i in reversed(range(1, len(self)-1)):
            for j in range(i+1, len(self)):
                mats[j] = np.matmul(mats[i], mats[j])
            
        
        
        
        for i in range(1, len(self)):
            print(i, "euler", np.degrees(self.eulers[i]), "test", np.degrees(m_to_euler(e_to_matrix(self.eulers[i], 'XYZ'), 'XYZ')), 'XYZ')

            m = np.linalg.inv(e_to_matrix(eulers[i], 'ZYX'))
            eulers[i] = m_to_euler(m, 'ZYX')
            
            if i < 10:
                print(i, np.degrees(self.eulers[i]), np.degrees(eulers[i]))
            for j in range(i+1, len(self)):
                eulers[j] = m_to_euler(np.matmul(m, e_to_matrix(eulers[j], 'ZYX')))
                pass
                
        self.verts_ = verts
        self.eulers = eulers
    
    # ---------------------------------------------------------------------------
    # To another chain
    
    def set_target(self, other):

        if len(other) != len(self):
            raise WError("Chain deformation error: the two chains must have the same number of points",
                         Class = "Chain", self="set_target",
                         len_of_self = len(self),
                         len_of_target = len(other))
            
        self.segments = other.segments
        self.eulers -= other.eulers
        self.location = other.location
    
    
    # ---------------------------------------------------------------------------
    # Bend
    
    def bend(self, t, vertices, axis=0, index=0):
        
        transpose = axis==1
        if transpose:
            verts = np.transpose(vertices, (1, 0, 2))
        else:
            verts = np.array(vertices)
            
        for i in range(1, len(self)):
            
            pivot = verts[i-1, index]
            
            if i == 1:
                verts[:] = e_rotate(-self.eulers[i]*t, verts[:] - pivot, 'ZYX')
            else:
                verts[i:] = e_rotate(-self.eulers[i]*t, verts[i:] - pivot, 'ZYX')
            
            # ----- Adjust segment length
            
            delta0 = np.array(verts[i, index])
            nrm = np.linalg.norm(delta0)
            if nrm > 0:
                delta1 = delta0 * (self.segments[i]*t + nrm*(1-t))/nrm
                verts[i:] += delta1 - delta0
                
            verts[i:] += pivot
            
            # ----- Twist
            
            if i > 0:
                pt = verts[i, index]
                twist_axis = verts[i, index] - verts[i-1, index]
                verts[i] = pt + q_rotate(quaternion(twist_axis, self.twists[i]*t), verts[i] - pt)
                
                if i == 1:
                    pt = verts[0, index]
                    verts[0] = pt + q_rotate(quaternion(twist_axis, self.twists[0]*t), verts[0] - pt)
                    

        #verts += (self.location - self.verts_[0])*t
        verts += self.verts_[0]*(1-t) + self.location*t
            
        if transpose:
            return verts.transpose((1, 0, 2))
        else:
            return verts
        
        
        
        
        
        
        
        
        
        
        
        
        transpose = axis==1
        if transpose:
            verts = np.transpose(vertices, (1, 0, 2))
        else:
            verts = np.array(vertices)
        
        for i in reversed(range(len(self))):
            
            if i == 0:
                pivot = np.zeros(3, np.float)
            else:
                if np.shape(verts) == 3:
                    pivot = verts[i-1, index]
                else:
                    pivot = verts[i-1]
                #pivot[0] += self.segments[i-1] - pivot[0]
                pivot[0] = self.segments[i-1]
            
            verts[i:] = pivot + e_rotate(self.eulers[i], verts[i:] - pivot, 'XYZ')
            
        if transpose:
            return verts.transpose((1, 0, 2))
        else:
            return verts

    @property
    def flat_verts(self):
        verts = np.zeros((len(self), 3), np.float)
        verts[:, 0] = self.segments
        return verts
    
    @property
    def vertsOLD(self):
        return np.array(self.vertices)
        return self.bend(self.flat_verts)
    
    @property
    def edges(self):
        return [[i, i+1] for i in range(len(self))]
    
    # ---------------------------------------------------------------------------
    # Initialize a chain to compute intermediary locations from one chain
    # to another one
    # A chained initialized as Deformation can use the deform method
    
    def deform(self, t, target, vertices, axis=0, index=0):
        
        if len(target) != len(self):
            raise WError("Chain deformation error: the two chains must have the same number of points",
                         Class = "Chain", self="deform",
                         len_of_self = len(self),
                         len_of_target = len(target))
            
        chain = Chain(len(self))
        chain.eulers   = self.eulers*(1-t) + target.eulers*t
        chain.twists   = self.twists*(1-t) + target.twists*t
        chain.segments = self.segments*(1-t) + target.segments*t
        
        return chain.bend(vertices, axis, index)





# ===========================================================================
# A chain is a series of segment. Each segment is made of:
# - an euler rotation and a distance
# - a next point
# The next point is on the x axis on the frame of the current point
# Each segment rotate all the following points. This allow to animate
# a curve
# The euler rotation needs only a y and z rotation inthe XYZ order
# An additional X rotation can be added to control the twist of the curve

class ChainOLD():
    
    def __init__(self, shape=(10,), axis=0, index=0):
        
        self.eulers = np.zeros((shape[axis], 3), np.float)
        
        if len(shape) == 1:
            shape = shape + (1,)
        
        self.vertices = np.zeros(shape + (3,), np.float)
        self.transpose = axis == 1
        if self.transpose:
            self.vertices = np.transpose(self.vertices, (1, 0, 2))
        self.index = index
        
        self.vertices[..., 0] = np.arange(shape[axis]).reshape(shape[axis], 1)
        self.vertices[..., 1] = np.arange(shape[1-axis])
        
        print(self.vertices)
        
        self.is_deformation    = False
        
    def __repr__(self):
        s = f"<Chain {self.shape}, length: {len(self)}, is_deformation: {self.is_deformation}"
        s += f", surface:{self.has_surface}, index={self.index}>"
        return s
    
    @property
    def backbone(self):
        return self.vertices[:, self.index].reshape(len(self), 3)
        
    @property
    def shape(self):
        shape = self.vertices.shape[:-1]
        if self.transpose:
            return (shape[1], shape[0])
        else:
            return shape
        
    def __len__(self):
        return self.vertices.shape[0]
    
    @property
    def has_surface(self):
        return self.vertices.shape[1] > 1
    
    def clone(self):
        chain = Chain(self.shape, 1 if self.transpose else 0, self.index)
        chain.eulers   = np.array(self.eulers)
        chain.vertices = np.array(self.vertices)
            
        return chain
    
    # ---------------------------------------------------------------------------
    # Twist
    
    def twist(self, angle):
        self.eulers[0, 0] = 0
        self.eulers[1:, 0] = angle/(len(self)-1)
        
    # ---------------------------------------------------------------------------
    # Compute the euler rotation to rotation from a point on the x axis to
    # the location passed in argument
    
    @staticmethod
    def yz_euler(v):
        return np.array([0, -np.arctan2(v[2], np.linalg.norm(v[:2])), np.arctan2(v[1], v[0])])

    # ---------------------------------------------------------------------------
    # Initialize a chain from a series of vertices
    
    @classmethod
    def FromVertices(cls, verts, axis=0, index=0):
        
        chain = cls(np.shape(verts)[:-1], axis, index)
        if chain.transpose:
            chain.vertices = np.array(np.transpose(np.reshape(verts, chain.shape + (3,)), (1, 0, 2)))
        else:
            chain.vertices = np.array(np.reshape(verts, chain.shape + (3,)))
            
        pivot = np.zeros(3, np.float)
        for i in range(len(chain)):
            chain.eulers[i] = Chain.yz_euler(chain.vertices[i, index] - pivot)
            chain.vertices[i:] = pivot + e_rotate(-chain.eulers[i], chain.vertices[i:] - pivot, 'ZYX')
            pivot = np.array(chain.vertices[i, index])
            
            
        return chain
    
    @classmethod
    def FromFunction(cls, f, count, t0=0, t1=1):
        return cls.FromVertices(f(np.linspace(t0, t1, count)))
    
    @property
    def verts(self):
        
        verts = np.array(self.vertices)
        for i in reversed(range(len(self))):
            
            if i == 0:
                pivot = np.zeros(3, np.float)
            else:
                pivot = verts[i-1, self.index]
            
            verts[i:] = pivot + e_rotate(self.eulers[i], verts[i:] - pivot, 'XYZ')
                

        if self.has_surface:
            if self.transpose:
                return verts.transpose((1, 0, 2))
            else:
                return verts
        else:
            return verts.reshape(len(self), 3)
    
    @property
    def edges(self):
        return [[i, i+1] for i in range(len(self))]
    
    # ---------------------------------------------------------------------------
    # Initialize a chain to compute intermediary locations from one chain
    # to another one
    # A chained initialized as Deformation can use the deform method
    
    def deformation(self, target):
        
        if len(target) != len(self):
            raise WError("Chain deformation error: the two chains must have the same number of points",
                         Class = "Chain", self="deformation",
                         len_of_self = len(self),
                         len_of_target = len(target),
                         self_shape = self.shape,
                         target_shape = target.shape)
            
        self.delta_eulers   = target.eulers - self.eulers
        self.delta_vertices = np.array(self.vertices)
        bb = target.backbone - 2*self.backbone
        self.delta_vertices += bb.reshape(len(self), 1, 3) #target.backbone[:, 0] - 2*self.backbone[:, 0]

        # DEBUG        
        #self.delta_vertices = target.vertices - self.vertices
        # EO DEBUG

        self.is_deformation = True
    
    # ---------------------------------------------------------------------------
    # Computes the intermediary vertices between two chains
    # Must be initialized with Deformation
    
    def deform(self, t):
        if not self.is_deformation:
            raise WError("The chain is not a deformation. Use Deformation initializer.",
                         Class = "Chain", Method="deform")
            
        chain = self.clone()
        chain.eulers   = self.eulers   + t*self.delta_eulers
        chain.vertices = self.vertices + t*self.delta_vertices 
        
        return chain.verts
    
    
    