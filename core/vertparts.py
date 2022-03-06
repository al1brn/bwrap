#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 18:54:37 2022

@author: alain
"""
     
import numpy as np

from .wroot import WRoot
from .vertgroups import VertGroups
from .varrays import VArrays
from .faces import Faces
from .commons import WError

from .maths.bezier import Polynoms
from .maths.transformations import Transformations

# ----------------------------------------------------------------------------------------------------
# A utility to clone a simple structure

def clone(p):
    class Q():
        pass
    
    q = Q()
    for k in dir(p):
        if k[:2] != '__':
            setattr(q, k, getattr(p, k))
            
    return q

# ----------------------------------------------------------------------------------------------------
# The verts are stored in an array of shape (parts_count, shapes_count, verts_count)

class VertParts(WRoot):
    
    def __init__(self, type = 'Mesh'):
        
        self.type      = type

        self.verts     = np.empty((0, 0, 0, 3), float) # Parts, Shapes, Verts
        self.sizes     = np.empty((0,), int)           # Number of vertices per part
        self.true_indices = None

        self.mats_     = []
        self.mat_i     = np.empty(0, int)
        
        if self.type == 'Mesh':
            self.faces  = Faces()
            self.uvmaps = {}
            self.groups = VertGroups()
            
        elif self.type == 'Curve':
            self.profile = np.zeros((0, 3), int)
            
            
    @classmethod
    def MeshFromData(cls, verts, faces, uvmaps={}):
        
        geo = cls(type = 'Mesh')

        vshape = np.shape(verts)
        vshape = (1, 1, 1, 1)[:4-len(vshape)] + vshape
            
        geo.verts = np.array(verts, float).reshape(vshape)
        
        geo.faces = Faces.FromOther(faces)
        geo.sizes = [geo.verts.shape[2]]
        
        geo.ensure_mat_i()
        
        if type(uvmaps) is not dict:
            uv_dict = {"UVMap": uvmaps}
        else:
            uv_dict = uvmaps 
        
        for name, uvs in uv_dict.items():
            geo.uvmaps[name] = uvs
        
        return geo

    @classmethod
    def CurveFromData(cls, verts, profile):
        
        geo       = cls(type = 'Curve')
        geo.verts = np.empty((1, 1, np.size(verts)//3, 3), float)
        geo.verts[0, 0, :] = verts
        
        geo.profile = np.array(profile)
        geo.sizes   = [geo.verts.shape[2]]
        
        geo.ensure_mat_i()
        
        return geo
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Basic shapes
    
    @staticmethod
    def polygon(count=16, radius=1., start_angle=0, inverse=False):
        if inverse:
            ags = np.linspace(start_angle, start_angle - 2*np.pi, count, endpoint=False)
        else:
            ags = np.linspace(start_angle, start_angle + 2*np.pi, count, endpoint=False)
        return radius*np.cos(ags), radius*np.sin(ags)
    
    @staticmethod
    def arc(count=16, radius=1., ag0=-np.pi/2, ag1=np.pi/2):
        ags = np.linspace(ag0, ag1, count, endpoint=True)
        return radius*np.cos(ags), radius*np.sin(ags)
        
            
    @classmethod
    def Default(cls, shape='CUBE', parts=1, shapes=1, materials=False, origin=0, **kwargs):
        
        shape = shape.upper()
        
        geo   = None
        verts = None
        faces = None
        uvs   = None
        
        if shape == 'CUBE':
            verts = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
            faces = Faces.FromList([[3, 0, 4, 7], [2, 3, 7, 6], [1, 2, 6, 5], [0, 1, 5, 4], [1, 0, 3, 2], [6, 7, 4, 5]])
            
            uvs = np.reshape(Faces.uvgrid(4, 4, rect=(1/8, 0, 1+1/8, 1)), (16, 4, 2))
            Faces.permute_uvs(uvs, 1)

            Faces.permute_uvs(uvs[8],  1)
            Faces.permute_uvs(uvs[10], 3)
            
            uvs = uvs[[1, 5, 9, 13, 8, 10]]
        
        elif shape == 'PLANE':            
            verts = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
            faces = Faces.FromList([[0, 1, 2, 3]])
            uvs   = Faces.uvsquare()
            
        elif shape == 'DISK':

            segms  = 32 if kwargs.get("segms")  is None else kwargs["segms"]
            radius = 1. if kwargs.get("radius") is None else kwargs["radius"]
            fans   = True if kwargs.get("fans") is None else kwargs["fans"]

            x_radius = radius if kwargs.get("x_radius") is None else kwargs["x_radius"]
            y_radius = radius if kwargs.get("y_radius") is None else kwargs["y_radius"]
            
            hx, hy = cls.polygon(segms, radius=1.)
            hx *= x_radius
            hy *= y_radius
            
            verts = np.zeros((segms, 3), float)

            verts[:, 0] = hx
            verts[:, 1] = hy
            
            if fans:
                verts = np.append(verts, np.resize(0, (1, 3)), axis=0)
                faces = Faces.Triangles(segms, np.arange(segms), close=True)
                uvs   = Faces.uvfans(segms)
            else:
                faces = Faces(1)
                faces.append(np.arange(segms), 0)
                uvs   = Faces.uvpolygon(segms)
        
        elif shape == 'GRID':
            
            r = 10 if kwargs.get("resolution") is None else kwargs["resolution"]
            s = (2., 2.) if kwargs.get("size") is None else kwargs["size"]
            
            res = np.zeros(2, int)
            res[:] = r
            
            size = np.zeros(2, float)
            size[:] = s
            
            verts = np.zeros((res[1], res[0], 3), float)
            verts[..., 0] = np.linspace(-size[0]/2, size[0]/2, res[0], endpoint=True)
            verts[..., 1] = np.expand_dims(np.linspace(-size[1]/2, size[1]/2, res[1], endpoint=True), axis=-1)
            verts = np.reshape(verts, (res[0]*res[1], 3))
            
            faces = Faces.Grid(res[0], res[1])
            
            uvs   = Faces.uvgrid(res[0]-1, res[1]-1)
        
        elif shape == 'PYRAMID':
            verts = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 1))
            faces = Faces.FromList([[3, 2, 1, 0], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
            
            a = .25
            b = .75
            c = .5
            
            uvs = Faces.uvsquare(rect=(a, a, b, b))
            
            uvs = np.append(uvs, ((b, b), (b, a), (1, c)), axis=0)
            uvs = np.append(uvs, ((a, b), (b, b), (c, 1)), axis=0)
            uvs = np.append(uvs, ((a, a), (a, b), (0, c)), axis=0)
            uvs = np.append(uvs, ((b, a), (a, a), (c, 0)), axis=0)
            
        elif shape in ['TETRA', 'TETRAEDRE']:
            
            radius = 1. if kwargs.get("radius") is None else kwargs["radius"]

            verts = np.zeros((4, 3), float)
            
            ags = np.linspace(-np.pi/3, 5*np.pi/3, 3, endpoint=False)
            verts[:3, 0] = np.cos(ags)
            verts[:3, 1] = np.sin(ags)
            
            verts[3, 2] = 4/3
            
            verts *= radius
            
            faces = Faces.FromList([[2, 1, 0], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
            
            uvs = np.empty((4, 3, 2), float)
            
            h = np.sqrt(3)/2
            # Triangle (0, 0) (1, 0), (0.5, h))
            
            A = (0.25, h/2)
            B = (0.75, h/2)
            C = (0.5, 0)
            
            uvs[0] = [A, B, C]
            uvs[1] = [C, B, (1, 0)]
            uvs[2] = [B, A, (.5, h)]
            uvs[3] = [A, C, (0, 0)]
            
        elif shape == 'UVSPHERE':
            
            segms  = 32 if kwargs.get("segms")  is None else kwargs["segms"]
            rings  = 16 if kwargs.get("rings")  is None else kwargs["rings"]
            radius = 1. if kwargs.get("radius") is None else kwargs["radius"]
            
            rings += 1
            
            
            hx, hy = cls.polygon(segms, radius=1.)
            vx, vy = cls.arc(rings, radius=radius, ag0=-np.pi/2, ag1=np.pi/2)
            
            verts = np.empty((rings-2, segms, 3), float)
            for i in range(1, rings-1):
                verts[i-1, :, 0] = hx*vx[i]
                verts[i-1, :, 1] = hy*vx[i]
                verts[i-1, :, 2] = vy[i]
                
            n = (rings-2)*segms
            verts = np.append(np.reshape(verts, (n, 3)), ((0, 0, -radius), (0, 0, radius)), axis=0)
            
            faces = Faces.Grid(segms, rings-2, x_close=True, y_close=False)
            faces.extend(Faces.Triangles(n,   np.flip(np.arange(segms)),        close=True))
            faces.extend(Faces.Triangles(n+1, np.arange(segms)+(rings-3)*segms, close=True))
            
            h_uv = 1/(rings-1)
            
            uvs = Faces.uvgrid(segms, rings-3, (0, h_uv, 1, 1-h_uv))
            
            uvs0 = Faces.uvtriangles(segms, (1, h_uv,   0, 0))
            uvs1 = Faces.uvtriangles(segms, (0, 1-h_uv, 1, 1))
            
            uvs = np.append(uvs, uvs0, axis=0)
            uvs = np.append(uvs, uvs1, axis=0)
            
        elif shape == 'CYLINDER':
            
            segms  = 32 if kwargs.get("segms")  is None else kwargs["segms"]
            rings  =  2 if kwargs.get("rings")  is None else kwargs["rings"]
            radius = 1. if kwargs.get("radius") is None else kwargs["radius"]
            height = 2. if kwargs.get("height") is None else kwargs["height"]
            fans   = True if kwargs.get("fans") is None else kwargs["fans"]
            
            h = height/2
            
            hx, hy = cls.polygon(segms, radius=1., start_angle=np.pi/2)
            
            verts = np.empty((rings, segms, 3), float)
            verts[..., 0] = hx
            verts[..., 1] = hy
            verts[..., 2] = np.expand_dims(np.linspace(-h, h, rings, endpoint=True), axis=-1)
            
            verts = verts.reshape(rings*segms, 3)
            faces = Faces.Grid(segms, rings, x_close=True, y_close=False)
            uvs   = Faces.uvgrid(segms, rings-1, (0, .5, 1, 1))
            
            if fans:
                n = len(verts)
                verts = np.append(verts, np.resize(((0, 0, -h), (0, 0, h)), (2, 3)), axis=0)
                
                faces.extend(Faces.Triangles(n,   np.flip(np.arange(segms)),        close=True))
                faces.extend(Faces.Triangles(n+1, np.arange(segms)+(rings-1)*segms, close=True))
                
                uvs0 = Faces.uvfans(segms, rect=(.01, .01, .49, .49))
                uvs1 = Faces.uvfans(segms, rect=(.51, .01, .99, .49))
                uvs  = np.append(uvs, uvs1, axis=0)
                uvs  = np.append(uvs, uvs0, axis=0)
                
            else:
                faces.append(np.flip(np.arange(segms)))
                faces.append(np.arange(segms)+(rings-1)*segms)
                
                uvs0 = Faces.uvpolygon(segms, rect=(.01, .01, .49, .49))
                uvs1 = Faces.uvpolygon(segms, rect=(.51, .01, .99, .49))
                uvs = np.append(uvs, uvs1, axis=0)
                uvs = np.append(uvs, uvs0, axis=0)
                
        elif shape == 'CONE':
            
            segms  = 32 if kwargs.get("segms")  is None else kwargs["segms"]
            rings  =  2 if kwargs.get("rings")  is None else kwargs["rings"]
            radius = 1. if kwargs.get("radius") is None else kwargs["radius"]
            height = 2. if kwargs.get("height") is None else kwargs["height"]
            
            disk = cls.Default(shape='DISK', segms=segms, radius=radius, fans=True)
            
            verts = np.reshape(disk.verts, (segms+1, 3))
            verts[-1, 2] = height
            
            faces = disk.faces
            faces.append(np.arange(segms))
            
            uvs  = Faces.uvfans(segms, rect=(.01, .01, .49, .49))
            uvs0 = Faces.uvpolygon(count=segms, rect=(.51, .01, .99, .49))
            uvs  = np.append(uvs, uvs0, axis=0)

        elif shape == 'TORUS':
            
            maj_segms   = 48  if kwargs.get("major_segments")  is None else kwargs["major_segments"]
            min_segms   = 12  if kwargs.get("minor_segments")  is None else kwargs["minor_segments"]
            maj_radius  = 1.  if kwargs.get("major_radius")    is None else kwargs["major_radius"]
            min_radius  = .25 if kwargs.get("minor_radius")    is None else kwargs["minor_radius"]
            
            
            base = np.zeros((min_segms, 3), float)
            base[:, 0], base[:, 2] = cls.polygon(min_segms, radius=min_radius, start_angle=0, inverse=True)
            base[:, 0] -= maj_radius
             
            
            cs, sn = cls.polygon(maj_segms, radius=maj_radius, start_angle=0)
            M = maj_radius*np.resize(np.identity(3, float), (maj_segms, 3, 3))
            
            M[:, 0, 0] = cs
            M[:, 0, 1] = -sn
            M[:, 1, 0] = sn
            M[:, 1, 1] = cs
            
            verts = np.matmul(M, base.T).transpose((2, 0, 1))
            faces = Faces.Grid(maj_segms, min_segms, x_close=True, y_close=True)
            uvs   = Faces.uvgrid(maj_segms, min_segms)
            
            
        else:
            return None
        
        # ----------------------------------------------------------------------------------------------------
        # Create the geometry
        
        verts = np.reshape(verts, (np.size(verts)//3, 3))
        
        geo = cls.MeshFromData(verts, faces)
        if uvs is not None:
            geo.uvmaps["UVMap"] = uvs
        #geo.check()
        
        # ----- Create the shapes
        
        if shapes > 1:
            verts = np.resize(geo.verts, (1, shapes, geo.part_size, 3))
            for i in range(1, shapes):
                verts[0, i, :, 0] = verts[0, i-1, :, 0]*0.9
                verts[0, i, :, 1] = verts[0, i-1, :, 1]*1.1
                verts[0, i, :, 2] = verts[0, i-1, :, 2]*2
                
            geo.verts = verts
        
        # ----- Origin
        
        O = np.zeros(3, float)
        O[:] = origin
        geo.verts += O
            
        # ----- Create the parts
        
        if parts > 1:
            geo = cls.Array(geo, count=parts)
            
            for i in range(parts):
                geo.verts[i, :, :, 0] += 2*i
            
        # ----- The materials
        
        geo.ensure_mat_i()
        if materials:
            for i in range(len(geo.mat_i)):
                if i < 5:
                    geo.mats_.append(f"Material {i}")
                geo.mat_i[i] = i % 5
        
        # ----- return the Geometry
        
        return geo

    
    # ---------------------------------------------------------------------------
    # A head of and arrow
    
    @classmethod
    def ArrowHead(cls, shaft_radius=0.04, head_radius=.14, height=0.2, recess=.3, segms=16, z=0., orient='Z', uv_rect=(0, 0, 1, 1)):
        
        # ----- Height is null : only two circles
        
        if height < 0.001:
            zs = np.zeros(2, float)
            rs = np.array([shaft_radius, head_radius])
            
        # ----- Height is not Null
        
        else:
            
            # ----- Vertical increment for external cone
            # Slope is z --> radius
            
            dz = min(.01, height/3)
            outer_slope = head_radius / height
            
            # ----- A single cone when the head radius is smaller than
            # the shaft radius
            
            spike = head_radius <= shaft_radius
            
            # ----- Spike: no recess, only one intermédiary circle 
            
            if spike:
                head_radius = shaft_radius
                outer_slope = head_radius / height
                
                head_z = 0
                zs = np.array([0.])
                rs = np.array([shaft_radius])
                
            # ----- Head with a recess
                
            else:
                
                # ----- Check that the recess doesn't make the shaft
                # pass through the external cone
                
                min_head_z = -height + shaft_radius / outer_slope
                head_z     = np.clip(-recess*height, min_head_z, 0.)
                
                # ----- Radius increment for intermediary circles
                # CAUTION: slope is radius --> z
                
                dr = min(.01, (head_radius - shaft_radius)/3)
                inner_slope = -head_z / (head_radius - shaft_radius)
                
                zs = [0.,           -inner_slope*dr,   head_z + inner_slope*dr, head_z     ]
                rs = [shaft_radius, shaft_radius + dr, head_radius - dr,        head_radius]
                
                
        if False:
            for t in np.linspace(dz, height-dz, 6):
                zs = np.append(zs, head_z + t)
                rs = np.append(rs, head_radius - t*outer_slope)
        else:
            zs = np.append(zs, (head_z + dz,                  head_z + height - dz))
            rs = np.append(rs, (head_radius - dz*outer_slope, dz*outer_slope))
            
        # ---------------------------------------------------------------------------
        # Let's build the head
        
        n = len(zs)
        verts = np.zeros((n, segms, 3), float)
        verts[..., 0], verts[..., 1] = cls.polygon(segms, radius=1.)
        verts[..., 2] = np.reshape(zs, (n, 1))
        verts[..., :2] *= np.reshape(rs, (n, 1, 1))
        
        # ----- Top point
        
        verts = np.append(np.reshape(verts, (n*segms, 3)), [[0, 0, head_z + height]], axis=0)

        # ----- Faces
        
        faces = Faces.Grid(segms, len(rs), x_close=True, y_close=False)
        faces.extend(Faces.Triangles(len(verts)-1, np.arange(segms)+(n-1)*segms, close=True))
        
        # ----- Uvs

        itv = np.zeros(len(zs)+1, float)
        itv[1:] = zs
        
        uvs = Faces.uvgrid(len(rs), itv[1:] - itv[:-1], uv_rect)
        
        # ---- The geometry
        
        geo = cls.MeshFromData(verts, faces)
        geo.uvmaps["UVMap"] = uvs
        geo.ensure_mat_i()
        
        return geo    
    
    # ---------------------------------------------------------------------------
    # An arrow
    #
    # - Base
    # - Shaft length
    # - head base
    # - head top = length
    
    @classmethod
    def Arrow(cls, length=1., radius=0.04, segms=16, head_height=0.3, head_angle=25., shapes=True):
        
        # ----- Animated arrow
        
        if shapes:
            
            def get_arrow(lg):
                return cls.Arrow(length=lg, radius=radius, segms=segms, head_height=head_height, head_angle=head_angle, shapes=False)
            
            arrow = get_arrow(length)
            
            for i, lg in enumerate([0, 2*length, 10*length]):
                
                ar = get_arrow(lg)
                index = arrow.add_shape()
                
                if i == 0:
                    index = 0
                    
                arrow.verts[:, index] = ar.verts[0, 0]
                    
            return arrow
                    
        # ----- Not animated arrow
        
        dz = min(.01, length/3)
        zs = np.array([0., 0., dz, length-dz])
        
        def f(t, der=0):
            
            if not hasattr(t, '__len__'):
                return f([t], der)[0]
            
            v = np.zeros((len(t), 3), float)
            
            if der == 0:
                v[:, 2] = zs[np.round(t).astype(int)]
                
            elif der == 1:
                v[:, 2] = 1
                
            return v
        
        # ----- The shaft
        
        radius = max(0.001, radius)
        dr = 0.0001
            
        shaft = cls.CurveToMesh(f, profile=radius, segms=segms, caps=False, t0=0, t1=3, count=4)
        # Inset of the base
        shaft.verts[0, 0, :segms, :2] *= 1 - dr/radius
        
        # ----- The arrow head
        
        angle = np.radians(np.clip(head_angle, 5, 89))
        head_radius = np.tan(angle)*head_height

        head  = cls.ArrowHead(shaft_radius=radius, head_radius=head_radius, height=head_height, segms=segms)
        head.verts[..., 2] += length
        
        # ----- Put the head on to the shaft with a seam
        
        ring = np.arange(segms)
        seam0 = ring + (len(zs)-1)*segms
        shaft.join(head, as_part=False, seam_from=seam0, seam_to=ring, seam_close=True)
        
        # The base cap
        base_face = shaft.faces.append(np.flip(ring))

        # ----- Cylinder uv on the shaft
        
        shaft.uvmaps["UVMap"] = Faces.uvgrid(segms, Faces.loc_to_delta(np.append(zs, length)))
        
        # ----- The material indices
        
        shaft_faces = segms*(len(zs))
        
        shaft.ensure_mat_i()
        
        # Arrow head
        shaft.mat_i[shaft_faces:] = 1
        
        # Base cap
        shaft.mat_i[base_face:]   = 2
        shaft.mat_i[:segms]       = 2
        
        shaft.mat_names = ['Arrow Shaft', 'Arrow head', 'Arrow cap']
        
        return shaft
        
    
    # ---------------------------------------------------------------------------
    # A cylindric shape around a backbone
    
    @classmethod
    def CurveToMesh(cls, f, profile=.1, segms=12, caps=True, t0=0, t1=1, count=100, uv_start=0, uv_end=1, scale=1, twist=None):
        
        # ----- The call arguments for the points
        
        ts = np.linspace(t0, t1, count, endpoint=True)
        
        # ---------------------------------------------------------------------------
        # Scale
        
        check_scales = False
        if hasattr(scale, '__call__'):
            scales = scale(ts)
            check_scales = True
            
        elif hasattr(scale, '__len__'):
            scales = Polynoms(np.linspace(t0, t1, len(scale), endpoint=True), scale)(ts)
            check_scales = True
            
        else:
            scales = scale
            
        if check_scales:
            if len(np.shape(scale)) == 1:
                scales = np.expand_dims(scales, axis=-1)
                
        # ---------------------------------------------------------------------------
        # Profile to use
        # - either the points to use
        # - or the radius of a circular profile
        
        if hasattr(profile, '__len__'):
            prof = np.zeros((len(profile), 3), float)
            prof[:, 0] = np.array(profile)[:, 0]
            prof[:, 1] = np.array(profile)[:, 1]
        else:
            n = segms
            ags = np.linspace(0, 2*np.pi, n, endpoint=False)
            prof = np.zeros((n, 3), float)
            prof[:, 0] = profile*np.cos(ags)
            prof[:, 1] = profile*np.sin(ags)
            
        # ---------------------------------------------------------------------------
        # The tangents
        
        try:
            ders = f(ts, der=1)
        except:
            dt = (t1-t0)/(10*count)
            ders = (f(ts+dt) - f(ts-dt))/2/dt
            
        # ---------------------------------------------------------------------------
        # Transformations matrices
        
        tm = Transformations(location=f(ts), scale=scales, count=count)
        
        # ----- Orientation
        
        tm.orient(ders, axis='Z', no_up=False)
        
        # ---------------------------------------------------------------------------
        # Twist
        
        if twist is not None:
            
            if hasattr(twist, '__call__'):
                twist_func = twist
            elif hasattr(twist, '__len__'):
                twist_func = Polynoms(np.linspace(t0, t1, len(twist), endpoint=True), twist)
            else:
                twist_func = twist

            tm.rotate(ders, twist_func(ts), center=f(ts))
            
        # ---------------------------------------------------------------------------
        # Let's create the geometry
        
        verts = np.reshape(tm.transform(prof), (count*len(prof), 3))
        faces = Faces.Grid(len(prof), count, x_close=True, y_close=False)
        if caps:
            faces.append(np.flip(np.arange(len(prof))))
            faces.append(np.arange(len(prof)) + (count-1)*len(prof))
        
        # ----- The uvs are based on the distances between the points of the profile
        
        pts = np.append(prof, [prof[0]], axis=0)
        dx  = np.linalg.norm(pts[1:] - pts[:-1], axis=-1) 
        
        uvs   = Faces.uvgrid(dx, count-1, (0, uv_start, 1, uv_end))
        
        geo = cls.MeshFromData(verts, faces)
        geo.uvmaps["UVMap"] = uvs
        geo.ensure_mat_i()
        if caps:
            geo.mat_i[-2:] = 1
        
        return geo
    
    # ---------------------------------------------------------------------------
    # Check
    
    def check(self):
        max_ind = np.max(self.faces.values)
        if max_ind >= self.true_verts_count:
            print(f"VertParts check error: max index of faces {max_ind} is greater that the number of vertices {self.true_verts_count}.")
            self.dump()
            raise RuntimeError("Check error")
    
    # ---------------------------------------------------------------------------
    # Tests
    
    @classmethod
    def Test(cls, num=0, with_shapes=False, name='Test'):
        
        O0 = [0, 0, num*1.5]
        O1 = [0, 2, num*1.5]
        
        shapes = 4 if with_shapes else 1
        
        if num == 0:
            test_name = "A single cube"
            geo = cls.Default('CUBE', shapes=shapes)
            
        elif num == 1:
            test_name = "Array of cubes"
            geo = cls.Default('CUBE', parts=5, shapes=shapes, origin=O0)
            
        elif num == 2:
            test_name = "Join a pyramid to a cube in a single part"
            geo = cls.Default('CUBE', shapes=shapes, origin=O0)
            geo.join(cls.Default('PYRAMID', shapes=shapes, origin=O1))
            
        elif num == 3:
            test_name = "Join a pyramid to a cube as a new part"
            geo = cls.Default('CUBE', shapes=shapes, origin=O0)
            geo.join(cls.Default('PYRAMID', shapes=shapes, origin=O1), as_part=True)
            
        elif num == 4:
            test_name = "Join a pyramid to an array of cubes as last part extension"
            geo = cls.Default('CUBE', shapes=shapes, parts=5, origin=O0)
            geo.join(cls.Default('PYRAMID', shapes=shapes, origin=O1))
            
        elif num == 5:
            test_name = "Join a pyramid to an array of cubes as new part"
            geo = cls.Default('CUBE', shapes=shapes, parts=5, origin=O0)
            geo.join(cls.Default('PYRAMID', shapes=shapes, origin=O1), as_part=True)
            
        elif num == 6:
            test_name = "Join an array of cubes to a pyramid as first part extension"
            geo = cls.Default('PYRAMID', shapes=shapes, origin=O1)            
            geo.join(cls.Default('CUBE', shapes=shapes, parts=5, origin=O0))
            
        elif num == 6:
            test_name = "Join an array of cubes to a pyramid as new parts"
            geo = cls.Default('PYRAMID', shapes=shapes, origin=O1)            
            geo.join(cls.Default('CUBE', shapes=shapes, parts=5, origin=O0), as_part=True)
            
        elif num == 7:
            test_name = "Join an array of cubes to an array of pyramid"
            geo = cls.Default('PYRAMID', shapes=shapes, parts=5, origin=O1)            
            geo.join(cls.Default('CUBE', shapes=shapes, parts=3, origin=O0))
            
        elif num == 8:
            test_name = "Array of arrays"
            base = cls.Default('CUBE', shapes=shapes, parts=1, origin=O1)            
            base.join(cls.Default('PYRAMID', shapes=shapes, parts=1, origin=O0), as_part=True)
            count = 7
            geo = cls.Array(base, count=count)
            for p in range(count):
                geo.verts[[2*p, 2*p+1], ..., 0] += 2*p
            
        else:
            return False
        
        print()
        print(f"Test {num:3d} - {test_name} {'with shapes' if with_shapes else ''}")
        print(geo)
        
        if with_shapes:
            shape = np.linspace(0, geo.shapes_count+1, geo.parts_count)
        else:
            shape = 0
            
        geo.set_to(f"{name} {num}", shape=shape)
        
        return True
    
    
    # ---------------------------------------------------------------------------
    # Dump the content
                
    def dump(self, title="Geometry dump"):
        
        def strv(vect):
            sv = "["
            sep = ""
            for v in vect:
                sv += f"{v:5.1f}{sep}"
                sep = ", "
            sv += "]"
            return sv
        
        def strs(p_i, v_i):
            s = ""
            for s_i in range(self.shapes_count):
                s += " " + strv(self.verts[p_i, s_i, v_i])
            return s
        
        print('-'*50)
        print(f"{title}\n")
        v_index = 0
        v_num   = 0
        print("===== Vertices")
        for p_i in range(self.parts_count):
            if p_i < 5 or p_i >= self.parts_count-5:
                print("Part", p_i)
                for v_i in range(self.part_size):
                    leave = False
                    if v_i < self.sizes[p_i]:
                        sok = "*"
                        snum = f"({v_num:3d})"
                        v_num += 1
                    else:
                        sok = " "
                        snum = "     "
                        leave = True
                        
                    print(f"   {sok} {v_index:3d}{snum}: {v_i:2d} {strs(p_i, v_i)}")
                    
                    v_index += 1
                    if leave:
                        break
                    
                print()
                    
        print()
        print("true_indices:", WRoot._str_indices(self.true_indices))
        print()
        
        if self.type == 'Mesh':
            print("===== Faces")
            self.faces.dump()
            print()
            
        else:
            print('===== Profile')
            print(self.profile)
            print()
            
        print("===== Materials")
        for i, name in enumerate(self.mat_names):
            print(f"{i:2d}: {name}")
        print()
        
        print(self.mat_i)
        print()
        
    # ----------------------------------------------------------------------------------------------------
    # Representation
        
    def __repr__(self):
        s = f"<Geometry of type {self.type} with:\n"
        s += f"   parts count:  {self.parts_count:4d} part(s)\n"
        s += f"   shapes count: {self.shapes_count:4d} shape(s)\n"
        s += f"   part size:    {self.part_size:4d} vertices; {self.sizes[:10]}{'...' if len(self.sizes) > 10 else ''}\n"
        s += f"   vertices:     {self.true_verts_count:4d} true on {self.verts_count} stored"
        if self.verts_count > 0:
            s += f", ratio = {self.true_verts_count/self.verts_count*100:.1f}%\n"
        else:
            s += "\n"
            
        s +=  "   true indices: "
        if self.true_indices is None:
            s += "None\n"
        else:
            #s += f"{self.true_indices[:50]}{'...' if len(self.true_indices)>50 else ''}\n"
            s += f"{WRoot._str_indices(self.true_indices)}\n"
            
        if self.type == 'Mesh':
            s += f"   faces:        {self.faces}\n"
        else:
            s += f"   profile:      {self.profile}\n"
        
        return s + ">"
    
    # ----------------------------------------------------------------------------------------------------
    # Dimensions
    
    @property
    def parts_count(self):
        return self.verts.shape[0]
    
    @property
    def part_size(self):
        return self.verts.shape[2]
    
    @property
    def shapes_count(self):
        return self.verts.shape[1]
    
    @property
    def verts_count(self):
        return self.verts.shape[0]*self.verts.shape[2]
        
    @property
    def true_verts_count(self):
        if self.true_indices is None:
            return self.verts.shape[0]*self.verts.shape[2]
        else:
            return len(self.true_indices)
        
    def update_true_indices(self):
        
        if np.min(self.sizes) == self.part_size:
            self.true_indices = None
        else:
            self.true_indices = np.empty(sum(self.sizes), int)
            offset = 0
            for i, size in enumerate(self.sizes):
                self.true_indices[offset:offset+size] = np.arange(size) + i*self.part_size
                offset += size
                
    @property
    def same_sizes(self):
        return self.true_indices is None
    
    # ---------------------------------------------------------------------------
    # Shapes
    
    def add_shape(self, from_shape=0):
        
        index = self.shapes_count
        
        shape = self.verts.shape
        new_shape = (shape[0], shape[1] + 1, shape[2], shape[3])
        self.verts = np.resize(self.verts, new_shape)
        
        self.verts[:, index] = self.verts[:, from_shape]
        
        return index
    
    # ----------------------------------------------------------------------------------------------------
    # Parts centers
    
    def centers(self, shape=0):
        
        if self.true_indices is None:
            return np.average(self.verts[:, shape], axis=1)
        
        centers = np.empty((len(self.sizes), 3), float)
        for i in range(self.parts_count):
            centers[i] = np.average(self.verts[i, shape, :self.sizes[i]], axis=0)
            
        return centers
    
    # ----------------------------------------------------------------------------------------------------
    # Translate vertices
    
    def translate(self, vector, shape=None):
        v = np.empty((self.parts_count, 3), float)
        v[:] = vector

        if shape is None:
            self.verts += np.reshape(v, (self.parts_count, 1, 1, 3))
        else:
            self.verts[:, shape] += np.reshape(v, (self.parts_count, 1, 3))
    
                
    # ----------------------------------------------------------------------------------------------------
    # Extract a part
    
    def get_part_info(self, part_index, shape_index=0):
        
        pi = {"type": self.type}
        
        pi["size"]  = self.sizes[part_index]
        pi["verts"] = np.array(self.verts[part_index, shape_index])

        offset = np.sum(self.sizes[:part_index])
        size   =self.sizes[part_index]
        pi["verts_indices"] = np.arange(size) + offset
        
        # ---------------------------------------------------------------------------
        # Mesh
        
        if self.type == 'Mesh':
            
            faces_i = self.faces.faces_of(np.arange(offset, offset+size))
            
            pi["faces_indices"] = faces_i
            pi["faces"]         = self.faces[faces_i]
            pi["mats_indices"]  = np.array(self.mat_i[faces_i])
            
            uvmaps = {}
            for name, uvs in self.uvmaps.items():
                uvmaps[name] = np.array(uvs[faces_i])
                
            pi["uvmaps"] = uvmaps
            
        # ---------------------------------------------------------------------------
        # Curve
            
        elif self.type == 'Curve':
            
            profile = []
            p_ofs  = 0
            for p_i, in range(len(self.profile)):
                if p_ofs >= offset + size:
                    break
                if p_ofs >= offset:
                    profile.append(self.profile[p_i])
                p_ofs += self.profile[p_i, 1]
                
            pi["profile"]      = np.array(profile, int)
            pi["mats_indices"] = np.array(self.mat_i[pi["profile"]])
            
        # ---------------------------------------------------------------------------
        # Return the result
        
        return pi
                
    
    # ----------------------------------------------------------------------------------------------------
    # Materials are defined
    
    @property
    def max_mat_i(self):
        return np.max(self.mat_i)
    
    @property
    def mat_names(self):
        mats = list(self.mats_)
        n = len(mats)
        imax = np.max(self.mat_i)
        if (imax > 0) and (n < imax+1):
            for i in range(n, imax+1):
                mats.append("Material")
                #mats.append(f"Material {i}")
        return mats
    
    @mat_names.setter
    def mat_names(self, value):

        n = len(self.mats_)
        
        for i in range(n):
            self.mats_[i] = value[i]
            
        for i in range(n, len(value)):
            self.mats_.append(value[i])
    
    def ensure_mat_i(self):
        if self.type == 'Mesh':
            target = len(self.faces)
        else:
            target = len(self.profile)
            
        if len(self.mat_i) < target:
            self.mat_i = np.append(self.mat_i, np.zeros(target - len(self.mat_i), int))
            
            
    # ----------------------------------------------------------------------------------------------------
    # Array
    
    @classmethod
    def Array(cls, model, count=1):
        
        count = max(1, count)
        
        geo = cls(type=model.type)
        
        # ---------------------------------------------------------------------------
        # Vertices

        geo.verts = np.resize(model.verts, (count*model.parts_count,) + model.verts.shape[1:])
        geo.sizes = np.resize(model.sizes, count*model.parts_count)
        
        if model.true_indices is not None:
            ti_count = len(model.true_indices)
            ti_size  = model.parts_count * model.part_size
            geo.true_indices = (np.resize(np.arange(count)*ti_size, (ti_count, count)).T + model.true_indices).reshape(count*ti_count)
            
        # ---------------------------------------------------------------------------
        # Materials

        geo.mat_names = model.mat_names
        geo.mat_i     = np.resize(model.mat_i, (count*len(model.mat_i),))
        
        # ---------------------------------------------------------------------------
        # Mesh
        
        if model.type == 'Mesh':

            # ----- Faces
            
            geo.faces = model.faces.array(count, verts_count = model.true_verts_count)
                
            # ----- uv maps
            
            geo.uvmaps = {}
            for name, uvs in model.uvmaps.items():
                geo.uvmaps[name] = np.resize(uvs, (count*len(uvs), 2))
                
            # ----- vertex groups
            
            geo.groups = model.groups.clone()
            geo.groups.array(count, model.true_verts_count)
                
        # ---------------------------------------------------------------------------
        # Curve
                
        elif model.type == 'Curve':
            
            # ----- Profile
            
            geo.profile = np.resize(model.profile, (count*len(model.profile), 3))

            # ----- Properties
            
            if hasattr(model, 'curve_properties'):
                geo.curve_properties = model.curve_properties
                
            if hasattr(model, 'splines_properties'):
                geo.splines_properties = [None] * (count*len(model.splines_properties))
                offset = 0 
                for i in range(count):
                    for j, props in enumerate(model.splines_properties):
                        geo.splines_properties[offset + j] = clone(props)
                    offset += len(model.splines_properties)
                    
        # ----- Return the geometry
        
        return geo
    
    # ----------------------------------------------------------------------------------------------------
    # Change the vertices
    
    def set_verts(self, verts, part=0, shape=0):
        self.verts[part, shape] = verts
    
    # ----------------------------------------------------------------------------------------------------
    # Join faces
    
    def join_faces(self, other, vcount, seam_from=None, seam_to=None, seam_close=False):
        
        if (seam_from is not None) and (seam_to is not None):
            self.faces.extend(Faces.Stripe(seam_from, np.array(seam_to) + vcount, seam_close))
            
        self.faces.join(other.faces, verts_count = vcount)
        
    # ----------------------------------------------------------------------------------------------------
    # Join materials
    
    def join_materials(self, other):
        
        # Map the other material indices to the new materials list
        # Append new materials when required
        
        mat_inds = []
        for i, name in enumerate(other.mats_):
            try:
                index = self.mats_.index(name)
                mat_inds.append(index) 
            except:
                mat_inds.append(len(self.mats_))
                self.mats_.append(name)
                
        # Append the new indices
        if len(mat_inds) > 0:
            self.mat_i = np.append(self.mat_i, np.array(mat_inds)[other.mat_i])
        else:
            self.mat_i = np.append(self.mat_i, other.mat_i)
            
    # ----------------------------------------------------------------------------------------------------
    # Join uv maps
            
    def join_uvmaps(self, other):
        
        new_uvs = {}
        for name, uvs in enumerate(self.uvmaps):
            o_uvs = other.uvmaps.get(name)
            if o_uvs is not None:
                new_uvs[name] = np.append(uvs, o_uvs, axis=0)
        
        del self.uvmaps
        self.uvmaps = new_uvs
        
    # ----------------------------------------------------------------------------------------------------
    # Join curve
        
    def join_curve(self, other):
        
        # ----- Profile
        
        self.profile = np.append(self.profile, other.profile, axis=0)
        
        # ----- Properties
        
        if not hasattr(self, 'curve_properties'):
            if hasattr(other, 'curve_properties'):
                self.curve_properties = other.curve_properties
            
        if hasattr(self, 'splines_properties'):
            if hasattr(other, 'splines_properties'):
                self.splines_properties.extend([clone(props) for props in other.splines_properties])
            else:
                del self.splines_properties
    
    
    # ----------------------------------------------------------------------------------------------------
    # Join another geometry
    
    def join(self, other, as_part=False, seam_from=None, seam_to=None, seam_close=False):
        
        # ---------------------------------------------------------------------------
        # Vertices
        
        vcount = self.true_verts_count
        
        as_part = as_part or ( (self.parts_count > 1) and (other.parts_count > 1) )
        if as_part:
            parts_count = self.parts_count + other.parts_count
            part_size   = max(self.part_size, other.part_size)
        else:
            parts_count = max(self.parts_count, other.parts_count)

            # -------------------------
            # other_to_last if other is joined to the last part of self
            # otherwise, self is joined before the first part of other
            
            other_to_last = other.parts_count == 1
            part_size     = self.part_size + other.part_size
            
        shapes_count = max(self.shapes_count, other.shapes_count)
        
        # ----- The new vertices array
        
        verts = np.zeros((parts_count, shapes_count, part_size, 3), float)
        
        # --------------------------------------------------
        # Append as a new part
        
        if as_part:
            
            verts[:self.parts_count, :self.shapes_count,  :self.part_size]  = self.verts
            verts[self.parts_count:, :other.shapes_count, :other.part_size] = other.verts
            
            self.sizes = np.append(self.sizes, other.sizes)
            
        # --------------------------------------------------
        # Join the other single part to the last part of self
        
        elif other_to_last:

            verts[:, :self.shapes_count,  :self.part_size]  = self.verts
            verts[-1, :other.shapes_count, self.part_size:] = other.verts[0]
            
            self.sizes[-1] += other.verts_count
            
        # --------------------------------------------------
        # Join the self single part to the first part of other
        
        else:
            
            verts[1:, :other.shapes_count, :other.part_size] = other.verts[1:]
            verts[0,  :other.shapes_count, self.part_size:]  = other.verts[0]
            verts[0,  :self.shapes_count,  :self.part_size]  = self.verts[0]
            
            self.sizes = np.array(other.sizes)
            self.sizes[0] += self.verts_count
            
        # ----- Let's update the vertices
            
        del self.verts
        self.verts = verts
            
        self.update_true_indices()

        # ---------------------------------------------------------------------------
        # Materials
        
        self.join_materials(other)
        
        # ---------------------------------------------------------------------------
        # Mesh
        
        if self.type == 'Mesh':
            
            # ----- Faces
            
            self.join_faces(other, vcount, seam_from=seam_from, seam_to=seam_to, seam_close=seam_close)
            
            # ----- uv maps
            
            self.join_uvmaps(other)
            
            # ----- Vertex groups
            
            self.groups.join(other.groups, vcount)
            
        # ----- Curve
                
        elif self.type == 'Curve':
            
            self.join_curve(other)
                    
        # ----- Ensure material indices are consistent
        
        self.ensure_mat_i()
        
        # ----- Enable chaining
        
        return self
    
    # ----------------------------------------------------------------------------------------------------
    # Extract faces without controls
    # Security must be ensured by the caller
    
    def quick_extract(self, i_faces):
        
        faces, i_verts = self.faces[i_faces].unique()
        
        uvmaps = {}
        for name, uvs in self.uvmaps.items():
            vas = VArrays.FromArrays(self.faces.sizes, uvs, copy=False)
            uvmaps[name] = vas[i_faces].values
        
        geo = self.MeshFromData(self.verts[..., i_verts, :], faces, uvmaps=uvmaps)
        geo.mat_i = self.mat_i[i_faces]
        
        return geo
    
    # ----------------------------------------------------------------------------------------------------
    # Extract a part
    
    def extract(self, faces_indices):
        if self.type != 'Mesh':
            raise WError("Explode can be called only for mesh geometries.",
                         Class = "VertParts", Method="extract", faces_indices=faces_indices)
            
        if self.parts_count > 1:
            raise WError(f"Explode can be called only for 1 part geometries, not {self.parts_count}.",
                         Class = "VertParts", Method="extract", faces_indices=faces_indices)
            
        i_faces = np.array(faces_indices)
        
        return self.quick_extract(np.reshape(i_faces, i_faces.size))

    # ----------------------------------------------------------------------------------------------------
    # Explode each face as individual part
    # NOTE : explode on the first part
    
    def explode(self, groups=None):
        if self.type != 'Mesh':
            raise WError("Explode can be called only for mesh geometries.",
                         Class = "VertParts", Method="Explode", groups=groups)
            
        if self.parts_count > 1:
            raise WError(f"Explode can be called only for 1 part geometries, not {self.parts_count}.",
                         Class = "VertParts", Method="explode", groups=groups)
            
        if groups is None:
            groups = np.arange(len(self.faces))
            
        geo = type(self)(type='Mesh')
        geo.mat_names = self.mat_names
        
        for group in range(np.max(groups)+1):
            
            i_faces = np.where(groups==group)[0]
            
            if len(i_faces) != 0:
                
                geo.join(self.quick_extract(i_faces), as_part=True)
                
        return geo
    
    
    # ----------------------------------------------------------------------------------------------------
    # Initialize parts from an array of geometries 
    
    @classmethod
    def Parts(cls, geometries):
        
        # ---------------------------------------------------------------------------
        # Dimensions
        
        parts_count = 0
        true_inds   = False
        part_size   = 0
        part_min    = geometries[0].part_size
        shapes      = 1
        for i, g in enumerate(geometries):
            parts_count += g.parts_count
            part_size = max(part_size, g.part_size)
            part_min  = min(part_min, g.part_size)
            shapes    = max(shapes, g.shapes_count)
            if g.true_indices is not None:
                true_inds = True
                
        if part_min != part_size:
            true_inds = True

        # ---------------------------------------------------------------------------
        # Vertices

        geo = cls(type=geometries[0].type)
            
        geo.verts = np.zeros((parts_count, shapes, part_size, 3))
        
        # ---------------------------------------------------------------------------
        # Loop on the geometries
        
        if true_inds:
            geo.true_indices = np.zeros((0,), int)
        
        parts_ofs = 0
        vcount    = 0
        for g_index, g in enumerate(geometries):
            
            # ----- Vertices
            
            geo.verts[parts_ofs:parts_ofs + g.parts_count, :g.shapes_count, :g.part_size] = g.verts
            geo.sizes = np.append(geo.sizes, g.sizes)
            
            if true_inds:
                if g.true_indices is None:
                    geo.true_indices = np.append(geo.true_indices, np.arange(g.verts_count) + parts_ofs*part_size)
                else:
                    geo.true_indices = np.append(geo.true_indices, g.true_indices + parts_ofs*part_size)

            parts_ofs += g.parts_count
            
            # ----- Materials
            
            geo.join_materials(g)
            
            # ----- Mesh
            
            if geo.type == 'Mesh':
                
                print("Parts init", g_index, vcount )
                
                # ----- Faces
                
                geo.join_faces(g, vcount)
                
                # ----- uv maps
                
                geo.join_uvmaps(g)
                
                # ----- Vertex groups
                
                geo.groups.join(g.groups, vcount)
                
            # ----- Curve
            
            elif geo.type == 'Curve':
                
                geo.join_curve(g)
                
            # ----- Update counters
                
            vcount += g.true_verts_count
                
            
        # ---------------------------------------------------------------------------
        # Final updates and return
            
        geo.ensure_mat_i()
        
        return geo
    
    # ----------------------------------------------------------------------------------------------------
    # Faces centers
    
    def faces_centers(self, shape=0):
        return self.faces.centers(self.verts[:, shape].reshape(self.parts_count*self.part_size, 3))

    # ----------------------------------------------------------------------------------------------------
    # Faces surfaces
    
    def faces_surfaces(self, shape=0):
        return self.faces.surfaces(self.verts[:, shape].reshape(self.parts_count*self.part_size, 3))

    # ----------------------------------------------------------------------------------------------------
    # Faces surfaces
    
    def faces_normals(self, shape=0):
        return self.faces.normals(self.verts[:, shape].reshape(self.parts_count*self.part_size, 3))

    # ----------------------------------------------------------------------------------------------------
    # Compute vertices based on shape transformation values
    # The computation can be relative or not
    # - relative : weigthed average of the shapes
    #              array shape of transformation is (n, shapes_count)
    # - absolute : continuous transformation between the shapes
    #              array shape of transformation is (n,)
    #
    # CAUTION: do not confuse the shapes of the matrices with the shapes of the vertices :-)
    
    def shaped_verts(self, shape=None, shapes=None, relative=False, extrapolation='CLIP'):
        
        if self.parts_count == 0:
            return None
        
        # ----- The shape of the resulting shape
        
        if shape is None:
            count = 1
            shape = (self.parts_count,)
        else:
            if hasattr(shape, '__len__'):
                shape = tuple(shape)
            else:
                shape = (shape,)
                
            size = int(np.product(shape))
            if size % self.parts_count != 0:
                raise WError(f"Impossible to shape {self.parts_count} parts into shape {shape}. " +
                             " The number of requested parts {size} must be a multiple of the number of geometry parts {self.parts_count}.",
                             Class = "Geometry",
                             Method = "shaped_verts"
                             )
            
            count = size //self.parts_count
        
        res_shape = shape + (self.part_size, 3)
        
        # ----- No shapes requested: the easiest way
        if (shapes is None) or (self.shapes_count == 1):
            return np.resize(self.verts[:, 0], res_shape)
        
        # ----- Total number of parts
        parts_total = count * self.parts_count
        
        # ----- Relative: weighted average of the shapes
        
        if relative:
            
            # Vertices initialized with the first shape
            verts = np.resize(self.verts[:, 0], (count, self.parts_count, self.part_size, 3))
            
            # Factors for the remaining shapes
            factors = np.zeros(shape + (self.shapes_count-1,))
            factors[:] = shapes
            factors = np.reshape(factors, (count, self.parts_count, self.shapes_count-1, 1, 1))
            
            # Loop on the shapes
            for s_i in range(0, self.shapes_count-1):
                verts += np.resize(self.verts[:, s_i+1] - self.verts[:, 0], (count, self.parts_count, self.part_size, 3))*factors[:, :, s_i]
            
            return np.reshape(verts, res_shape)            
            
        # ----- Absolute: interpolations between shapes
                                   
        else:
            factors    = np.zeros(shape, float)
            factors[:] = shapes
            factors    = np.reshape(factors, (count, self.parts_count))
            
            if extrapolation == 'CLIP':
                factors = np.clip(factors, 0, self.shapes_count-1)
                
            elif extrapolation == 'LOOP':
                factors %= self.shapes_count-1
                
            elif extrapolation == 'BACK':
                factors %= 2*(self.shapes_count-1)
                after = factors > self.shapes_count-1
                factors[after] = 2*(self.shapes_count-1) - factors[after]
                
            else:
                raise WError(f"Unknwon extrapolation code: '{extrapolation}'. " + 
                             "valid values are: 'CLIP', 'LOOP', 'BACK'.",
                             Class = "Geometry",
                             Method = "shaped_verts")
                
            # ----- Shape indices
                
            inds = np.clip(np.floor(factors).astype(int), 0, self.shapes_count-2)
            p    = np.reshape(factors - inds, (parts_total, 1, 1))
            inds = np.reshape(inds + np.arange(self.parts_count)*self.shapes_count, parts_total)
            
            # ----- Compute the vertices
            
            verts = np.resize(self.verts, (self.parts_count*self.shapes_count, self.part_size, 3))
            
            return np.reshape((1-p)*verts[inds] + p*verts[inds + 1], res_shape)
    
    # ----------------------------------------------------------------------------------------------------
    # True vertices from shaped vertices
    
    def true_verts(self, verts):
        if self.true_indices is None:
            return np.reshape(verts, (np.size(verts)//3, 3))
        else:
            return np.reshape(verts, (np.size(verts)//3, 3))[self.true_indices]
            
    


  