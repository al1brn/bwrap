#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 18:55:52 2022

@author: alain
"""
            
import numpy as np

from ..core.commons   import WError
from ..core.vertparts import VertParts, clone
from .wrap_function   import wrap
from .wmesh           import WMesh

# ----------------------------------------------------------------------------------------------------
# The Geometry class
#
# The verts are store in an array of shape (parts_count, shapes_count, verts_count)

class Geometry(VertParts):
    
    def __init__(self, type = 'Mesh'):
        super().__init__(type=type)

    # ----------------------------------------------------------------------------------------------------
    # Initialize from a mesh
        
    @classmethod
    def FromMesh(cls, mesh):

        wmesh = WMesh(mesh)
        
        # ---------------------------------------------------------------------------
        # Vertices
        
        geo = cls(type='Mesh')
        
        vcount = wmesh.verts_count
        
        verts = wmesh.wshape_keys.verts()
        if verts is None:
            geo.verts = wmesh.verts.reshape(1, 1, vcount, 3)
        else:
            geo.verts = np.expand_dims(verts, axis=0)

        geo.sizes = np.array([vcount], int)
        
        # ---------------------------------------------------------------------------
        # Materials
        
        geo.mat_names = wmesh.wmaterials.mat_names
        geo.mat_i     = wmesh.material_indices
        
        # ---------------------------------------------------------------------------
        # Import from a Mesh
            
        geo.faces = wmesh.faces

        geo.uvmaps = {}
        for name in wmesh.uvmaps:
            geo.uvmaps[name] = wmesh.get_uvs(name)
        
        # ---------------------------------------------------------------------------
        # Done

        return geo

        
    # ----------------------------------------------------------------------------------------------------
    # Initialize from an object, Mesh or Curve
        
    @classmethod
    def FromObject(cls, object):

        wobj = wrap(object, create=False)
        
        # ---------------------------------------------------------------------------
        # Vertices
        
        geo = cls(type=wobj.object_type)
        
        vcount = wobj.wdata.verts_count
        
        verts = wobj.wdata.wshape_keys.get_verts()
        if verts is None:
            geo.verts = wobj.wdata.verts.reshape(1, 1, vcount, 3)
        else:
            geo.verts = np.expand_dims(verts, axis=0)

        geo.sizes = np.array([vcount], int)
        
        # ---------------------------------------------------------------------------
        # Materials
        
        geo.mat_names = wobj.wmaterials.mat_names
        geo.mat_i     = wobj.wdata.material_indices
        
        # ---------------------------------------------------------------------------
        # Import from a Mesh
        
        if wobj.object_type == 'Mesh':
            
            wmesh = wobj.wdata
            
            geo.faces = wmesh.faces

            geo.uvmaps = {}
            for name in wmesh.uvmaps:
                geo.uvmaps[name] = wmesh.get_uvs(name)
                
            geo.groups = wobj.vert_groups
            
        # ---------------------------------------------------------------------------
        # Import from a Curve
        
        elif wobj.object_type == 'Curve':
            
            wcurve = wobj.wdata
            
            geo.profile            = wcurve.profile
            
            geo.curve_properties   = wcurve.curve_properties
            geo.splines_properties = wcurve.splines_properties
            
        else:
            
            return None
        
        # ---------------------------------------------------------------------------
        # Done

        return geo
    
    # ----------------------------------------------------------------------------------------------------
    # Set to an object
    
    def set_to(self, object, verts=None, replace_materials=False, shape_keys=True):
        
        wobj = wrap(object, create=self.type.upper())
        
        # ----- Materials
        
        if replace_materials:
            wobj.wmaterials.mat_names = self.mat_names
        else:
            n = np.max(self.mat_i)
            if (n > 0) and (len(wobj.wmaterials) < n+1):
                wobj.wmaterials.append(self.mat_names[len(wobj.wmaterials):])
        
        # ----- Verts
        
        if verts is None:
            verts = self.true_verts(self.shaped_verts())
            
        # ----- Mesh
        
        if self.type == 'Mesh':
            
            wobj.new_geometry(verts, self.faces)
            
            wobj.material_indices = self.mat_i
            for name, uvs in self.uvmaps.items():
                wobj.create_uvmap(name)
                wobj.set_uvs(name, uvs)
                
            wobj.vert_groups = self.groups
            
        # ----- Curve
            
        elif self.type == 'Curve':
            
            wobj.wdata.profile = self.profile
            wobj.wdata.verts   = verts
            wobj.wdata.material_indices = self.mat_i

            # ----- Properties

            if hasattr(self, 'curve_properties'):
                wobj.wdata.curve_properties = self.curve_properties
                
            if hasattr(self, 'splines_properties'):
                wobj.wdata.splines_properties = self.splines_properties
                
        # ----- Shape keys
        
        if shape_keys and (self.shapes_count > 1):
            verts = np.reshape(np.transpose(self.verts, (1, 0, 2, 3)), (self.shapes_count, self.parts_count*self.part_size, 3))
            if self.true_indices is None:
                wobj.wshape_keys.set_verts(verts)
            else:
                wobj.wshape_keys.set_verts(verts[:, self.true_indices])
                
                
        return wobj

# ======================================================================================================================================================
# OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 

class OLD():    
        
    @classmethod
    def MeshFromData(cls, verts, faces):
        
        geo = cls(type = 'Mesh')
        geo.verts = np.empty((1, 1, np.size(verts)//3, 3), float)
        geo.verts[0, 0, :] = verts
        geo.faces = faces
        geo.sizes = [geo.verts.shape[2]]
        
        geo.ensure_mat_i()
        
        return geo

            
    @classmethod
    def Default(cls, shape='CUBE', parts=1, shapes=1, origin=0):
        
        geo = None
        
        if shape == 'CUBE':
            geo = cls.MeshFromData(
                ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
                [[0, 1, 2, 3], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])
        
        elif shape == 'PLANE':
            geo = cls.MeshFromData(
                ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
                [[0, 1, 2, 3]])
        
        elif shape == 'PYRAMID':
            geo = cls.MeshFromData(
                ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0.5, 0.5, 1)),
                [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
            
        if geo is None:
            return None
        
        # ----- Create the shapes
        
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
            
        # ----- return the Geometry
        
        return geo
    
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
        
        print(f"\n{title}")
        v_index = 0
        v_num   = 0
        for p_i in range(self.parts_count):
            print("Part", p_i)
            for v_i in range(self.part_size):
                if v_i < self.sizes[p_i]:
                    sok = "*"
                    snum = f"({v_num:3d})"
                    v_num += 1
                else:
                    sok = " "
                    snum = "     "
                    
                print(f"   {sok} {v_index:3d}{snum}: {v_i:2d} {strs(p_i, v_i)}")
                
                v_index += 1
                    
            print()
                    
        print()
        print("true_indices")
        print(self.true_indices)
        
        print()
        print("faces")
        for face in self.faces:
            print(face)
        
        
        
    def __repr__(self):
        s = f"<Geometry of type {self.type} with:\n"
        s += f"   parts count:  {self.parts_count:4d} part(s)\n"
        s += f"   shapes count: {self.shapes_count:4d} shape(s)\n"
        s += f"   part size:    {self.part_size:4d} vertices; {self.sizes[:10]}{'...' if len(self.sizes) > 10 else ''}\n"
        if self.verts_count > 0:
            sratio = f", ratio = {self.true_verts_count/self.verts_count*100:1f}%"
        else:
            sratio = ""
        s += f"   vertices:     {self.true_verts_count:4d} true on {self.verts_count} stored{sratio}\n"
        s +=  "   true indices: "
        if self.true_indices is None:
            s += "None\n"
        else:
            s += f"{self.true_indices[:50]}{'...' if len(self.true_indices)>50 else ''}\n"
        
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
    
    # ----------------------------------------------------------------------------------------------------
    # Materials are defined
    
    @property
    def materials_count(self):
        return len(self.materials)
    
    @property
    def has_mat_i(self):
        return len(self.mat_i) > 0
    
    def ensure_mat_i(self):
        if self.type == 'Mesh':
            target = len(self.faces)
        else:
            target = len(self.profile)
            
        if len(self.mat_i) <target:
            self.mat_i = np.append(self.mat_i, np.zeros(target - len(self.mat_i)))
    
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
        
        geo.materials = list(model.materials)
        geo.mat_i     = np.resize(model.mat_i, (count*len(model.mat_i),))

        # ---------------------------------------------------------------------------
        # Mesh
        
        if model.type == 'Mesh':
            
            # ----- Faces
            
            geo.faces = []
            offset = 0
            for i in range(count):
                geo.faces.extend([[offset + f for f in face] for face in model.faces])
                offset += model.true_verts_count
            
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
    # Join faces
    
    def join_faces(self, other, vcount):
        
        self.faces.extend([[vcount + f for f in face] for face in other.faces])
        
    # ----------------------------------------------------------------------------------------------------
    # Join materials
    
    def join_materials(self, other):
        
        # Map the other material indices to the new materials list
        # Append new materials when required
        
        mat_inds = []
        for i, name in enumerate(other.materials):
            try:
                index = self.materials.index(name)
                mat_inds.append(index) 
            except:
                mat_inds.append(len(self.materials))
                self.materials.append(name)
                
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
    
    def join(self, other, as_part=False):
        
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
            
            self.join_faces(other, vcount)
            
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
    # Compute vertices based on shape transformation values
    # The computation can be relative or not
    # - relative : weigthed average of the shapes
    #              array shape of transformation is (n, shapes_count)
    # - absolute : continuous transformation between the shapes
    #              array shape of transformation is (n,)
    #
    # CAUTION: do not confuse the shapes of the matrices with the shapes of the vertices :-)
    
    def shaped_verts(self, shape=None, shapes=None, relative=False, extrapolation='CLIP'):
        
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
    


        