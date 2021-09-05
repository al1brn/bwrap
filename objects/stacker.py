#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:28:15 2021

@author: alain
"""

import numpy as np

from ..wrappers.wrap_function import wrap

# =============================================================================================================================
# Stacker : feed the stack
# From an existing object
# From another source of geometry

class Stacker():
    
    def __init__(self, nverts):
        
        self.inst_indices = np.zeros(0, int) # instance indices
        
        # Vertices
        
        self.nverts       = nverts           # Vertices count
        self.verts_       = None
        
        # Material

        self.mat_offset   = 0        
        self.mat_count_   = 0
        self.mat_indices_ = []
        
        # Mesh
        
        self.faces_       = []
        self.uvs_size_    = 0
        self.uvs_         = {}
        
        # Curve
        
        self.profile_     = np.zeros((0, 3), int)
        
    def __repr__(self):
        return f"<Stacker: {len(self):3d} x {self.nverts:3d} = {len(self)*self.nverts:4d} vertices, '{self.name}'>"
        
    def __len__(self):
        return len(self.inst_indices)
    
    @property
    def name(self):
        return "No name"

    # ------------------------------------------------------------------------------------------
    # Mesh and curve properties
    
    @property
    def verts(self):
        return self.verts_
    
    @verts.setter
    def verts(self, value):
        self.verts_ = value
        
    @property
    def faces(self):
        return self.faces_
    
    @faces.setter
    def faces(self, value):
        self.faces_ = value

    @property
    def profile(self):
        return self.profile_
    
    @profile.setter
    def profile(self, value):
        self.profile_ = value
        
    @property
    def profile_size(self):
        return len(self.profile)
        
    @property
    def mat_count(self):
        return self.mat_count_
    
    @mat_count.setter
    def mat_count(self, value):
        self.mat_count_ = value
        
    @property
    def mat_indices(self):
        return self.mat_indices_
    
    @mat_indices.setter
    def mat_indices(self, value):
        self.mat_indices_ = value
        
    @property
    def mat_indices_count(self):
        return len(self.mat_indices)
        
    @property
    def uvs_size(self):
        return self.uvs_size_
    
    @uvs_size.setter
    def uvs_size(self, value):
        self.uvs_size_ = value
        
    @property
    def uvmaps(self):
        return list(self.uvs_.keys())
    
    @property
    def empty_uvs(self):
        return np.zeros((self.uvs_size, 2), np.float)
    
    def get_uvs(self, name):
        uvs = self.uvs_.get(name)
        #print("stacker get_uvs", name, len(uvs), [len(uv) for uv in uvs])
        if uvs is None:
            uvs = self.empty_uvs
            self.uvs_[name] = uvs
        return uvs
    
    def set_uvs(self, name, uvs):
        self.uvs_[name] = uvs
        
    # ----------------------------------------------------------------------------------------------------
    # Build the target object
    # index is the index with the property verts_indices
    
    def get_shifted_faces(self, offset):
        return [[offset + f for f in face] for face in self.faces]

    # ----------------------------------------------------------------------------------------------------
    # Specific to curves
    
    def setup_spline(self, spline_index, target_spline):
        pass


# =============================================================================================================================
# Object stacker
# From an existing object

class ObjectStacker(Stacker):
    
    def __init__(self, name):
        self.wobject = wrap(name)
        super().__init__(self.wobject.verts_count)
        
    @property
    def name(self):
        return self.wobject.name
        
    @property
    def verts(self):
        if self.wobject.object_type == 'Mesh':
            return self.wobject.verts
        else:
            return self.wobject.wdata.ext_verts
        
    @property
    def faces(self):
        return self.wobject.poly_indices
    
    @property
    def profile(self):
        return self.wobject.profile
    
    @property
    def profile_size(self):
        return len(self.wobject.wdata.splines)
    
    @property
    def mat_count(self):
        return len(self.wobject.wmaterials)
    
    @property
    def mat_indices(self):
        return self.wobject.material_indices
    
    @property
    def uvs_size(self):
        return self.wobject.uvs_size
    
    @property
    def uvmaps(self):
        return self.wobject.uvmaps
    
    def get_uvs(self, name):
        return self.wobject.get_uvs(name)
    
    def setup_spline(self, spline_index, target_spline):
        target_spline.copy_from(self.wobject.data.splines[spline_index])
    
    
# =============================================================================================================================
# A char stacker as a mesh

class MeshCharStacker(Stacker):
    
    def __init__(self, char, vfu):
        
        super().__init__(len(vfu[0]))
        
        self.char   = char
        
        self.verts_ = vfu[0]
        self.nverts = len(self.verts)
        
        self.faces_ = vfu[1]
        self.mat_indices_ = np.zeros(len(self.faces_), int)
        
        self.uvs_ = {"uv_chars": vfu[2]}
        self.uvs_size_ = len(vfu[2])
        
    @property
    def name(self):
        return self.char

# =============================================================================================================================
# A char stacker as a curve

class CurveCharStacker(Stacker):
    
    def __init__(self, char, beziers):
        
        verts = np.zeros((0, 3), np.float)
        profile = np.zeros((len(beziers), 3), int)
        for i, bz in enumerate(beziers):
            verts = np.append(verts, bz[:, 0], axis=0)
            verts = np.append(verts, bz[:, 1], axis=0)
            verts = np.append(verts, bz[:, 2], axis=0)
            profile[i] = (3, len(bz), 0)
            
        super().__init__(len(verts))
        
        self.char     = char
        self.verts_   = verts
        self.profile_ = profile
        
        self.mat_indices_ = np.zeros(len(beziers), int)
        
    @property
    def name(self):
        return self.char
    
    def setup_spline(self, spline_index, target_spline):
        target_spline.use_cyclic_u = True
        target_spline.fill_mode    = 'FULL'


# =============================================================================================================================
# Stack of objects

class Stack():
    
    def __init__(self, object_type):
        
        self.object_type = object_type

        self.stackers  = [] # The list of stackers
        
        self.mat_count   = 0
        self.mat_indices = np.zeros(0, int)
        
    def reset(self):
        
        del self.stackers
        self.stackers  = []
        
        self.mat_count   = 0
        self.mat_indices = np.zeros(0, int)
        
        
    def __repr__(self):
        
        s = f"<Stack of {len(self)} objects:\n"
        for i, stacker in enumerate(self):
            s += f" {i:3d}: {stacker}\n"
        return s + ">"
        
    def __len__(self):
        return len(self.stackers)
    
    def __getitem__(self, index):
        return self.stackers[index]
    
    @property
    def is_mesh(self):
        return self.object_type == 'Mesh'
    
    @property
    def is_curve(self):
        return self.object_type == 'Curve'
    
    @property
    def dupli_count(self):
        count = 0
        for stacker in self:
            count += len(stacker)
        return count
    
    @property
    def max_nverts(self):
        mx = 0
        for stacker in self:
            mx= max(stacker.nverts, mx)
        return mx
    
    @property
    def curve_dim(self):
        for stacker in self:
            if np.min(stacker.profile[:, 0]) == 1:
                return 6
        return 3
        
    
    # ------------------------------------------------------------------------------------------
    # Indices : an entry per instance. The value gives the stacker index
    #
    # stackers:
    # - 0: [3, 6, 2]
    # - 1: [1, 7]
    # - 2: [0, 4, 5]
    #
    # indices = [2, 1, 0, 0, 2, 2, 0, 1]
    #
    # Can be then shuffled and set back
    
    @property
    def indices(self):
        indices = np.zeros(self.dupli_count, int)
        for i, stacker in enumerate(self):
            indices[stacker.inst_indices] = i
        return indices
    
    @indices.setter
    def indices(self, indices):
        indices = np.array(indices)
        rg = np.arange(len(indices))
        for i, stacker in enumerate(self):
            stacker.inst_indices = rg[indices == i]
            
    # ------------------------------------------------------------------------------------------
    # When we need to stack arrays such as material indices or profile in the instances order
    # we need to know the slices where to insert the instance
    
    def slices(self, attr, indices=None):

        if indices is None:        
            indices = self.indices
            
        indices = np.reshape(indices, np.size(indices))
            
        n = len(indices)
        nrg = np.arange(n)

        slices = np.zeros((n, 2), int)
        for i, stacker in enumerate(self):
            slices[nrg[indices == i], 1] = getattr(stacker, attr)
            
        slices[1:, 0] = np.cumsum(slices[:-1, 1])
        return slices
            
            
    # ------------------------------------------------------------------------------------------
    # Stack a new stacker for a given number of instances
    
    def stack(self, stacker, count=1):
        
        self.stackers.append(stacker)
        
        n = self.dupli_count
        stacker.inst_indices = np.arange(n, n+count)
        
        return stacker
    
    def stack_object(self, name, count=1):
        
        return self.stack(ObjectStacker(name), count)
    
    
    # ------------------------------------------------------------------------------------------
    # Shuffle the instances
    
    def shuffle(self, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            
        indices = self.indices
        np.random.shuffle(indices)
        self.indices = indices
        
    # ------------------------------------------------------------------------------------------
    # Set the materials to the target
    
    def create_materials_in(self, name):
        
        wtarget = wrap(name)
        
        # ----- Create the material list in the target 
        
        wtarget.wmaterials.clear()
        mat_offset = 0
        for stacker in self:
            if hasattr(stacker, 'wobject'):
                wtarget.wmaterials.copy_materials_from(stacker.wobject, append=True)
                stacker.mat_offset = mat_offset
            mat_offset += stacker.mat_count
        
    # ------------------------------------------------------------------------------------------
    # Set the materials to the target
    
    def set_materials_indices(self, name):
        
        wtarget = wrap(name)
            
        # ----- Build the material indices array
        
        indices = self.indices
        slices  = self.slices("mat_indices_count", indices)
        n = slices[-1, 0] + slices[-1, 1]
        mat_indices = np.zeros(n, int)
        
        for i_stacker, (index, size) in zip(indices, slices):
            stacker = self[i_stacker]
            mat_indices[index:index+size] = stacker.mat_indices + stacker.mat_offset
        
        wtarget.material_indices = mat_indices
        
    # ====================================================================================================
    # Set to an object
    
    def set_to_object(self, name):

        if self.object_type == 'Mesh':
            self.set_to_mesh_object(name)
        
        elif self.object_type == 'Curve':
            self.set_to_curve_object(name)
        
    # ----------------------------------------------------------------------------------------------------
    # Initialize a mesh object with the content

    def set_to_mesh_object(self, name):
        
        # ---------------------------------------------
        # The target object
        
        wtarget = wrap(name)

        # ---------------------------------------------------------------------------
        # Indices

        indices = self.indices
        
        # ---------------------------------------------------------------------------
        # Vertices
        
        slices  = self.slices("nverts", indices)
        n = slices[-1, 0] + slices[-1, 1]
        
        verts = np.zeros((n, 3), np.float)
        
        for i_stacker, (index, size) in zip(indices, slices):
            stacker = self[i_stacker]
            verts[index:index+size] = stacker.verts
            
        # ---------------------------------------------------------------------------
        # Faces
        
        faces = []
        for stacker in self:
            for i in stacker.inst_indices:
                faces.extend(stacker.get_shifted_faces(slices[i, 0]))
                
        # ---------------------------------------------------------------------------
        # The new geometry
        
        wtarget.new_geometry(verts, faces)
        
        del verts
        del faces
        
        # ---------------------------------------------------------------------------
        # The material indices
        
        self.create_materials_in(wtarget)
        self.set_materials_indices(wtarget)
        
        # ---------------------------------------------------------------------------
        # uvs
        
        # ----- All the uvmap names
        
        uvmaps = []
        for stacker in self:
            for name in stacker.uvmaps:
                if name not in uvmaps:
                    uvmaps.append(name)
                    
        # ----- Transfer the uvmaps into the target
        
        slices = self.slices("uvs_size", indices)
        n = slices[-1, 0] + slices[-1, 1]
        uvs = np.zeros((n, 2), np.float)
        
        for name in uvmaps:
            uvs[:] = 0
            for i_stacker, (index, size) in zip(indices, slices):
                stacker = self[i_stacker]
                uvs[index:index+size] = stacker.get_uvs(name)
                
            uvmap = wtarget.get_uvmap(name, create=True)
            uvmap.data.foreach_set('uv', uvs.reshape(2*n))
            
    # ----------------------------------------------------------------------------------------------------
    # Initialize a curve object with the content
    
    def set_to_curve_object(self, name):
        
        # ---------------------------------------------
        # The target object
        
        wtarget = wrap(name)
        wsplines = wtarget.wdata.wsplines

        # ---------------------------------------------------------------------------
        # Indices

        indices = self.indices
        
        # ---------------------------------------------------------------------------
        # Profile
        
        prof_slices  = self.slices("profile_size", indices)
        n = prof_slices[-1, 0] + prof_slices[-1, 1]
        
        profile = np.zeros((n, 3), int)
        
        for i_stacker, (index, size) in zip(indices, prof_slices):
            stacker = self[i_stacker]
            profile[index:index+size] = stacker.profile
        
        wsplines.profile = profile
        
        # ---------------------------------------------------------------------------
        # Vertices
        
        only_bezier = np.min(profile[:, 0]) == 3
        ndim = 3 if only_bezier else 6

        slices  = self.slices("nverts", indices)
        n = slices[-1, 0] + slices[-1, 1]
        
        verts = np.zeros((n, ndim), np.float)
        
        for i_stacker, (index, size) in zip(indices, slices):
            stacker = self[i_stacker]
            vs = stacker.verts
            if vs.shape[-1] == 3:
                verts[index:index+size, :3] = vs
            else:
                verts[index:index+size] = vs
                
        wsplines.verts = verts
        
        # ---------------------------------------------------------------------------
        # Create the material list into the target
        
        self.create_materials_in(wtarget)
        
        # ----- Splines attributes
        
        for i_stacker, (index, size) in zip(indices, prof_slices):
            stacker = self[i_stacker]
            for i_spl in range(size):
                stacker.setup_spline(spline_index = i_spl, target_spline=wsplines[index + i_spl])
                
            continue
            
            if hasattr(self[i_stacker], 'wobject'):
                sps = self[i_stacker].wobject.data.splines
                for i_spl in range(size):
                    wsplines[index + i_spl].copy_from(sps[i_spl])
                
        # ----- Material indices
        # Since spline.copy_from copy the material_index attribute
        # this must be done after the copy_from loop
        
        self.set_materials_indices(wtarget)
        
        # ----- Curve parameter read from the first object
        
        if hasattr(self[0], 'wobject'):
            wtarget.wdata.copy_from(self[0].wobject.wdata)
        
    # ====================================================================================================
    # Get the crowd base vertices
    
    def get_crowd_bases(self):

        if self.object_type == 'Mesh':
            return self.get_crowd_mesh_bases()
        
        elif self.object_type == 'Curve':
            return self.get_crowd_curve_bases()
        
        return None
        
    # ----------------------------------------------------------------------------------------------------
    # The mesh base vertices for a crowd

    def get_crowd_mesh_bases(self):
        
        if len(self) == 1:
            return self[0].verts.reshape(1, self[0].nverts, 3)
        
        indices = self.indices
        n = len(indices)
        nmax = self.max_nverts
        verts = np.zeros((n, nmax, 3))
        
        for i, i_stacker in enumerate(indices):
            verts[i, :self[i_stacker].nverts, :] = self[i_stacker].verts
            
        return verts
        
    # ----------------------------------------------------------------------------------------------------
    # The curve base vertices for a crowd

    def get_crowd_curve_bases(self):
        
        ndim = self.curve_dim
        
        if len(self) == 1:
            return self[0].verts.reshape(1, self[0].nverts, ndim)
        
        indices = self.indices
        n = len(indices)
        nmax = self.max_nverts
        verts = np.zeros((n, nmax, ndim))
        
        for i, i_stacker in enumerate(indices):
            vs = self[i_stacker].verts
            verts[i, :self[i_stacker].nverts, :vs.shape[-1]] = vs
            
        return verts
    
    # ----------------------------------------------------------------------------------------------------
    # Get crowd vertex indices to extract the true vertices
    # Return None when all vertices are true (all instance hav the same size)
    
    def get_true_vert_indices(self):
        
        indices = self.indices
        slices  = self.slices("nverts", indices)
        
        # ----- All instances have the same size
        
        nmax = np.max(slices[:, 1])
        if np.min(slices[:, 1]) == nmax:
            return None
        
        # ----- Build the true indices array
        
        n = np.sum(slices[:, 1])
        extract = np.zeros(n, int)
        
        for stacker in self:
            for i_dupli in stacker.inst_indices:
                index = slices[i_dupli, 0]
                size  = slices[i_dupli, 1]
                vert_index = i_dupli * nmax
                extract[index:index+size] = np.arange(vert_index, vert_index+size)
            
        return extract
        
        
                    

        
