#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:28:15 2021

@author: alain
"""

import numpy as np

#from ..wrappers.wmeshobject import WMeshObject
#from ..wrappers.wcurveobject import WCurveObject
from ..wrappers.wrap_function import wrap

# =============================================================================================================================
# Stacker : feed the stack
# From an existing object
# From another source of geometry

class Stacker():
    def __init__(self, nverts):
        
        self.nverts       = nverts
        self.vert_indices = np.zeros(0, int)
        
        self.faces_       = []
        self.profile_     = np.zeros((0, 2), np.float)
        
        self.mat_count_   = 0
        self.mat_indices_ = 0
        
        self.uvs_size_    = 0
        self.uvs_         = {}
        
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        return self.vert_indices[index]
        
    def __setitem__(self, index, value):
        self.vert_indices[index] = value
        
    def add_instance(self, vert_index):
        self.vert_indices.append(vert_index)

    def add_instances(self, vert_indices):
        self.vert_indices.extend(vert_indices)
        
    @property
    def faces(self):
        return None
    
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
        if uvs is None:
            uvs = self.empty_uvs
            self.uvs_[name] = uvs
        return uvs
    
    def set_uvs(self, name, uvs):
        self.uvs_[name] = uvs
        
    # ----------------------------------------------------------------------------------------------------
    # Build the target object
    # index is the index with the property verts_indices
    
    def get_shifted_faces(self, index=None):

        if index is None:
            faces = []
            for i in self:
                faces.extend(self.get_shifted_faces(i))
            return faces
        
        base = self.vert_indices[index]
        return [[base + f for f in face] for face in self.faces]

# =============================================================================================================================
# Object stacker
# From an existing object

class ObjectStacker():
    def __init__(self, name):
        
        self.wobject = wrap(name)
        super().__init__(self.wobject.verts_count)
        
    @property
    def faces(self):
        return self.wobject.poly_indices
    
    @property
    def profile(self):
        return self.wobject.profile
    
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


# =============================================================================================================================
# Stack of objects

class Stack():
    def __init__(self):

        self.wobjects = []
        self.dupli_count = 0
        
        self.mat_count   = 0
        self.mat_indices = np.zeros(0, int)
        
        
    def __len__(self):
        return len(self.wobjects)
    
    def __getitem__(self, index):
        return self.wobjects[index]
    
    def stack(self, name, count=1):
        
        wobj = wrap(name)
        self.wobjects.append(wobj)
        
        wobj.stk_count = count
        wobj.stk_dupli_slice = (self.dupli_count, self.dupli_count + count)
        self.dupli_count += count
        
        # ---------------------------------------------------------------------------
        # Material indices
        
        mat_count = len(wobj.wmaterials)
        mat_index = len(self.mat_indices)
        mat_indices = self.mat_count + wobj.material_indices
        self.mat_indices = np.append(self.mat_indices, np.resize(mat_indices, count*len(mat_indices)), axis=0)
        wobj.stk_mat_indices_slice = (mat_index, len(self.mat_indices))
        wobj.stK_mat_slice = (self.mat_count, self.mat_count + mat_count)
        self.mat_count += mat_count
        
        del mat_indices
        
        return wobj
        
    
# =============================================================================================================================
# Stack of mesh objects

class MeshStack(Stack):
    def __init__(self):
        
        super().__init__()
        
        self.verts       = np.zeros((0, 3), np.float)
        self.faces       = []
        self.mat_count   = 0
        self.mat_indices = np.zeros(0, int)
        self.uvs_index   = 0
        self.uvmaps      = {}
        
    def __repr__(self):
        
        s = f"<MeshStack of {len(self)} objects:\n"
        for i, wobj in enumerate(self):
            s += f" {i:3d}: {wobj.stk_count:3d} x {wobj.stk_nverts:4d} vertices @ {wobj.stk_verts_slice} '{wobj.name}'\n"
        s += f"   verts      : {len(self.verts)}\n"
        s += f"   faces      : {len(self.faces)}\n"
        s += f"   mat indices: {len(self.mat_indices)}\n"
        s += f"   uv maps    : {len(self.uvmaps)}: {list(self.uvmaps.keys())}\n"
        return s + ">"
    
    # ----------------------------------------------------------------------------------------------------
    # Stack a new mesh
    
    def stack(self, name, count=1):
        
        wobj = super().stack(name, count)
        
        # ---------------------------------------------------------------------------
        # Vertices

        verts_index = len(self.verts)
        verts = wobj.verts
        wobj.stk_nverts = len(verts)
        self.verts = np.append(self.verts, np.resize(verts, (count*wobj.stk_nverts, 3)), axis=0)
        wobj.stk_verts_slice = (verts_index, len(self.verts))
        del verts

        # ---------------------------------------------------------------------------
        # Faces
        
        faces_index = len(self.faces)
        faces = wobj.poly_indices
        self.faces.extend([[verts_index + i*wobj.stk_nverts + f for f in face]for i in range(count) for face in faces])
        wobj.stk_faces_slice = (faces_index, len(self.faces))
        
        # ---------------------------------------------------------------------------
        # uvs
        
        uvs_size = count * wobj.uvs_size
        wobj.stk_uvs_slice = (self.uvs_index, self.uvs_index + uvs_size)
        
        for name in wobj.uvmaps:
            
            stacked_uvs = self.uvmaps.get(name)
            if stacked_uvs is None:
                stacked_uvs = np.zeros((self.uvs_index, 2), np.float)
                
            self.uvmaps[name] = np.append(stacked_uvs, np.resize(wobj.get_uvs(name), (uvs_size, 2)), axis=0)

        uvs_size += uvs_size
        
        return wobj
    
    # ----------------------------------------------------------------------------------------------------
    # Initialize an object with the content

    def set_to_object(self, name):
        
        wtarget = wrap(name)
        
        wtarget.new_geometry(self.verts, self.faces)
        
        wtarget.wmaterials.clear()
        for wobj in self:
            wtarget.wmaterials.copy_materials_from(wobj, append=True)
        wtarget.material_indices = self.mat_indices
        
        for name, uvs in self.uvmaps.items():
            uvmap = wtarget.get_uvmap(name, create=True)
            uvmap.data.foreach_set('uv', uvs.reshape(2*len(uvs)))
            

# =============================================================================================================================
# Stack of curve objects

class CurveStack(Stack):
    def __init__(self):
        
        super().__init__()
        
        self.verts       = np.zeros((0, 6), np.float)
        self.profile     = np.zeros((0, 2), np.float)
        
    def __repr__(self):
        
        s = f"<CurveStack of {len(self)} objects:\n"
        for i, wobj in enumerate(self):
            s += f" {i:3d}: {wobj.stk_count:3d} x [{wobj.stk_nsplines} splines and {wobj.stk_nverts:4d} vertices] @ {wobj.stk_verts_slice} '{wobj.name}'\n"
        s += f"   verts      : {len(self.verts)}\n"
        s += f"   profile    : {len(self.profile)}\n"
        s += f"   mat indices: {len(self.mat_indices)}\n"
        return s + ">"
    
    # ----------------------------------------------------------------------------------------------------
    # Stack a new curve
    
    def stack(self, name, count=1):
        
        wobj = super().stack(name, count)
        
        # ---------------------------------------------------------------------------
        # Profile
        
        profile_index = len(self.profile)
        profile = wobj.profile
        wobj.stk_nsplines = len(profile)
        self.profile = np.append(self.profile, np.resize(profile, (count*wobj.stk_nsplines, 2)), axis=0)
        wobj.stk_profile_slice = (profile_index, len(self.profile))
        
        # ---------------------------------------------------------------------------
        # Vertices
        # Profile was reader first to use it to get the vertices

        verts_index = len(self.verts)
        verts = wobj.wdata.wsplines.get_vertices(profile=profile, extended=True)
        wobj.stk_nverts = len(verts)
        appnd = np.zeros((count, wobj.stk_nverts, 6), np.float)
        if verts.shape[-1] == 3:
            appnd[..., :3] = verts
        else:
            appnd[:] = verts
        self.verts = np.append(self.verts, appnd.reshape(count*wobj.stk_nverts, 6), axis=0)
        wobj.stk_verts_slice = (verts_index, len(self.verts))
        del verts
        
    # ----------------------------------------------------------------------------------------------------
    # Initialize an object with the content

    def set_to_object(self, name):
        
        wtarget = wrap(name)
        wsplines = wtarget.wdata.wsplines
        
        # ----- Set the splines at the right type and length
        
        wsplines.profile = self.profile

        # ----- Set the splines vertices
        
        wsplines.verts = self.verts

        # ----- Create the materials list

        wtarget.wmaterials.clear()
        for wobj in self:
            wtarget.wmaterials.copy_materials_from(wobj, append=True)
        
        # ----- Splines attributes
        
        i_spline = 0
        for wobj in self:
            for i_spl, spline in enumerate(wobj.wdata.splines):
                for i in range(wobj.stk_count):
                    #print("index", i_spline, i, i_spl, '-->', i_spline + i*wobj.stk_nsplines + i_spl)
                    wsplines[i_spline + i*wobj.stk_nsplines + i_spl].copy_from(spline)
                
            i_spline += wobj.stk_count * wobj.stk_nsplines
                
        # ----- Material indices
        # Since spline.copy_from copy the material_index attribute
        # this must be done after the copy_from loop
        
        wtarget.material_indices = self.mat_indices
        
        # ----- Curve paramter read from the first object
        
        wtarget.wdata.copy_from(self[0].wdata)
        
        
        
        
