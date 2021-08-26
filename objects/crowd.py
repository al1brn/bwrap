#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np
import bpy
import bpy_types

from ..blender import blender
from ..maths.transformations import Transformations, SlaveTransformations
from ..maths.shapes import get_full_shape

from ..wrappers.wrap_function import wrap

from ..core.commons import WError

# ====================================================================================================
# Group transo class utility

class GroupTransfo():
    
    def __init__(self, crowd, indices, pivot):
        
        self.crowd   = crowd
        self.indices = indices
        self.transfo = SlaveTransformations(crowd)
        
        self.pivot = np.zeros(4, np.float) # No one at the end !
        if pivot is None:
            verts = self.crowd.wmodels[0].verts[indices]
            n = len(verts)
            if n != 0:
                self.pivot[:3] = np.sum(verts, axis=0)/n
        else:
            self.pivot[:3] = pivot
        
    def __repr__(self):
        return f"<GroupTransfo: {self.transfo}, pivot: {self.pivot}, indices: {np.shape(self.indices)}>"
    
    
# ====================================================================================================
# Crowd: duplicates a mesh or a curve in a single object
# MeshCrowd and CurveCrowd implement specificites

class Crowd(Transformations):
    
    SUPPORTED = ['Mesh', 'Curve', 'TextCurve']
    
    """Manages a crowd of copies of a model made of vertices in a single object.
    
    The model can be either:
        - A single model which is duplicated according the shape of the Tranformation
        - An array of models. At each frame, one version is selected for each duplicate
        - As many models as the shape of the transfpormations
        
    Additional transformation can be made on groups
    
    Copies of the model can be considered as individual objects which van be transformed
    individually using the method of the Transformations class:
        - locations
        - rotations
        - scales
    Plus the shortcuts: x, y, z, rx, ry, rz, rxd, ryd, rzd, sx, sy, sz
    
    Examples width a shape of (100, 70) of duplicates
    
    - SINGLE : Crowd of a single shape (default)
        model    = 10 vertices --> (10, 3)
        produces an array of shape (100, 70, 10, 3)
        
    - ANIMATION : Crowd of a deformed mesh with different shapes
        model    = 5 times 10 vertices --> (5, 10, 3)
        uses indices of shape (100, 70) of int between 0 and 4 
        produces an array of shape (100, 70, 10, 3)
        
    - VARIABLE : Versions can have different shapes
        model = list of models width various shapes
        uses indices of shape (100, 70) of int between 0 and 4 
        produces an array of shape (100 x 70 x ?, 3)
        
    - INDIVIDUALS : each individual has its own shape
        model = array(100, 70) of objects
        produces an array of shape (100 x 70 x ?, 3)
    """
    
    def __init__(self, model, shape=(), name=None, animation_key=None):
        """Initialize the crowd with by creating count duplicates of the model.
        
        These three initializations are valid:
            - Crowd("Model", 1000)
            - Crowd(bw.wrap("Model"), 1000)
            - Crowd(bpy.data.objects["Model"], 1000)
            
        The crowd is created if if doesn't already exists
        If not provided, the name is based on the model name: 'Crowd of models'

        Parameters
        ----------
        model : str or Wrapper
            Name of an existing object or wrapper of an existing object. The wrapper version allows
            to use the evaluated version
        count : shape, optional
            Shape of the crowd. The default is 10.
        name : str
            Name of the crowd object. If not provided (default), 'Crowd of models' is used.
        evaluated : bool
            Take the evaluated vertices of the model. Default is False.

        Raises
        ------
        RuntimeError
            If the model doesn't exist or is not a mesh.

        Returns
        -------
        None.
        """
        
        # ---------------------------------------------------------------------------
        # ----- Super initialization
        
        super().__init__(count=shape)
        
        # ---------------------------------------------------------------------------
        # ----- The model can be a single model (SINGLE or ANIMATION)
        # or a list of different models
        
        if type(model) in (list, tuple, bpy_types.Collection, np.ndarray):
            self.mode = 'INDIVIDUALS'
            self.wmodels = [wrap(mdl) for mdl in model]
            
            if self.size == len(self.wmodels):
                self.mdl_indices = np.arange(len(self.wmodels)).reshape(self.shape)
            else:
                self.mdl_indices = np.random.choice(np.arange(len(self.wmodels)), self.shape)
        else:
            self.mode = 'SINGLE'
            self.wmodels = [wrap(model)]
        
        # ---------------------------------------------------------------------------
        # ----- Wrap the models
        
        for i, wmdl in enumerate(self.wmodels):
            if wmdl.object_type not in Crowd.SUPPORTED:
                raise WError(f"Crowd initialization error: {wmdl.name} must be in {Crowd.SUPPORTED}, not {wmdl.object_type}.",
                    Class = "Crowd",
                    Method = "__init__",
                    model = model,
                    shape = shape)

            if wmdl.object_type != self.wmodels[0].object_type:
                raise WError(f"Crowd initialization error: all the models must have the same type. {wmdl.name} is not of the same type as {self.wmodels[0].object_type}.",
                    Class = "Crowd",
                    Method = "__init__",
                    model = model,
                    shape = shape)
            
        if name is None:
            name = "Crowd of " + self.wmodels[0].name + 's'
            
        self.extract_indices = None
            
        if self.is_mesh:
            self.set_mesh_model(name = name, animation_key=animation_key)
            
        if self.is_curve:
            self.set_curve_model(name = name, animation_key=animation_key)
            
        if self.is_text:
            self.wobject = wrap(name, create = "BEZIER" )
        
        # ----- Group transformations
        
        self.group_transfos = {}


    # ====================================================================================================
    # Content
    
    def __repr__(self):
        if len(self.shape) > 1:
            size = f"{self.shape} ({self.size} duplicates) of"
        else:
            size = f"of {self.size}"
        if len(self.wmodels) == 1:
            name = f"'{self.wmodels[0].name}'"
            model = self.wmodels[0].name
        else:
            name = f"{len(self.wmodels)} models ('{self.wmodels[0].name}' ... '{self.wmodels[-1].name}')"
            model = [wmdl.name for wmdl in self.wmodels]
        s  = f"<Crowd {size} {name}, type={self.wmodels[0].object_type}: '{self.wobject.name}'\n"
        s += f"   model:{model}\n"
        s += f"   vertices per duplicate: {self.dupli_verts}\n"
        s += f"   total vertices        : {self.total_verts}\n"
        s += f"   mode: {self.mode}"
        if self.mode == 'ANIMATION':
            s += f" with {self.models.shape[0]} shapes\n"
        if self.mode == 'INDIVIDUALS':
            if self.extract_indices is None:
                s += f" of same size {self.dupli_verts}\n"
            else:
                tot_verts = self.size * self.dupli_verts
                s += f" of different sizes. Max is {self.dupli_verts}\n"
                s += f"      total vertices: {tot_verts}, true vertices ratio: {self.total_verts / tot_verts * 100:5.1f}%\n"
                inds = np.reshape(self.mdl_indices, self.size)
                for i_mdl, wmdl in enumerate(self.wmodels):
                    n = len(np.where(inds == i_mdl)[0])
                    s += f"         {wmdl.name:20s}: {n:3d} duplicates of {wmdl.verts_count:4d} vertices: ({n*wmdl.verts_count:5d} {(n*wmdl.verts_count)/tot_verts*100:5.1f}%)\n"
                    
        return s + ">"
    
    
    # ====================================================================================================
    # Takes the models in wmodels into account to build the model vertices
    # wobject has already been created
    
    def set_model(self, animation_key=None):
        
        # ---------------------------------------------------------------------------
        # Copy the materials and store the indices
        # The indices are incremented to take the fusion of the materials lists
        # Note that identical materials will be duplicated
        #
        # Also get the dimension of the vectors
        
        mat_count = np.zeros(len(self.wmodels), int)
        mdl_mat_indices = [None for i in range(len(self.wmodels))]
        mat_base = 0
        self.verts_dim = 3
        for i_mdl, wmdl in enumerate(self.wmodels):
            self.wobject.wmaterials.copy_materials_from(wmdl, append=True)
            
            mat_count[i_mdl] = len(wmdl.wmaterials)
            mdl_mat_indices[i_mdl] = wmdl.material_indices + mat_base
            mat_base += mat_count[i_mdl]
            
            self.verts_dim = max(self.verts_dim, wmdl.verts_dim)
        
        # ---------------------------------------------------------------------------
        # The initialization can be done from a single model or from a set of models
        
        if self.mode == 'INDIVIDUALS':
            
            # ---------------------------------------------------------------------------
            # ----- The size of each duplicate will be the max size of the individuals
            
            mdl_indices = self.mdl_indices.reshape(self.size)
            
            # ---------------------------------------------------------------------------
            # ----- Let's loop on the models to prepare the main loop
            
            models_count = len(self.wmodels)
            
            self.verts_dim  = 3
            dupli_per_model = np.zeros(models_count, int)
            sizes           = np.zeros(models_count, int)
            
            if self.is_curve:
                mdl_profiles = np.zeros(len(self.wmodels), object)
                
            mat_base = 0                                                # To count yhe total number of materials
            mat_count = np.zeros(len(self.wmodels), int)                # Number of materials per model
            mat_indices_count = 0                                       # Number of material indices
            mdl_mat_indices = [None for i in range(len(self.wmodels))]  # Mat indices of models
            
            for i_mdl in range(len(self.wmodels)):
                
                # ----- Duplicates per model
                
                inds = np.where(mdl_indices == i_mdl)[0]
                dupli_per_model[i_mdl] = len(inds)
                
                # ----- Number of vertices per models
                
                sizes[i_mdl] = wmdl.verts_count
                
                # ----- The profile of the model
                
                if self.is_curve:
                    mdl_profiles[i_mdl] = wmdl.profile
                    
                # ----- Materials
                
                self.wobject.wmaterials.copy_materials_from(wmdl, append=True)

                mat_indices_count += len(inds)
            
                mat_count[i_mdl] = len(wmdl.wmaterials)
                mdl_mat_indices[i_mdl] = wmdl.material_indices + mat_base
                mat_base += mat_count[i_mdl]
                
                # ----- Vector size
                
                self.verts_dim = max(self.verts_dim, wmdl.verts_dim)
                
                
            # ---------------------------------------------------------------------------
            # We have info to prepare the build loop
            
            # ----- ind_sizes: start vertex, vertices count in a shape (size, 2): 
            
            self.dupli_verts = np.max(sizes)
            
            ind_map          = np.zeros((self.size, 2), int)
            ind_map[:, 1]    = sizes[mdl_indices]
            ind_map[1:, 0]   = np.cumsum(ind_map[:-1, 1])
            self.total_verts = np.sum(ind_map[:, 1])
            
            del sizes    
                    
            # ----- We can initialize a proper array to get the vertices
            
            self.models = np.zeros((self.size, self.dupli_verts, self.verts_dim + 1))
            self.models[..., 3] = 1
            
            # ----- Faces will be built by stacking all the faces of the same model
            # faces = [model0, ... model0, model1, ... model1, model2, ... model(n-1), ... model(n-1)]
            
            # ----- uv names are stored to create as uv maps as requireds
            # uv maps will be shifted as for the faces
            
            if self.is_mesh:
                faces    = []
                uv_names = []
                
            elif self.is_curves:
                profile = np.zeros(models_count, object)
                profile_length = 0

            mat_indices = np.zeros(mat_indices_count, int)
            mat_index = 0
            
            self.extract_indices = np.zeros(self.total_verts, int)
            
            # ----------------------------------------------------------------------------------------------------
            # ----- Let's build the new geometry
            
            for i_mdl, wmdl in enumerate(self.wmodels):
                
                # ----- Individual indices using the model
                 
                inds = np.where(mdl_indices == i_mdl)[0]
                
                # ----- The extract indices
                
                for i in inds:
                    start = ind_map[i, 0]
                    n = ind_map[i, 1]
                    self.extract_indices[start:start + n] = np.arange(i*self.dupli_verts, i*self.dupli_verts + n)
                    
                # ----- Vertices at all the duplicate indices
                verts = wmdl.verts[:, :3]
                self.models[inds, :wmdl.verts_count, :3] = verts[:, :3]
                if self.verts_dim > 3:
                    self.models[inds, :wmdl.verts_count, 4:] = verts[:, 3:]
                    
                # ----- Loop on the individuals indices
                
                if self.is_mesh:
                    polys = wmdl.poly_indices
                elif self.is_curve:
                    prof = wmdl.profile
                    profile_length += len(prof) * len(inds)
                    
                for i_dupli in inds:
                    
                    # Faces
                    if self.is_mesh:
                        vert_index = ind_map[i_dupli, 0]
                        faces.extend([[vert_index + index for index in face] for face in polys])
                        
                    # Profile
                    
                    elif self.is_curve:
                        profile[i_dupli] = prof
                        
                # ----- uv map names
                
                if self.is_mesh:
                    for uv_name in wmdl.uvmaps:
                        if uv_name not in uv_names:
                            uv_names.append(uv_name)
                        
                # ----- Material indices
                
                n = dupli_per_model[i_mdl] * len(mdl_mat_indices[i_mdl])
                mat_indices[mat_index:mat_index+n] = np.resize(mdl_mat_indices[i_mdl], n)
                
                mat_index += n
                
                    
            # ---------------------------------------------------------------------------
            # We can set the new geometry of the mesh object
            
            if self.is_mesh:
                self.wobject.new_geometry(self.models[..., :3].reshape(self.size*self.dupli_verts, 3)[self.extract_indices], faces)
                
            elif self.is_curve:
                prof = np.zeros((profile_length, 2), int)
                index = 0
                for pf in profile:
                    prof[index:index + len(pf)] = pf
                    index += len(pf)
                
            self.wobject.material_indices = mat_indices
            
            # ---------------------------------------------------------------------------
            # Now the uv maps

            if len(uv_names) > 0:
                
                # ----- Create all the uv_maps exist
                
                for uv_name in uv_names:
                    self.wobject.create_uvmap(uv_name)
                    
                # ----- uv map size of each model
                    
                sizes = np.array([wmdl.uvs_size for wmdl in self.wmodels], int)
                
                
                # ----- An empty uv map
                
                uvs = np.zeros((self.wobject.uvs_size, 2))

                # ----- set all the required maps
                
                for uv_name in uv_names:
                    
                    uvs[:] = 0
                    index  = 0
                    
                    for i_mdl, wmdl in enumerate(self.wmodels):
                        
                        if uv_name in wmdl.uvmaps:
                            mdl_uvs = wmdl.get_uvs(uv_name)
                            for i in range(dupli_per_model[i_mdl]):
                                uvs[index:index+sizes[i_mdl]] = mdl_uvs
                                index += sizes[i_mdl]
                        else:
                            
                            index += sizes[i_mdl] * dupli_per_model[i_mdl]
                        
                    # We can set the mapping
                        
                    self.wobject.set_uvs(uv_name, uvs)
            
            # ----- Let'ts put the models at teh right shape
            
            self.models = self.models.reshape(self.shape + (self.dupli_verts, 4))
            
        # ---------------------------------------------------------------------------
        # One single model

        else:
            
            wmodel = self.wmodels[0]
            
            # ---------------------------------------------------------------------------
            # ----- Create the geometry
    
            # ----- Vertices from the model
            
            verts = np.array(wmodel.verts)
            
            # ----- Create the new mesh
            
            self.dupli_verts = len(verts)
            self.total_verts = self.size * self.dupli_verts
            
            # ---------------------------------------------------------------------------
            # ----- Build the new geometry
            
            # ----- Polygons
            
            polys = wmodel.poly_indices
            self.p_count = len(polys)
            
            faces = [[index + i*self.dupli_verts for index in face] for i in range(self.size) for face in polys]
    
            # ----- New geometry
            
            self.wobject.new_geometry(np.resize(verts, (self.total_verts, 3)), faces)
            
            # ----- uv mapping
            
            for name in wmodel.uvmaps:
                uvs = wmodel.get_uvs(name)
                self.wobject.create_uvmap(name)
                self.wobject.set_uvs(name, np.resize(uvs, (self.size*len(uvs), 2)))
    
            self.wobject.wmaterials.copy_materials_from(wmodel)   
            self.wobject.material_indices = np.resize(wmodel.material_indices, self.wobject.poly_count)
                
            # ---------------------------------------------------------------------------
            # Initialize the models vertices
            
            if animation_key is None:
                self.set_mode('SINGLE', verts)
            else:
                kverts = wmodel.wshape_keys.verts(animation_key)
                if kverts is None:
                    raise WError(f"Crowd initialization error: {wmodel.name} doesn't have shape keys namde '{animation_key}'",
                        Class = "Crowd",
                        Method = "set_mesh_model",
                        wmodel = wmodel,
                        mode = self.mode,
                        name = name,
                        animation_key = animation_key)
                self.set_mode('ANIMATION', kverts)
            
            
        
    # ====================================================================================================
    # Initialize a mesh
    
    def set_mesh_model(self, name, animation_key=None):
        
        # ---------------------------------------------------------------------------
        # ----- Create the recipient mesh
            
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj = wrap(name)
            if obj.object_type != 'Mesh':
                obj.select_set(True)
                bpy.ops.object.delete() 
                obj = None
                
        if obj is None:
            obj = wrap(name, create="CUBE")
            
        self.wobject = obj
        blender.copy_collections(self.wmodels[0].wrapped, self.wobject.wrapped)
        
        # ---------------------------------------------------------------------------
        # Copy the materials and store the indices
        # The indices are incremented to take the fusion of the materials lists
        # Note that identical materials will be duplicated
        
        mat_count = np.zeros(len(self.wmodels), int)
        mdl_mat_indices = [None for i in range(len(self.wmodels))]
        mat_base = 0
        for i_mdl, wmdl in enumerate(self.wmodels):
            self.wobject.wmaterials.copy_materials_from(wmdl, append=True)
            
            mat_count[i_mdl] = len(wmdl.wmaterials)
            mdl_mat_indices[i_mdl] = wmdl.material_indices + mat_base
            mat_base += mat_count[i_mdl]
        
        # ---------------------------------------------------------------------------
        # The initialization can be done from a single model or from a set of models
        
        if self.mode == 'INDIVIDUALS':
            
            # ---------------------------------------------------------------------------
            # ----- The size of each duplicate will be the max size of the individuals
            
            mdl_indices = self.mdl_indices.reshape(self.size)
            
            sizes = np.array([wmdl.verts_count for wmdl in self.wmodels])
            self.dupli_verts = np.max(sizes)
            
            # ----- ind_sizes: start vertex, vertices count in a shape (size, 2): 
            
            ind_map        = np.zeros((self.size, 2), int)
            ind_map[:, 1]  = sizes[mdl_indices]
            ind_map[1:, 0] = np.cumsum(ind_map[:-1, 1])
            self.total_verts    = np.sum(ind_map[:, 1])
            
            # ----- The indices to extract the individuals indices
            self.extract_indices = np.zeros(self.total_verts, int)
            for i_mdl in range(len(self.wmodels)):
                inds = np.where(mdl_indices == i_mdl)[0]
                n = sizes[i_mdl]
                for i in inds:
                    start = ind_map[i, 0]
                    self.extract_indices[start:start + n] = np.arange(i*self.dupli_verts, i*self.dupli_verts + n)
                    
            del sizes
            
            # ----- We can initialize a proper array to get the vertices
            
            self.models = np.zeros((self.size, self.dupli_verts, 4))
            self.models[..., 3] = 1
            
            # ----- Faces will be built by stacking all the faces of the same model
            # faces = [model0, ... model0, model1, ... model1, model2, ... model(n-1), ... model(n-1)]
            faces = []
            
            # Let's store how many duplicates per model
            dupli_per_model = [0 for i in range(len(self.wmodels))]
            
            # ----- uv names are stored to create as uv maps as requireds
            # uv maps will be shifted as for the faces
            
            uv_names = []
            
            # ----- Materials indices are stored in a array
            count = 0
            for i_mdl in range(len(self.wmodels)):
                n = len(np.where(mdl_indices == i_mdl)[0])
                count += n * len(mdl_mat_indices[i_mdl])
                
            mat_indices = np.zeros(count, int)
            mat_index = 0
            
            # ----- Let's build the new geometry
            
            for i_mdl, wmdl in enumerate(self.wmodels):
                
                # Individual indices using the model
                inds = np.where(mdl_indices == i_mdl)[0]
                
                # Remember how many
                dupli_per_model[i_mdl] = len(inds)
                
                # ----- Vertices at all the duplicate indices
                self.models[inds, :wmdl.verts_count, :3] = wmdl.verts
                
                # ----- Faces
                polys = wmdl.poly_indices
                for i_dupli in inds:
                    vert_index = ind_map[i_dupli, 0]
                    faces.extend([[vert_index + index for index in face] for face in polys])
                    
                # ----- uv map names
                
                for uv_name in wmdl.uvmaps:
                    if uv_name not in uv_names:
                        uv_names.append(uv_name)
                        
                # ----- Material indices
                n = dupli_per_model[i_mdl] * len(mdl_mat_indices[i_mdl])
                mat_indices[mat_index:mat_index+n] = np.resize(mdl_mat_indices[i_mdl], n)
                
                mat_index += n
                
                    
            # ---------------------------------------------------------------------------
            # We can set the new geometry of the mesh object
            
            self.wobject.new_geometry(self.models[..., :3].reshape(self.size*self.dupli_verts, 3)[self.extract_indices], faces)
            self.wobject.material_indices = mat_indices
            
            # ---------------------------------------------------------------------------
            # Now the uv maps

            if len(uv_names) > 0:
                
                # ----- Create all the uv_maps exist
                
                for uv_name in uv_names:
                    self.wobject.create_uvmap(uv_name)
                    
                # ----- uv map size of each model
                    
                sizes = np.array([wmdl.uvs_size for wmdl in self.wmodels], int)
                
                
                # ----- An empty uv map
                
                uvs = np.zeros((self.wobject.uvs_size, 2))

                # ----- set all the required maps
                
                for uv_name in uv_names:
                    
                    uvs[:] = 0
                    index  = 0
                    
                    for i_mdl, wmdl in enumerate(self.wmodels):
                        
                        if uv_name in wmdl.uvmaps:
                            mdl_uvs = wmdl.get_uvs(uv_name)
                            for i in range(dupli_per_model[i_mdl]):
                                uvs[index:index+sizes[i_mdl]] = mdl_uvs
                                index += sizes[i_mdl]
                        else:
                            
                            index += sizes[i_mdl] * dupli_per_model[i_mdl]
                        
                    # We can set the mapping
                        
                    self.wobject.set_uvs(uv_name, uvs)
            
            # ----- Let'ts put the models at teh right shape
            
            self.models = self.models.reshape(self.shape + (self.dupli_verts, 4))
            
        # ---------------------------------------------------------------------------
        # One single model

        else:
            
            wmodel = self.wmodels[0]
            
            # ---------------------------------------------------------------------------
            # ----- Create the geometry
    
            # ----- Vertices from the model
            
            verts = np.array(wmodel.verts)
            
            # ----- Create the new mesh
            
            self.dupli_verts = len(verts)
            self.total_verts = self.size * self.dupli_verts
            
            # ---------------------------------------------------------------------------
            # ----- Build the new geometry
            
            # ----- Polygons
            
            polys = wmodel.poly_indices
            self.p_count = len(polys)
            
            faces = [[index + i*self.dupli_verts for index in face] for i in range(self.size) for face in polys]
    
            # ----- New geometry
            
            self.wobject.new_geometry(np.resize(verts, (self.total_verts, 3)), faces)
            
            # ----- uv mapping
            
            for name in wmodel.uvmaps:
                uvs = wmodel.get_uvs(name)
                self.wobject.create_uvmap(name)
                self.wobject.set_uvs(name, np.resize(uvs, (self.size*len(uvs), 2)))
    
            self.wobject.wmaterials.copy_materials_from(wmodel)   
            self.wobject.material_indices = np.resize(wmodel.material_indices, self.wobject.poly_count)
                
            # ---------------------------------------------------------------------------
            # Initialize the models vertices
            
            if animation_key is None:
                self.set_mode('SINGLE', verts)
            else:
                kverts = wmodel.wshape_keys.verts(animation_key)
                if kverts is None:
                    raise WError(f"Crowd initialization error: {wmodel.name} doesn't have shape keys namde '{animation_key}'",
                        Class = "Crowd",
                        Method = "set_mesh_model",
                        wmodel = wmodel,
                        mode = self.mode,
                        name = name,
                        animation_key = animation_key)
                self.set_mode('ANIMATION', kverts)
            
        
    # ====================================================================================================
    # Initialize a curve
    
    def set_curve_model(self, name=None, animation_key=None):
        
        # ----- Create the recipient mesh
        
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj = wrap(name)
            if obj.object_type != 'Curve':
                obj.select_set(True)
                bpy.ops.object.delete() 
                obj = None
                
        if obj is None:
            obj = wrap(name, create="BEZIER")
            
        self.wobject = obj
        blender.copy_collections(self.wmodels[0].wrapped, self.wobject.wrapped)
        
        
        self.set_model(animation_key=animation_key)
        return

        #self.wobject.material_indices = self.wmodel.material_indices # Will be properly broadcasted
            
        # ---------------------------------------------------------------------------
        # ----- Get the model geometry
        
        profile  = wmodel.wsplines.profile
        nsplines = len(profile)
        verts    = wmodel.verts
        
        self.dupli_verts = len(verts)
        self.total_verts = self.size * self.dupli_verts

        # ---------------------------------------------------------------------------
        # ----- Set the model geometry
        
        self.wobject.profile = np.resize(profile, (self.size*nsplines, 2))
        
        #self.wobject.set_verts_types(np.resize(verts, (self.total_verts, 3)), np.resize(types, (self.size*len(types), 2)) )
        
        # ----- Groups
        
        #group_indices = self.wmodel.group_indices()
        #self.groups = {group_name: {'indices': group_indices[group_name], 'center': self.wmodel.group_center(group_name)} for group_name in group_indices}
            
        # ---------------------------------------------------------------------------
        # Initialize the models vertices

        if animation_key is None:
            self.set_mode('SINGLE', verts)
        else:
            kverts = wmodel.wshape_keys.verts(animation_key)
            if kverts is None:
                raise WError(f"Crowd initialization error: {wmodel.name} doesn't have shape keys namde '{animation_key}'",
                    Class = "Crowd",
                    Method = "set_mesh_model",
                    wmodel = wmodel,
                    mode = self.mode,
                    name = name,
                    animation_key = animation_key)
            self.set_mode('ANIMATION', kverts)
    
    # ---------------------------------------------------------------------------
    # Type
    
    @property
    def is_curve(self):
        return self.wmodels[0].object_type == 'Curve'

    @property
    def is_mesh(self):
        return self.wmodels[0].object_type == 'Mesh'
    
    @property
    def is_text(self):
        return self.wmodels[0].object_type == 'CurveText'

    @property
    def is_mesh_text(self):
        if self.is_text:
            return self.wobject.object_type == 'Mesh'
        return False

    @property
    def is_curve_text(self):
        if self.is_text:
            return self.wobject.object_type == 'Curve'
        return False

    # ---------------------------------------------------------------------------
    # Euler order
    
    @property
    def euler_order(self):
        """Overrides the supper class method.
        
        The euler order is read from the Crowd object.

        Returns
        -------
        str
            The euler order to use in the transformations.
        """

        return self.wobject.rotation_euler.order
    
    @property
    def track_axis(self):
        """Overrides the supper class method.

        The track axis is read from the Crowd object.

        Returns
        -------
        str
            The track axis to use in tracking methods.
        """
        
        return self.wobject.track_axis
    
    @property
    def up_axis(self):
        """Overrides the supper class method.

        The up axis is read from the Crowd object.

        Returns
        -------
        str
            The up axis to use in tracking methods.
        """
        
        return self.wobject.up_axis
    
    # ---------------------------------------------------------------------------
    # Change the mode
    
    def set_mode(self, mode, verts, indices=None):
        
        self.mode = mode
        
        # ----- curve verts can have additional info
        
        models = verts[..., :3]
        self.radius_tilt = None
        
        if self.is_curve:
            if verts.shape[-1] == 5:
                self.radius_tilt = verts[..., 3:5]
                
        
        # Store the models as 4-vectors
        
        if mode in ['SINGLE', 'ANIMATION']:
            self.models = np.insert(models, 3, 1, axis=-1)
            if len(np.shape(self.models))== 2:
                self.models = np.expand_dims(self.models, axis=0)

            if len(np.shape(self.models)) != 3:
                raise WError(f"Model shape {np.shape(self.models)} is incorrect fro Crow initialization.\n {mode} required a shape (steps, vertices, 3).",
                        Class="Crowd", Method="set_mode", mode=mode, models_shape=np.shape(models))
                
            if mode == 'ANIMATION':
                self.models_indices = np.zeros(self.shape, np.float)
                self.base_shape = get_full_shape(self.shape, (len(self.models[0]), 4))
                
            self.verts_count = self.size
                
        elif mode == 'VARIABLE':
            
            # ----- Indices to the available models
            
            inds = np.zeros(self.shape, int)
            inds[:] = indices
            inds = inds.reshape(np.size(inds))

            # ----- Will store max size with filter on valid vertices
            
            max_len = 0
            self.verts_count = 0
            for model in models:
                self.verts_count += len(model)
                max_len = max(max_len, len(model))
            
            self.var_indices = np.zeros(0, int)
            for i, i_model in enumerate(inds):
                self.var_indices = np.append(self.var_indices, np.arange(i*max_len, i*max_len + len(models[i_model])))
                
            self.models = np.zeros((self.size, max_len, 4))
            self.models[..., 3] = 1
            
            for i, index in enumerate(inds):
                self.models[i, :len(models[index]), :3] = models[index]
                
            self.models= self.models.reshape(get_full_shape(self.shape, (max_len, 4)))
            self.models_verts_count = self.size * max_len
                
        elif mode == 'INDIVIDUALS':
            
            mdls = np.zeros(self.shape, object)
            mdls[:] = models
            
            mdls = mdls.reshape(np.size(mdls))
            
            self.set_mode('VARIABLE', mdls, np.arange(len(mdls)).reshape(self.shape))
            self.mode = 'INDIVIDUALS'

        else:
            raise WError(f"Unknwon Crowd mode: {mode}. Valid modes are 'SINGLE', 'ANIMATION', 'VARIABLE', 'INDIVIDUALS'",
                        Class="Crowd", Method="set_mode", mode=mode, models_shape=np.shape(models))
            
    # ---------------------------------------------------------------------------
    # Add a group of vertices which can be additionally transformed
    
    def add_sub_transformation(self, name, indices, pivot=None):

        self.group_transfos[name] = GroupTransfo(self, indices, pivot=None)

    def add_group_transformation(self, group_name, pivot_name=None):
        
        pivot = None
        if pivot_name is not None:
            verts = self.wmodels[0].verts[self.wmodels[0].group_indices[pivot_name]]
            n = len(verts)
            if n == 0:
                pivot = np.zeros(3, np.float)
            else:
                pivot = np.sum(verts, axis=-1)/n
        
                
        self.add_sub_transformation(group_name, self.wmodels[0].group_indices(group_name), pivot=pivot)
            
    # ---------------------------------------------------------------------------
    # Set the vertices once transformed
    
    def set_vertices(self, verts):
        
        if self.is_mesh:
            self.wobject.verts = verts
            
        elif self.is_curve:
            self.wobject.verts = verts.reshape(self.total_verts, 3)
            
    # ---------------------------------------------------------------------------
    # Overrides matrices transformations
        
    def apply(self):
        """Overrides the super class method.
        
        Compute vertice of shape (shape, n 3) where shape is the shape of the Crowd and n the number of vertices in a duplicate.
        
        The steps are the following:
            1) build the base (shape, n, 3) from models
            2) Apply group transformation before global transformation
            3) Apply transformations
    
        The first step depends upon the mode:
            SINGLE      : a simple broadcast of the single model. No broadcast if there are no group transformation.
            ANIMATION   : build the base by computing the verts from models and  the models_indices
            INDIVIDUALS : simple copy of the models which are at the correct shape
            VARIABLE    : simple copy of the models without group transformation
            
        Returns
        -------
        None.
        """
        # The slave transformation can call apply()
        
        mem_locked = self.locked
        self.locked = 1
        
        # -----------------------------------------------------------------------------------------------------------------------------
        # Step 1 : prepare the base vertices
        
        # Since the model can manage extra info after the 4 component, we need the length of teh actual vectors
        ndim = np.shape(self.models)[-1]
        
        base = None
        if self.mode == 'SINGLE':
            
            if len(self.group_transfos) == 0:
                base = self.models
            else:
                base = np.resize(self.models, self.shape + (self.dupli_verts, ndim))
                
        elif self.mode == 'ANIMATION':
            
            imax = len(self.models)-1
            if imax == 0:
                base = self.models[np.zeros(self.shape, int)]
            else:
                #t   = np.clip(self.models_indices, 0, imax)

                t   = np.array(self.models_indices, np.float)
                
                i0  = np.floor(t).astype(int)
                i0[i0 < 0]     = 0
                i0[i0 >= imax] = imax-1
                p  = np.expand_dims(np.expand_dims(t - i0, axis=-1), axis=-1)
                
                base = self.models[i0]*(1-p) + self.models[i0+1]*p
                
        else:
            base = self.models
            
                
        # -----------------------------------------------------------------------------------------------------------------------------
        # Step 2 : group transformations
        
        if base is not None:
            
            for gt in self.group_transfos.values():
                #verts[..., gt.indices, :] = self.compose(gt.transfo, center=gt.pivot).transform_verts43(self.verts[gt.indices])
                base[..., gt.indices, :4] = gt.pivot + gt.transfo.transform_verts4(base[..., gt.indices, :4] - gt.pivot)
            
        # -----------------------------------------------------------------------------------------------------------------------------
        # Step 3 : transformation
        
        if base is not None:
            
            # ----- Compute the vertices
            
            verts = self.transform_verts43(base[..., :4])
            
            # ----- Add extra information
            
            if ndim == 6:
                verts = np.insert(verts, (4, 4), 0, axis=-1)
                verts[..., 4] = base[..., 4]
                verts[..., 5] = base[..., 5]
                
            # ----- Apply
            
            if self.extract_indices is None:
                
                self.set_vertices(verts.reshape(self.total_verts, ndim-1))

            else:
            
                self.set_vertices(verts.reshape(self.size*self.dupli_verts, ndim-1)[self.extract_indices])
                
            # ----- Some cleaning of big arrays
            
            del verts
        
        # ----------------------------------------------------------------------------------------------------
        # Restore locked state

        self.locked = mem_locked
        
    # ---------------------------------------------------------------------------
    # Animation with keyshapes
    
    def animate_with_keyshapes(self, name=None):
        """Animate the duplicates by changing the base vertices use for each of them.
        
        When not animated, there is one copy of the vertices.
        When animated, there are several versions of the base vertices.
        The animate method select one of the available version for each of the duplicates.
        
        Phases ans speeds are use to compute the index of the version to use:
            - version of duplicate i at frame f: (phases[i] + f * speeds[i]) % (number of versions)
            
        If no phases is provided, random phases are generated. If the animation must be synchronized
        pass 0 as an argument.
        If no speeds is None, random speeds are generated.
        
        This mechanism is an alternative to the use of eval_time on shape keys with a Duplicator.
        Performances can be higher for simple animations.

        Parameters
        ----------
        points : array of array of vertices
            An array fo shape (steps, v_count, 3) containing the possible versions of the vertices.
        phases : array of int, optional
            The phases to use for each duplicate. The default is None.
        speeds : array of int, optional
            The animation speed of each duplicate. The default is 1.
        seed : any, optional
            Random seed if not None. The default is 0.

        Returns
        -------
        None.
        """
        
        if self.is_mesh:
            verts = self.wmodel.wdata.get_mesh_vertices(name=name)
            
        elif self.is_curve:
            verts = self.wmodel.wdata.get_curve_vertices(name=name)
            
        self.set_mode('ANIMATION', verts)
    



