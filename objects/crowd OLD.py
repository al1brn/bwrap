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
            verts = self.crowd.wmodel.verts[indices]
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
        # ----- Wrap the model
        
        self.wmodel = wrap(model)
        
        if self.wmodel.object_type not in Crowd.SUPPORTED:
            raise WError(f"Crowd initialization error: {self.wmodel.name} must be in {Crowd.SUPPORTED}, not {self.wmodel.object_type}.",
                Class = "Crowd",
                Method = "__init__",
                model = model,
                shape = shape)
            
        if name is None:
            name = "Crowd of " + self.wmodel.name + 's'
            
        self.animation = False
            
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
            
        s  = f"<Crowd {self.shape}, {self.size} duplicates, of '{self.wmodel.name}', type={self.wmodel.object_type}: '{self.wobject.name}'\n"
        s += f"   model                 : {self.wmodel.name}\n"
        s += f"   vertices per duplicate: {self.dupli_verts}\n"
        s += f"   total vertices        : {self.total_verts}\n"
        s += f"   animation             : {self.animation}"
        if self.animation:
            s += f", {len(self.models)} steps\n"
        else:
            s += "\n"
        return s + ">"
    
    # ====================================================================================================
    # Set the models vertices
    
    def set_models(self, verts, animation=False):
        
        # ----------------------------------------------------------------------------------------------------
        # Initialize the models vertices: an array of steps base model: (steps, dupli_verts, verts_dim)
        
        if len(verts.shape) == 2:
            verts = verts.reshape((1,) + verts.shape)
            
        # Store the verts and add the 1 to have 4-vectors
        # Note that curve can have extra info after the 3-vector
            
        self.models = np.insert(verts, 3, 1, axis=-1)
        
        # ----------------------------------------------------------------------------------------------------
        # Animation
        
        self.animation = animation
        if self.animation:
            self.models_indices = np.zeros(self.shape, np.float)
        

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
        blender.copy_collections(self.wmodel.wrapped, self.wobject.wrapped)
        
        # ---------------------------------------------------------------------------
        # Copy the materials
        
        self.wobject.wmaterials.copy_materials_from(self.wmodel)
            
        # ---------------------------------------------------------------------------
        # ----- Geometry dimension
        
        verts = np.array(self.wmodel.verts)
        
        self.dupli_verts = len(verts)
        self.total_verts = self.size * self.dupli_verts
        
        # ---------------------------------------------------------------------------
        # ----- Build the new geometry
        
        # ----- Polygons
        
        polys = self.wmodel.poly_indices
        faces = [[index + i*self.dupli_verts for index in face] for i in range(self.size) for face in polys]

        # ----- New geometry
        
        self.wobject.new_geometry(np.resize(verts, (self.total_verts, 3)), faces)
        
        # ---------------------------------------------------------------------------
        # ----- Material indices
        
        self.wobject.wmaterials.copy_materials_from(self.wmodel, append=False)   
        self.wobject.material_indices = np.resize(self.wmodel.material_indices, self.wobject.poly_count)
        
        # ---------------------------------------------------------------------------
        # ----- uv mapping
        
        for name in self.wmodel.uvmaps:
            uvs = self.wmodel.get_uvs(name)
            self.wobject.create_uvmap(name)
            self.wobject.set_uvs(name, np.resize(uvs, (self.size*len(uvs), 2)))
        
        # ---------------------------------------------------------------------------
        # ----- Animation key
        
        if animation_key is None:
            
            self.set_models(verts, animation=False)
            
        else:
            
            kverts = self.wmodel.wshape_keys.verts(animation_key)
            if kverts is None:
                raise WError(f"Crowd initialization error: {self.wmodel.name} doesn't have shape keys namde '{animation_key}'",
                    Class = "Crowd",
                    Method = "set_mesh_model",
                    wmodel = self.wmodel,
                    name = name,
                    animation_key = animation_key)
            self.set_models(kverts, animation=True)
        
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
        blender.copy_collections(self.wmodel.wrapped, self.wobject.wrapped)
        self.wobject.wdata.copy_from(self.wmodel.data)
        
        # ---------------------------------------------------------------------------
        # Copy the materials
        
        self.wobject.wmaterials.copy_materials_from(self.wmodel, append=False)
        
        # ---------------------------------------------------------------------------
        # ----- Geometry dimension
        
        profile  = self.wmodel.wsplines.profile
        nsplines = len(profile)
        verts    = self.wmodel.verts
        
        self.dupli_verts = len(verts)
        self.total_verts = self.size * self.dupli_verts

        # ---------------------------------------------------------------------------
        # ----- Build the new geometry
        
        self.wobject.profile = np.resize(profile, (self.size*nsplines, 2))
        self.wobject.verts   = np.resize(verts, (self.size*len(verts), verts.shape[-1]))
        
        # ---------------------------------------------------------------------------
        # ----- Splines properties

        self.wobject.wmaterials.copy_materials_from(self.wmodel)   
        
        model_splines = self.wmodel.wsplines
        splines = self.wobject.wsplines
        for i_spline, spline_model in enumerate(model_splines):
            for i in range(0, len(splines), len(model_splines)):
                splines[i].copy_from(spline_model)
        
        # ---------------------------------------------------------------------------
        # ----- Material indices
        
        #mat_indices = self.wmodel.material_indices
        #self.wobject.material_indices = np.resize(mat_indices, self.size*len(mat_indices))
        
        # ---------------------------------------------------------------------------
        # ----- Animation key
            
        if animation_key is None:
            
            self.set_models(verts, animation=False)
            
        else:
            
            kverts = self.wmodel.wshape_keys.verts(animation_key)
            if kverts is None:
                raise WError(f"Crowd initialization error: {self.wmodel.name} doesn't have shape keys namde '{animation_key}'",
                    Class = "Crowd",
                    Method = "set_mesh_model",
                    wmodel = self.wmodel,
                    name = name,
                    animation_key = animation_key)
            self.set_models(kverts, animation=True)
    
    # ---------------------------------------------------------------------------
    # Type
    
    @property
    def is_curve(self):
        return self.wmodel.object_type == 'Curve'

    @property
    def is_mesh(self):
        return self.wmodel.object_type == 'Mesh'
    
    @property
    def is_text(self):
        return self.wmodel.object_type == 'CurveText'

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
    # Add a group of vertices which can be additionally transformed
    
    def add_sub_transformation(self, name, indices, pivot=None):

        self.group_transfos[name] = GroupTransfo(self, indices, pivot=None)

    def add_group_transformation(self, group_name, pivot_name=None):
        
        pivot = None
        if pivot_name is not None:
            verts = self.wmodel.verts[self.wmodel.group_indices[pivot_name]]
            n = len(verts)
            if n == 0:
                pivot = np.zeros(3, np.float)
            else:
                pivot = np.sum(verts, axis=-1)/n
        
                
        self.add_sub_transformation(group_name, self.wmodel.group_indices(group_name), pivot=pivot)
            
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
        
        Compute vertices of shape (shape, n 3) where shape is the shape of the Crowd and n the number of vertices in a duplicate.
        
        The steps are the following:
            1) build the base (shape, n, 3) from models
            2) Apply group transformation before global transformation
            3) Apply transformations
    
        The first step depends upon the mode:
            SINGLE      : a simple broadcast of the single model. No broadcast if there are no group transformation.
            ANIMATION   : build the base by computing the verts from models and  the models_indices
            
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
        
        if self.animation:
            
            imax = len(self.models)-1
            if imax == 0:
                base = self.models[np.zeros(self.shape, int)]
            else:
                t   = np.array(self.models_indices, np.float)
                
                i0  = np.floor(t).astype(int)
                i0[i0 < 0]     = 0
                i0[i0 >= imax] = imax-1
                p  = np.expand_dims(np.expand_dims(t - i0, axis=-1), axis=-1)
                
                base = self.models[i0]*(1-p) + self.models[i0+1]*p        
        else:

            if len(self.group_transfos) == 0:
                base = self.models
            else:
                base = np.resize(self.models, self.shape + (self.dupli_verts, ndim))
 
        # -----------------------------------------------------------------------------------------------------------------------------
        # Step 2 : group transformations
        
        for gt in self.group_transfos.values():
            #verts[..., gt.indices, :] = self.compose(gt.transfo, center=gt.pivot).transform_verts43(self.verts[gt.indices])
            base[..., gt.indices, :4] = gt.pivot + gt.transfo.transform_verts4(base[..., gt.indices, :4] - gt.pivot)
            
        # -----------------------------------------------------------------------------------------------------------------------------
        # Step 3 : transformation
        
        # ----- Compute the vertices
        
        verts = self.transform_verts43(base[..., :4])
        
        # ----- Add extra information
        
        if ndim == 6:
            verts = np.insert(verts, (4, 4), 0, axis=-1)
            verts[..., 4] = base[..., 4]
            verts[..., 5] = base[..., 5]
            
        # ----- Apply
        
        self.set_vertices(verts.reshape(self.total_verts, ndim-1))
            
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
    



