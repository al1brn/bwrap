#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np
import bpy

from ..blender import blender
from ..maths.transformations import Transformations, SlaveTransformations
from .stacker import Stack

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
        return f"<GroupTransfo: pivot: {self.pivot}, indices: {np.shape(self.indices)}>"
    
    
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
    
    def __init__(self, model, shape=(), name=None):
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
        # ----- The model is a stack

        wmodel = None        
        if isinstance(model, Stack):
            
            self.stack = model
            
            if name is None:
                name = "Crowd"
                
            if hasattr(self.stack, "wobject"): wmodel = self.stack[0].wobject
                
        # ---------------------------------------------------------------------------
        # ----- Wrap the model
        
        else:
        
            wmodel = wrap(model)
            
            if wmodel.object_type not in Crowd.SUPPORTED:
                raise WError(f"Crowd initialization error: {self.wmodel.name} must be in {Crowd.SUPPORTED}, not {self.wmodel.object_type}.",
                    Class = "Crowd",
                    Method = "__init__",
                    model = model,
                    shape = shape)
                
            # ---------------------------------------------------------------------------
            # ----- Use a stack to build the vertices
            
            self.stack = Stack(wmodel.object_type)
            self.stack.stack_object(wmodel, self.size)
            
        
        # ---------------------------------------------------------------------------
        # ----- Create the target object
            
        if name is None:
            name = "Crowd of " + self.wmodel.name + 's'
        
        obj = bpy.data.objects.get(name)
        if obj is not None:
            wobj = wrap(name)
            if wobj.object_type != self.stack.object_type:
                obj.select_set(True)
                bpy.ops.object.delete() 
                obj = None
                
        if obj is None:
            if self.is_mesh:
                wobj = wrap(name, create="CUBE")
            elif self.is_curve:
                wobj = wrap(name, create="BEZIER")
                if wmodel is not None:
                    wobj.wdata.copy_from(wmodel.data)
        
        self.wobject = wobj
        
        wmodel = self.stack[0].wobject
        blender.copy_collections(wmodel.wrapped, self.wobject.wrapped)
        
        # ----- Other initialization
        
        self.animation = False
        self.group_transfos = {}
        
        # ----- Set the geometry from the stack
        
        self.init_from_stack()


    # ====================================================================================================
    # Content
    
    def __repr__(self):
            
        s  = f"<Crowd {self.shape}, {self.size} duplicates, type={self.stack.object_type}: '{self.wobject.name}'\n"
        s += f"   stack                  : {self.stack}\n"
        s += f"   total vertices         : {self.nverts}\n"
        s += f"   animation              : {self.animation}"
        if self.animation:
            s += f", {len(self.models)} steps\n"
        else:
            s += "\n"
        s += f"Group transformations: {len(self.group_transfos)}\n"
        for name, gt in self.group_transfos.items():
            s += f"{name}: {gt}\n"
        return s + ">"
    
    # ====================================================================================================
    # Init from a stack
    
    @classmethod
    def FromStack(cls, stack, name="Crowd"):
        return cls(stack, shape=stack.dupli_count, name=name)
        
    # ====================================================================================================
    # Init from a list of objects
    
    @classmethod
    def Crowds(cls, names, count=1, name="Crowd", shuffle=True, seed=None):
        
        first = wrap(names[0])
        stack = Stack(first.object_type)
        
        counts = np.ones(len(names), int)
        counts[:] = count
        for nm, n in zip(names, counts):
            stack.stack_object(nm, n)
            
        if shuffle:
            stack.shuffle(seed)
            
        return cls.FromStack(stack, name)
    
    # ====================================================================================================
    # The main model is the first one in the stack
    
    @property
    def wmodel(self):
        if len(self.stack) != 1:
            raise WError("Crowd model error. The crowd doesn't have a single model.",
                         Class = "Crowd",
                         Property = "wmodel",
                         stack = self.stack,
                         crowd = self)
        
        if not hasattr(self.stack[0], "wobject"):
            raise WError("Crowd model error. The stack doesn't contain an object.",
                         Class = "Crowd",
                         Property = "wmodel",
                         stack = self.stack,
                         crowd = self)
        
        return self.stack[0].wobject
    
    # ====================================================================================================
    # Set the models vertices
    # If animation exists, the models are the available key shapes to use to compute actual base vertices
    # If no animation, the models are either:
    # - an array (1, nverts, ndim) of base vertices to duplicates in base array of shape duplicates
    # - an array of (shape, nverts, ndim) which is the base array of the duplicates
    
    def set_models(self, models):
        
        
        # Make sure the models shape is 3 long
        
        if len(models.shape) == 2:
            models = models.reshape((1,) + models.shape)

        # Store the verts and add the 1 to have 4-vectors
        # Note that curve can have extra info after the 3-vector
            
        self.models = np.insert(models, 3, 1, axis=-1)

    # ====================================================================================================
    # Set the models vertices

    def init_from_stack(self):
        
        # ---------------------------------------------------------------------------
        # Set the geometry to the object
        
        self.stack.set_to_object(self.wobject)
        self.nverts      = self.wobject.verts_count
        self.block_size  = self.stack.max_nverts
        self.total_verts = self.size * self.block_size
        
        # ---------------------------------------------------------------------------
        # Set the base vertices
        
        self.set_models(self.stack.get_crowd_bases())
        self.animation     = False
        self.true_vertices = self.stack.get_true_vert_indices()
        
    # ====================================================================================================
    # Set an animation
    
    def set_animation(self, key_shapes):
        
        # ----- Some controls
        
        if len(self.stack) > 1:
            raise WError("Crowd animation initialization error: animation can be set only with one model.",
                    Class = "Crowd",
                    Method = "set_animation",
                    key_shapes = np.shape(key_shapes),
                    stack = self.stack)
            
        if self.stack[0].nverts != np.shape(key_shapes)[-2]:
            raise WError("Crowd animation initialization error: the number of vertices don't macth between the base and the animation shapes",
                    Class = "Crowd",
                    Method = "set_animation",
                    vertices_count = self.stack[0].nverts,
                    key_shapes = np.shape(key_shapes),
                    stack = self.stack)
            
        self.set_models(key_shapes)
        self.animation      = True
        self.models_indices = np.zeros(self.shape, np.float)
        

    # ====================================================================================================
    # Utilities
    
    @property
    def is_curve(self):
        return self.stack.is_curve

    @property
    def is_mesh(self):
        return self.stack.is_mesh
    
    @property
    def is_text(self):
        return self.stack.object_type == 'CurveText'

    @property
    def is_mesh_text(self):
        if self.is_text:
            return self.stack.object_type == 'Mesh'
        return False

    @property
    def is_curve_text(self):
        if self.is_text:
            return self.stack.object_type == 'Curve'
        return False
    
    # ====================================================================================================
    # Transformations interface
    
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
    
    # ====================================================================================================
    # Partial transformations on group of vertices

    # ---------------------------------------------------------------------------
    # Add a group of vertices which can be additionally transformed
    
    def add_sub_transformation(self, name, indices, pivot=None):

        self.group_transfos[name] = GroupTransfo(self, indices, pivot=None)
        return self.group_transfos[name]

    def add_group_transformation(self, group_name, pivot_name=None):
        
        pivot = None
        if pivot_name is not None:
            verts = self.wmodel.verts[self.wmodel.group_indices[pivot_name]]
            n = len(verts)
            if n == 0:
                pivot = np.zeros(3, np.float)
            else:
                pivot = np.sum(verts, axis=-1)/n
        
                
        return self.add_sub_transformation(group_name, self.wmodel.group_indices(group_name), pivot=pivot)
        
    # ====================================================================================================
    # Set the vertices once transformed
    
    def set_vertices(self, verts):
        
        if self.is_mesh:
            if self.true_vertices is None:
                self.wobject.verts = verts.reshape(self.nverts, 3)
            else:
                print(self.true_vertices)
                self.wobject.verts = verts.reshape(self.total_verts, 3)[self.true_vertices]
            
        elif self.is_curve:
            
            ndim = np.shape(verts)[-1]
            
            if self.true_vertices is None:
                self.wobject.verts = verts.reshape(self.nverts, ndim)
            else:
                self.wobject.verts = verts.reshape(self.total_verts, ndim)[self.true_vertices]
            
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
        
        # Since the model can manage extra info after the 4 component, we need the length of the actual vectors
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
            if (len(self.group_transfos) == 0) and (len(self.stack) == 1):
                # Models is shape (1, verts, ndim)
                base = self.models
            else:
                # Models is shape (shape, verts, dim)
                # Make sure the shape is correct
                base = np.resize(self.models, self.shape + (self.block_size, ndim))
                
 
        # -----------------------------------------------------------------------------------------------------------------------------
        # Step 2 : group transformations
        
        for gt in self.group_transfos.values():
            base[..., gt.indices, :4] = gt.pivot + gt.transfo.transform_verts4(base[..., gt.indices, :4] - gt.pivot)
            
        # -----------------------------------------------------------------------------------------------------------------------------
        # Step 3 : transformation
        
        # ----- Compute the vertices
        
        verts = self.transform_verts43(base[..., :4])
        
        # ----- Add extra information
        # if ndim == 6 : extra info is only radius and tilt
        # if ndim == 7 : radius, tilt and w
        
        if ndim >= 6:
            # base is structure : V4 extra info
            # We must copy extra info after V3
            
            if ndim == 7:
                verts = np.insert(verts, (3, 3, 3), 0, axis=-1)
            else:
                verts = np.insert(verts, (3, 3), 0, axis=-1)
                
            verts[..., 3:] = base[..., 4:]
            
        # ----- Apply
        
        self.set_vertices(verts.reshape(self.total_verts, ndim-1))
            
        # ----- Some cleaning of big arrays
        
        del verts
        
        # ----------------------------------------------------------------------------------------------------
        # Restore locked state

        self.locked = mem_locked
        
    # ---------------------------------------------------------------------------
    # Animation with keyshapes
    
    def animate_with_shapekeys(self, name=None):
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
        
            
        self.set_animation(self.wmodel.wshape_keys.verts(name))
    



