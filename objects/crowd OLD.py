#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np
import bpy

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
        - A single block which is duplicated according the shape of the Tranformation
        - An array of blocks. At each frame, one version is selected for each duplicate
        - As many blocks as the shape of the transfpormations
        
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
    
    def __init__(self, crowd_type='Mesh', shape=(), name="Crowd", model=None, var_blocks=False):
        """Initialize an empty crowd of a given type.
        
        Parameters
        ----------
        crowd_type : str in 'Mesh', 'Curve'
            Type of crowd
        shape : shape, optional
            Shape of the crowd. The default is 10.
        name : str
            Name of the crowd object. If not provided (default), 'Crowd of models' is used.

        Returns
        -------
        None.
        """
        
        # ---------------------------------------------------------------------------
        # ----- Super initialization
        
        super().__init__(count=shape)

        # ---------------------------------------------------------------------------
        # ----- If var_blocks is True, the blocks are not the same size and the
        # ----- the transformations must be computed with a loop, not with np.matmul
        
        self.var_blocks = var_blocks
        
        # ---------------------------------------------------------------------------
        # ----- No stack
        
        self.stack = None
        
        # ---------------------------------------------------------------------------
        # ----- Create the target object
            
        obj = bpy.data.objects.get(name)
        if obj is not None:
            wobj = wrap(name)
            if wobj.object_type != crowd_type:
                obj.select_set(True)
                bpy.ops.object.delete() 
                obj = None
                
        if obj is None:
            if crowd_type == 'Mesh':
                wobj = wrap(name, create="CUBE")
            elif crowd_type == 'Curve':
                wobj = wrap(name, create="BEZIER")
        
        self.wobject = wobj
        
        # ---------------------------------------------------------------------------
        # ----- Model
        
        self.wmodel = wrap(model)
        
        # ----- Other initialization
        
        self.animation = False
        self.group_transfos = {}
        self.deformations   = []
        self.deformation_time = 0. # parameter used in deformation
        
    # ====================================================================================================
    # Content
    
    def __repr__(self):
            
        s  = f"<Crowd {self.shape}, {self.size} duplicates, type={self.object_type}: '{self.wobject.name}'\n"
        s += f"   locked                 : {self.locked}\n"
        s += f"   stack                  : {self.stack}\n"
        s += f"   vertices count         : {self.nverts}\n"
        s += f"   var blocks             : {self.var_blocks}\n"
        if not self.var_blocks:
            s += f"   total vertices         : {self.total_verts}\n"
            s += f"   block size             : {self.block_size}\n"
    
            ratio = self.nverts/self.total_verts*100
            s += f"   use vertices           : {ratio:.1f}%\n"
        
            if self.true_vertices is None:
                tvs = None
            else:
                tvs = f"{len(self.true_vertices)} -> {self.true_vertices}"
                if len(tvs) > 40:
                    tvs = tvs[:40] + '...]'
            
            s += f"   true vertices          : {tvs}\n"
        
        s += f"   animation              : {self.animation}"
        if self.animation:
            s += f", {len(self.blocks)} steps\n"
        else:
            s += "\n"
        s += f"Group transformations: {len(self.group_transfos)}\n"
        for name, gt in self.group_transfos.items():
            s += f"{name}: {gt}\n"
        return s + ">"
        
    # ====================================================================================================
    # Initialize from an object
    
    @classmethod
    def FromObject(cls, model, shape=(), name=None):
        """Initialize the crowd by creating shape duplicates of the model.
        
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
        # ----- The model
        
        wmodel = wrap(model)
        
        if wmodel.object_type not in Crowd.SUPPORTED:
            raise WError(f"Crowd initialization error: {wmodel.name} must be in {Crowd.SUPPORTED}, not {wmodel.object_type}.",
                Class = "Crowd",
                Method = "FromObject",
                model = model,
                shape = shape)
            
        if name is None:
            name = "Crowd of " + wmodel.name + 's'
        
        # ---------------------------------------------------------------------------
        # ----- Create the crowd
        
        crowd = Crowd(wmodel.object_type, shape, name, wmodel, var_blocks=False)
        
        # ---------------------------------------------------------------------------
        # ----- Use a stack to build the vertices
        
        crowd.stack = Stack(wmodel.object_type, var_blocks=crowd.var_blocks)
        crowd.stack.stack_object(wmodel, crowd.size)
        
        # ----- Set the geometry from the stack
        
        crowd.init_from_stack(shape)
        
        # ----- Done
        
        return crowd


    # ====================================================================================================
    # Init from a stack
    
    @classmethod
    def FromStack(cls, stack, model, name="Crowd", var_blocks=False):
        crowd = cls(stack.object_type, shape=(stack.dupli_count,), name=name, model=model, var_blocks=var_blocks)
        crowd.stack = stack
        crowd.init_from_stack()
        
        return crowd
    
        
    # ====================================================================================================
    # Init from a list of objects
    
    @classmethod
    def Crowds(cls, names, count=1, name="Crowd", shuffle=True, seed=None, var_blocks=False):
        
        first = wrap(names[0])
        stack = Stack(first.object_type)
        
        counts = np.ones(len(names), int)
        counts[:] = count
        for nm, n in zip(names, counts):
            stack.stack_object(nm, n)
            
        if shuffle:
            stack.shuffle(seed)
            
        return cls.FromStack(stack, name, var_blocks=var_blocks)
    
    # ====================================================================================================
    # Initialize from the faces of a Mesh object
    
    @classmethod
    def Faces(cls, model, name=None, var_blocks=False):
        """Initialize the crowd from the faces of a Mesh object.
        """
        
        # ---------------------------------------------------------------------------
        # ----- The model
        
        wmodel = wrap(model)
        
        if wmodel.object_type != 'Mesh':
            raise WError(f"Crowd from mesh faces initialization error: {wmodel.name} must be a mesh, not {wmodel.object_type}.",
                Class = "Crowd",
                Method = "Faces",
                model = model)
            
        if name is None:
            name = "Faces of " + wmodel.name
        
        # ---------------------------------------------------------------------------
        # ----- Create the crowd
        
        crowd = Crowd(wmodel.object_type, None, name, wmodel, var_blocks=var_blocks)
        
        # ---------------------------------------------------------------------------
        # ----- Use a stack to build the vertices
        
        crowd.stack = Stack(wmodel.object_type, var_blocks=crowd.var_blocks)
        crowd.stack.stack_faces(wmodel)
        
        # ----- Set the geometry from the stack
        
        crowd.init_from_stack(shape=None)
        
        # ----- Initial locations matter
        
        crowd.location = [stacker.location for stacker in crowd.stack]
        
        
        # ----- Done
        
        return crowd   
    
    # ====================================================================================================
    # Initialize from group of faces of a Mesh object
    
    @classmethod
    def GroupedFaces(cls, model, groups, name=None, var_blocks=False):
        """Initialize the crowd from the faces of a Mesh object.
        """
        
        # ---------------------------------------------------------------------------
        # ----- The model
        
        wmodel = wrap(model)
        
        if wmodel.object_type != 'Mesh':
            raise WError(f"Crowd from mesh faces initialization error: {wmodel.name} must be a mesh, not {wmodel.object_type}.",
                Class = "Crowd",
                Method = "Faces",
                model = model)
            
        if name is None:
            name = "Faces of " + wmodel.name
            
        # ---------------------------------------------------------------------------
        # ----- Create the crowd
        
        crowd = Crowd(wmodel.object_type, None, name, wmodel, var_blocks=var_blocks)
        
        # ---------------------------------------------------------------------------
        # ----- Set the geometry from the grouped faces
        
        crowd.init_from_faces(groups=groups, shape=None)
        
        # ----- Done
        
        return crowd        
    
    # ====================================================================================================
    # The main model is the first one in the stack
    
    @property
    def wmodel(self):
        return self.wmodel_
    
    @wmodel.setter
    def wmodel(self, value):
        self.wmodel_ = value
        if self.wmodel_ is None:
            return
        
        if (self.wobject.object_type == 'Curve') and (self.wmodel.object_type == 'Curve'):
            self.wobject.wdata.copy_from(self.wmodel.data)
            
        #blender.copy_collections(self.wmodel.wrapped, self.wobject.wrapped)
        
    def copy_materials_from_model(self):
        if self.wmodel_ is None:
            return
        
        self.wobject.wmaterials.clear()
        self.wobject.wmaterials.copy_materials_from(self.wmodel, append=True)
    
    # ====================================================================================================
    # Set the blocks vertices
    # If animation exists, the blocks are the available key shapes to use to compute actual base vertices
    # If no animation, the blocks are either:
    # - if var_blocks, a list of array(n, ndim)
    # - if not var_blocks:
    #   - an array (1, nverts, ndim) of base vertices to duplicates in base array of shape duplicates
    #   - an array of (shape, nverts, ndim) which is the base array of the duplicates
    
    def set_block(self, index, block):
        
        if self.var_blocks:
            
            if len(self.blocks[index]) != len(block):
                chars = self.ftext.array_of_chars
                raise WError(f"Impossible to set a block of length {len(block)} in the Crowd '{self.wobject.name}'.\n" +
                             f"Expected length is {len(self.blocks[index])}.",
                             Class = "Crowd", Method="set_block",
                             index = index,
                             block_shape = np.shape(block),
                             lengths=[(i, chars[i], len(self.blocks[i])) for i in range(len(chars))])
                
            # Insert a fourth dimension
            self.blocks[index] = np.insert(block, 3, 1, axis=-1)
            
        else:
            raise WError("Not yet implemented")
        
    
    def set_blocks(self, blocks):
        
        if self.var_blocks:
            
            if len(blocks) != self.size:
                raise WError("With variable size blocks, the number of blocks must match the size of the transformation.",
                            Class = "Crowd", Method="set_blocks",
                            shape = self.shape, size = self.size, blocks_count=len(blocks))

            self.blocks = [None] * len(blocks)
            self.blocks_slices = [None] * len(blocks)
            self.verts_count = 0
            
            index = 0
            for i, block in enumerate(blocks):
                
                # Insert a fourth dimension
                self.blocks[i]   = np.insert(block, 3, 1, axis=-1)
                
                # Create the slices
                self.verts_count += len(block)
                self.blocks_slices[i] = slice(index, index + len(block))
                index += len(block)
            
        else:
        
            # Make sure the blocks shape is 3 long
            
            if len(blocks.shape) == 2:
                blocks = blocks.reshape((1,) + blocks.shape)
    
            # Store the verts and add the 1 to have 4-vectors
            # Note that curve can have extra info after the 3-vector
                
            self.blocks = np.insert(blocks, 3, 1, axis=-1)

    # ====================================================================================================
    # Init the Crowd from stackers stacked in a stcak :-)

    def init_from_stack(self, shape=None):
        
        self.lock()
        
        self.resize(self.stack.dupli_count)
        
        # ---------------------------------------------------------------------------
        # Set the geometry to the object
        
        self.copy_materials_from_model()  

        self.stack.set_to_object(self.wobject)
        self.nverts      = self.wobject.verts_count
        self.block_size  = self.stack.max_nverts
        self.total_verts = self.size * self.block_size
        
        # ---------------------------------------------------------------------------
        # Set the base vertices
        
        self.set_blocks(self.stack.get_crowd_bases())
        self.animation     = False
        self.true_vertices = self.stack.get_true_vert_indices()
        
        self.unlock()
        
        if shape is not None:
            self.reshape(shape)
            
    # ====================================================================================================
    # Init the Crowd from faces, grouped or not, of an object

    def init_from_faces(self, groups=None, shape=None):
        
        wmesh = self.wmodel.wmesh
        
        self.lock()
        
        if groups is None:
            groups = [[i] for i in range(wmesh.poly_count)]
        count = len(groups)
        
        self.resize(count)
        
        # ---------------------------------------------------------------------------
        # Get faces and vertices from the model
        
        verts, faces = wmesh.explode(groups)
        self.wobject.new_geometry(verts, faces)
        
        # ---------------------------------------------------------------------------
        # New vertices can have been created but faces / uv maps are identical
        
        self.copy_materials_from_model()
        self.wobject.wmesh.material_indices = wmesh.material_indices
        self.wobject.wmesh.all_uvs = wmesh.all_uvs
        
        # ---------------------------------------------------------------------------
        # Now we can set the blocks to animate de groups
        
        block_size = 0
        b_inds = [None] * count
        for i_group, group in enumerate(groups):
            inds = self.wobject.wmesh.get_group_indices(group)
            block_size = max(len(inds), block_size)
            b_inds[i_group] = inds
            
        blocks = np.zeros((count, block_size, 3), float)
        verts  = self.wobject.wmesh.verts
        locs   = np.zeros((count, 3), float)
        true_inds = []
        true_inds_offset = 0
        for i_block, inds in enumerate(b_inds):
            n = len(inds)
            vs = verts[inds]
            locs[i_block] = np.average(vs, axis=0)
            blocks[i_block, :n, :] = vs - locs[i_block]
            
            true_inds.extend(list(np.array(inds) + true_inds_offset))
            true_inds_offset += block_size - n
            
            
        self.var_blocks = False
        self.set_blocks(blocks)
        
        self.animation     = False

        # Dimensions
        
        self.nverts        = self.wobject.verts_count
        self.block_size    = block_size
        self.total_verts   = self.size * self.block_size
        self.true_vertices = true_inds
        
        # Blocks initial location
        
        self.location = locs
        self.unlock()
        
        if shape is not None:
            self.reshape(shape)            
        
    # ====================================================================================================
    # Set an animation
    
    def set_animation(self, key_shapes):
        
        if self.var_blocks:
            raise WError("Impossible to animate a Crowd width blocks of variable size")
        
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
            
        self.set_blocks(key_shapes)
        self.animation      = True
        self.blocks_indices = np.zeros(self.shape, np.float)
        
    # ====================================================================================================
    # Set a deformation
    # deform a function of template :
    # f(block, index=None) --> block
    
    def add_deformation(self, deform):
        self.deformations.append(deform)
        

    # ====================================================================================================
    # Utilities
    
    @property
    def object_type(self):
        if self.stack is None:
            return self.wmodel.type
        else:
            return self.stack.object_type
    
    @property
    def is_curve(self):
        if self.stack is None:
            return self.wmodel.type == 'CURVE'
        else:
            return self.stack.is_curve

    @property
    def is_mesh(self):
        if self.stack is None:
            return self.wmodel.type == 'MESH'
        else:
            return self.stack.is_mesh
    
    @property
    def is_text(self):
        if self.stack is None:
            return False
        else:
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
    
    @euler_order.setter
    def euler_order(self, value):
        self.wobject.rotation_euler.order = value
        
    
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
    
    @track_axis.setter
    def track_axis(self, value):
        if value in ['X', '+X', 'POS_X']:
            self.wobject.track_axis = 'POS_X'
        elif value in ['-X', 'NEG_X']:
            self.wobject.track_axis = 'NEG_X'
        elif value in ['Y', '+Y', 'POS_Y']:
            self.wobject.track_axis = 'POS_Y'
        elif value in ['-Y', 'NEG_Y']:
            self.wobject.track_axis = 'NEG_Y'
        elif value in ['Z', '+Z', 'POS_Z']:
            self.wobject.track_axis = 'POS_Z'
        elif value in ['-Z', 'NEG_Z']:
            self.wobject.track_axis = 'NEG_Z'
        
    
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
    
    @up_axis.setter
    def up_axis(self, value):
        self.wobject.up_axis = value
        
    
    # ====================================================================================================
    # Duplicates have vertices, faces, material indices...
    # We need to get access to these items through duplicates coordinates
    
    def verts_indices(self, dupl_indices=None):
        return self.stack.verts_indices(dupl_indices)
    
    def mats_indices(self, dupl_indices=None):
        return self.stack.mats_indices(dupl_indices)
    
    def faces_indices(self, dupl_indices=None):
        return self.stack.faces_indices(dupl_indices)
    
    def uvs_indices(self, dupl_indices=None):
        return self.stack.uvs_indices(dupl_indices)
    
    def prof_indices(self, dupl_indices=None):
        return self.stack.prof_indices(dupl_indices)
    
    
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
        
        # ----------------------------------------------------------------------------------------------------
        # Variable blocks
        
        if self.var_blocks:
            
            ndim = np.shape(self.blocks[0])[-1]
            
            verts = np.zeros((self.verts_count, ndim-1))
            
            mem_shape = self.shape
            self.reshape(self.size)
            
            for dupl_index in range(self.size):
                
                block = np.array(self.blocks[dupl_index])
                
                # ----- Group transformation
                
                for gt in self.group_transfos.values():
                    block[gt.indices, :4] = gt.pivot + gt.transfo[dupl_index].transform_verts4(block[gt.indices, :4] - gt.pivot)
                    
                # ----- Deformations
                
                for deform in self.deformations:
                    block[..., :3] = deform(self.deformation_time, block[..., :3], dupl_index)
                    
                # ---- Transformation
                
                block[..., :3] = self[dupl_index].transform_verts43(block[..., :4])
                
                # ----- Set in the target array while suppressing the 4th dim
                
                verts[self.blocks_slices[dupl_index]] = np.delete(block, 3, axis=-1)
                
            self.reshape(mem_shape)
                
            # ----- Set the vertices to the object
            
            self.set_vertices(verts.reshape(self.verts_count, ndim-1))

        # ----------------------------------------------------------------------------------------------------
        # Fixed size blocks
        
        else:
        
            # -----------------------------------------------------------------------------------------------------------------------------
            # Step 1 : prepare the base vertices
            
            # Since the model can manage extra info after the 4th component, we need the length of the actual vectors
            ndim = np.shape(self.blocks)[-1]
            
            if self.animation:
                
                imax = len(self.blocks)-1
                if imax == 0:
                    base = self.blocks[np.zeros(self.shape, int)]
                else:
                    t   = np.array(self.blocks_indices, np.float)
                    
                    i0  = np.floor(t).astype(int)
                    i0[i0 < 0]     = 0
                    i0[i0 >= imax] = imax-1
                    p  = np.expand_dims(np.expand_dims(t - i0, axis=-1), axis=-1)
                    
                    base = self.blocks[i0]*(1-p) + self.blocks[i0+1]*p
                    
            else:
                if self.stack is None:
                    stack1 = False
                else:
                    stack1 = len(self.stack) == 1
                
                if (len(self.group_transfos) == 0) and (len(self.deformations) == 0) and stack1:
                    
                    # Models is shape (1, verts, ndim)
                    base = self.blocks
                else:
                    # Models is shape (shape, verts, dim)
                    # Make sure the shape is correct
                    base = np.resize(self.blocks, self.shape + (self.block_size, ndim))
                    
            # -----------------------------------------------------------------------------------------------------------------------------
            # Step 2 : group transformations and deformations
            
            for gt in self.group_transfos.values():
                base[..., gt.indices, :4] = gt.pivot + gt.transfo.transform_verts4(base[..., gt.indices, :4] - gt.pivot)
                
            for deform in self.deformations:
                base = deform(self.deformation_time, base)
                
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
    