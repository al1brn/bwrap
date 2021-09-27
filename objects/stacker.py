#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:28:15 2021

@author: alain
"""

import numpy as np

from ..wrappers.wrap_function import wrap

from ..core.commons import WError

# =============================================================================================================================
# Duplicates in a crowd are generated from stackers stacked in a stack.
#
# The purpose is to generate a block of vertices per duplicate, each block having the same
# size to allow transformations by transformation matrices.
#
# The target object is made only with the true vertices plus de complementary informations
# such as faces for meshes or splines types for curves.
#
# A stacker provides the geometry of the duplicates. This geometry comes from an object or
# from something else, for instance from a True Type Font file.
#
# The geometry includes necessarilty:
# - vertices
# - material indices
#
# For mesh geometry:
# - faces
# - uv maps
#
# For curve geometry:
# - profile
#
# Let's remind that profiles are used to define the geometry of splines in a curve.
# The length of a profile is the number of splines. Each entry provides 3 informations:
# - 1 or 3 : 1 for nurbs and 3 for bezier curves
# - n      : the number of vertices in the spline. The actual number of vertices in the verts
#            array is to be multiplied by 1 or 3
# - type   : the code of the spline type, 0 for bezier curves
# Note that bezier curves are coded redundantly with [3, n, 0]
#
# The stack contains s stackers and each stacker generates n_s duplicates. The total
# number of duplicates is n.
# n = n_0 + n_1 + ... + n_s
#
# The duplicates are ordered as required by the transformation matrices. For instance, in 
# a string (Crowd of chars), the duplicates of char 'e' are located as the 'e' are in the string.
#
# Each stacker stores the array of the indices of its duplicates
# - stacker 0 : dupl_indices = [3, 6, 7]
# - stacker 1 : dupl_indices = [1, 2, 5]
# - stacker 2 : dupl_indices = [0, 4]
#
# Duplicate 5 is the third duplicate of stacker 1
#
# The geometry is built in the target object by stacking the geometry provided by stackers.
# The geometry don't have necessarily the same size. The transformation matrices will compute
# unnecessary vertices. The get_true_vert_indices method allows to select the actual vertices
# within the vertices block:
#
# block size true vertices
#   0   10    7
#   1   10   10
#   2   10    3
# True vertices : 0, ... 6, 10 ... 19, 20, ... 22 in an array of 30 vertices.
#
# If all the duplicates have the same number of vertices, the get_true_vert_indices return None
# to avoid unncessary filter.
#
# ----- Building target object
#
# The target object is built by creating the vertices and the complementary information such
# as faces and profile.
# Two ways are possible to build the target: adding all the vertices in the order of the stack
# or in the order of the duplicates. With the following stack:
#
# - 0 : [3, 6, 7]
# - 1 : [1, 2, 5]
# - 2 : [0, 4]
#
# The duplicates order is : 0, 1, 2, 3, 4, 5, 6, 7
# The stack order gives   : 3, 6, 7, 1, 2, 5, 0, 4
#
# The vertices are stacked in the duplicates order to allow direct transformation by transformation
# matrices. Other geometry is not necesssarily built following the duplicates order:
#
#   Geometry               Order
# 
#   vertices               duplicates
#   faces                  stack
#   splines                duplicates
#   mat indices            faces / splines
#   uvs                    stack (as for faces)
#
# The slices give the start index and the size of the geometry item within the target object.


# =============================================================================================================================
# Stacker : feed the stack
# From an existing object
# From another source of geometry

class Stacker():
    """A generic Class providing geometry for Crowd duplicates.
    
    The geometry can come from an object or something else such as True Type chars.
    """
    
    def __init__(self, nverts):
        """Initialize as Stacker.
        
        In addition to the vertices, the stacker provides:
            - material indices
            - faces (Mesh only)
            - uv map (Mesh only)
            - profile (Curve only)

        Parameters
        ----------
        nverts : int
            The number of vertices.

        Returns
        -------
        None.
        """
        
        self.dupl_indices = np.zeros(0, int) # instance indices
        
        # Vertices
        
        self.nverts       = nverts           # Vertices count
        self.verts_       = None
        
        # Material

        self.mat_offset   = 0        
        self.mat_count_   = 0
        self.mat_indices_ = []
        
        # Mesh
        
        self.faces_       = []
        self.nfaces_      = None
        self.uvs_size_    = 0
        self.uvs_         = {}
        
        # Curve
        
        self.profile_     = np.zeros((0, 3), int)
        
    def __repr__(self):
        return f"<Stacker: {len(self):3d} x {self.nverts:3d} = {len(self)*self.nverts:4d} vertices, faces: {self.nfaces} '{self.name}'>"
        
    def __len__(self):
        return len(self.dupl_indices)
    
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
    def nfaces(self):
        if self.nfaces_ is None:
            self.nfaces_ = len(self.faces)
        return self.nfaces_
        
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
    """Object Stacker provides the geometry from an object: Mesh or Curve.
    """
    
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
        """Copy spline attribute from a spline in the stacker to a spline in the target curve.
        """
        
        target_spline.copy_from(self.wobject.data.splines[spline_index])
    
    
# =============================================================================================================================
# A char stacker as a mesh

class MeshCharStacker(Stacker):
    """A class which provides mesh geometry of a char in a True Type Font.
    """
    
    def __init__(self, char, vfu):
        """Initialize the stacker with the vertices, the faces and the uvs map.
        This triplet is produced by the method 'raster' of a Glyphe with
        the argument return_uvmap = True.

        Parameters
        ----------
        char : str
            The character.
        vfu : triplet of array of vertices, array of faces, array of uvs:
            - vertices: array(n, 3) of floats,
            - faces:    list of list of ints, one list of int per face.
            - uv:       array(?, 2) of floats, one couple of float per vertex per face

        Returns
        -------
        None.
        """
        
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
    """A class which provides curve geometry of a char in a True Type Font.
    
    The geometry contains only Bezier curves.
    """
    
    def __init__(self, char, beziers):
        """Initialize the stacker with the beziers control points.
        This beziers is produced by the eponym method 'beziers' of a Glyphe.
        
        Note that the Glyphe provides the control points as a (n, 3) array of
        vertices when the vertices must be stacked as an array (n*3) of vertices.
        
        The initializer put all the arrays in one single array of vertices. The profile
        of the curve stores the split in different curves.
        
        Parameters
        ----------
        char : str
            The character.
        beziers : list of arrays (n, 3) of vertices =list of arrays(n, 3, 3) of floats
            The Beziers curves control points:
                - 0 : the vertices
                - 1 : left handles
                - 2 : right handles

        Returns
        -------
        None.
        """
        
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
        """The Bezier curves must be closed and the have a surface.

        Parameters
        ----------
        spline_index : int
            Index of the spline to copy.
        target_spline : Spline
            The spline in the target to set.

        Returns
        -------
        None.
        """
        
        target_spline.use_cyclic_u = True
        target_spline.fill_mode    = 'FULL'


# =============================================================================================================================
# Stack of objects

class Stack():
    """Manage a stacke of Stackers to build the target object of a Crowd and the vertices array.
    """
    
    def __init__(self, object_type, var_blocks=False):
        """Initialize the Stack to an empty stack.

        Parameters
        ----------
        object_type : str in 'Mesh', 'Curve'
            The type of target pbject.
        var_blocks : bool
            The blocks have a variable size rather than the max block size

        Returns
        -------
        None.
        """
        
        self.object_type = object_type
        self.var_blocks  = var_blocks

        self.stackers  = [] # The list of stackers
        
        self.mat_count   = 0
        self.mat_indices = np.zeros(0, int)
        
    def reset(self):
        """Reset the stack to empty. 

        Returns
        -------
        None.
        """
        
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
        """Total number of generated duplicates.
        
        Computed by adding the number of duplicates per Stacker.

        Returns
        -------
        count : int
            Total number of suplicates.
        """
        
        count = 0
        for stacker in self:
            count += len(stacker)
        return count
    
    @property
    def max_nverts(self):
        """Give the maximum number of vertices of the stackers.

        Returns
        -------
        mx : int
            Maximum number of vertices.
        """
        
        mx = 0
        for stacker in self:
            mx= max(stacker.nverts, mx)
        return mx
    
    @property
    def curve_dim(self):
        """Give the dimension of the curves vertices.
        
        Note that non bezier curves manage complementary information: 4th dim
        of the vectors, radius and tilt.

        Returns
        -------
        int
            3 or 6.
        """
        
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
    def stack_indices(self):
        """Compute the stacker index of each duplicate.

        Returns
        -------
        indices : array of ints
            Stacker index of the duplicates.
        """
        
        indices = np.zeros(self.dupli_count, int)
        for i, stacker in enumerate(self):
            indices[stacker.dupl_indices] = i
        return indices
    
    @stack_indices.setter
    def stack_indices(self, value):
        """Set the stacker indices of the duplicates.
        
        The list [2, 1, 0, 0, 2, 2, 0, 1] will create 8 duplicates: 3 for stacker #2,
        3 for stacker #0 and 2 for stacker #1. This list can be generated by shuffling
        an array containing n times the int i when we need n duplicates of stacker i.
        
        Full randomization list can obtained with : np.random.randint(0, 3, 100) for
        100 duplicates randomly generated from 3 stackers.
        
        Parameters
        ----------
        value : array of ints
            A valid stacker index per entry. The length of the array is the number of duplicates.

        Returns
        -------
        None.
        """
        
        indices = np.array(value)
        rg = np.arange(len(indices))
        for i, stacker in enumerate(self):
            stacker.dupl_indices = rg[indices == i]
            
    # ------------------------------------------------------------------------------------------
    # The duplicates are numbered from 0 to n-1
    # 'indices' property gives the index in the stack ordered from 0 to n-1
    # The coordinated combine the 2 infos:
    # - index in the stack
    # - index in the stacker indices
    # coordinates[i] = (stack_index, index)
    # stacker.dupl_indices[index] = i
    
    @property
    def coordinates(self):
        """Return the stacker index and in the index within the stacker of each duplicates.
        
        As the property stack_indices, this method returns the stacker index of each duplicate.
        In addition, it provides the order of the duplicate for its stacker. Hence, the following is True:
            
            i = stacker[coordinates[i, 0]].dupl_index[coordinates[i, 1]]
            
        The following is also True:
            
            stack_indices = coordinates[:, 0]

        Returns
        -------
        array (n, 2) of ints
            stack index and order in stacker of each duplicate.
        """
        
        coords = np.zeros((self.dupli_count, 2), int)
        for i_stacker, stacker in enumerate(self.stackers):
            coords[stacker.dupl_indices, 0] = i_stacker
            coords[stacker.dupl_indices, 1] = np.arange(len(stacker.dupl_indices))
        
        return coords
            
    # ------------------------------------------------------------------------------------------
    # When we need to stack arrays such as material indices or profile in the instances order
    # we need to know the slices where to insert the instance
    # order can take two values:
    # - DUPLICATES: values are stacked in the order of the duplicates. This is for instance
    #               the case for the vertices. The vertices of duplicate i are at the ith location
    # - STACK     : the values are stacked in the order of the stackers in the stack. This is for
    #               instance the case for ce faces. All the faces of the duplicates of the same
    #               stacker are stacked all together
    
    def slices(self, attr, order='DUPLICATES'):
        """Compute offset and size of geometry data for each duplicate.
        
        The size is read from the attr: nverts, nfaces,...
        
        Depending on the order, the duplicate sizes are added per Stacker or in the
        order of the duplicates.

        Parameters
        ----------
        attr : str
            Stacker attribute name giving the size of the geometry data.
        order : str, optional
            'DUPLICATES' or 'STACK'. The default is 'DUPLICATES'.

        Raises
        ------
        WError
            If order is not valid.

        Returns
        -------
        slices : array (n, 2) of ints
            Offset and size of geometry data for each duplicate.
        """
        
        stack_indices = self.stack_indices
        nrg = np.arange(self.dupli_count)

        slices = np.zeros((self.dupli_count, 2), int)
        
        # Both algorithm place the size of the geometry data at dupl_index location
        # - DUPLICATES : Offset is the cumulative sum of the sizes
        # - STACK      : Offset is the rank of the duplicate within the stacker plus
        #                the total size of the previous stackers
        
        if order == 'DUPLICATES':
            
            for i_stacker, stacker in enumerate(self):
                slices[nrg[stack_indices == i_stacker], 1] = getattr(stacker, attr)
                
            # Offset is the cumulative sum of the sizes
            
            slices[1:, 0] = np.cumsum(slices[:-1, 1])                
                
        elif order == 'STACK':
            
            offset = 0
            for i_stacker, stacker in enumerate(self):
                
                # Values used twice in the algorithm
                inds = nrg[stack_indices == i_stacker]
                v    = getattr(stacker, attr)
                
                # Offset starts from global offset and depends upon the rank
                # of the duplicate within the stacker
                slices[inds, 0] = offset + np.arange(len(stacker))*v
                
                # Size is the same as for DUPLICATES algorithm
                slices[inds, 1] = v
                
                # Update of the global offset
                offset += v*len(stacker)
            
        else:
            raise WError(f"Unknown slices order: '{order}'. Must be 'DUPLICATES' or 'STACK'.",
                         Class = 'Stack', Method='slices')
        
        
        return slices
    
    # ------------------------------------------------------------------------------------------
    # Geometry slices
    
    @property
    def verts_slices(self):
        return self.slices("nverts", order='DUPLICATES')

    @property
    def mats_slices(self):
        order = 'DUPLICATES' if self.is_curve else 'STACK'
        return self.slices("mat_indices_count", order=order)
    
    @property
    def faces_slices(self):
        return self.slices("nfaces", order='STACK')
    
    @property
    def uvs_slices(self):
        return self.slices("uvs_size", order='STACK')
    
    @property
    def prof_slices(self):
        return self.slices("profile_size", order='DUPLICATES')
    
    @staticmethod
    def slices_to_indices(slices, dupl_indices=None):
        
        if dupl_indices is None:
            sls = slices
        else:
            sls = slices[np.reshape(dupl_indices, np.size(dupl_indices))]
            
        n = np.sum(sls[:, 1])
        inds = np.zeros(n, int)
        index = 0
        for ofs, size in sls:
            inds[index:index+size] = np.arange(ofs, ofs+size)
            index += size
            
        return inds
    
    # ------------------------------------------------------------------------------------------
    # Gives the indices of the geometry data within the target object
    #
    # Example of use : change the appearance of all the 'i' character in
    # a crowd of chars:
    #
    # --- Get the indices of the duplicates of char 'i'
    # i_indices = chars.chars_in('i')
    # --- Get the material indices of the faces of the duplicates
    # mats_indices = chars.stack.mats_indices(i_indices)
    # --- Get all the material indices
    # all_mats_indices = chars.wobject.material_indices
    # --- Change the material indices of the faces of the duplicates
    # all_mats_indices[mats_indices] += 1
    # --- Set back the materail indices to the object
    # chars.wobject.material_indices = all_mats_indices
            
    def verts_indices(self, dupl_indices=None):
        return Stack.slices_to_indices(self.verts_slices, dupl_indices)

    def mats_indices(self, dupl_indices=None):
        return Stack.slices_to_indices(self.mats_slices, dupl_indices)
    
    def faces_indices(self, dupl_indices=None):
        return Stack.slices_to_indices(self.faces_slices, dupl_indices)
    
    def uvs_indices(self, dupl_indices=None):
        return Stack.slices_to_indices(self.uvs_slices, dupl_indices)
    
    def prof_indices(self, dupl_indices=None):
        return Stack.slices_to_indices(self.prof_slices, dupl_indices)
    
    # ---------------------------------------------------------------------------
    # Utility to provide information on a duplicate
    
    def dupl_info(self, dupl_index):
        
        class Info():
            def __repr__(self):
                s = self.name
                s += "\n" + f"verts: {self.nverts:3d}, slice: {self.verts_slice}"
                s += "\n" + f"faces: {self.nfaces:3d}, slice: {self.faces_slice}"
                s += "\n" + f"mats:  {self.nmats:3d}, slice: {self.mats_slice}"
                s += "\n" + f"uvs:   {self.nuvs:3d}, slice: {self.uvs_slice}"
                s += "\n" + f"prof:  {self.nprof:3d}, slice: {self.prof_slice}"
                return s
        
        info    = Info()
        coords  = self.coordinates
        stacker = self.stackers[coords[dupl_index, 0]]
        
        info.coords       = coords
        info.stacker      = stacker
        info.name         = f"Duplicate {dupl_index} of stacker {coords[dupl_index, 0]} [{coords[dupl_index, 1]}/{len(stacker)}]: '{stacker.name}' of type '{self.object_type}'"
        info.index        = dupl_index
        
        info.nverts       = stacker.nverts
        info.verts_slice  = self.verts_slices[dupl_index]
        
        info.nfaces       = stacker.nfaces
        info.faces_slice  = self.faces_slices[dupl_index]
        
        info.nmats        = stacker.mat_indices_count
        info.mats_slice   = self.mats_slices[dupl_index]
        
        info.nuvs         = stacker.uvs_size
        info.uvs_slice    = self.uvs_slices[dupl_index]

        info.nprof        = stacker.profile_size
        info.prof_slice   = self.prof_slices[dupl_index]
        
        return info
            
    # ------------------------------------------------------------------------------------------
    # Stack a new stacker for a given number of instances
    
    def stack(self, stacker, count=1):
        """Stack a new stacker.

        Parameters
        ----------
        stacker : Stacker
            Feed the geometry of its duplicates.
        count : int, optional
            Number of duplicates to create. The default is 1.

        Returns
        -------
        Stacker
            The stacked stacker for chaining instructions.
        """
        
        self.stackers.append(stacker)
        
        n = self.dupli_count
        stacker.dupl_indices = np.arange(n, n+count)
        
        return stacker
    
    def stack_object(self, name, count=1):
        """Stack a Blender object stacker by its name.

        Parameters
        ----------
        name : str
            Blender object name.
        count : int, optional
            Number of duplicates to create. The default is 1.

        Returns
        -------
        ObjectStacker
            The ObjectStacker created for this object.
        """
        
        return self.stack(ObjectStacker(name), count)
    
    # ------------------------------------------------------------------------------------------
    # Shuffle the instances
    
    def shuffle(self, seed=None):
        """Shuffle the duplicates. 
        
        At stacking time, on can set the number of duplicates to create per stacker.
        If nothing else is done, the duplicates are created in the order of theur stacker.
        
        The shuffling keeps the number of instances per stacker and shuffle the order
        in which the duplicates will be created.

        Parameters
        ----------
        seed : any, optional
            The seed to use to randomly shuffle the duplicates. No seed initialization
            if None. The default is None.

        Returns
        -------
        None.
        """
        
        if seed is not None:
            np.random.seed(seed)
            
        indices = self.stack_indices
        np.random.shuffle(indices)
        self.stack_indices = indices
        
    # ------------------------------------------------------------------------------------------
    # Set the materials to the target
    
    def create_materials_in(self, name):
        """Create the material list into the target object.

        Parameters
        ----------
        name : str
            Name of the object to set the materials to.

        Returns
        -------
        None.
        """
        
        wtarget = wrap(name)
        
        # ----- Create the material list in the target 
        
        ok_clear = True
        
        mat_offset = 0
        for stacker in self:
            if hasattr(stacker, 'wobject'):
                
                if ok_clear:
                    wtarget.wmaterials.clear()
                    ok_clear = False
                
                wtarget.wmaterials.copy_materials_from(stacker.wobject, append=True)
                stacker.mat_offset = mat_offset
            mat_offset += stacker.mat_count
        
    # ------------------------------------------------------------------------------------------
    # Set the materials to the target
    
    def set_materials_indices(self, name):
        """Set the material indices into the target object.

        Parameters
        ----------
        name : str
            Name of the object to set the materials to.

        Returns
        -------
        None.
        """
        
        wtarget = wrap(name)
            
        # ----- Build the material indices array
        
        stack_indices = self.stack_indices
        slices  = self.slices("mat_indices_count")
        n = slices[-1, 0] + slices[-1, 1]
        mat_indices = np.zeros(n, int)
        
        for i_stacker, (index, size) in zip(stack_indices, slices):
            stacker = self[i_stacker]
            mat_indices[index:index+size] = stacker.mat_indices + stacker.mat_offset
        
        wtarget.material_indices = mat_indices
        
    # ====================================================================================================
    # Set to an object
    
    def set_to_object(self, name):
        """Build the geometry from the duplicates.
        
        This method simpliy calls set_to_mesh_object or set_to_curve_object,
        depending upon the object type.

        Parameters
        ----------
        name : str
            Name of the Blender object.

        Returns
        -------
        None.
        """

        if self.object_type == 'Mesh':
            self.set_to_mesh_object(name)
        
        elif self.object_type == 'Curve':
            self.set_to_curve_object(name)
        
    # ----------------------------------------------------------------------------------------------------
    # Initialize a mesh object with the content

    def set_to_mesh_object(self, name):
        """Build the geometry of a mesh from mesh stackers.
        
        The vertices are stacked in the duplicates order. Faces, material indices and
        uv maps are defined in the order of the stackers in the stack.
        
        The materials list is built from the stacker if the stackers have objects.
        Otherwise, it is up to the caller to build the materials list.
        
        Simirarily, several uvmaps with different names can be built from the stackers.

        Parameters
        ----------
        name : str
            Name of the Blender mesh object.

        Returns
        -------
        None.
        """
        
        # ---------------------------------------------
        # The target object
        
        wtarget = wrap(name)

        # ---------------------------------------------------------------------------
        # Stacker indices

        stack_indices = self.stack_indices
        
        # ---------------------------------------------------------------------------
        # Vertices
        
        slices  = self.slices("nverts")
        n = slices[-1, 0] + slices[-1, 1]
        
        verts = np.zeros((n, 3), np.float)
        
        for i_stacker, (index, size) in zip(stack_indices, slices):
            stacker = self[i_stacker]
            verts[index:index+size] = stacker.verts
            
        # ---------------------------------------------------------------------------
        # Faces
        
        faces = []
        for stacker in self:
            for i in stacker.dupl_indices:
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
        
        slices = self.slices("uvs_size")
        n = slices[-1, 0] + slices[-1, 1]
        uvs = np.zeros((n, 2), np.float)
        
        for name in uvmaps:
            uvs[:] = 0
            for i_stacker, (index, size) in zip(stack_indices, slices):
                stacker = self[i_stacker]
                uvs[index:index+size] = stacker.get_uvs(name)
                
            uvmap = wtarget.get_uvmap(name, create=True)
            uvmap.data.foreach_set('uv', uvs.reshape(2*n))
            
    # ----------------------------------------------------------------------------------------------------
    # Initialize a curve object with the content
    
    def set_to_curve_object(self, name):
        """Build the geometry of a mesh from curve stackers.
        
        Vertices are stakced in the duplicates order. The splines are built
        using the profile mechanism.

        Parameters
        ----------
        name : str
            Name of the Blender curve object.

        Returns
        -------
        None.
        """
        
        # ---------------------------------------------
        # The target object
        
        wtarget = wrap(name)
        wsplines = wtarget.wdata.wsplines

        # ---------------------------------------------------------------------------
        # Indices

        stack_indices = self.stack_indices
        
        # ---------------------------------------------------------------------------
        # Profile
        
        prof_slices  = self.slices("profile_size")
        n = prof_slices[-1, 0] + prof_slices[-1, 1]
        
        profile = np.zeros((n, 3), int)
        
        for i_stacker, (index, size) in zip(stack_indices, prof_slices):
            stacker = self[i_stacker]
            profile[index:index+size] = stacker.profile
        
        wsplines.profile = profile
        
        # ---------------------------------------------------------------------------
        # Vertices
        
        only_bezier = np.min(profile[:, 0]) == 3
        ndim = 3 if only_bezier else 6

        slices  = self.slices("nverts")
        n = slices[-1, 0] + slices[-1, 1]
        
        verts = np.zeros((n, ndim), np.float)
        
        for i_stacker, (index, size) in zip(stack_indices, slices):
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
        
        for i_stacker, (index, size) in zip(stack_indices, prof_slices):
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
        """Build the vertices array used by a Crowd.
        
        The array has a shape of 3 dimensions (l, m, n):
            - l is the number of duplicates
            - m is the size of a block of vertices
            - n is the size of the vertices.
            
        If there is only one stacker in the stack, all the blocks are the same.
        Broadcasting capabilities of numpy is used, and an array of shape (1, m, n)
        is returned.
        
        The block size m is the maximum number of vertices of the stackers.
        If the stackers don't have all the same size, the total number of computed
        vertices is greater than the actual number of vertices in the object. The
        actual vertices are given by the method 'get_true_vert_indices':
            crowd.wobject.verts = blocks[stack.get_true_vert_indices()]
            
        The size of the vertices is 3 but can be 6 for curve object with spline of
        another type than Bezier. The 3 additional floats are:
            - vertex radius
            - vertex tilt
            - 4th component of the vertex
        The 4th component of the vertex is placed ad the end for compatibility with
        shape keys since spline shape keys don't store the 4th compenent.
        
        This method simply calls 'get_crowd_mesh_bases' or 'get_crowd_curve_bases'
        depending upon the object type.

        Returns
        -------
        array(l, m, n) of floats.
            The blocks to be used by the crowd.

        """

        if self.object_type == 'Mesh':
            return self.get_crowd_mesh_bases()
        
        elif self.object_type == 'Curve':
            return self.get_crowd_curve_bases()
        
        return None
        
    # ----------------------------------------------------------------------------------------------------
    # The mesh base vertices for a crowd

    def get_crowd_mesh_bases(self):
        """Build the vertices array used by a mesh Crowd.
        
        The results depends upon the property var_blocks.
        - if var_blocks is False (default):
            All the blocks have the same size, the shape result is (l, m, 3)
            where m is the block size
        - if var_blocks is True:
            The result is a list of np.arrays of shape (s, 3) where s is the
            size of the duplicate
            

        Returns
        -------
        array (l, m, 3) of floats or list of array(s, 3) of floats
            The l data blocks of m vertices.
        """
        
        if self.var_blocks:
            if len(self) == 1:
                return [self[0].verts]
            
            stack_indices = self.stack_indices
            
            verts = [None] * self.dupli_count
            for i, i_stacker in enumerate(stack_indices):
                verts[i] = self[i_stacker].verts
                
            return verts
        
        else:
            if len(self) == 1:
                return self[0].verts.reshape(1, self[0].nverts, 3)
            
            stack_indices = self.stack_indices
            n = len(stack_indices)
            nmax = self.max_nverts
            verts = np.zeros((n, nmax, 3))
            
            for i, i_stacker in enumerate(stack_indices):
                verts[i, :self[i_stacker].nverts, :] = self[i_stacker].verts
                
            return verts
        
    # ----------------------------------------------------------------------------------------------------
    # The curve base vertices for a crowd

    def get_crowd_curve_bases(self):
        """Build the vertices array used by a curve Crowd.
        
        The size of the vertices can be 6 if there exists a least a spline of
        another type than Bezier. The 3 additional floats are:
            - vertex radius
            - vertex tilt
            - 4th component of the vertex
        The 4th component of the vertex is placed ad the end for compatibility with
        shape keys since spline shape keys don't store the 4th compenent.

        Returns
        -------
        array (l, m, n) of floats
            The l data blocks of m vertices.
        """
        
        ndim = self.curve_dim
        
        if self.var_blocks:
            if len(self) == 1:
                return [self[0].verts.reshape(self[0].nverts, ndim)]
            
            stack_indices = self.stack_indices
            
            verts = [None] * len(stack_indices)
            
            for i, i_stacker in enumerate(stack_indices):
                verts[i] = self[i_stacker].verts
                
            return verts
        else:
            if len(self) == 1:
                return self[0].verts.reshape(1, self[0].nverts, ndim)
            
            stack_indices = self.stack_indices
            n = len(stack_indices)
            nmax = self.max_nverts
            verts = np.zeros((n, nmax, ndim))
            
            for i, i_stacker in enumerate(stack_indices):
                vs = self[i_stacker].verts
                verts[i, :self[i_stacker].nverts, :vs.shape[-1]] = vs
                
            return verts
    
    # ----------------------------------------------------------------------------------------------------
    # Get crowd vertex indices to extract the true vertices
    # Return None when all vertices are true (all instances have the same size)
    
    def get_true_vert_indices(self):
        """Get the indices of the actual vertices in the target object.
        
        Since the data blocks used in the crowd can contain more vertices than
        the actual ones provided by the stackers, one needs to select the true vertices
        from the data blocks:
            
            target_object.verts = blocks[stack.get_true_vert_indices()]
            
        Returns None if all vertices are actual.

        Returns
        -------
        extract : array of int or None
            The indices with the data blocks which exist in the target object.
        """
        
        # ----- If variables blocks, no lost vertices
        
        if self.var_blocks:
            return None

        # ----- The slices
        
        slices  = self.slices("nverts")
        
        # ----- All instances have the same size
        
        nmax = np.max(slices[:, 1])
        if np.min(slices[:, 1]) == nmax:
            return None
        
        # ----- Build the true indices array
        
        n = np.sum(slices[:, 1])
        extract = np.zeros(n, int)
        
        for stacker in self:
            for i_dupli in stacker.dupl_indices:
                index = slices[i_dupli, 0]
                size  = slices[i_dupli, 1]
                vert_index = i_dupli * nmax
                extract[index:index+size] = np.arange(vert_index, vert_index+size)
            
        return extract

