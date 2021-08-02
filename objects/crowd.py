#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np

from ..maths.transformations import Transformations, SlaveTransformations
from ..maths.shapes import get_full_shape

from ..wrappers.wrap_function import wrap

from ..core.commons import WError

# ====================================================================================================
# Group transo class utility

class GroupTransfo():
    def __init__(self, indices, center):
        self.indices = indices
        self.center  = center
        self.transfo = None
        self.pivot   = None
        
    def __repr__(self):
        return f"<GroupTransfo: {self.transfo}, pivot: {self.pivot}, indices: {np.shape(self.indices)}>"

# ====================================================================================================
# Crowd: duplicates a mesh in a single mesh

class Crowd(Transformations):
    """Manages a crowd of copies of a mesh model in a single mesh object.
    
    The model can be the evaluated version of the mesh to take the modifiers into account.
    
    The geometry of the model are duplicated as required by the argument count.
    The individual copies can be later on animated using the set_animation method.
    
    Copies of the model can be considered as individual objects which van be transformed
    individually using the method of the Transformations class:
        - locations
        - rotations
        - scales
    Plus the shortcuts: x, y, z, rx, ry, rz, rxd, ryd, rzd, sx, sy, sz
    """
    
    def __init__(self, model, count=100):
        """Initialize the crowd with by creating count duplicates of the model mesh.
        
        The model can be the wrapped of an evaluated mesh to take into account the 
        modifiers applied on the model.
        
        These three initializations are valid:
            - Crowd("Model", 1000)
            - Crowd(bw.wrap("Model"), 1000)
            - Crowd(bw.wrap("Model").evaluated, 1000)
            
        The crowd is created if if doesn't already exists
        The name is based on the model name by suffixing with Crowd

        Parameters
        ----------
        model : str or Wrapper
            Name of an existing object or wrapper of an existing object. The wrapper version allows
            to use the evaluated version
        count : TYPE, optional
            DESCRIPTION. The default is 100.

        Raises
        ------
        RuntimeError
            If objects don't exist or are not meshes.

        Returns
        -------
        None.
        """
        
        super().__init__(count=count)
        
        # ----- Wrap the model create the crowd
        
        wmodel = wrap(model)
        if not wmodel.is_mesh:
            raise WError(f"Crowd init error: {model} must be a mesh object",
                Class = "Crowd",
                Method = "__init__",
                model = model,
                count = count)
            
        self.wobject = wrap(wmodel.name + " Crowd", create="CUBE")
            
        # ----- Build the new geometry made of stacking vertices and polygons
        
        # Vertices
        verts = np.array(wmodel.verts)
        self.v_count = len(verts)
        self.total_verts = self.size * self.v_count
        
        # Base vertices as 4D-vectors
        self.base_verts = np.column_stack((verts, np.ones(self.v_count)))

        # Polygons
        polys = wmodel.poly_indices
        self.p_count = len(polys)
        
        faces = [[index + i*self.v_count for index in face] for i in range(self.size) for face in polys]

        # New geometry
        self.wobject.new_geometry(np.resize(verts, (self.total_verts, 3)), faces)
        
        # ----- Materials
        
        self.wobject.copy_materials_from(wmodel)
        self.wobject.material_indices = wmodel.material_indices # Will be properly broadcasted
        
        # ----- uv mapping
        
        for name in wmodel.uvmaps:
            uvs = wmodel.get_uvs(name)
            self.wobject.create_uvmap(name)
            self.wobject.set_uvs(name, np.resize(uvs, (self.size*len(uvs), 2)))
            
        # ----- No animation
        self.animated = False
        
        # ----- Groups
        group_indices = wmodel.group_indices()
        self.groups = {group_name: GroupTransfo(group_indices[group_name], center=wmodel.group_center(group_name)) for group_name in group_indices}
            
    def __repr__(self):
        s = "<"
        s += f"Crowd of {self.size} [{self.shape}] meshes of {self.v_count} vertices, vertices: {self.total_verts}"
        if self.animated:
            s += "\n"
            s += f"Animation of {self.steps} steps: {self.base_verts.shape}"
        return s + ">"    
        
        
    @property
    def verts(self):
        """The vertices of the crowd object.

        Returns
        -------
        array of vertices
            The vertices of the crowd object.
        """
        
        return self.wobject.verts
    
    @verts.setter
    def verts(self, value):
        self.wobject.verts = value
        
    # ---------------------------------------------------------------------------
    # Groups can benefit from additional transformations
    
    def group_transformation(self, group_name, center_group_name=None):
        
        if not group_name in self.groups:
            raise WError(f"The model object of '{self.wobject.name}' doesn't have a vertex group named '{group_name}'")
            
        if center_group_name is None:
            center_group_name = group_name
        else:
            if not center_group_name in self.groups:
                raise WError(f"The model object of '{self.wobject.name}' doesn't have a vertex group named '{center_group_name}'")
            
            
        gt = self.groups[group_name]
        if gt.transfo is None:
            gt.transfo = SlaveTransformations(self)
            gt.pivot   = self.groups[center_group_name].center

        return gt.transfo
        
    # ---------------------------------------------------------------------------
    # Overrides matrices transformations
        
    def apply(self):
        """Overrides the supper class method.
        
        The transformations matrices are applies on the base vertices

        Returns
        -------
        None.
        """
        
        # The slave transformation can call apply()
        mem_locked = self.locked
        self.locked = 1
        
        verts = self.transform_verts43(self.base_verts)
        
        # ----- Group transformation
        
        for group_name, gt in self.groups.items():
            if gt.transfo is not None:
                verts[..., gt.indices, :] = self.compose(gt.transfo, center=gt.pivot).transform_verts43(self.base_verts[gt.indices])
            
        self.wobject.wdata.verts = verts
        
        self.locked = mem_locked
        
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
    # Set an animation
    # The animation is made of steps set of vertices. The number of vertices
    # must match the number of vertices used for initialization.
    # The shape of verts is hence: (steps, v_count, 3)
    
    def set_animation(self, animation, phases=None, speeds=1, seed=0):
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
        animation : array of array of vertices
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
        
        self.animated  = True
        self.steps     = len(animation)
        n              = self.steps*self.v_count
        self.animation = np.resize(np.column_stack(
                    (animation.reshape(n, 3), np.ones(n))
                ), (self.steps, self.v_count, 4))
        
        if seed is not None:
            np.random.seed(seed)
            
        if phases is None:
            self.phases = np.random.randint(0, self.steps, self.shape)
        else:
            self.phases = np.resize(phases, self.shape)
            
        if speeds is None:
            self.speeds = np.random.randint(1, self.steps, self.shape)
        else:
            self.speeds = np.resize(speeds, self.shape)
            
        # ----- Reshape the base vertices
        self.base_verts = np.resize(self.base_verts, get_full_shape(self.shape, (self.v_count, 4)))
            
    # ---------------------------------------------------------------------------
    # Set an animation
    
    def animate(self, frame):
        """Change the base verts with the vertices corresponding to the frame.
        
        See methode set_animation

        Parameters
        ----------
        frame : int
            The frame at which the animation is computed.

        Returns
        -------
        None.
        """
        
        if self.animated:
            indices = (self.phases + frame*self.speeds).astype(np.int) % self.steps
            self.base_verts = self.animation[indices]
            self.lock_apply()


        
    
    
    
    
    
