#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:03:31 2021

@author: alain
"""

import numpy as np

from ..blender import blender
from ..maths.transformations import Transformations, SlaveTransformations
from ..maths.shapes import get_full_shape
from ..maths.key_shapes import KeyShapes
#from ..maths.operations import morph, check_shape_keys

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
# Crowd: duplicates a mesh or a curve in a single object
# MeshCrowd and CurveCrowd implement specificites

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
    
    def __init__(self, model, count=10, name=None, evaluated=False):
        """Initialize the crowd with by creating count duplicates of the model mesh.
        
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
        
        # ----- Super initialization
        
        super().__init__(count=count)
        
        # ----- Wrap the model create the crowd
        
        wmodel = wrap(model)
        if wmodel.object_type not in  ['Mesh', 'Curve', 'SurfaceCurve']:
            raise WError(f"Crowd init error: {model} must be a mesh, a curve or a surface.",
                Class = "Crowd",
                Method = "__init__",
                model = model,
                count = count)
            
        wmodel.set_evaluated(evaluated)
            
        if name is None:
            name = "Crowd of " + wmodel.name + 's'
            
        if wmodel.object_type == 'Mesh':
            self.wobject = wrap(name, create = "CUBE" )
            self.otype = 'Mesh'
            
        elif wmodel.object_type == 'Curve':
            self.wobject = wrap(name, create = "BEZIER" )
            self.otype = 'Curve'
            
        elif wmodel.object_type == 'SurfaceCurve':
            self.wobject = wrap(name, create = "SURFACE" )
            self.otype = 'Surface'
            
        blender.copy_collections(wmodel.wrapped, self.wobject.wrapped)

        # ----- Materials
        self.wobject.copy_materials_from(wmodel)   
        
        # ----- Keep model track
        self.wmodel = wmodel
        
        # ----- No animation yet
        self.animated = False


        
    def __repr__(self):
        s = "<"
        s += f"Crowd of {self.size} [{self.shape}] {self.otype}."
        return s + ">"    

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
    
    def set_animation(self, keyshapes):
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
        
        self.animated  = True
        self.keyshapes = keyshapes
        n = self.keyshapes.points_per_shape

        if n != self.v_count:
            raise WError(f"Incorrect number of vertices ({n})in the shape keys.",
                    Class = "Crowd",
                    Method = "set_animation",
                    keyshapes = keyshapes,
                    expected_count = self.v_count,
                    passed_count = n)
            
    # ---------------------------------------------------------------------------
    # Animated at time t
    
    def animate(self, t):
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
        
        pass
    

# ====================================================================================================
# Crowd: duplicates a mesh in a single mesh

class MeshCrowd(Crowd):
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
    
    def __init__(self, model, count=10, name=None, evaluated=False):
        """Initialize the crowd with by creating count duplicates of the model mesh.
        
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
        
        super().__init__(model, count, name, evaluated)
        
        # ----- Vertices from the model
        
        verts = np.array(self.wmodel.verts)
        self.v_count = len(verts)
        self.total_verts = self.size * self.v_count
        
        # ---------------------------------------------------------------------------
        # ----- Build the new geometry
        
        # Polygons
        polys = self.wmodel.poly_indices
        self.p_count = len(polys)
        
        faces = [[index + i*self.v_count for index in face] for i in range(self.size) for face in polys]

        # New geometry
        self.wobject.new_geometry(np.resize(verts, (self.total_verts, 3)), faces)
        
        # ---------------------------------------------------------------------------
        # ----- uv mapping
        
        for name in self.wmodel.uvmaps:
            uvs = self.wmodel.get_uvs(name)
            self.wobject.create_uvmap(name)
            self.wobject.set_uvs(name, np.resize(uvs, (self.size*len(uvs), 2)))
            
        # ---------------------------------------------------------------------------
        # ----- Material indices
            
        self.wobject.material_indices = self.wmodel.material_indices # Will be properly broadcasted
        
        # ---------------------------------------------------------------------------
        # ----- Groups

        group_indices = self.wmodel.group_indices()
        self.groups = {group_name: GroupTransfo(group_indices[group_name], center=self.wmodel.group_center(group_name)) for group_name in group_indices}
            
        # ---------------------------------------------------------------------------
        # ----- Base vertices as 4-vectors

        self.base_verts = np.insert(verts, 3, 1, axis=-1)
        
    # ---------------------------------------------------------------------------
    # A pretty repr
            
    def __repr__(self):
        s = "<"
        s += f"Crowd of {self.size} [{self.shape}] meshes of {self.v_count} vertices, vertices: {self.total_verts}"
        if self.animated:
            s += "\n"
            s += f"Animation of {self.steps} steps: {self.base_verts.shape}"
        return s + ">"    
        
    # ---------------------------------------------------------------------------
    # Vertex groups can benefit from additional transformations
    
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
        """Overrides the super class method.
        
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
    # Set an animation
    # The animation is made of steps set of vertices. The number of vertices
    # must match the number of vertices used for initialization.
    # The shape of verts is hence: (steps, v_count, 3)
    
    def set_animation(self, keyshapes):
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
        
        super().set_animation(keyshapes)
        
        # With animation, each instance has now its own base verts
            
        self.base_verts = np.resize(self.base_verts, get_full_shape(self.shape, (self.v_count, 4)))
            
    # ---------------------------------------------------------------------------
    # Animated at time t
    
    def animate(self, t):
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
        
        if not self.animated:
            return
        
        kss = self.keyshapes.interpolate(t)
        self.base_verts[..., 1:] = kss.verts
        
        self.lock_apply()            

    
# ====================================================================================================
# Crowd: duplicates a mesh in a single mesh

class CurveCrowd(Transformations):
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
    
    def __init__(self, model, count=10, name=None, evaluated=False):
        """Initialize the crowd with by creating count duplicates of the model mesh.
        
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
        
        super().__init__(count=count)
        
        # ----- Wrap the model create the crowd
        
        wmodel = wrap(model)
        if not wmodel.object_type in ['Mesh', 'Curve', 'SurfaceCurve']:
            raise WError(f"Crowd init error: {model} must be a mesh or curve object",
                Class = "Crowd",
                Method = "__init__",
                model = model,
                count = count)
            
        wmodel.set_evaluated(evaluated)
            
        if name is None:
            name = "Crowd of " + wmodel.name + 's'
            
        if wmodel.object_type == 'Mesh':
            self.wobject = wrap(name, create = "CUBE" )
            self.otype = 'Mesh'
            
        elif wmodel.object_type == 'Curve':
            self.wobject = wrap(name, create = "BEZIER" )
            self.otype = 'Curve'
            
        elif wmodel.object_type == 'SurfaceCurve':
            self.wobject = wrap(name, create = "SURFACE" )
            self.otype = 'Surface'
            
        blender.copy_collections(wmodel.wrapped, self.wobject.wrapped)

        # ----- Materials
        self.wobject.copy_materials_from(wmodel)
        
        # ----- Get the base shapes from the models
        
        
        # ---------------------------------------------------------------------------
        # Crowd of meshes
        
        if wmodel.object_type == 'Mesh':
            
            # Vertices
            verts = np.array(wmodel.verts)
            self.total_verts = self.size * self.v_count
            
            # Base vertices as 4-vectors
            self.base_verts = np.insert(verts, 3, 1, axis=0)
            self.v_count = self.base_shapes.points_per_shape
            
            
            # Polygons
            polys = wmodel.poly_indices
            self.p_count = len(polys)
            
            faces = [[index + i*self.v_count for index in face] for i in range(self.size) for face in polys]
    
            # New geometry
            self.wobject.new_geometry(np.resize(verts, (self.total_verts, 3)), faces)
            
            # ----- uv mapping
            
            for name in wmodel.uvmaps:
                uvs = wmodel.get_uvs(name)
                self.wobject.create_uvmap(name)
                self.wobject.set_uvs(name, np.resize(uvs, (self.size*len(uvs), 2)))
                
            # ----- Material indices
                
            self.wobject.material_indices = wmodel.material_indices # Will be properly broadcasted
            
            # ----- Groups
            group_indices = wmodel.group_indices()
            self.groups = {group_name: GroupTransfo(group_indices[group_name], center=wmodel.group_center(group_name)) for group_name in group_indices}
                
        # ---------------------------------------------------------------------------
        # Crowd of curves
        
        elif wmodel.object_type == 'Curve':
            
            self.base_shapes = KeyShapes.FromObject(wmodel.wrapped)
            
            for i in range(self.size):
                self.wobject.copy_splines_from(wmodel, add = i>0)
        
        # ----- No animation
        self.animated = False
        
            
    def __repr__(self):
        
        s = "<"
        s += f"Crowd of {self.size} [{self.shape}] {self.otype} of {self.v_count} vertices, vertices: {self.total_verts}"
        if self.animated:
            s += "\n"
            s += f"Animation of {self.steps} steps: {self.base_verts.shape}"
        return s + ">"    
        
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
    
    def set_animation(self, keyshapes, interpolation_shape='/', interpolation_name=None):
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
        
        self.animated  = True
        self.keyshapes = keyshapes
        n = self.keyshapes.points_per_shape

        if n != self.v_count:
            raise WError(f"Incorrect number of vertices ({n})in the shape keys.",
                    Class = "Crowd",
                    Method = "set_animation",
                    keyshapes = keyshapes,
                    expected_count = self.v_count,
                    passed_count = n)
            
        # ----- Reshape the base vertices
        
        if self.otype == 'Mesh':
            self.base_verts = np.resize(self.base_verts, get_full_shape(self.shape, (self.v_count, 4)))
        else:
            pass
            
    # ---------------------------------------------------------------------------
    # Set an animation
    
    def animate(self, t):
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
        
        if not self.animated:
            return
        
        
        
        if self.otype == 'Mesh':
            self.base_verts = kss.verts4.reshape(1, self.v_count , 4)
        self.lock_apply()
            
            
        
        points = self.keyshapes.interpolation(t).reshape(self.v_count, 3)
        
        points = points.reshape(1, 12, 3)
        print('-'*10, "animate", self.shape, self.keyshapes.points.shape, points.shape, "\n")
        
        self.base_verts[:] = KeyShapes(points).verts4
        
        return
        
        self.base_verts[:] = kss.verts4.reshape(1, self.v_count , 4)
        self.lock_apply()
                
# ---------------------------------------------------------------------------
# Lead to the right class

def crowd(model, count=10, name=None, evaluated=False):
    
    wmodel = wrap(model)
    
    if wmodel.object_type == 'Mesh':
        return MeshCrowd(model, count, name, evaluated)
    
    elif wmodel.object_type in ['Curve', 'SurfaceCurve']:
        return CurveCrowd(model, count, name, evaluated)
    
    else:
        raise WError(f"Crowd init error: {model} must be a mesh, a curve or a surface.",
            Funciton = "crowd",
            model = model,
            count = count,
            name = name,
            evaluated = evaluated)
        
