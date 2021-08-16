#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:07:20 2021

@author: alain
"""

import numpy as np
import bpy

from .wid import WID

from ..core.plural import to_shape, getattrs, setattrs
from .wmaterials import WMaterials

from ..core.commons import WError


# ---------------------------------------------------------------------------
# Mesh mesh wrapper
# wrapped : data block of mesh object

class WMesh(WID, WMaterials):
    """Wrapper of a Mesh structure.
    """

    def __init__DEPR(self, wrapped, evaluated=False):
        if evaluated:
            super().__init__(wrapped, name=wrapped.name)
        else:
            super().__init__(name=wrapped.name, coll=bpy.data.meshes)
            
    @property
    def wrapped(self):
        """The wrapped Blender instance.

        Returns
        -------
        Struct
            The wrapped object.
        """
        return self.blender_object.data
            
    @staticmethod
    def get_mesh(thing, **kwargs):
        
        if issubclass(type(thing), WMesh):
            return thing
        
        if hasattr(thing, 'wmesh'):
            return thing.wmesh
        
        raise WError(f"Object {thing} is not a mesh!", **kwargs)

    # Mesh vertices update

    def mark_update(self):
        super().mark_update()
        self.wrapped.update()

    # Vertices count

    @property
    def verts_count(self):
        """The number of vertices in the mesh.
        """
        
        return len(self.wrapped.vertices)

    # Vertices (uses verts not to override vertices attributes)

    @property
    def verts(self):
        """The vertices of the mesh

        Returns
        -------
        array(len, 3) of floats
            numpy array of the vertices.
        """
        
        verts = self.wrapped.vertices
        a    = np.empty(len(verts)*3, np.float)
        verts.foreach_get("co", a)
        return np.reshape(a, (len(verts), 3))

    @verts.setter
    def verts(self, vectors):
        verts = np.empty((self.verts_count, 3), np.float)
        count = np.size(vectors)
        if count >= 3:
            verts[:] = np.reshape(vectors, (count//3, 3))
        else:
            verts[:] = vectors
        
        self.wrapped.vertices.foreach_set("co", verts.reshape(self.verts_count*3))
        self.mark_update()
        
    # x, y, z vertices access

    @property
    def xs(self):
        """x locations of the vertices
        """
        
        return self.verts[:, 0]

    @xs.setter
    def xs(self, values):
        locs = self.verts
        locs[:, 0] = to_shape(values, self.vcount)
        self.verts = locs

    @property
    def ys(self):
        """y locations of the vertices
        """
        
        return self.verts[:, 1]

    @ys.setter
    def ys(self, values):
        locs = self.verts
        locs[:, 1] = to_shape(values, self.vcount)
        self.verts = locs

    @property
    def zs(self):
        """z locations of the vertices
        """
        
        return self.verts[:, 2]

    @zs.setter
    def zs(self, values):
        locs = self.verts
        locs[:, 2] = to_shape(values, self.vcount)
        self.verts = locs

    # vertices attributes

    @property
    def bevel_weights(self):
        """bevel weights of the vertices
        """
        
        return getattrs(self.wrapped.vertices, "bevel_weight", 1, np.float)

    @bevel_weights.setter
    def bevel_weights(self, values):
        setattrs(self.wrapped.vertices, "bevel_weight", values, 1)

    # edges as indices

    @property
    def edge_indices(self):
        """A python array with the edges indices.
        
        Indices can be used in th 
        
        Returns
        -------
        array of couples of ints 
            A couple of ints per edge.
        """
        
        edges = self.wrapped.edges
        return [e.key for e in edges]

    # edges as vectors

    @property
    def edge_vertices(self):
        """The couple of vertices of the edges.

        Returns
        -------
        numpy array of couple of vertices (n, 2, 3)
            The deges vertices.

        """
        
        return self.verts[np.array(self.edge_indices)]
    
    # polygons as indices
    
    @property
    def poly_count(self):
        return len(self.wrapped.polygons)

    @property
    def poly_indices(self):
        """The indices of the polygons

        Returns
        -------
        python array of array of ints
            Shape (d1, ?) where d1 is the number of polygons and ? is the number of vertices
            of the polygon.
        """

        polygons = self.wrapped.polygons
        return [tuple(p.vertices) for p in polygons]

    # polygons as vectors

    @property
    def poly_vertices(self):
        """The vertices of the polygons.

        Returns
        -------
        python array of array of triplets.
            Shape is (d1, ?, 3) where d1 is the number of polygons and ? is the number of vertices
            of the polygon.
        """
        
        polys = self.poly_indices
        verts = self.verts
        return [ [list(verts[i]) for i in poly] for poly in polys]
    
    # ---------------------------------------------------------------------------
    # Polygons centersand normals

    @property
    def poly_centers(self):
        """Polygons centers.
        """

        polygons = self.wrapped.polygons
        a = np.empty(len(polygons)*3, np.float)
        polygons.foreach_get("center", a)
        return np.reshape(a, (len(polygons), 3))

    @property
    def normals(self):
        """Polygons normals
        """
        
        polygons = self.wrapped.polygons
        a = np.empty(len(polygons)*3, np.float)
        polygons.foreach_get("normal", a)
        return np.reshape(a, (len(polygons), 3))


        
    # ---------------------------------------------------------------------------
    # uv management
    
    @property
    def uvmaps(self):
        return [uvl.name for uvl in self.wrapped.uv_layers]
    
    def get_uvmap(self, name, create=False):
        try:
            return self.wrapped.uv_layers[name]
        except:
            pass
        
        if create:
            self.wrapped.uv_layers.new(name=name)
            return self.wrapped.uv_layers[name]
        
        raise RuntimeError(f"WMesh error: uvmap '{name}' doesn't existe for object '{self.name}'")
    
    def create_uvmap(self, name):
        return self.get_uvmap(name, create=True)
    
    def get_uvs(self, name):
        uvmap = self.get_uvmap(name)
        
        count = len(uvmap.data)
        uvs = np.empty(2*count, np.float)
        uvmap.data.foreach_get("uv", uvs)
        
        return uvs.reshape((count, 2))
    
    def set_uvs(self, name, uvs):
        uvmap = self.get_uvmap(name)

        count = len(uvmap.data)
        uvs = np.resize(uvs, count*2)
        uvmap.data.foreach_set("uv", uvs)
        
    def get_poly_uvs(self, name, poly_index):
        uvmap = self.get_uvmap(name)
        
        poly = self.wrapped.polygons[poly_index]
        return np.array([uvmap.data[i].uv for i in poly.loop_indices])
    
    def set_poly_uvs(self, name, poly_index, uvs):
        uvmap = self.get_uvmap(name)
        
        poly = self.wrapped.polygons[poly_index]
        uvs = np.resize(uvs, (poly.loop_total, 2))
        for i, iv in enumerate(poly.loop_indices):
            uvmap.data[iv].uv = uvs[i]
            
    def get_poly_uvs_indices(self, poly_index):
        return self.wrapped.polygons[poly_index].loop_indices

    # ---------------------------------------------------------------------------
    # Set new points

    def new_geometry(self, verts, polygons=[], edges=[]):
        """Replace the existing geometry by a new one: vertices and polygons.
        
        Parameters
        ----------
        verts : array(n, 3) of floats
            The new vertices of the mesh.
        polygons : array of array of ints, optional
            The new polygons of the mesh. The default is [].
        edges : array of couples of ints, optional
            The new edges of the mesh. The default is [].

        Returns
        -------
        None.
        """

        mesh = self.wrapped
        obj  = self.blender_object

        # Clear
        obj.shape_key_clear()
        mesh.clear_geometry()

        # Set
        mesh.from_pydata(verts, edges, polygons)

        # Update
        mesh.update()
        mesh.validate()

    # ---------------------------------------------------------------------------
    # Detach geometry to create a new mesh
    # polygons: an array of arrays of valid vertex indices

    def detach_geometry(self, polygons, independant=False):
        """Detach geometry to create a new mesh.
        
        The polygons is an array of array of indices within the array of vertices.
        Only the required vertices are duplicated.
        
        The result can then be used to create a mesh with indenpendant meshes.

        Parameters
        ----------
        polygons : array of array of ints
            The polygons to detach.
        independant : bool, optional
            Make resulting polygons independant by duplicating the vertices instances if True.

        Returns
        -------
        array(, 3) of floats
            The vertices.
        array of array of ints
            The polygons with indices within the new vertices array.
        """

        new_verts = []
        new_polys = []
        
        if independant:
            for poly in polygons:
                new_polys.append([i for i in range(len(new_verts), len(new_verts)+len(poly))])
                new_verts.extend(poly)
                
        else:
            new_inds  = np.full(self.verts_count, -1)
            for poly in polygons:
                new_poly = []
                for vi in poly:
                    if new_inds[vi] == -1:
                        new_inds[vi] = len(new_verts)
                        new_verts.append(vi)
                    new_poly.append(new_inds[vi])
                new_polys.append(new_poly)

        return self.verts[new_verts], new_polys

    # ---------------------------------------------------------------------------
    # Copy

    def copy_mesh(self, mesh, replace=False):
        """Copy the geometry of another mesh.

        Parameters
        ----------
        mesh : a mesh object or a mesh.
            The geometry to copy from.
        replace : bool, optional
            Replace the existing geometry if True or extend the geometry otherwise. The default is False.

        Returns
        -------
        None.
        """
        
        if type(mesh) is str:
            wmesh = WMesh(bpy.data.objects[mesh].data)
        elif type(mesh) is WMesh:
            wmesh = mesh
        else:
            try:
                wmesh = WMesh(bpy.data.objects[mesh.name].data)
            except:
                raise WError(f"Impossible to read mesh from argument {mesh}.",
                    Class = "WMesh",
                    Method = "copy_mesh",
                    mesh = mesh,
                    replace = replace)
        
        #wmesh = wrap(mesh)

        verts = wmesh.verts
        edges = wmesh.edge_indices
        polys = wmesh.poly_indices

        if not replace:
            x_verts = self.verts
            x_edges = self.edge_indices
            x_polys = self.poly_indices

            verts = np.concatenate((x_verts, verts))

            offset = len(x_verts)

            x_edges.extennd([(e[0] + offset, e[1] + offset) for e in edges])
            edges = x_edges

            x_polys.extend([ [p + offset for p in poly] for poly in polys])
            polys = x_polys

        self.new_geometry(verts, polys, edges)

    # ---------------------------------------------------------------------------
    # Materials indices
        
    @property
    def material_indices(self):
        """Material indices from the faces.
        """
        
        inds = np.zeros(self.poly_count)
        self.wrapped.polygons.foreach_get("material_index", inds)
        return inds
    
    @material_indices.setter
    def material_indices(self, value):
        inds = np.resize(value, self.poly_count)
        self.wrapped.polygons.foreach_set("material_index", inds)            

    # ---------------------------------------------------------------------------
    # To python source code

    def python_source_code(self):
        
        def gen():
            verts = self.verts

            s      = "verts = ["
            count  = 3
            n1     = len(verts)-1
            for i, v in enumerate(verts):
                s += f"[{v[0]:.8f}, {v[1]:.8f}, {v[2]:.8f}]"
                if i < n1:
                    s += ", "

                count -= 1
                if count == 0:
                    yield s
                    count = 3
                    s = "\t"

            yield s + "]"
            polys = self.poly_indices
            yield f"polys = {polys}"

        source = ""
        for s in gen():
            source += s + "\n"

        return source


    # ---------------------------------------------------------------------------
    # Layers

    def get_floats(self, name, create=True):
        """Get values of a float layer.        

        Parameters
        ----------
        name : str
            Layer name.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        vals : array of floats
            The values in the layer.
        """
        
        layer = self.wrapped.vertex_layers_float.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_float.new(name=name)
            else:
                return None
        count = len(layer.data)
        vals  = np.zeros(count, np.float)
        layer.data.foreach_get("value", vals)

        return vals

    def set_floats(self, name, vals, create=True):
        """set values to a float layer.        

        Parameters
        ----------
        name : str
            Layer name.
        vals: array of floats
            The values to set.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        None
        """
        
        layer = self.wrapped.vertex_layers_float.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_float.new(name=name)
            else:
                return

        layer.data.foreach_set("value", np.resize(vals, len(layer.data)))

    def get_ints(self, name, create=True):
        """Get values of an int layer.        

        Parameters
        ----------
        name : str
            Layer name.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        vals : array of ints
            The values in the layer.
        """
        
        layer = self.wrapped.vertex_layers_int.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_int.new(name=name)
            else:
                return None
        count = len(layer.data)
        vals  = np.zeros(count, np.int)
        layer.data.foreach_get("value", vals)

        return vals

    def set_ints(self, name, vals, create=True):
        """set values to an int layer.        

        Parameters
        ----------
        name : str
            Layer name.
        vals: array of ints
            The values to set.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        None
        """
        
        layer = self.wrapped.vertex_layers_int.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_int.new(name=name)
                print("creation", layer)
            else:
                return
            
        layer.data.foreach_set("value", np.resize(vals, len(layer.data)))
        
    # ---------------------------------------------------------------------------
    # Groups
    # Return the indices of the vertices belonging to a group
    
    def group_indices(self, group_index):
        if group_index is None:
            return None
        
        verts = self.wrapped.vertices

        idx = np.zeros(len(verts), bool)
        for i, v in enumerate(verts):
            for g in v.groups:
                if g.group == group_index:
                    idx[i] = True
                    break

        return np.arange(len(verts))[idx]
        
    # ===========================================================================
    # Properties and methods to expose to WMeshObject
    
    @classmethod
    def exposed_methods(cls):
        return ["copy_materials_from", "get_uvmap", "create_uvmap", "get_uvs", "set_uvs",
             "get_poly_uvs", "set_poly_uvs", "get_poly_uvs_indices", "new_geometry",
             "detach_geometry", "copy_mesh", "python_source_code",
             "get_floats", "set_floats", "get_ints", "set_ints"]

    @classmethod
    def exposed_properties(cls):
        return {"verts_count": 'RO', "verts": 'RW', "xs": 'RW', "ys": 'RW', "zs": 'RW',
             "bevel_weights": 'RW',"edge_indices": 'RO', "edge_indices": 'RO', "poly_count": 'RO',
             "poly_indices": 'RO', "poly_vertices": 'RO', "poly_centers": 'RO', "normals": 'RO', 
             "materials": 'RO', "material_indices": 'RW', "uvmaps": 'RO'}
        
    # ===========================================================================
    # Generated source code for WMesh class

    @property
    def rna_type(self):
        return self.wrapped.rna_type

    @property
    def name_full(self):
        return self.wrapped.name_full

    @property
    def original(self):
        return self.wrapped.original

    @property
    def users(self):
        return self.wrapped.users

    @property
    def use_fake_user(self):
        return self.wrapped.use_fake_user

    @use_fake_user.setter
    def use_fake_user(self, value):
        self.wrapped.use_fake_user = value

    @property
    def is_embedded_data(self):
        return self.wrapped.is_embedded_data

    @property
    def tag(self):
        return self.wrapped.tag

    @tag.setter
    def tag(self, value):
        self.wrapped.tag = value

    @property
    def is_library_indirect(self):
        return self.wrapped.is_library_indirect

    @property
    def library(self):
        return self.wrapped.library

    @property
    def asset_data(self):
        return self.wrapped.asset_data

    @property
    def override_library(self):
        return self.wrapped.override_library

    @property
    def preview(self):
        return self.wrapped.preview

    @property
    def vertices(self):
        return self.wrapped.vertices

    @property
    def edges(self):
        return self.wrapped.edges

    @property
    def loops(self):
        return self.wrapped.loops

    @property
    def polygons(self):
        return self.wrapped.polygons

    @property
    def loop_triangles(self):
        return self.wrapped.loop_triangles

    @property
    def texture_mesh(self):
        return self.wrapped.texture_mesh

    @texture_mesh.setter
    def texture_mesh(self, value):
        self.wrapped.texture_mesh = value

    @property
    def uv_layers(self):
        return self.wrapped.uv_layers

    @property
    def uv_layer_clone(self):
        return self.wrapped.uv_layer_clone

    @uv_layer_clone.setter
    def uv_layer_clone(self, value):
        self.wrapped.uv_layer_clone = value

    @property
    def uv_layer_clone_index(self):
        return self.wrapped.uv_layer_clone_index

    @uv_layer_clone_index.setter
    def uv_layer_clone_index(self, value):
        self.wrapped.uv_layer_clone_index = value

    @property
    def uv_layer_stencil(self):
        return self.wrapped.uv_layer_stencil

    @uv_layer_stencil.setter
    def uv_layer_stencil(self, value):
        self.wrapped.uv_layer_stencil = value

    @property
    def uv_layer_stencil_index(self):
        return self.wrapped.uv_layer_stencil_index

    @uv_layer_stencil_index.setter
    def uv_layer_stencil_index(self, value):
        self.wrapped.uv_layer_stencil_index = value

    @property
    def vertex_colors(self):
        return self.wrapped.vertex_colors

    @property
    def sculpt_vertex_colors(self):
        return self.wrapped.sculpt_vertex_colors

    @property
    def vertex_layers_float(self):
        return self.wrapped.vertex_layers_float

    @property
    def vertex_layers_int(self):
        return self.wrapped.vertex_layers_int

    @property
    def vertex_layers_string(self):
        return self.wrapped.vertex_layers_string

    @property
    def polygon_layers_float(self):
        return self.wrapped.polygon_layers_float

    @property
    def polygon_layers_int(self):
        return self.wrapped.polygon_layers_int

    @property
    def polygon_layers_string(self):
        return self.wrapped.polygon_layers_string

    @property
    def face_maps(self):
        return self.wrapped.face_maps

    @property
    def skin_vertices(self):
        return self.wrapped.skin_vertices

    @property
    def vertex_paint_masks(self):
        return self.wrapped.vertex_paint_masks

    @property
    def attributes(self):
        return self.wrapped.attributes

    @property
    def remesh_voxel_size(self):
        return self.wrapped.remesh_voxel_size

    @remesh_voxel_size.setter
    def remesh_voxel_size(self, value):
        self.wrapped.remesh_voxel_size = value

    @property
    def remesh_voxel_adaptivity(self):
        return self.wrapped.remesh_voxel_adaptivity

    @remesh_voxel_adaptivity.setter
    def remesh_voxel_adaptivity(self, value):
        self.wrapped.remesh_voxel_adaptivity = value

    @property
    def use_remesh_smooth_normals(self):
        return self.wrapped.use_remesh_smooth_normals

    @use_remesh_smooth_normals.setter
    def use_remesh_smooth_normals(self, value):
        self.wrapped.use_remesh_smooth_normals = value

    @property
    def use_remesh_fix_poles(self):
        return self.wrapped.use_remesh_fix_poles

    @use_remesh_fix_poles.setter
    def use_remesh_fix_poles(self, value):
        self.wrapped.use_remesh_fix_poles = value

    @property
    def use_remesh_preserve_volume(self):
        return self.wrapped.use_remesh_preserve_volume

    @use_remesh_preserve_volume.setter
    def use_remesh_preserve_volume(self, value):
        self.wrapped.use_remesh_preserve_volume = value

    @property
    def use_remesh_preserve_paint_mask(self):
        return self.wrapped.use_remesh_preserve_paint_mask

    @use_remesh_preserve_paint_mask.setter
    def use_remesh_preserve_paint_mask(self, value):
        self.wrapped.use_remesh_preserve_paint_mask = value

    @property
    def use_remesh_preserve_sculpt_face_sets(self):
        return self.wrapped.use_remesh_preserve_sculpt_face_sets

    @use_remesh_preserve_sculpt_face_sets.setter
    def use_remesh_preserve_sculpt_face_sets(self, value):
        self.wrapped.use_remesh_preserve_sculpt_face_sets = value

    @property
    def use_remesh_preserve_vertex_colors(self):
        return self.wrapped.use_remesh_preserve_vertex_colors

    @use_remesh_preserve_vertex_colors.setter
    def use_remesh_preserve_vertex_colors(self, value):
        self.wrapped.use_remesh_preserve_vertex_colors = value

    @property
    def remesh_mode(self):
        return self.wrapped.remesh_mode

    @remesh_mode.setter
    def remesh_mode(self, value):
        self.wrapped.remesh_mode = value

    @property
    def use_mirror_x(self):
        return self.wrapped.use_mirror_x

    @use_mirror_x.setter
    def use_mirror_x(self, value):
        self.wrapped.use_mirror_x = value

    @property
    def use_mirror_y(self):
        return self.wrapped.use_mirror_y

    @use_mirror_y.setter
    def use_mirror_y(self, value):
        self.wrapped.use_mirror_y = value

    @property
    def use_mirror_z(self):
        return self.wrapped.use_mirror_z

    @use_mirror_z.setter
    def use_mirror_z(self, value):
        self.wrapped.use_mirror_z = value

    @property
    def use_mirror_vertex_groups(self):
        return self.wrapped.use_mirror_vertex_groups

    @use_mirror_vertex_groups.setter
    def use_mirror_vertex_groups(self, value):
        self.wrapped.use_mirror_vertex_groups = value

    @property
    def use_auto_smooth(self):
        return self.wrapped.use_auto_smooth

    @use_auto_smooth.setter
    def use_auto_smooth(self, value):
        self.wrapped.use_auto_smooth = value

    @property
    def auto_smooth_angle(self):
        return self.wrapped.auto_smooth_angle

    @auto_smooth_angle.setter
    def auto_smooth_angle(self, value):
        self.wrapped.auto_smooth_angle = value

    @property
    def has_custom_normals(self):
        return self.wrapped.has_custom_normals

    @property
    def texco_mesh(self):
        return self.wrapped.texco_mesh

    @texco_mesh.setter
    def texco_mesh(self, value):
        self.wrapped.texco_mesh = value

    @property
    def shape_keys(self):
        return self.wrapped.shape_keys

    @property
    def use_auto_texspace(self):
        return self.wrapped.use_auto_texspace

    @use_auto_texspace.setter
    def use_auto_texspace(self, value):
        self.wrapped.use_auto_texspace = value

    @property
    def use_mirror_topology(self):
        return self.wrapped.use_mirror_topology

    @use_mirror_topology.setter
    def use_mirror_topology(self, value):
        self.wrapped.use_mirror_topology = value

    @property
    def use_paint_mask(self):
        return self.wrapped.use_paint_mask

    @use_paint_mask.setter
    def use_paint_mask(self, value):
        self.wrapped.use_paint_mask = value

    @property
    def use_paint_mask_vertex(self):
        return self.wrapped.use_paint_mask_vertex

    @use_paint_mask_vertex.setter
    def use_paint_mask_vertex(self, value):
        self.wrapped.use_paint_mask_vertex = value

    @property
    def use_customdata_vertex_bevel(self):
        return self.wrapped.use_customdata_vertex_bevel

    @use_customdata_vertex_bevel.setter
    def use_customdata_vertex_bevel(self, value):
        self.wrapped.use_customdata_vertex_bevel = value

    @property
    def use_customdata_edge_bevel(self):
        return self.wrapped.use_customdata_edge_bevel

    @use_customdata_edge_bevel.setter
    def use_customdata_edge_bevel(self, value):
        self.wrapped.use_customdata_edge_bevel = value

    @property
    def use_customdata_edge_crease(self):
        return self.wrapped.use_customdata_edge_crease

    @use_customdata_edge_crease.setter
    def use_customdata_edge_crease(self, value):
        self.wrapped.use_customdata_edge_crease = value

    @property
    def total_vert_sel(self):
        return self.wrapped.total_vert_sel

    @property
    def total_edge_sel(self):
        return self.wrapped.total_edge_sel

    @property
    def total_face_sel(self):
        return self.wrapped.total_face_sel

    @property
    def is_editmode(self):
        return self.wrapped.is_editmode

    @property
    def auto_texspace(self):
        return self.wrapped.auto_texspace

    @auto_texspace.setter
    def auto_texspace(self, value):
        self.wrapped.auto_texspace = value

    @property
    def texspace_location(self):
        return self.wrapped.texspace_location

    @texspace_location.setter
    def texspace_location(self, value):
        self.wrapped.texspace_location = value

    @property
    def texspace_size(self):
        return self.wrapped.texspace_size

    @texspace_size.setter
    def texspace_size(self, value):
        self.wrapped.texspace_size = value

    @property
    def materials(self):
        return self.wrapped.materials

    @property
    def cycles(self):
        return self.wrapped.cycles

    def animation_data_clear(self, *args, **kwargs):
        return self.wrapped.animation_data_clear(*args, **kwargs)

    def animation_data_create(self, *args, **kwargs):
        return self.wrapped.animation_data_create(*args, **kwargs)

    @property
    def bl_rna(self):
        return self.wrapped.bl_rna

    def calc_loop_triangles(self, *args, **kwargs):
        return self.wrapped.calc_loop_triangles(*args, **kwargs)

    def calc_normals(self, *args, **kwargs):
        return self.wrapped.calc_normals(*args, **kwargs)

    def calc_normals_split(self, *args, **kwargs):
        return self.wrapped.calc_normals_split(*args, **kwargs)

    def calc_smooth_groups(self, *args, **kwargs):
        return self.wrapped.calc_smooth_groups(*args, **kwargs)

    def calc_tangents(self, *args, **kwargs):
        return self.wrapped.calc_tangents(*args, **kwargs)

    def clear_geometry(self, *args, **kwargs):
        return self.wrapped.clear_geometry(*args, **kwargs)

    def copy(self, *args, **kwargs):
        return self.wrapped.copy(*args, **kwargs)

    def count_selected_items(self, *args, **kwargs):
        return self.wrapped.count_selected_items(*args, **kwargs)

    def create_normals_split(self, *args, **kwargs):
        return self.wrapped.create_normals_split(*args, **kwargs)

    @property
    def edge_keys(self):
        return self.wrapped.edge_keys

    def evaluated_get(self, *args, **kwargs):
        return self.wrapped.evaluated_get(*args, **kwargs)

    def flip_normals(self, *args, **kwargs):
        return self.wrapped.flip_normals(*args, **kwargs)

    def free_normals_split(self, *args, **kwargs):
        return self.wrapped.free_normals_split(*args, **kwargs)

    def free_tangents(self, *args, **kwargs):
        return self.wrapped.free_tangents(*args, **kwargs)

    @property
    def from_pydata(self):
        return self.wrapped.from_pydata

    def make_local(self, *args, **kwargs):
        return self.wrapped.make_local(*args, **kwargs)

    def normals_split_custom_set(self, *args, **kwargs):
        return self.wrapped.normals_split_custom_set(*args, **kwargs)

    def normals_split_custom_set_from_vertices(self, *args, **kwargs):
        return self.wrapped.normals_split_custom_set_from_vertices(*args, **kwargs)

    def override_create(self, *args, **kwargs):
        return self.wrapped.override_create(*args, **kwargs)

    def override_template_create(self, *args, **kwargs):
        return self.wrapped.override_template_create(*args, **kwargs)

    def split_faces(self, *args, **kwargs):
        return self.wrapped.split_faces(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.wrapped.transform(*args, **kwargs)

    def unit_test_compare(self, *args, **kwargs):
        return self.wrapped.unit_test_compare(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self.wrapped.update(*args, **kwargs)

    def update_gpu_tag(self, *args, **kwargs):
        return self.wrapped.update_gpu_tag(*args, **kwargs)

    def update_tag(self, *args, **kwargs):
        return self.wrapped.update_tag(*args, **kwargs)

    def user_clear(self, *args, **kwargs):
        return self.wrapped.user_clear(*args, **kwargs)

    def user_of_id(self, *args, **kwargs):
        return self.wrapped.user_of_id(*args, **kwargs)

    def user_remap(self, *args, **kwargs):
        return self.wrapped.user_remap(*args, **kwargs)

    def validate(self, *args, **kwargs):
        return self.wrapped.validate(*args, **kwargs)

    def validate_material_indices(self, *args, **kwargs):
        return self.wrapped.validate_material_indices(*args, **kwargs)

    # End of generation
    # ===========================================================================
        
        
