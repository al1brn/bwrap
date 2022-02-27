#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 10:11:11 2022

@author: alain
"""

import numpy as np

from .wroot import WRoot
from .varrays import VArrays

# -----------------------------------------------------------------------------------------------------------------------------
# Faces made of an array oif vertices indices and another of the faces sizes

class Faces(VArrays):
    
    # ---------------------------------------------------------------------------
    # some stats
    
    @property
    def tris_count(self):
        return self.size_equal(3)
    
    @property
    def quads_count(self):
        return self.size_equal(4)
    
    @property
    def ngons_count(self):
        return self.size_at_least(5)

    # ---------------------------------------------------------------------------
    # Content
                
    def __repr__(self):
        if len(self) == 0:
            return "<Faces Empty>"
        
        if len(self.values) == 0:
            interval = "[Empty indices]"
        else:
            interval = f"[{np.min(self.values)} ... {np.max(self.values)}]"
        
        s = f"<Faces: {len(self)} face(s) on {len(self.values)} indices in {interval}"
        s += f" tris: {self.tris_count}, quads: {self.quads_count}, n-gons: {self.ngons_count}"
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # Full dump
    
    def dump(self):
        print('-'*20)
        print(self)
        print()
        print("sizes:", self.sizes)
        print("indices shapes:", self.values.shape)
        print()
        for i, face in enumerate(self):
            if i < 20 or i >= len(self)-20:
                if i == len(self)-20 and len(self) > 40:
                    print(f"   ... {len(self)} ...")
                print(f"   {i:2d} ({len(face):3d}): {WRoot._str_indices(face)}")
        print()
    
    # ---------------------------------------------------------------------------
    # Create the faces for an array of duplicates
    
    def array(self, count, verts_count=None):
        
        if verts_count is None:
            if len(self.values) is None:
                verts_count = 0
            else:
                verts_count = max(self.values)+1
        
        faces = Faces()
        
        faces.sizes  = np.resize(self.sizes, count*len(self.sizes))
        faces.values = np.reshape(
            np.resize(np.arange(count)*verts_count, (len(self.values), count)).T + self.values,
            count*len(self.values))
        
        return faces

    # ---------------------------------------------------------------------------
    # Join two set of faces
    
    def join(self, faces, verts_count=None):
        
        if len(faces) == 0:
            return
        
        elif self.values is None:
            if faces.values is None:
                return
            
            self.sizes  = np.array(faces.sizes)
            self.values = np.array(faces.values)
            
            if verts_count is not None:
                self.values += verts_count
                
        else:
            if verts_count is None:
                verts_count = np.max(self.values)+1
                
            self.values = np.append(self.values, faces.values + verts_count)
            self.sizes  = np.append(self.sizes,  faces.sizes)
        
    # ---------------------------------------------------------------------------
    # Return the faces indices containing a set of indices
    
    def faces_of(self, indices):
        faces_inds = []
        offset = 0
        for i, size in enumerate(self.sizes):
            if len(np.intersect1d(indices, self.values[offset:offset + size])) > 0:
                faces_inds.append(i)
            offset += size
            
        return faces_inds
    
    # ---------------------------------------------------------------------------
    # Build an empty uv map
    
    def uvmap(self):
        return np.zeros((len(self.values), 2), float)
    
    
    # ---------------------------------------------------------------------------
    # Edges
    
    def edges(self):
        
        if len(self.sizes) == 0:
            return np.zeros((0, 2), int)
        
        es = np.zeros((len(self.values), 2), int)
        es[:,   0] = self.values
        es[:-1, 1] = self.values[1:]
        
        ofs = np.cumsum(self.sizes)
        
        es[ofs[1:-1]-1, 1] = self.values[ofs[:-2]]
        
        es[ofs[0]-1, 1]    = self.values[0]
        es[ofs[-1]-1, 1]   = self.values[ofs[-2]]

        return np.stack((np.min(es, axis=1), np.max(es, axis=1))).T
    
    
    # ---------------------------------------------------------------------------
    # Return faces towards edges and not faces
    # The number of edges per face is equal to the number of vertices
    # if unique, return the array of uniqe edges
    
    def faces_of_edges(self, unique=False, edges=None):
        if edges is None:
            edges = self.edges()
            
        vas = VArrays.FromArrays(self.sizes, edges, copy=False)
        if unique:
            return vas.unique()
        else:
            return vas
        
    # ---------------------------------------------------------------------------
    # Return neighbours as faces of faces
    # Each edge is shared by two faces
    
    def neighbours(self):
        
        # --------------------------------------------------
        # - edges:  faces of indices on uniq
        # - uniq:   the uniques edges
        
        edges, uniq = self.faces_of_edges(unique=True)
        
        # --------------------------------------------------
        # Each edge has now an index
        # We want to identify faces sharing the same edge
        # We sort the edges per their index.
        # The neighbour faces will be successors in the resulting array
        # - inds : give the arguments for the sorted edges
        
        inds = np.argsort(edges.values)
        
        # --------------------------------------------------
        # Let's use this on the faces indices
        # - faces : indices of faces sorted by the growing index of
        #           their edges
        
        faces = edges.arrays_indices()[inds]
        
        # --------------------------------------------------
        # Normally, one edge is shared by exactly thwo faces
        # if it not the case, the algorithm won't work
        # - faces : array(n, 2) of couples of faces indices
        #           sharing the same edge
        
        n = len(uniq)
        faces = np.reshape(faces, (n, 2))
        
        # --------------------------------------------------
        # Let's sort from growing faces indices
        # Each couple appears in two faces, we must duplicate
        # the array with the couples ordered reversely
        # - faces : array (2*n, 2) of couples of faces indices
        #           up to n: (lower face index, higher face index)
        #           from n : (highr face index, lower face index)
        
        fmin  = np.min(faces, axis=1)
        fmax  = np.max(faces, axis=1)
        faces = np.empty((2*n, 2), int)
        faces[:n, 0] = fmin
        faces[:n, 1] = fmax
        faces[n:, 0] = fmax
        faces[n:, 1] = fmin
        
        # --------------------------------------------------
        # The faces must be reordered to their initial order
        # - faces : concatened arrays of faces neighbours (couples)
        
        faces = faces[np.argsort(faces[:, 0])]
        
        # --------------------------------------------------
        # Let's return as variable arrays
        # Note that the first member of the couples is the face index
        # we need only the second membre
        
        return VArrays.FromData(self.sizes, faces[:, 1], duplicate=True)
    
    # ---------------------------------------------------------------------------
    # Demo
     
    def random_groups(self, size=3, count=None, seed=0):
        
        if seed is not None:
            np.random.seed(seed)
            
        faces_count = len(self.sizes)
        
        if count is None:
            count = int(round(faces_count / size))
            
        count = max(1, min(faces_count, count))
        
        # ---------------------------------------------------------------------------
        # Very simple
        
        if count == faces_count:
            return np.arange(count)
        
        # ---------------------------------------------------------------------------
        # Randomly select count faces
        
        faces  = np.random.choice(np.arange(faces_count), count, replace=False)
        groups = np.ones(faces_count, int) * -1
        groups[faces] = np.arange(count)
        
        # ---------------------------------------------------------------------------
        # Get the neighbours of the faces
        
        neighbours = self.neighbours()
        
        # ---------------------------------------------------------------------------
        # In the loop, we select randomly one neighbour face par face
        # If the face belongs to a group, the selected neighbour is added to the group
        
        loops = faces_count-count
        for wd in range(loops):
            
            # ----- The faces already in a group
            # shape = (n,)

            grouped_faces = np.where(groups >= 0)[0]
            if len(grouped_faces) == len(groups):
                break

            #print(f"Random groups, loop {wd}/{loops} ({wd/loops*100:5.1f}%): grouped {len(grouped_faces)}/{len(groups)} ({len(grouped_faces)/len(groups)*100:5.1f}%)")
            
            # ----- Random neighbours per face belonging to a group
            # shape = (n,)
            
            nbs = neighbours.random_values()[grouped_faces]
            
            # ----- The groups of these faces
            # shape = (n,)
            
            grps = groups[nbs]
            
            # ----- Only for groups not already initialized
            # inds shape = (m,)
            
            inds = np.where(grps < 0)[0]
            groups[nbs[inds]] = groups[grouped_faces[inds]]
            
            
        # ---------------------------------------------------------------------------
        # Ok, we can return the groups
        
        return groups   
    
    # ---------------------------------------------------------------------------
    # A stripe between two rows of vertices
    
    @classmethod
    def Stripe(cls, row0, row1=None, close=False):
        if row1 is None:
            row1 = row0 + len(row0)
            
        if len(row0) == len(row1):
            count = len(row0)
                
            inds = np.empty((count, 4), int)
            
            inds[:,   0] = row0
            
            inds[:-1, 1] = row0[1:]
            inds[ -1, 1] = row0[0]
            
            inds[:-1, 2] = row1[1:]
            inds[ -1, 2] = row1[0]
            
            inds[:,   3] = row1
            
            if close:
                return Faces.FromData(np.ones(count, int)*4, np.reshape(inds, count*4))
            else:
                count -= 1
                return Faces.FromData(np.ones(count, int)*4, np.reshape(inds[:count], count*4))
                                  
        else:
            pass
        
    # ---------------------------------------------------------------------------
    # Triangles to one center points
    
    @classmethod
    def Triangles(cls, pole, row, close=False):
            
        count = len(row)
        inds = np.empty((count, 3), int)
        
        inds[:,   0] = row
        
        inds[:-1, 1] = row[1:]
        inds[ -1, 1] = row[0]
        
        inds[:,   2] = pole
        
        if close:
            return Faces.FromData(np.ones(count, int)*3, np.reshape(inds, count*3))
        else:
            count -= 1
            return Faces.FromData(np.ones(count, int)*3, np.reshape(inds[:count], count*3))
    
    # ---------------------------------------------------------------------------
    # Quads grid
    
    @classmethod
    def Grid(cls, x=10, y=10, x_close=False, y_close=False, offset=0):
        
        if x_close:
            x_count = x
        else:
            x_count = x-1
            
        if y_close:
            y_count = y
        else:
            y_count = y-1
        
        faces = Faces.Stripe(np.arange(x), close=x_close)
        
        quads = np.reshape(faces.values + offset, (1, x_count, 4)) + (np.arange(y_count)*x).reshape(y_count, 1, 1)
        
        if y_close:
            quads[y-1:, :, 2:] -= x*y
            
        return cls.FromData(np.ones(x_count*y_count, int)*4, np.reshape(quads, x_count*y_count*4))
        
    # ---------------------------------------------------------------------------
    # Utility which permute the uvs
    
    @staticmethod
    def permute_uvs(uvs, n = 1):
        
        old_uvs = np.array(uvs)
        
        count = np.shape(uvs)[-2]
        for i in range(count):
            uvs[..., i, :] = old_uvs[..., (i-n)%count, :]
    
    
    # ---------------------------------------------------------------------------
    # uv grid
    # x and y give either the number of segments per dimension or the widths of
    # each interval
    # example
    # - x = 10        --> 11 points from 0 to 1
    # - x = [2, 3, 7] -->  4 points proportionnaly set into the interval 0 to 1
        
    @staticmethod
    def uvgrid(x=10, y=10, rect=(0, 0, 1, 1)):
        
        if hasattr(x, '__len__'):
            dx = np.empty(len(x), float)
            dx[:] = x
        else:
            dx = np.ones(x, float)/x
    
        if hasattr(y, '__len__'):
            dy = np.empty(len(y), float)
            dy[:] = y
        else:
            dy = np.ones(y, float)/y
        
        xs = np.insert(np.cumsum(dx), 0, 0)
        ys = np.insert(np.cumsum(dy), 0, 0)
        
        xs = rect[0] + xs/np.sum(dx)*(rect[2] - rect[0])
        ys = rect[1] + ys/np.sum(dy)*(rect[3] - rect[1])
        
        uvs = np.empty((len(dy), len(dx), 4, 2), float)
        
        uvs[..., 0, 0] = xs[:-1]
        uvs[..., 0, 1] = np.expand_dims(ys[:-1], axis=-1)

        uvs[..., 1, 0] = xs[1:]
        uvs[..., 1, 1] = np.expand_dims(ys[:-1], axis=-1)

        uvs[..., 2, 0] = xs[1:]
        uvs[..., 2, 1] = np.expand_dims(ys[1:], axis=-1)

        uvs[..., 3, 0] = xs[:-1]
        uvs[..., 3, 1] = np.expand_dims(ys[1:], axis=-1)
        
        return np.reshape(uvs, (len(dx)*len(dy)*4, 2))
    
    # ---------------------------------------------------------------------------
    # uv triangle
    # x gives either the number of segments or the widths of each interval
    
    @staticmethod
    def uvtriangles(x=10, rect=(0, 0, 1, 1)):
        
        if hasattr(x, '__len__'):
            dx = np.empty(len(x), float)
            dx[:] = x
        else:
            dx = np.ones(x, float)/x
    
        xs = np.insert(np.cumsum(dx), 0, 0)
        xs = rect[0] + xs/np.sum(dx)*(rect[2] - rect[0])

        y0 = rect[1]
        y1 = rect[3]
        
        uvs = np.empty((len(dx), 3, 2), float)
        
        uvs[..., 0, 0] = xs[:-1]
        uvs[..., 0, 1] = y0
        
        uvs[..., 1, 0] = xs[1:]
        uvs[..., 1, 1] = y0
        
        uvs[..., 2, 0] = (xs[:-1] + xs[1:])/2
        uvs[..., 2, 1] = y1
        
        return np.reshape(uvs, (len(dx)*3, 2))

    # ---------------------------------------------------------------------------
    # uv fans
    # Triangle fans around the center
    # x gives either the number of segments or the widths of each interval
    
    @staticmethod
    def uvfans(count=10, rect=(0, 0, 1, 1)):
        
        a = (rect[0] + rect[2])/2
        b = (rect[1] + rect[3])/2
        w = (rect[2] - rect[0])/2
        h = (rect[3] - rect[1])/2

        ags = np.linspace(0, 2*np.pi, count+1, endpoint=True)
        x = a + w*np.cos(ags)
        y = b + h*np.sin(ags)

        uvs = np.empty((count, 3, 2), float)

        uvs[:, 0, 0] = x[:-1]
        uvs[:, 0, 1] = y[:-1]

        uvs[:, 1, 0] = x[1:]
        uvs[:, 1, 1] = y[1:]
        
        uvs[:, 2, 0] = a
        uvs[:, 2, 1] = b

        return np.reshape(uvs, (count*3, 2))

    
    # ---------------------------------------------------------------------------
    # uv polygon
    # x gives either the number of segments or the widths of each interval
    
    @staticmethod
    def uvpolygon(count=10, rect=(0, 0, 1, 1)):
        
        ags = np.linspace(0, 2*np.pi, count, endpoint=False)

        uvs = np.empty((count, 2))
        uvs[:, 0] = (rect[0] + rect[2])/2 + (rect[2]-rect[0])*np.cos(ags)/2
        uvs[:, 1] = (rect[1] + rect[3])/2 + (rect[3]-rect[1])*np.sin(ags)/2
        
        return uvs
    
    @staticmethod
    def uvsquare(rect=(0, 0, 1, 1)):
        
        uvs = np.zeros((4, 2), float)

        uvs[[0, 3], 0] = rect[0]
        uvs[[1, 2], 0] = rect[2]
        
        uvs[[0, 1], 1] = rect[1]
        uvs[[2, 3], 1] = rect[3]
        
        return uvs
    
    # ---------------------------------------------------------------------------
    # Some geometries with vertices
    
    def centers(self, verts):
        
        if len(self) == 0:
            return None
        
        if self.same_sizes:
            size = self.sizes[0]
            return np.average(verts[self.values.reshape(len(self), size)], axis=1)
        
        centers = np.empty((len(self.sizes), 3), float)
        
        for size in np.unique(self.sizes):
            
            arrays, indices = self.arrays_of_size(size)
            
            # Indices is shaped (count, size) where:
            # - count is the number of faces
            # - size the number of vertices per face
            #
            # verts is shape (n, 3) where n is the number of available vertices
            #
            # self.values[indices] is shaped (count, size)
            #
            # verts[self.values[indices]] is shape (count, size, 3)
            #
            # The average is computed on axis 1 (size) to obtain shape (count, 3)
            
            centers[arrays] = np.average(verts[self.values[indices]], axis=1)
        
        return centers
    
    
    # ---------------------------------------------------------------------------
    # Demo
    
    @staticmethod
    def _demo():
        
        cube = Faces.FromList([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
        
        print("-"*30)
        cube._dump("Cube\n")
        print()
        print("edges")
        edges = cube.edges()
        for i in range(len(cube)):
            print(cube._str_matrix(edges[cube.slice(i)], fmt="3d"))
            
        vedges = VArrays.FromData(cube.sizes, edges, duplicate=True)
        vuedges, uedges = vedges.unique()
        print('-'*30)
        vuedges._dump("Unique edges\n")
        for i in range(len(uedges)):
            print(f"{i:2d}: {uedges[i]}")
        print()
        for i in range(len(vuedges)):
            print(cube._str_matrix(uedges[vuedges[i]], fmt="3d"))
            
        cubes = cube.array(3)
        print("-"*30)
        cubes._dump("3 Cubes\n")
        print()
            
        pyramid = Faces.FromList([[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        print("-"*30)
        pyramid._dump("Pyramid\n")
        print()
        print("edges")
        edges = pyramid.edges()
        for i in range(len(pyramid)):
            print(pyramid._str_matrix(edges[pyramid.slice(i)], fmt="3d"))
        
        cube.join(pyramid)
        print("-"*30)
        cube._dump("Cube + Pyramid\n")
        print()
        print("edges")
        edges = cube.edges()
        for i in range(len(cube)):
            print(cube._str_matrix(edges[cube.slice(i)], fmt="3d"))
            
            
        
        
    