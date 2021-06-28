#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helper to build simple meshes with Python

Created: Jul 2020
"""

__author__     = "Alain Bernard"
__copyright__  = "Copyright 2020, Alain Bernard"
__credits__    = ["Alain Bernard"]

__license__    = "GPL"
__version__    = "1.0"
__maintainer__ = "Alain Bernard"
__email__      = "wrapanime@ligloo.net"
__status__     = "Production"



import numpy as np
from math import cos, sin, radians, pi

from .bezier import tangent, PointsInterpolation

two_pi  = pi*2
half_pi = pi/2

#from wrapanime.surface import Surface
#from wrapanime.vert_array import VertArray
#from wrapanime import blender
#from wrapanime import geometry as wgeo

# -----------------------------------------------------------------------------------------------------------------------------
# Get a normalized vector in a direction possibly given by a letter

def get_axis(axis):
    """Returns a 3D np.array corresponding to the required axis
    
    Parameters
    ----------
    axis : str or 3-vector
        Either a string axis spec in ['-x' .. '+z']
        or a 3-vector
        
    Returns
    -------
    3-np.array of floats
        A normalized vecor along the required axis
    """
    
    if type(axis) is str:
        if axis in ['x', 'X', '+x', '+X']:
            return np.array((1., 0., 0.))
        elif axis in ['y', 'Y', '+y', '+Y']:
            return np.array((0., 1., 0.))
        elif axis in ['z', 'Z', '+z', '+Z']:
            return np.array((0., 0., 1.))
        elif axis in ['-x', '-X']:
            return np.array((-1., 0., 0.))
        elif axis in ['-y', '-Y']:
            return np.array((0., -1., 0.))
        elif axis in ['-z', '-Z']:
            return np.array((0., 0., -1.))
        else:
            raise RuntimeError(f"get_axis: invalid axis specification = {axis}.")
    else:
        v = np.array(axis)
        return v/np.linalg.norm(v)

# -----------------------------------------------------------------------------------------------------------------------------
# Get a matrix rotating an axis towards another one

def orient_axis_matrix(from_axis, to_axis):
    """Build a matrix rotating an axis to another one
    
    Parameters
    ----------
    from_axis : str or 3-tuple
        The axis to rotate
        
    to_axis : str or 3-tuple
        Where to rotate
        
    Returns
    -------
    (3,3) matrix
        The rotation matrix
    """

    # Normalize input axis
    x0 = get_axis(from_axis)
    x1 = get_axis(to_axis)

    # Vector perpendicular to the two vectors
    # Rotation will occur around this axis
    y  = np.cross(x0, x1)

    # If angle is null, returns identity matrix
    sn = np.linalg.norm(y)
    if abs(sn) < 0.0001:
        return np.identity(3)

    # Normalize perpendicular vector
    y = y/sn

    # First Matrix :
    # - Orient x axis along from_axis
    # - Target axis is in plane (x, z)
    # - Rotation around y

    M0 = np.zeros((3, 3), np.float)
    M0[:, 0] = x0
    M0[:, 1] = y
    M0[:, 2] = np.cross(x0, y)

    # Target axis is the rotated basis
    x1r = M0.transpose().dot(x1)

    # Rotation Matrix around y
    # Since rotation is around y, y component is null
    # x and z compenents are cos and sin

    MR = np.identity(3)
    MR[0, 0] =  x1r[0]
    MR[0, 2] = -x1r[2]
    MR[2, 0] =  x1r[2]
    MR[2, 2] =  x1r[0]

    # Returns the composition of the 3 matrices

    return np.matmul(np.matmul(M0, MR), M0.transpose())


# -----------------------------------------------------------------------------------------------------------------------------
# Get a matrix rotating around a given axis

def rotation_around_matrix(axis, angle):
    """Build a matrixe for rotation around a given axis
    
    Parameters
    ----------
    axis : str or 3-vector
        The axis to rotate around
        
    angle : float
        The angle to rotate
        
    Returns
    -------
    (3,3) matrix
        The rotation matrix
    """

    M0 = orient_axis_matrix('Z', axis)

    MR = np.identity(3)
    cs = np.cos(angle)
    sn = np.sin(angle)
    MR[0, 0] =  cs
    MR[0, 1] = -sn
    MR[1, 0] =  sn
    MR[1, 1] =  cs

    # Returns the composition of the 3 matrices

    return np.matmul(np.matmul(M0, MR), M0.transpose())

# -----------------------------------------------------------------------------------------------------------------------------
# Utility function

def z_axis_function(t):
    a = np.zeros((np.size(t), 3), np.float)
    a[:, 2] = t
    return a

# -----------------------------------------------------------------------------------------------------------------------------
# A simple mesh builder to build meshes

class MeshBuilder():
    """A class to build a mesh with verts and faces
    
    Vertices can be added using numpy arrays.
    General rule is to return the indices of the created / added vertices.
    
    Example: builder.add([(1, 2, 3), (4, 5, 6)]) return (6, 7) if 6 vertices
    already exist in the mesh.
    """

    def __init__(self):
        self.verts   = np.zeros((0, 3), np.float) # The vertices
        self.faces   = []                         # The faces : tuples of vertices indices
        self.uvs     = []                         # uv mapping. Must be the size of faces

    def __repr__(self):
        return f"MeshBuilder[verts: {len(self.verts)}, faces: {len(self.faces)}]"

    # =============================================================================================================================
    # Vertices management

    # ----------------------------------------------------------------------------------------------------
    # If a list of vertices indices is None, return all the indices

    def indices(self, inds=None):
        """Return the list of vertices indices
        
        Used to select a subset of the vertices, None meaning all.
        
        Parameters
        ----------
        inds : array of int, default None
            The indices to use. If None, return all the vertices indices
            
        Returns
        -------
        array of int
        """

        if inds is None:
            return np.arange(len(self.verts))
        else:
            return np.array(inds)

    # ----------------------------------------------------------------------------------------------------
    # Selected vertices

    def sel_verts(self, inds=None):
        """Return a selection of vertices.
        
        Parameters
        ----------
        inds : array of int
            The vertices to return. Return all the vertices if inds is None
            
        Returns
        -------
        array of vertices
            The selected vertices
        """
        return self.verts[self.indices(inds)]

    # ----------------------------------------------------------------------------------------------------
    # Add new vertices in the list
    # return the indices of the newly created vertices which can be used to create a face

    def add_verts(self, v):
        """Add vertices to tyhe mesh
        
        Parameters
        ----------
        v : array of vertices
            The vertices to add to the mesh

        Returns
        -------
        array of int
            Indices of the vertices just added.
        """
        vs = np.array(v)
        n  = np.size(vs) // 3

        l = len(self.verts)
        self.verts = np.concatenate((self.verts, vs.reshape((n, 3))))

        return [l + i for i in range(n)]

    # ----------------------------------------------------------------------------------------------------
    # Add a single vertex

    def vert(self, v):
        """Add a single vertex.
        
        Parameters
        ----------
        v : vertex
        
        Returns
        -------
        int
            The index of the added vertex
        """
        
        return self.add_verts(v)[0]

    # =============================================================================================================================
    # Faces management

    # ----------------------------------------------------------------------------------------------------
    # Add a new face
    # Return the index of the newly created face

    def face(self, f, inverse=False, uv=None):
        """Add a face to the mesh.
        
        Parameters
        ----------
        f : array of int
            The indices of the vertices making the face
            
        inverse : bool
            Invert the order of the vertices when adding the face
            
        uv : array of couples of floats
            The uv mapping of the face
            
        Returns
        -------
        int
            The index of the newly created face
        """

        # Inverse the face if required
        uv1 = None
        if inverse:
            f1  = tuple(a for a in reversed(f))
            if uv is not None:
                uv1 = tuple(a for a in reversed(uv)) 
        else:
            f1  = tuple(f)
            if uv is not None:
                uv1 = tuple(uv)

        # Append the face and the uv
        self.faces.append(f1)
        self.uvs.append(uv1)

        # Return the index of the newly registered face
        return len(self.faces)-1

    # ----------------------------------------------------------------------------------------------------
    # Add a list of faces

    def add_faces(self, fs, uvs=None):
        """Add a list of faces.
        
        Parameters
        ----------
        fs : array of tuples
            The faces to add to the mesh
            
        uvs : array of tuples of couples of floats
            The uv maps of the faces
        
        Returns
        -------
        array of int
            The indices of the created faces
        """

        n = len(self.faces)
        self.faces += fs
        if uvs is None:
            self.uvs += [None for i in range(len(fs))]
        else:
            self.uvs += uvs

        return [n + i for i in range(len(fs))]

    # =============================================================================================================================
    # Merge and clone

    # ----------------------------------------------------------------------------------------------------
    # Merge

    def merge(self, other):
        """Merge the mesh with another one.
        
        Parameters
        ----------
        other : Meshbuilder
            The mesh to merge with
            
        Returns
        -------
        two arrays of int
            Indices of added vertices and indices of added faces
        """

        n = len(self.verts)
        vs = self.add_verts(other.verts)
        faces = [[n + iv for iv in face] for face in other.faces]
        fs = self.add_faces(faces, other.uvs)
        
        return vs, fs

    # ----------------------------------------------------------------------------------------------------
    # Clone the builder

    def clone(self, clone_faces=True):
        """Clone the mesh into another one
        
        Parameters
        ----------
        clone_faces : bool
            Don't clone faces nor uvs if False
            
        Returns
        -------
        Meshbuilder
            The cloned Meshbuilder class
        """

        builder = MeshBuilder()
        builder.add_verts(self.verts)
        if clone_faces:
            builder.faces = [tuple(face) for face in self.faces]
            builder.uvs   = [None if mp is None else [uv for uv in mp] for mp in self.uvs]

        return builder

    # =============================================================================================================================
    # Basic shapes

    # ---------------------------------------------------------------------------
    # Polygon / circle

    def polygon(self, radius=1, count=6, axis='Z'):
        """Build a polygon.
        
        Parameters
        ----------
        radius : float
            Polygon radius
            
        count : int
            Number of vertices on the circumference
            
        axis : str of 3-vector
            Face perpendicular
            
        Returns
        -------
        array of int
            The created vertices
        """

        angles = np.arange(count)*two_pi/count
        verts  = radius*np.stack((np.cos(angles), np.sin(angles), np.zeros(count))).transpose()
        
        rotate = type(axis) is not str
        if rotate:
            rotate = axis not in ['Z', 'z', '+Z', '+z']
        
        if rotate:
            M = orient_axis_matrix('Z', axis)
            verts = np.matmul(verts, M.transpose())

        # Add the vertices
        return self.add_verts(verts)

    # ---------------------------------------------------------------------------
    # Grid

    def grid(self, size=(2., 2.), count=(10, 10), faces=True, axis='Z'):
        """Create a grid.
        
        The grid is centered on the origin and perpendicular to the
        provdied axis.
        
        Parameters
        ----------
        size : couple of float
            The size of the grid
            
        count : couple of int
            The number of vertices to create in each direction
            
        faces : bool
            Generate the faces if True
            
        axis : str or vector
            The axis perpendicular to the polygon
            
        Returns
        -------
        two arrays of int
            The indices of the created vertices and of the created faces
        """

        x = np.linspace(-size[0]/2, size[0]/2, count[0])
        y = np.linspace(-size[1]/2, size[1]/2, count[1])
        xx, yy = np.meshgrid(x, y)

        verts = np.stack((xx, yy, np.zeros(xx.shape)), axis=2).reshape(np.size(xx), 3)
        inds  = self.add_verts(verts)
        ifaces = []

        fs = []
        if faces:
            fs = [[i*count[0]+j, (i+1)*count[0]+j, (i+1)*count[0]+j+1, i*count[0]+j+1] for j in range(count[0]-1) for i in range(count[1]-1)]
            ifaces = self.add_faces(fs)

        return inds, ifaces

    # =============================================================================================================================
    # Geometry computations

    # ---------------------------------------------------------------------------
    # Compute the center of a face

    def face_center(self, face=None):
        """Compute the center of an array of vertices.
        
        Parameters
        ----------
        face : array of int, default None
            Indices of the vertices. All the vertices if the parameter is None.
            
        Returns
        -------
        vector
            The center of the vertices
        """
        
        inds = self.indices(face)
        V = np.einsum('ij->j', self.verts[inds])
        return V / len(inds)

    # ---------------------------------------------------------------------------
    # Compute the normal of a face

    def face_normal(self, face=None):
        """Compute the normal to a list of vertices.
        
        Parameters
        ----------
        face : array of int, default None
            Indices of the vertices. All the vertices if the parameter is None.
            
        Returns
        -------
        vector
            The normal to the vertices
        """
        
        if len(face) < 3:
            return None

        # Vectors from center to vertices
        C     = self.face_center(face)
        verts = self.verts[face] - C

        # All perps
        perps = np.cross(verts[0], verts[1:])

        # Norms
        norms = np.linalg.norm(perps, axis=1, keepdims=True)

        # Max index
        ind = np.argmax(norms)
        nrm = perps[ind] / norms[ind]

        # Check the Orientation
        if np.dot(perps[0], nrm) < 0:
            nrm = -nrm

        return nrm

    # ---------------------------------------------------------------------------
    # Bounding box

    def bounding_box(self, inds=None):
        """Compute the boundig box of a list of vertices.
        
        Parameters
        ----------
        inds : array of int, default None
            Indices of the vertices. All the vertices if the parameter is None.
            
        Returns
        -------
        array of two vertices
            The two vertices at the opposite corners of the bouding box
        """
        
        vs = self.self_verts(inds)
        return np.array(
                [np.min(vs[:, 0]), np.min(vs[:, 1]), np.min(vs[:, 2])],
                [np.max(vs[:, 0]), np.max(vs[:, 1]), np.max(vs[:, 2])]
                )

    # =============================================================================================================================
    # uv mapping

    def uv_from_face(self, face):
        """Compute the uv coordinates of a face.
        
        CAUTION: Not tested
        
        Parameters
        ----------
        face : array of int
            Indices of the vertices.
        """
        
         # CAUTION : NOT TESTED

        if len(face) < 3:
            return None

        c = self.verts[0]
        vx = [self.verts[iv]-c for iv in face]

        p = self.face_normal(face)
        vp = [v - v.dot(p)*p for v in vx]

        vi = vp[1].normalized()
        vj = p.cross(vi)

        uvx = np.array([vi.dot(v) for v in vp])
        uvy = np.array([vj.dot(v) for v in vp])

        try:
            uvx /= max(abs(uvx))
        except:
            pass

        try:
            uvy /= max(abs(uvy))
        except:
            pass

        uv = [[uvx[i], uvy[i]] for i in range(len(face))]
        return uv

    # =============================================================================================================================
    # Transformations

    # ---------------------------------------------------------------------------
    # Translation

    def translate(self, selection, translation):
        """Translate vertices.
        
        Parameters
        ----------
        selection : array of int
            Indices of the vertices to translate. All the vertices if None.

        translation : vector
            The translation vector
        """    
        
        inds = self.indices(selection)
        self.verts[inds] += np.array(translation)

    # ---------------------------------------------------------------------------
    # Rotation with Matrix

    def rotate(self, selection, M, center=(0., 0., 0.)):
        """Rotate vertices with a matrix
        
        Parameters
        ----------
        selection : array of int
            Indices of the vertices to translate. All the vertices if None.

        M : (3, 3) matrix
            The rotation matrix
            
        center : vertex
            Vertex to rotate around
        """
        
        C = np.array(center)
        
        inds = self.indices(selection)
        self.verts[inds] = C + np.matmul(self.verts[inds] - C, M)

    # ---------------------------------------------------------------------------
    # Rotation around a given axis

    def rotate_axis(self, selection, axis='Z', angle=0., center=(0., 0., 0.)):
        """Rotate vertices around an axis with a given angle.
        
        Parameters
        ----------
        selection : array of int
            Indices of the vertices to translate. All the vertices if None.

        axis : str or vector
            The axis to rotate the vertices around
            
        angle : float
            Value of the rotation around the axis
            
        center : vertex
            Vertex to rotate around
        """
        
        self.rotate(selection, rotation_around_matrix(axis, angle), center)

    # ---------------------------------------------------------------------------
    # scale

    def scale(self, selection, factor=1., center=None):
        """Scale vertices of a given factor relatively to a center.
        
        Parameters
        ----------
        selection : array of int
            Indices of the vertices to translate. All the vertices if None.

        factor : float
            The scale factor.
            
        center : vertex, default None
            The vertex to use fo scale computation. If None, use the
            center of the vertices.
        """
        
        inds = self.indices(selection)
        if center is None:
            C = self.face_center(inds)
        else:
            C = np.array(center)

        self.verts[inds] = C + (self.verts[inds] - C)*np.resize(factor, (3))

    # =============================================================================================================================
    # Extrusion
    # Returns the vertices indices of the extruded face plus the faces created along the extrusion

    def extrude(self, selection, translation=(0., 0., 0.)):
        """Extrude vertices along a given vector.
        
        Parameters
        ----------
        selection : array of int
            Indices of the vertices to translate. All the vertices if None.
            
        translation : vector
            The extrusion vector
            
        Returns
        -------
        two arrays of int
            The indices of the created vertices and the indices of the created faces
        """
        
        face = self.indices(selection)

        verts = np.array(self.verts[face]) + np.array(translation)
        newf  = self.add_verts(verts)

        faces = []
        n = len(selection)
        for i in range(n):
            faces.append(self.face([face[i], face[(i+1)%n], newf[(i+1)%n], newf[i]]))

        return newf, faces

    # =============================================================================================================================
    # Advanced meshes

    # ---------------------------------------------------------------------------
    # Cylinder
    # A cylinder is defined by a backbone functions
    # Bot and Top can be capped with CAP, SPIKE and ARROW (SPHERE TO BE DONE)
    
    def cylinder(self, f=z_axis_function, t0=-1., t1=1., radius=1., steps=2, segments=8, bot=None, top=None):
        """Create a cylindric envelop around a curve.
        
        Can be used to represent un curve.
        
        Parameters
        ----------
        f : function of template f(t) = vector
            The function representing the cylinder axis.
            
        t0 : float, default 0.
            Lower bound of the interval
            
        t1 : float, default 1.
            Upper bound of the interval
            
        radius : float of function of template f(t) = float, default 1.
            The constant radius (if float) or varying radius of the cylinder
            
        steps : int, default 2
            The number of rings in the cylinder
            
        segments : int, default 8
            The number of segments / vertices per ring
            
        bot : str in ['CAP', 'SPIKE', 'ARROW'] or None, default None
            How to  close the bottom of the cylinder
            
        top : str in ['CAP', 'SPIKE', 'ARROW'] or None, default None
            How to  close the top of the cylinder
        
        Returns
        -------
        array of array of int, array of int
            The indices of the created vertices and the indices of the created faces
        """
        
        # Base radius
        var_radius = hasattr(radius, '__call__')
        base_radius = 1. if var_radius else radius
            
        # Number of rings
        steps = max(2, steps)
        
        # Rings centers along the backbone
        ts = np.linspace(t0, t1, steps)
        try:
            cs = f(ts)
        except:
            vf = np.vectorize(f)
            cs = vf(ts)
            
        # Tangents
        print("Cylinder", ts, (t1-t0)/100)
        tgs = tangent(f, ts, dt=(t1-t0)/100)

        # Base face
        base = self.polygon(radius=base_radius, count=segments, axis=tgs[0])
        self.translate(base, cs[0])

        # Will return an array with all the created sections
        sections = [base]

        # Array of created faces
        faces    = []

        # Initialize the loop variable section
        section = base
        
        # Loop
        for i in range(1, steps):
            
            # Extrude the last ring
            section, newf = self.extrude(sections[-1], cs[i]-cs[i-1])
            
            # Rotation to new tangent
            M = orient_axis_matrix(tgs[i], tgs[i-1])
            self.rotate(section, M, center=cs[i])
            
            # Update the arrays
            faces.extend(newf)
            sections.append(section)
            
        # Scale if variable radius
        if var_radius:
            for i in range(steps):
                self.scale(sections[i], factor=radius(ts[i]), center=cs[i])
            
            
        # Close to two extremities
        radius0 = radius(t0) if var_radius else radius
        radius1 = radius(t1) if var_radius else radius
        self.close(sections[0],  cap=bot, length=radius0*5, center=cs[0],  direction=-tgs[0])
        self.close(sections[-1], cap=top, length=radius1*5, center=cs[-1], direction= tgs[-1])

        # Return the sections and faces
        return sections, faces

    # ---------------------------------------------------------------------------
    # Close a cylinder with a CAP, a SPIKE or an ARROW

    def close(self, face, cap=None, length=None, center=None, direction=None):
        """Close a cylinder with CAP, SPIKE or ARROW
        
        Parameters
        ----------
        
        face : array of int
            The face to close
            
        cap : str in ['CAP', 'SPIKE', 'ARROW']
            Shape of the closing
            
        length : float
            Length of the SPIKE or ARROW
            
        center : vertex, default None
            Center of the face. Compute the face center if none
            
        direction : str of vector
            The direction of the SPIKE or ARROW
        """

        # Center of the face
        if center is None:
            C = self.face_center(face)
        else:
            C = np.array(center)

        # Direction of extrusion
        if direction is None:
            tg = self.face_normal(face)
        else:
            tg = np.array(direction)
        tg /= np.linalg.norm(tg)
        
        # Number of vertices@
        n = len(face)

        # Let's go

        # ----- caps
        if (cap == 'CAP') or (cap == True):
            self.face(face)

        # ----- Spike
        elif cap == 'SPIKE':
            X  = C + tg*length
            ix = self.vert(X)
            for i in range(n):
                self.face((ix, face[i], face[(i+1)%n]))

        # ----- Arrow
        elif cap == 'ARROW':
            trans = -tg*length/3

            section, _ = self.extrude(face, translation=trans)
            self.scale(section, 2.6, C + trans)

            self.close(section, cap='SPIKE', length=length*4/3, center= C + trans, direction=tg)
            
            
    # ---------------------------------------------------------------------------
    # Demo
    
    @classmethod
    def Demo(cls, shape='VAR_SIN'):
        
        # Cylinder backbone
        def bb(t):
            vs = np.zeros((len(t), 3), float)
            vs[:, 0] = t
            vs[:, 2] = np.sin(t)
            return vs
        
        # Envelop shape
        def env(t):
            return 0.1 + abs(np.sin(5.1*t)*0.3)
        
        builder = MeshBuilder()
        if shape == 'ARROW':
            builder.cylinder(radius=0.1, segments=16, bot='CAP', top='ARROW')
        elif shape == 'SINUSOID':
            builder.cylinder(bb, 0., 20, radius=0.1, steps=100, segments=8) #, bot='SPIKE', top='ARROW')
        elif shape == 'RANDOM':
            count = 15
            pint = PointsInterpolation(np.random.uniform(-3, 3, (count, 3)))
            builder.cylinder(pint, 0., 1., radius=0.1, steps=500, segments=16, bot='CAP', top='CAP')
        else:
            builder.cylinder(bb, 0., 20, radius=env, steps=500, segments=32, bot='SPIKE', top='ARROW')
            
        
        return builder


# ---------------------------------------------------------------------------
# Some tests

def tests():
    
    # orient_axis_matrix
    print('-'*30)
    print("orient_axis_matrix test")
    print()
    print("For random axis, compute cross product between rotated vector and")
    print("target vector. Must be zero.")

    
    count = 1000
    ns = np.zeros(count, np.float)
    for i in range(count):
        # Two random vectors
        v0 = np.random.uniform(-10, 10, 3)
        v1 = np.random.uniform(-10, 10, 3)
        
        # Rotation matrix
        m = orient_axis_matrix(v0, v1)
        
        # Test
        v = m.dot(v0)
        ns[i]= np.linalg.norm(np.cross(v, v1))
    
    print(f"Result for {count} computations:")
    print(f"Avg: {np.average(ns):.6f}")
    print(f"Min: {np.min(ns):.6f}")
    print(f"Max: {np.max(ns):.6f}")
    print(f"Std: {np.std(ns):.6f}")
    print()
    
    
    # rotation_around_matrix
    print('-'*30)
    print("rotation_around_matrix test")
    print()
    
    count = 1000
    axs = np.zeros(count, np.float)
    rts = np.zeros(count, np.float)
    for i in range(count):
        
        v0   = np.random.uniform(-3, 3, 3)
        v1   = np.random.uniform(-3, 3, 3)
        axis = np.cross(v0, v1)
        
        v0 /= np.linalg.norm(v0)
        v1 /= np.linalg.norm(v1)
        ag  = np.arcsin(np.linalg.norm(np.cross(v0, v1)))
        
        m = rotation_around_matrix(axis, ag)
        
        # Angle is not ok
        nrm = np.linalg.norm(m.dot(v0)-v1)
        if nrm > 0.1:
            m = rotation_around_matrix(axis, np.pi-ag)
        
        axs[i] = np.linalg.norm(m.dot(axis) - axis)
        rts[i] = np.linalg.norm(m.dot(v0) - v1)
        
        if rts[i] > 0.1:
            print(f"Strange at step {i}: angle = ({np.degrees(ag):.1f})")
            print("v0:", v0)
            print("v1:", v1)
            print("axis:", axis)
            print("m.dot(v0):", m.dot(v0), np.linalg.norm(m.dot(v0)-v1))
            print("m.dot(v1):", m.dot(v1), np.linalg.norm(m.dot(v1)-v0))
            print()
        
    print(f"Result for {count} computations:")
    print(f"Avg: {np.average(axs):.6f} {np.average(rts):.6f}")
    print(f"Min: {np.min(axs):.6f} {np.min(rts):.6f}")
    print(f"Max: {np.max(axs):.6f} {np.max(rts):.6f}")
    print(f"Std: {np.std(axs):.6f} {np.std(rts):.6f}")
    print()
        
    








class VERY_OLD():

    # =============================================================================
    # Map a vertices loop along the x-axis
    # =============================================================================

    def verts_uloc(self, verts, u_bounds=[0., 1.]):
        """u of uv map coordinates for a sequence of vertices.

        The successive vertices are mapped along the u-axis within the intervalle u_bounds.
        The vertices are flattened : the distance (always positive) between two
        successive vertices is used.

        Example : for a regular polygon of 4 edges, the result is:
            u_bounds = [0.0, 1.0] --> [0.0, 0.25,  0.5,  0.75,  1.]
            u_bounds = [0.5, 0.6] --> [0.5, 0.525, 0.55, 0.575, 0.6]

        Parameters
        ----------
        verts : array of int
            Indices of the vertices to manage
        u_bounds : couple of float
            min and max x values for the mapping

        Returns
        -------
        array of float
            The x abscisses for the uv map of the vertices sequences

        """

        n = len(verts)
        if n <= 2:
            return tuple(u_bounds)

        # Vertices copy
        VX = [self.verts[iv] for iv in verts]

        # Length of each side of the face
        # Note that if n == 2, there is no loop
        u_len = np.array([len(VX[(i+1)%n]-VX[i]) for i in range(n)], np.float)

        # The total length must match the x amplitude
        L = sum(u_len)
        u_len *= (u_bounds[1]-u_bounds[0]) / L

        # X Location is mapped between uvx[0] and uvx[1]
        u_loc = [u_bounds[0] + sum(u_len[:i]) for i in range(n)]

        # Append the max limit to have n+1 locations
        u_loc.append(u_bounds[1])

        return u_loc

    # =============================================================================
    # Create faces between a loop and a vertex
    # =============================================================================

    def cone_faces(self, loop, topv):
        """Create faces between a loop and a top vertex.

        Parameters
        ----------
        loop: array of int
            Indices of the vertices within the loop
        topv: int
            Index of the top vertex

        Returns
        -------
        array of array of int
            The cteated faces
        """

        faces = []
        n = len(loop)
        for i in range(n):
            faces.append(self.face([loop[i], loop[(i+1)%n], topv]))
        return faces

    # =============================================================================
    # Link two loops with faces
    # =============================================================================

    def link_with_faces(self, verts0, verts1, u_bounds=[0., 1.], v_bounds=[0., 1.]):
        """Create faces between two sequences.

        The two sequences must be of the same size.

        Parameters
        ----------
        verts0 : array of int
            The first sequence to link with faces
        verts1 : array of int
            The second sequence to link with faces
        u_bounds : couple of float
            The u bounds for the uv map
        v_bounds : couple of float
            The v bounds for the uv map

        Returns
        array of int
            The indices of the created faces
        """

        # ---------------------------------------------------------------------------
        # u locations

        u_loc0 = self.verts_uloc(verts0, u_bounds=u_bounds)
        u_loc1 = self.verts_uloc(verts1, u_bounds=u_bounds)

        # ---------------------------------------------------------------------------
        # Faces

        n = len(verts0)
        y0 = u_bounds[0]
        y1 = u_bounds[1]
        faces = [[verts0[i], verts0[(i+1)%n], verts1[(i+1)%n], verts1[i]] for i in range(n)]
        uvs   = [[(u_loc0[i], y0), (u_loc0[i+1], y0), (u_loc1[i+1], y1), (u_loc1[i], y1)] for i in range(n)]

        return self.add_faces(faces, uvs)


    # =============================================================================
    # Extrusion
    # =============================================================================

    def extrude(self, verts, amount, steps=1, u_bounds=[0., 1.], v_bounds=[0., 1.], close=False):
        """Extrude vertices of certain amount.

        The extruded faces are uv mmaped along u for the ring and along v
        for the extrusion. u_bounds and v_bounds allow to constraint the mapping
        to a sub area.

        Extrude uses an existing sequence of vertices. Cylinder create all the sequences,
        including the first one.

        Parameters
        ----------
        verts : array of int
            Indices of the vertex to extrude
        amount : vector-like or function(step_index, section_index, vertex) -> vertex
            The vector to use for extrusion.
            If amount is a function, it takes three parameters:
                Parameters
                ----------
                step_index : int
                    The extrusion step
                section_index : int
                    The index in the section (not the index of the vertex)
                vertex : Vector
                    The vertex value

                Returns
                -------
                Vector
                    The extruded vertex

        steps : int
            The extrusion can be made in several steps, ie by extruding several
            vertex on the extrusion path
        u_bounds : tuple of float
            The min and max for x uv mapping
        v_bounds : tuple of float
            Tthe min and max for y uv mapping
        close : boolean
            The end of the extrusion must be linked to the begining

        Returns
        -------
        array of array of int, array of int
            array of the created vertices arraged in edges and array of the created faces.
            In the created array of vertices, the index 0 if for the initial section
        """

        # ---------------------------------------------------------------------------
        # x uv mapping

        u_loc = self.verts_uloc(verts, u_bounds)

        # ---------------------------------------------------------------------------
        # y uv mapping is easier :-)

        dy = v_bounds[1]-v_bounds[0]
        if close:
            dy /= steps+1
        else:
            dy /= steps
        v0 = v_bounds[0]

        # ---------------------------------------------------------------------------
        # Extrude one line of vertices per vertex

        with_function = type(amount).__name__ == "function"

        if with_function:
            lines = [[iv] + self.add_verts([amount(j, sec_i, self.verts[iv]) for j in range(steps)]) for sec_i, iv in enumerate(verts)]
        else:
            V = Vector(amount)/steps
            lines = [[iv] + self.add_verts([self.verts[iv] + (j+1)*V for j in range(steps)]) for iv in verts]

        # ---------------------------------------------------------------------------
        # Create the faces when they ara at least 2 vertices in the list

        n = len(verts)
        if n == 2:
            uvs = [ [[u_bounds[0], v0+dy*i], [u_bounds[1], v0+dy*i], [u_bounds[1], v0+dy*(i+1)], [u_bounds[0], v0+dy*(i+1)]] for i in range(steps)]
            fs  = [ [lines[0][i], lines[1][i], lines[1][i+1], lines[0][i+1]] for i in range(steps)]
            faces = self.add_faces(fs, uvs)

        elif n > 2:
            uvs = [ [[u_loc[j], v0+dy*i], [u_loc[j+1], v0+dy*i], [u_loc[j+1], v0+dy*(i+1)], [u_loc[j], v0+dy*(i+1)]] for i in range(steps) for j in range(n)]
            fs  = [ [lines[j][i], lines[(j+1)%n][i], lines[(j+1)%n][i+1], lines[j][i+1]] for i in range(steps) for j in range(n)]
            faces = self.add_faces(fs, uvs)

        # ---------------------------------------------------------------------------
        # Close the extrusion

        if close:
            #self.link_with_faces([lines[i][-1] for i in range(n)], verts, u_bounds=u_bounds, v_bounds=[v_bounds[1]-dy, v_bounds[1]])
            self.face([lines[i][-1] for i in range(n)])

        # ---------------------------------------------------------------------------
        # Return lines and faces

        return lines, faces

    # =============================================================================
    # Extrusion
    # =============================================================================

    def cylinder(self, section, path, steps=1, u_bounds=[0., 1.], v_bounds=[0., 1.], close=False):
        """Create a cylinder with a given section.

        The extruded faces are uv mmaped along u for the ring and along v
        for the extrusion. u_bounds and v_bounds allow to constraint the mapping
        to a sub area.

        Extrude uses an existing sequence of vertices. Cylinder create all the sequences,
        including the first one.

        Parameters
        ----------
        section : array of vector-like
            The vertices forming the section of the cylinder
        path : vector-like or function(step_index, section_index, vertex) -> vertex
            If vector-like, it represents to total amount of the extrusion to perform

            If path is a function, it takes three parameters:
                Parameters
                ----------
                step_index : int
                    The extrusion step
                section_index : int
                    The index in the section (not the index of the vertex)
                vertex : Vector
                    The vertex value

                Returns
                -------
                Vector
                    The extruded vertex

        steps : int
            The extrusion can be made in several steps, ie by extruding several
            vertex on the extrusion path
        u_bounds : tuple of float
            The min and max for x uv mapping
        v_bounds : tuple of float
            Tthe min and max for y uv mapping
        close : boolean
            The end of the extrusion must be linked to the begining

        Returns
        -------
        array of array of int, array of int
            array of the created vertices arraged in edges and array of the created faces.
        """

        n = len(section)

        # ---------------------------------------------------------------------------
        # Extrude one line of vertices per vertex

        with_function = type(path).__name__ == "function"
        if with_function:
            lines = [self.add_verts([path(j, i, section[i]) for j in range(steps)]) for i in range(n)]
        else:
            V = Vector(path)/steps
            lines = [self.add_verts([section[i] + j*V for j in range(steps)]) for i in range(n)]

        # ---------------------------------------------------------------------------
        # x uv mapping

        u_loc = self.verts_uloc([lines[i][0] for i in range(n)], u_bounds)

        # ---------------------------------------------------------------------------
        # y uv mapping is easier :-)

        dy = v_bounds[1]-v_bounds[0]
        if close:
            dy /= steps
        else:
            dy /= (steps-1)
        v0 = v_bounds[0]

        # ---------------------------------------------------------------------------
        # Create the faces when they ara at least 2 vertices in the list

        if n == 2:
            uvs = [ [[u_bounds[0], v0+dy*i], [u_bounds[1], v0+dy*i], [u_bounds[1], v0+dy*(i+1)], [u_bounds[0], v0+dy*(i+1)]] for i in range(steps)]
            fs  = [ [lines[0][i], lines[1][i], lines[1][i+1], lines[0][i+1]] for i in range(steps)]
            faces = self.add_faces(fs, uvs)

        elif n > 2:
            uvs = [ [[u_loc[j], v0+dy*i], [u_loc[j+1], v0+dy*i], [u_loc[j+1], v0+dy*(i+1)], [u_loc[j], v0+dy*(i+1)]] for i in range(steps-1) for j in range(n)]
            fs  = [ [lines[j][i], lines[(j+1)%n][i], lines[(j+1)%n][i+1], lines[j][i+1]] for i in range(steps-1) for j in range(n)]
            faces = self.add_faces(fs, uvs)

        # ---------------------------------------------------------------------------
        # Close the extrusion

        if close:
            self.link_with_faces([lines[i][-1] for i in range(n)], [lines[i][0] for i in range(n)], u_bounds=u_bounds, v_bounds=[v_bounds[1]-dy, v_bounds[1]])

        # ---------------------------------------------------------------------------
        # Return lines and faces

        return lines, faces

    # =============================================================================
    # Twist
    # =============================================================================

    def twist(self, axis, angle):
        """Twist the shape along an axis and with a given angle.

        The twist is made around an axis and is of a certain angle.
        The middle of the mesh is unchanged, twisting is made half of the angle
        on each half of the mesh.

        Parameters
        ----------
        axis : vector-like
            The axis to twist the mesh around
        angle: float
            The angle to rotate
        """

        A = Vector(axis).normalized()

        mn = None
        mx = None
        for v in self.verts:
            x = A.dot(v)
            mn = x if mn is None else min(x, mn)
            mx = x if mx is None else max(x, mx)

        amp = mx-mn
        if amp < 0.0001:
            return

        for i, v in enumerate(self.verts):
            x = A.dot(v)
            ag = -angle/2. + angle*(x-mn)/amp
            q = Quaternion(A, ag)
            self.verts[i].rotate(q)

    # =============================================================================
    # Bend around z axis
    # =============================================================================

    def bendz(self, angle=0.):
        """Bend the shape around the Z axis and towards X axis.

        The length along x is bended such as forming an arc of the same length under the
        given angle. Given the length of the arc and the angle, the radius is computed with arc/angle.
        The center is located on the y-axis at radius distance of the y-middle of the mesh.

        A vertex is projected with the following algorithm:
            - Theta = x / radius
            - (x', y') = rotation of (x, y) around the center

        Parameters
        ----------
        angle : float
            The angle to bend
        """

        if abs(angle) < radians(0.1):
            return

        # ----- Length of the mesh
        b0, b1 = self.bounding_box()
        length = b1.x - b0.x
        if length < 0.001:
            return

        # ---- Bending location
        # The radius is so that the length of the arc under angle is the length of the mesh
        radius = length/angle
        cy = (b0.y + b1.y)/2. + radius

        # Bend loop
        for i, v in enumerate(self.verts):
            rho   = cy - v.y
            theta = v.x/radius
            self.verts[i] = Vector((rho*sin(theta), -rho*cos(theta)+cy, v.z))
