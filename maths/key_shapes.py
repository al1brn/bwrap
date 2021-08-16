#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:53:32 2021

@author: alain
"""

import numpy as np

if True:
    from .shapes import get_full_shape
    from ..core.commons import WError
else:
    def get_full_shape(main_shape, item_shape):
        return (tuple(main_shape) if hasattr(main_shape, '__len__') else (main_shape, )) + (tuple(item_shape) if hasattr(item_shape, '__len__') else (item_shape, ))
    
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# This module encapsulate the shape keys in a single array
# The shape of the array is (number of shape keys, number of points, end_size)
# end_size depends upon the type of shapes
# - mesh                 end_size =  3 - vertices only
# - curve without bezier end_size =  5 - vertices plus tilt and radius
# - curve with bezier    end_size = 11 - vertices plus tilt and radius plus left and right handles
#
# The interpoloation is made in one computation only
#
# A curve object can have both bezier and non bezier curves. In that case, foreach_set and foreach_get
# don't work for handles. The compute_beziers checks if :
# - NONE   : There is no bezier curves
# - ALL    : All curves are bezier
# - SOME   : There are both bezier and non bezier curves
#
# For SOME, the algorith performs a loop on the key points, otherwise, it uses foreach access
#
# KeyShapes encapsulates the whole array.
# KeyShapes points to such an aray and performe read / write on the object shapes keys
#
# Typical use is:
# kss = KeyShapes.Read(object)
# ks = ks.get_key_shape(5)
# ... perform operations on ks
# kss.write(object.data)
#
# ----- Interpolation
# Interpolation can be computed with intermediary values betweent the various key shapes
# The compputation accepts an array of intermediary values.
# This allows to compute at once several interpolations
# The result is an array of shapes with an array shape based on array shape of t:
#
# Example: t = array of shape (2, 3)
# KeyShape.interpolation(t) returns an array of shape (2, 3, number of points, end_size)
#
# An interpolation can be applied to an object using the method set_to
#
# points : KeyShape(...).interpolations(t)
# for ... :
#   KeyShape(points, index).set_to(target_object)
#
# The points can be used in a Crowd
#
# ----- Key names
# The key shapes to read in the KeyShapes class can be controlled with the key_names array
# For instance, take all the key names corresponding to an animation to exclude other key shapes
# If key names are not provided, all the key shapes are loaded
#
# Note that KeyShapes and KeyShape can be initiated without object:
# - Create ShapeKeys from an array
# - Compute interpolations
# --> Set to an object shape


# -----------------------------------------------------------------------------------------------------------------------------
# Constant

VERTS_SLICE  = slice( 0,  3)
RADIUS_SLICE = slice( 3,  4)
TILTS_SLICE  = slice( 4,  5)
LEFTS_SLICE  = slice( 5,  8)
RIGHTS_SLICE = slice( 8, 11)

MESH_SIZE    = 3
CURVE_SIZE   = 5
BEZIER_SIZE  = 11 

# -----------------------------------------------------------------------------------------------------------------------------
# An individual key shape which points to an array of key shapes values

class KeyShape():
    
    def __init__(self, points, index, name=None):
        self.points  = points
        self.index   = index
        self.name    = name
        self.total   = np.product(np.shape(points)[:-2])
        self.linear  = (self.total, np.shape(points)[-2], np.shape(points)[-1])
        
    # ---------------------------------------------------------------------------
    # The length corresponds to the number of points in a key shape
        
    def __len__(self):
        return self.points.shape[-2]

    # ---------------------------------------------------------------------------
    # Check if the end_size allows attributes to be read in the points array
    
    def check_attr(self, attr_name, **kwargs):
        
        end_size = self.points.shape[-1]
        
        if attr_name in ['tilts', 'radius']:
            if end_size > MESH_SIZE:
                return True
            
        if attr_name in ['lefts', 'rights']:
            if end_size == BEZIER_SIZE:
                return True
        
        raise WError(f"Attribute '{attr_name}' unsupported for key shape",
                     Class = "KeyShape",
                     **kwargs)
        
    # ---------------------------------------------------------------------------
    # Vertices        
        
    @property
    def verts(self):
        return self.points.reshape(self.linear)[self.index, :, VERTS_SLICE]
    
    @verts.setter
    def verts(self, value):
        self.points.reshape(self.linear)[self.index, :, VERTS_SLICE] = value
        
    # ---------------------------------------------------------------------------
    # Tilts and radius        
        
    @property
    def tilts(self):
        self.check_attr('tilts')       
        return np.squeeze(self.points.reshape(self.linear)[self.index, :, TILTS_SLICE])
    
    @tilts.setter
    def tilts(self, value):
        self.check_attr('tilts')       
        self.points.reshape(self.linear)[self.index, :, TILTS_SLICE] = np.expand_dims(value, axis=-1)
        
    @property
    def radius(self):
        self.check_attr('radius')       
        return np.squeeze(self.points.reshape(self.linear)[self.index, :, RADIUS_SLICE])
    
    @radius.setter
    def radius(self, value):
        self.check_attr('radius')       
        self.points.reshape(self.linear)[self.index, :, RADIUS_SLICE] = np.expand_dims(value, axis=-1)
        
    # ---------------------------------------------------------------------------
    # Left and right handles     
        
    @property
    def lefts(self):
        self.check_attr('lefts')       
        return self.points.reshape(self.linear)[self.index, :, LEFTS_SLICE]
    
    @lefts.setter
    def lefts(self, value):
        self.check_attr('radius')       
        self.points.reshape(self.linear)[self.index, :, LEFTS_SLICE] = value
        
    @property
    def rights(self):
        self.check_attr('rights')       
        return self.points.reshape(self.linear)[self.index, :, RIGHTS_SLICE]
    
    @rights.setter
    def rights(self, value):
        self.check_attr('rights')       
        self.points.reshape(self.linear)[self.index, :, RIGHTS_SLICE] = value
        
    # ---------------------------------------------------------------------------
    # Encapsulation of foreach get
    
    def read_attr(self, data, attr_name, dim):
        a = np.empty(len(data)*dim, np.float)
        data.foreach_get(attr_name, a)
        if dim > 1:
            return a.reshape(len(data), dim)
        else:
            return a
        
    # ---------------------------------------------------------------------------
    # Encapsulation of foreach set
    
    def write_attr(self, data, attr_name, value, dim):
        data.foreach_set(attr_name, np.reshape(value, len(data)*dim))        
        
    # ---------------------------------------------------------------------------
    # Read from data
        
    def read(self, data, beziers=None):
        
        if len(data) != len(self):
            raise WError(f"Impossible to read the shape. shape length is {len(self)} when the length to read is {len(data)}",
                         Class = "KeyShape",
                         Method = "read",
                         data = data,
                         beziers = beziers)
        
        # ----- The vertices
        
        self.verts  = self.read_attr(data, 'co',     3)
        
        # ----- Continue for splines
        
        if not hasattr(data[0], 'radius'):
            return
            
        self.radius = self.read_attr(data, 'radius', 1)
        self.tilts  = self.read_attr(data, 'tilt',   1)
        
        if beziers is None:
            beziers = KeyShapes.compute_beziers(data)
        
        # ----- Only bezier curves
    
        if beziers == 'ALL':
            
            self.lefts  = self.read_attr(data, 'handle_left',  3)
            self.rights = self.read_attr(data, 'handle_right', 3)
            
        # ---- Some Bezier curves
            
        elif beziers != 'NONE':
            lefts  = np.zeros((len(data), 3), np.float)
            rights = np.zeros((len(data), 3), np.float)
            for i, item in enumerate(data):
                if hasattr(item, 'handle_left'):
                    lefts[i]  = item.handle_left
                    rights[i] = item.handle_right
                    
            self.lefts = lefts
            self.rights = rights
                    
    # ---------------------------------------------------------------------------
    # Write to data
        
    def write(self, data, beziers=None):
        
        # ----- Vertices
        
        self.write_attr(data, 'co', self.verts,  3)
        
        if not hasattr(data[0], 'radius'):
            return
        
        # ----- Spline shape point
        
        self.write_attr(data, 'radius', self.radius, 1)
        self.write_attr(data, 'tilt',   self.tilts,  1)
        
        # ----- Only bezier points
        
        if beziers is None:
            beziers = KeyShapes.compute_beziers(data)
        
        
        if beziers == 'NONE':
            
            self.write_attr(data, 'handle_left',  self.lefts,  3)
            self.write_attr(data, 'handle_right', self.rights, 3)
            
        # ----- Some bezier points
            
        elif beziers != 'NONE':
            for i, item in enumerate(data):
                if hasattr(item, 'handle_left'):
                    item.handle_left  = self.lefts[i]
                    item.handle_right = self.rights[i]
                    
    # ---------------------------------------------------------------------------
    # Set to an object
    
    def set_to(self, obj):
        
        data = obj.data
        
        if type(data).__name__ == 'Mesh':
            
            data.vertices.foreach_set('co', np.reshape(self.verts, len(self)*3))
            data.update()
            
        elif type(data).__name__ in ['Curve', 'SurfaceCurve']:
            
            splines = data.splines
            
            index = 0
            for spline in splines:
                
                if spline.type == 'BEZIER':

                    n = len(spline.bezier_points)
                    
                    spline.bezier_points.foreach_set('co', np.reshape(self.verts[index:index+n], n*3))

                    spline.bezier_points.foreach_set('tilt',   np.array(self.tilts[index:index+n]))
                    spline.bezier_points.foreach_set('radius', np.array(self.radius[index:index+n]))
                    
                    spline.bezier_points.foreach_set('handle_left',  np.reshape(self.lefts[index:index+n], n*3))
                    spline.bezier_points.foreach_set('handle_right', np.reshape(self.rights[index:index+n], n*3))
                
                else:
                    
                    n = len(spline.points)
                    
                    v4 = np.empty(n*4, np.float)
                    spline.points.foreach_get('co', v4)
                    v4 = v4.reshape(n, 4)
                    v4[:, :3] = self.verts[index:index+n]
                    
                    spline.points.foreach_set('co', np.reshape(v4, n*4))
                    
                    spline.points.foreach_set('tilt',   np.array(self.tilts[index:index+n]))
                    spline.points.foreach_set('radius', np.array(self.radius[index:index+n]))

                index += n
                
            data.update_tag()
            
        else:
            raise WError(f"Impossible to set the Key Shape to object '{obj.name}'. Data type '{type(data).__name__}' is not supported.",
                         Class = "KeyShape",
                         Method = "set_to",
                         length = len(self))
                    
        obj.update_tag()
        
    
# -----------------------------------------------------------------------------------------------------------------------------
# Shapes : an array of arrays of points

class KeyShapes():
    
    def __init__(self, points, beziers=None):
        
        if len(np.shape(points)) == 2:
            points = points.reshape(1, np.shape(points)[0], np.shape(points)[1])

        acceptable = [MESH_SIZE, CURVE_SIZE, BEZIER_SIZE]
        
        if (len(np.shape(points)) < 3) or (points.shape[-1] not in acceptable):
            raise WError("Array shape incorrect to initialize KeyShapes class.",
                Class = "KeyShapes",
                points_shape = np.shape(points),
                required_shape = "(shape count, points per shape, 3 | 5 | 11)")

        self.points  = points
        self.beziers = beziers
        
    def clone(self):
        return type(self)(np.array(self.points), self.beziers)
        
    def __repr__(self):
        skc = self.points.shape[:-2]
        vc = self.points.shape[1]
        
        return f"<KeyShapes '{self.shape_type}' of {skc} shapes of {vc} points>"
    
    @property
    def shape_type(self):
        if self.points.shape[-1] == MESH_SIZE:
            return "Mesh"
        
        elif self.points.shape[-1] == CURVE_SIZE:
            return "Curve without Beziers"
        
        else:
            if self.beziers == 'ALL':
                return "Curve with only Beziers"
            elif self.beziers == 'NONE':
                return "Curve without Beziers"
            else:
                return "Curve with some Beziers"
            
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Dimensions

    # ---------------------------------------------------------------------------
    # End size : shape point attributes aggregated in one vector
    
    @property
    def end_size(self):
        return self.points.shape[-1]
    
    # ---------------------------------------------------------------------------
    # How many points per key shape
        
    @property
    def points_per_shape(self):
        return self.points.shape[-2]

    # ---------------------------------------------------------------------------
    # Total points
    
    @property
    def total_points(self):
        return np.product(self.points.shape[:-1])   

    # ---------------------------------------------------------------------------
    # How many shapes
    
    @property
    def shapes_count(self):
        return np.product(self.points.shape[:-2])
    
    @property
    def main_shape(self):
        return self.points.shape[:-2]
    
    def reshape(self, shape):
        self.points = self.points.reshape(get_full_shape(shape, (self.points.shape[-2], self.points.shape[-1])))
            
    # ---------------------------------------------------------------------------
    # Initialize a structure by reading the vertices of an object
        
    @classmethod
    def FromObject(cls, obj, key_names=None):
        
        otype = type(obj.data).__name__
        if not otype in ['Mesh', 'Curve', 'SurfaceCurve']:
            raise WError(f"Impossible to create ShapeKeys from object of type {otype}",
                    Class = "KeyShapes",
                    Methode = "FromObject",
                    object = obj.name,
                    object_type = otype,
                    supported_types = ['Mesh', 'Curve', 'SurfaceCurve'])
            
        if otype == 'Mesh':
            
            vertices = obj.data.vertices
            points = np.empty(len(vertices)*3)
            vertices.foreach_get('co', points)
            points = points.reshape(len(vertices), 3)
            beziers = 'NONE'

        else:
            bcount = 0
            ccount = 0
            size   = 0
            for spline in obj.data.splines:
                if spline.type == 'BEZIER':
                    bcount += 1
                    size += len(spline.bezier_points)
                else:
                    ccount += 1
                    size += len(spline.points)
                    
            if bcount == 0:
                beziers = 'NONE'
                end_size = CURVE_SIZE
            elif ccount == 0:
                beziers = 'ALL'
                end_size = BEZIER_SIZE
            else:
                beziers = 'SOME'
                end_size = BEZIER_SIZE
            
            points = np.zeros((1, size, end_size))
            index = 0
            for spline in obj.data.splines:
                
                if spline.type == 'BEZIER':
                    pts = spline.bezier_points
                else:
                    pts = spline.points 
                
                slc = slice(index, index+len(pts))
                
                a = np.empty(len(pts), np.float)

                pts.foreach_get('tilt', a)
                points[0, slc, TILTS_SLICE] = np.expand_dims(a, axis=-1)
                
                pts.foreach_get('radius', a)
                points[0, slc, RADIUS_SLICE] = np.expand_dims(a, axis=-1)
                
                a = np.empty(len(pts)*3, np.float)
                
                pts.foreach_get('co', a)
                points[0, slc, VERTS_SLICE] = a.reshape(len(pts), 3)
                
                if spline.type == 'BEZIER':
                    
                    pts.foreach_get('handle_left', a)
                    points[0, slc, LEFTS_SLICE] = a.reshape(len(pts), 3)
                
                    pts.foreach_get('handle_right', a)
                    points[0, slc, RIGHTS_SLICE] = a.reshape(len(pts), 3)
                    
                index += len(pts)
                
        return KeyShapes(points, beziers)

    # ---------------------------------------------------------------------------
    # Initialize a structure by reading the shape keys of an object
        
    @classmethod
    def FromShapeKeys(cls, obj, key_names=None):
        
        # ----- shape keys must exist
        
        if obj.data is None: return None
        if obj.data.shape_keys is None: return None
        
        blocks = obj.data.shape_keys.key_blocks
        
        if blocks is None or len(blocks) == 0: return None
        
        # ----- Which keys to load
        
        if key_names is None: key_names = [block.name for block in blocks]
        if len(key_names) == 0: return None
        
        # ----- Ok, we have something to load. What is the end size
        
        b0 = blocks[0].data
        
        if type(obj.data).__name__ == 'Mesh':
            beziers  = 'NONE'
            end_size = 3
        else:
            beziers  = KeyShapes.compute_beziers(b0)
            end_size = 5 if beziers == 'NONE' else 11

        # ----- We can initialize with the array of points
            
        shapes = cls(np.zeros((len(key_names), len(b0), end_size), np.float), beziers)
            
        # ----- Let's read the key shapes

        for i, name in enumerate(key_names):
            shapes.get_key_shape(i).read(blocks[name].data, shapes.beziers)
            
        shapes.key_names = list(key_names)
            
        return shapes
    
    # ---------------------------------------------------------------------------
    # Write the shape keys back to an object
    # Note the the shape keys must exist before
    
    def to_shape_keys(self, obj):
        if not hasattr(self, 'key_names'):
            raise WError("Impossible to write key shapes which have not be initialized with Read method.\n" +
                    "key_names must be initialized.",
                    Class = "KeyShapes",
                    Method = "to_shape_keys")
            
        blocks = obj.data.shape_keys.key_blocks
        for i, name in enumerate(self.key_names):
            self.get_key_shape(i).write(blocks[name].data, self.beziers)
            
    # ---------------------------------------------------------------------------
    # For curves, there can exist a mix of Bezier and non bezier curves
        
    @staticmethod
    def compute_beziers(data):
        
        bcount = 0
        ccount = 0
        
        for sk in data:
            
            s = type(sk).__name__
            
            if s == 'ShapeKeyBezierPoint':
                bcount += 1
                if ccount != 0:
                    return 'SOME'
                
            elif s == 'ShapeKeyCurvePoint':
                ccount += 1
                if bcount != 0:
                    return 'SOME'
                
            elif s == 'ShapeKeyPoint':
                return 'NONE'
            
            else:
                raise WError(f"Unsupported shape key point: '{s}'",
                    Class = "Shapes",
                    Method = "compute_beziers")
                
        if bcount == 0:
            return 'NONE'
        elif ccount == 0:
            return 'ALL'
        else:
            return 'SOME'
        
    # ---------------------------------------------------------------------------
    # Access the shapes by the major index
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        return self.points[index]
    
    def __setitem__(self, index, value):
        self.points[index] = value
        
    # ---------------------------------------------------------------------------
    # Create a key shape pointing to the array of points
        
    def get_key_shape(self, *args):
        if len(self.main_shape) != len(args):
            raise WError(f"key shape index error. The shape of the shape keys is {self.main_shape}.\n" +
                         f"{len(self.main_shape)} int(s) must be provided",
                        Class = "KeyShapes",
                        Method = "get_key_shape",
                        args = args)
            
        return KeyShape(self.points, np.ravel_multi_index(args, self.main_shape))
    
    # ---------------------------------------------------------------------------
    # Check that the possible attributes are implemented
    
    def check_attr(self, attr_name, **kwargs):
        
        if attr_name in ['tilts', 'radius']:
            if self.end_size > MESH_SIZE:
                return True
            
        if attr_name in ['lefts', 'rights']:
            if self.end_size == BEZIER_SIZE:
                return True
        
        raise WError(f"Attribute '{attr_name}' unsupported for key shape.",
                     Class = "KeyShapes",
                     endf_size = self.end_size,
                     **kwargs)
    
    # ---------------------------------------------------------------------------
    # The vertices
    
    @property
    def verts(self):
        return self.points[..., VERTS_SLICE]
        
    @verts.setter
    def verts(self, value):
        self.points[..., VERTS_SLICE] = value
        
    # ---------------------------------------------------------------------------
    # Tilts and radius
    
    @property
    def tilts(self):
        self.check_attr('tilts')
        return np.squeeze(self.points[..., TILTS_SLICE])
        
    @tilts.setter
    def tilts(self, value):
        self.check_attr('tilts')
        self.points[..., TILTS_SLICE] = np.expand_dims(value, axis=-1)
    
    @property
    def radius(self):
        self.check_attr('radius')
        return np.squeeze(self.points[..., RADIUS_SLICE])
        
    @radius.setter
    def radius(self, value):
        self.check_attr('radius')
        self.points[..., RADIUS_SLICE] = np.expand_dims(value, axis=-1)
        
    # ---------------------------------------------------------------------------
    # Left and right handles
    
    @property
    def lefts(self):
        self.check_attr('lefts')
        return self.points[..., LEFTS_SLICE]
        
    @lefts.setter
    def lefts(self, value):
        self.check_attr('lefts')
        self.points[..., LEFTS_SLICE] = value
        
    @property
    def rights(self):
        self.check_attr('rights')
        return self.points[..., RIGHTS_SLICE]
        
    @rights.setter
    def rights(self, value):
        self.check_attr('rights')
        self.points[..., RIGHTS_SLICE] = value
        
    # ---------------------------------------------------------------------------
    # Interpolation
        
    def interpolation(self, t):
        
        sh = np.shape(t)
        if sh == ():
            return self.interpolation([t])

        target_sh = get_full_shape(sh, (self.points.shape[1], self.points.shape[2]))
        
        count = np.size(t)
        if len(self) == 1:
            return self.points[np.zeros(count, int)].reshape(target_sh)
        
        ts   = np.clip(np.reshape(t, count), 0, len(self)-1)
        inds = np.clip(np.floor(ts).astype(int), 0, len(self)-2)
        p    = np.reshape(ts-inds, (count, 1, 1))
        
        return KeyShapes( (self.points[inds]*(1-p) + self.points[inds+1]*p).reshape(target_sh) )
    
    # ---------------------------------------------------------------------------
    # Verts 4 for transformation
    
    @property
    def verts4(self):
        
        n = self.total_points
        
        if self.end_size == BEZIER_SIZE:
            a = np.ones((n, 3, 4), np.float)
            a[..., 0, 1:] = self.lefts
            a[..., 1, 1:] = self.verts
            a[..., 2, 1:] = self.rights
            return a.reshape(n*3, 4)
            
        else:
            a = np.ones((n, 4), np.float)
            a[..., 1:] = self.verts
            
        return a
    
    @verts4.setter
    def verts4(self, value):
        
        n = self.total_points
        
        if self.end_size == BEZIER_SIZE:
            
            a = np.reshape(value, (n, 3, 4))
            self.lefts  = a[..., 0, 1:]
            self.verts  = a[..., 1, 1:]
            self.rights = a[..., 2, 1:]
            
        else:
            self.verts  = value[..., 1:]
        
