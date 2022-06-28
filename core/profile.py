#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:53:43 2022

@author: alain
"""

import numpy as np


# ----------------------------------------------------------------------------------------------------
# Iterator of profile, returns:
# - curve type
# - offset (of points or vertices)
# - length (of points)
# - nverts (if on_verts)

class ProfileIter():

    def __init__(self, profile, on_verts=True):
        self.profile = profile
        self.on_verts = on_verts

    def __iter__(self):
        self.index = 0
        self.offset = 0
        return self

    def __next__(self):
        if self.index >= len(self.profile):
            raise StopIteration

        ofs = self.offset
        (ctype, length) = self.profile[self.index]

        self.index += 1

        if self.on_verts:
            nverts = length*3 if ctype == 0 else length
            self.offset += nverts
            return ctype, ofs, length, nverts
        else:
            self.offset += length
            return ctype, ofs, length

# ----------------------------------------------------------------------------------------------------
# Manages an array of couples:
# - curve type
# - curve length
#
# type 0 is for Bezier
# For Bezier curves, the number of managed vertices is multiplied by 3 to take the handles
# into account


class Profile():

    CURVE_TYPES = ['BEZIER', 'POLY', 'BSPLINE', 'CARDINAL', 'NURBS']

    def __init__(self, ctype='BEZIER', length=2, count=1):
        self.curves = np.zeros((count, 2), int)
        self.curves[:, 0] = self.ctype_code(ctype)
        self.curves[:, 1] = length

        self.verts = None

    @classmethod
    def ctype_name(cls, code):
        if type(code) is int:
            return cls.CURVE_TYPES[code]
        else:
            return code.upper()

    @classmethod
    def ctype_code(cls, name):
        if type(name) is int:
            return name
        else:
            return cls.CURVE_TYPES.index(name)

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, index):
        return self.curves[index]

    def ctype_count(self, ctype):
        return len(np.where(self.curves[:, 0] == self.ctype_code(ctype))[0])

    def __repr__(self):
        s = f"<Curve profile of {len(self)} curve(s) with {self.verts_count} vertices:\n"
        for ctype in self.CURVE_TYPES:
            s += f"   {ctype:8s} : {self.ctype_count(ctype):3d}\n"
        return s + ">"

    def points_iter(self):
        return iter(ProfileIter(self, on_verts=False))

    def verts_iter(self):
        return iter(ProfileIter(self, on_verts=True))
    
    def ctype(self, index, as_str=False):
        if as_str:
            return self.ctype_name(self.curves[index, 0])
        else:
            return self.curves[index, 0]

    def length(self, index):
        return self.curves[index, 1]

    # ===========================================================================
    # Beziers vs other type

    @property
    def only_bezier(self):
        return np.max(self.curves[:, 0]) == 0

    @property
    def only_nurbs(self):
        return np.min(self.curves[:, 0]) > 0
    
    @property
    def has_bezier(self):
        return np.min(self.curves[:, 0]) == 0

    @property
    def has_nurbs(self):
        return np.max(self.curves[:, 0]) > 0

    @property
    def is_mix(self):
        return self.has_bezier and self.has_nurbs

    # ===========================================================================
    # Manage the curves

    # ---------------------------------------------------------------------------
    # Append somes curves

    def append(self, ctype, length, count=1):

        if count == 1:
            self.curves = np.append(
                self.curves, [[self.ctype_code(ctype), length]], axis=0)
            return len(self.curves) - 1
        else:
            a = np.empty((count, 2), int)
            a[:, 0] = self.ctype_code(ctype)
            a[:, 1] = length
            self.curves = np.append(self.curves, a, axis=0)
            return np.arange(count) + len(self.curves) - count

    # ---------------------------------------------------------------------------
    # Join with another set of curves

    def join(self, other):
        self.curves = np.append(other.curves, axis=0)

    # ---------------------------------------------------------------------------
    # Access to vertices

    @property
    def points_count(self):
        return np.sum(self.curves[:, 1])

    @property
    def verts_count(self):
        return np.sum(self.curves[:, 1]) + np.sum(self.curves[self.curves[:, 0] == 0][:, 1])*2

    @property
    def verts_sizes(self):
        sizes = np.array(self.curves[:, 1])
        sizes[np.where(self.curves[:, 0] == 0)[0]] *= 3
        return sizes

    @property
    def verts_slices(self):

        slices = np.zeros((len(self)+1, 2), int)

        sizes = self.verts_sizes
        slices[1:,  0] = np.cumsum(sizes)
        slices[:-1, 1] = sizes
        return slices[:-1]

    @property
    def verts_offsets(self):
        return self.slices[:, 0]

    # ---------------------------------------------------------------------------
    # Set / get the vertices

    def set_verts(self, index, verts, slices=None):
        if self.verts is None:
            raise RuntimeError("Verts is None")

        if slices is None:
            slices = self.verts_slices

        v0 = slices[index, 0]
        n = slices[index, 1]
        v1 = v0 + n

        self.verts[v0:v1] = np.reshape(verts, (n, 3))

    def get_verts(self, index, verts, slices=None):
        if self.verts is None:
            raise RuntimeError("Verts is None")

        if slices is None:
            slices = self.verts_slices

        v0 = slices[index, 0]
        n = slices[index, 1]
        v1 = v0 + n

        if self.curves[index, 0] == 0:
            shape = (3, self.curves[index, 1], 3)
        else:
            shape = (n, 3)

        return np.reshape(self.verts[v0:v1], shape)

    # ===========================================================================
    # From splines

    @classmethod
    def FromSplines(cls, splines):
        prof = cls(count=len(splines))

        for i, spline in enumerate(splines):
            ctype = cls.ctype_code(spline.type)
            n = len(spline.bezier_points) if ctype == 0 else len(spline.points)

            prof.curves[i] = [ctype, n]

        return prof

    # ===========================================================================
    # Read the vertices from the splines

    def get_splines_verts(self, splines, ndim=3):

        ndim = np.clip(ndim, 3, 4)

        verts = np.zeros((self.verts_count, ndim), float)

        for (ctype, offset, length, nverts), spline in zip(self.verts_iter(), splines):

            if ctype == 0:
                a = np.empty((3, length * 3), float)

                spline.bezier_points.foreach_get('co',           a[0])
                spline.bezier_points.foreach_get('handle_left',  a[1])
                spline.bezier_points.foreach_get('handle_right', a[2])

                verts[offset:offset+nverts, :3] = np.reshape(a, (nverts, 3))

            else:
                a = np.empty(length*4, float)
                spline.points.foreach_get('co', a)

                verts[offset:offset +
                      nverts] = np.reshape(a, (length, 4))[:, :ndim]

        return verts

    # ===========================================================================
    # Read the vertices from the splines

    def set_splines_verts(self, splines, verts):

        ndim = np.shape(verts)[-1]

        for (ctype, offset, length, nverts), spline in zip(self.verts_iter(), splines):

            if ctype == 0:
                a = np.reshape(verts[offset:offset+nverts, :3], (3, length*3))

                spline.bezier_points.foreach_set('co',           a[0])
                spline.bezier_points.foreach_set('handle_left',  a[1])
                spline.bezier_points.foreach_set('handle_right', a[2])

            else:
                a = np.ones((length, 4), float)
                a[:, :ndim] = verts[offset:offset+nverts]
                spline.points.foreach_set('co', np.reshape(a, (length*4)))

        splines.data.update_tag()

    # ===========================================================================
    # Read attributes

    def get_splines_attrs(self, splines, attrs=['radius', 'tilt', 'weight', 'weight_softbody']):

        n = len(attrs)
        if n == 0:
            return None

        values = {}
        for attr in attrs:
            values[attr] = np.zeros(self.points_count, float)

        for (ctype, offset, length), spline in zip(self.points_iter(), splines):

            for attr, vals in values.items():

                pts = spline.bezier_points if ctype == 0 else spline.points

                if (ctype != 0) or (attr != 'weight'):
                    pts.foreach_get(attr, vals[offset:offset+length])

        return values

    # ===========================================================================
    # Write attributes

    def set_splines_attrs(self, splines, values):

        offset = 0

        for (ctype, offset, length), spline in zip(self.points_iter(), splines):

            for attr, vals in values.items():

                pts = spline.bezier_points if ctype == 0 else spline.points

                if (ctype != 0) or (attr != 'weight'):
                    pts.foreach_set(attr, vals[offset:offset+length])
