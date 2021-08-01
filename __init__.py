#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

bl_info = {
    "name":     "Blender wrap",
    "author":   "Alain Bernard",
    "version":  (1, 0),
    "blender":  (2, 80, 0),
    "location": "View3D > Sidebar > Wrap",
    "description": "Wrapanime commands and custom parameters",
    "warning":   "",
    "wiki_url":  "",
    "category":  "3D View"}

import numpy as np
import bpy

from .wrappers.wrap_function import wrap
from .objects.wcollection import WCollection

from .core.animation import Interval, Engine, Animator


"""
from .core.wrappers import wrap

from .core.utils import dicho
from .core.utils import npdicho

from .core.interpolation import Rect, Easing, BCurve

from .core.blender import create_collection
from .core.blender import get_collection
from .core.blender import get_object_collections
from .core.blender import put_object_in_collection

from .core.blender import wrap_collection
from .core.blender import control_collection

from .core.blender import get_frame

from .core.blender import create_object
from .core.blender import get_object
from .core.blender import get_create_object
from .core.blender import copy_modifiers
from .core.blender import delete_object
from .core.blender import smooth_object
from .core.blender import hide_object
from .core.blender import show_object
from .core.blender import set_material

from .core.transformations import Transformations, ObjectTransformations
from .core.duplicator import Duplicator
from .core.crowd import Crowd


from .core.bezier import PointsInterpolation
from .core.bezier import from_points
from .core.bezier import from_function

from .core.curspaces import CurvedSpace

from .core.meshes import arrow, curved_arrow

from .core.meshbuilder import MeshBuilder
from .core.markers import markers

from .core.d4 import enable_4D, disable_4D

from .core.astrodb import Planets

from .core.commons import base_error_title
error_title = base_error_title % "main.%s"

"""
