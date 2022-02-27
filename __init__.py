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

from .wrappers.wrap_function import wrap, wcollection
from .objects.crowd import Crowd, Crowds
from .text.chars import Chars
from .objects.duplicator import Duplicator

from .core.animation import Interval, Engine, Animator



