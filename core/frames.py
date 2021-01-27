#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 08:05:16 2021

@author: alain
"""

import bpy

from .commons import base_error_title

error_title = base_error_title % "frames.%s"


# -----------------------------------------------------------------------------------------------------------------------------
# Get a frame

def get_frame(frame_or_str, delta=0):
    
    if frame_or_str is None:
        return None
    
    if type(frame_or_str) is str:
        marker = bpy.context.scene.timeline_markers.get(frame_or_str)
        if marker is None:
            raise RuntimeError(
                error_title % "get_frame" +
                F"Marker '{frame_or_str}' doesn't exist."
                )
        return marker.frame + delta
    
    return frame_or_str + delta
