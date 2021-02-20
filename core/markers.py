#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

import bpy

def markers(text, clear=True, start=0, end=True):

    scene = bpy.context.scene

    if clear:
        scene.timeline_markers.clear()

    frame_max = 0
    lines = text.split('\n')
    for line in lines:
        try:
            frame, name = line.split(',')
            frame = int(frame)
            scene.timeline_markers.new(name.strip(), frame=frame)
            frame_max = max(frame_max, frame)
        except:
            pass

    scene.frame_start = 0
    scene.frame_end   = frame_max
