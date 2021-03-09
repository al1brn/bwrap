#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

import bpy

def markers(text, clear=True, start=0, end=True):

    print('-'*10)
    print("Markers...")
    fails = 0
    total = 0

    scene = bpy.context.scene

    if clear:
        scene.timeline_markers.clear()

    frame_min = 1000000
    frame_max = 0

    # ----- Split the lines
    lines = text.split('\n')

    mks = []

    # ----- First pass to compute min an max
    for line in lines:

        total += 1

        # Split with possible separators
        ko = True
        for sep in [',', '\t', ';']:
            # Empty line !
            if (line == "") or (line.strip() == sep):
                total -= 1
                ko = False
                break

            # Not empty line
            else:
                fr_na = line.split(sep)
                if len(fr_na) == 2:
                    try:
                        frame = int(fr_na[0])
                        mks.append((frame, fr_na[1].strip()))
                        frame_min = min(frame_min, frame)
                        frame_max = max(frame_max, frame)

                        ko = False
                        break
                    except:
                        pass

        if ko:
            fails += 1
            print(f"Markers: unable to handle the line: '{line}'")

    # ----- Loop on the markers to set
    for fr_na in mks:
        scene.timeline_markers.new(fr_na[1], frame=start + fr_na[0] - frame_min)

    # ----- Start and end frames
    scene.frame_start = start
    scene.frame_end   = start + frame_max - frame_min

    # ----- Synthesis
    print(f"Markers from {start} to {start + frame_max - frame_min}: {total} line(s), {fails} fail(s)")
