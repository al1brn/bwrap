#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:38:55 2021

@author: alain
"""

import numpy as np

# ----------------------------------------------------------------------------------------------------
# RGB to HSV conversion

def rgb2hsv(rgb):

    delta_min = 1e-5
    
    # ------------------------------------------------------------------------------------------
    # Alpha channel exists
    
    if np.shape(rgb)[-1] == 4:
        hsva = np.array(rgb)
        hsva[..., :3] = rgb2hsv(hsva[..., :3])
        return hsva
    
    # ---------------------------------------------------------------------------
    # A single value
    
    if np.shape(rgb) == (3,):
        
        cmin = np.min(rgb)
        cmax = np.max(rgb)
        
        delta = cmax - cmin
        
        if delta < delta_min:
            return np.array((0., 0., cmax))
        
        if cmax <= 0.:
            return np.array((0., 0., 0.))
        
        s = delta/cmax
        v = cmax
        
        if cmax == rgb[0]:
            h = (rgb[1] - rgb[2])/delta
        elif cmax == rgb[1]:
            h = (rgb[2] - rgb[0])/delta + 2
        else:
            h = (rgb[0] - rgb[1])/delta + 4
            
        h *= 60
        if h < 0:
            h += 360
        h /= 360
        
        return np.array((h, s, v))   
    
    # ---------------------------------------------------------------------------
    # An array of rgbs

    rgb = np.array(rgb)
    hsv = np.zeros(np.shape(rgb), float)
    
    cmin  = np.min(rgb, axis=-1)
    cmax  = np.max(rgb, axis=-1)
    
    delta = cmax - cmin
    
    # ----- small delta
    
    flt = delta < delta_min 
    hsv[flt, 2] = cmax[flt]
    
    # ----- cmax > 0
    # (for cmax < 0, initialization to 0. is ok)
    
    flt = np.logical_and(np.logical_not(flt), cmax > 0.)
    
    hsv[flt, 1] = delta[flt]/cmax[flt]  # s
    hsv[flt, 2] = cmax[flt]             # v
    
    # h
    
    wr = np.logical_and(flt, cmax == rgb[..., 0])
    wg = np.logical_and(flt, cmax == rgb[..., 1])
    wb = np.logical_and(flt, cmax == rgb[..., 2])
    
    hsv[wr, 0] = (rgb[wr , 1] - rgb[wr, 2]) / delta[wr]
    hsv[wg, 0] = (rgb[wg , 2] - rgb[wg, 0]) / delta[wg] + 2
    hsv[wb, 0] = (rgb[wb , 0] - rgb[wb, 1]) / delta[wb] + 4
    
    hsv[np.logical_and(flt, hsv[..., 0] < 0), 0] += 6
    hsv[flt, 0] /= 6
    
    return hsv

# ----------------------------------------------------------------------------------------------------
# HSV to RGB Conversion

def hsv2rgb(hsv):
    
    # ------------------------------------------------------------------------------------------
    # Alpha channel exists
    
    if np.shape(hsv)[-1] == 4:
        rgba = np.array(hsv)
        rgba[..., :3] = hsv2rgb(rgba[..., :3])
        return rgba
    
    # ------------------------------------------------------------------------------------------
    # Non alpha channel
    
    
    if np.shape(hsv) == (3,):
        
        hsv = np.array(hsv, float)
        
        hsv[0] *= 360
        
        c = hsv[2] * hsv[1]
        x = c * (1 - abs(((hsv[0]/60.0) % 2) - 1))
        m = hsv[2] - c
        
        if 0.0 <= hsv[0] < 60:
            rgb = (c, x, 0)
        elif 0.0 <= hsv[0] < 120:
            rgb = (x, c, 0)
        elif 0.0 <= hsv[0] < 180:
            rgb = (0, c, x)
        elif 0.0 <= hsv[0] < 240:
            rgb = (0, x, c)
        elif 0.0 <= hsv[0] < 300:
            rgb = (x, 0, c)
        elif 0.0 <= hsv[0] < 360:
            rgb = (c, 0, x)
            
        return np.array(rgb) + m
            
        #return list(map(lambda n: (n + m), rgb))
        
    hsv = np.array(hsv, float)
    hsv[..., 0] *= 360
    rgb = np.zeros(np.shape(hsv), float)
    
    c = hsv[..., 2] * hsv[..., 1]
    x = c * (1 - abs(((hsv[..., 0]/60.0) % 2) - 1))
    m = hsv[..., 2] - c
    
    flt = np.logical_and(  0 <= hsv[..., 0], hsv[..., 0] <  60)
    rgb[flt] = np.stack((c[flt], x[flt], np.zeros(np.count_nonzero(flt), float))).transpose()
    
    flt = np.logical_and( 60 <= hsv[..., 0], hsv[..., 0] < 120)
    rgb[flt] = np.stack((x[flt], c[flt], np.zeros(np.count_nonzero(flt), float))).transpose()

    flt = np.logical_and(120 <= hsv[..., 0], hsv[..., 0] < 180)
    rgb[flt] = np.stack((np.zeros(np.count_nonzero(flt), float), c[flt], x[flt])).transpose()
    
    flt = np.logical_and(180 <= hsv[..., 0], hsv[..., 0] < 240)
    rgb[flt] = np.stack((np.zeros(np.count_nonzero(flt), float), x[flt], c[flt])).transpose()
    
    flt = np.logical_and(240 <= hsv[..., 0], hsv[..., 0] < 300)
    rgb[flt] = np.stack((x[flt], np.zeros(np.count_nonzero(flt), float), c[flt])).transpose()
    
    flt = np.logical_and(300 <= hsv[..., 0], hsv[..., 0] < 360)
    rgb[flt] = np.stack((c[flt], np.zeros(np.count_nonzero(flt), float), x[flt])).transpose()
    
    return rgb + np.expand_dims(m, axis=-1)

# ----------------------------------------------------------------------------------------------------
# Encode / decode a float on the four components of colors

def float2rgba(v):
    
    shape = (1,) if np.shape(v) == () else np.shape(v)
    
    vs = np.empty(shape, float)
    vs[:] = v*4

    rgba = np.empty(shape + (4,), float)
    
    rgba[..., 0] = np.clip(vs,     0, 1)
    rgba[..., 1] = np.clip(vs - 1, 0, 1) 
    rgba[..., 2] = np.clip(vs - 2, 0, 1)
    rgba[..., 3] = np.clip(vs - 3, 0, 1)
    
    return rgba

def rgba2float(rgba):
    
    return np.sum(np.array(rgba)*.25, axis=-1)


def float2rgb(v):
    
    shape = (1,) if np.shape(v) == () else np.shape(v)
    
    vs = np.empty(shape, float)
    vs[:] = v*3

    rgb = np.empty(shape + (3,), float)
    
    rgb[..., 0] = np.clip(vs,     0, 1)
    rgb[..., 1] = np.clip(vs - 1, 0, 1) 
    rgb[..., 2] = np.clip(vs - 2, 0, 1)
    
    return rgb

def rgb2float(rgb):
    
    return np.sum(np.array(rgb)/3, axis=-1)

