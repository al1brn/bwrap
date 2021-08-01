#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:02:08 2021

@author: alain
"""

import numpy as np
try:
    from .geometry import m_to_euler, m_to_quat, e_to_matrix
except:
    from geometry import m_to_euler, m_to_quat, e_to_matrix

# =============================================================================================================================
# Get and set from Blender matrices
# Not sure set is usefull, but it exists
# Works fro matrix_worl and matrix_basis

# Caution:
# Two ways to read the transformation matrix of an object
# Wnen read from attribute, the casting to np.array gives a matrix which is transposed
# compared to reading through foreach_get.

# ----------------------------------------------------------------------------------------------------
# Location

def bmx_location(bmatrix):
    return np.array(bmatrix)[:3, 3]

def bmx_x(bmatrix):
    return np.array(bmatrix)[0, 3]
def bmx_y(bmatrix):
    return np.array(bmatrix)[1, 3]
def bmx_z(bmatrix):
    return np.array(bmatrix)[2, 3]

def bmx_location_set(bmatrix, loc):
    loc = np.resize(loc, 3)
    bmatrix[0][3] = loc[0]
    bmatrix[1][3] = loc[1]
    bmatrix[2][3] = loc[2]
    return bmatrix
    
def bmx_x_set(bmatrix, value):
    bmatrix[0][3] = value
    return bmatrix

def bmx_y_set(bmatrix, value):
    bmatrix[1][3] = value
    return bmatrix

def bmx_z_set(bmatrix, value):
    bmatrix[2][3] = value
    return bmatrix

# ----------------------------------------------------------------------------------------------------
# Scale read

def bmx_scale(bmatrix):
    return np.linalg.norm(np.array(bmatrix)[:3, :3], axis=0)

def bmx_sx(bmatrix):
    return np.linalg.norm(np.array(bmatrix)[:3, 0])
def bmx_sy(bmatrix):
    return np.linalg.norm(np.array(bmatrix)[:3, 1])
def bmx_sz(bmatrix):
    return np.linalg.norm(np.array(bmatrix)[:3, 2])

# ----------------------------------------------------------------------------------------------------
# Mat read
    
def bmx_mat_scale(bmatrix):
    tmat  = np.array(bmatrix)
    scale = np.linalg.norm(tmat[:3, :3], axis=0)
    mat   = tmat[:3, :3] / (np.expand_dims(scale, 0))
    return mat.transpose(), scale

def bmx_mat(bmatrix):
    return bmx_mat_scale(bmatrix)[0]

def bmx_euler(bmatrix, order):
    return m_to_euler(bmx_mat(bmatrix), order)
def bmx_rx(bmatrix, order):
    return bmx_euler(bmatrix, order)[0]
def bmx_ry(bmatrix, order):
    return bmx_euler(bmatrix, order)[1]
def bmx_rz(bmatrix, order):
    return bmx_euler(bmatrix, order)[2]

def bmx_quat(bmatrix):
    return m_to_quat(bmx_mat(bmatrix))

# ----------------------------------------------------------------------------------------------------
# Mat scale write

def bmx_mat_scale_set(bmatrix, mat, scale):
    
    m4 = np.identity(4, np.float)

    #m4[:3, :3] = (np.array(mat)) * (np.expand_dims(np.resize(np.array(scale, np.float), 3), 1).transpose())
    m4[:3, :3] = (np.array(mat)) * (np.expand_dims(np.resize(np.array(scale, np.float), 3), 1))
    m4[:3, 3]  = bmx_location(bmatrix)
    
    return m4

# ----------------------------------------------------------------------------------------------------
# Scale write

def bmx_scale_set(bmatrix, scale):
    return bmx_mat_scale_set(bmatrix, bmx_mat(bmatrix), scale)
    
def bmx_sx_set(bmatrix, value):
    scale = bmx_scale(bmatrix)
    scale[0] = value
    return bmx_scale_set(bmatrix, scale)

def bmx_sy_set(bmatrix, value):
    scale = bmx_scale(bmatrix)
    scale[1] = value
    return bmx_scale_set(bmatrix, scale)

def bmx_sz_set(bmatrix, value):
    scale = bmx_scale(bmatrix)
    scale[2] = value
    return bmx_scale_set(bmatrix, scale)

# ----------------------------------------------------------------------------------------------------
# Rotation write

def bmx_mat_set(bmatrix, mat):
    return bmx_mat_scale_set(bmatrix, mat, bmx_scale(bmatrix))

def bmx_euler_set(bmatrix, euler, order):
    m = e_to_matrix(euler, order)
    return bmx_mat_set(bmatrix, m)

def bmx_rx_set(bmatrix, angle, order):
    e = bmx_euler(bmatrix, order)
    e[0] = angle
    return bmx_euler_set(bmatrix, e, order)

def bmx_ry_set(bmatrix, angle, order):
    e = bmx_euler(bmatrix, order)
    e[1] = angle
    return bmx_euler_set(bmatrix, e, order)

def bmx_rz_set(bmatrix, angle, order):
    e = bmx_euler(bmatrix, order)
    e[2] = angle
    return bmx_euler_set(bmatrix, e, order)
