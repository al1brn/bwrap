#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:06:17 2021

@author: alain
"""

res0 = """

Alpha edt = 2

==============================
Perihelion without Doppler

rotation:  -3.1415323466082636 rad
        :  -179.9965446644832 °
        :  -647987.5607921395 sec
period  : 43.969 days

[-45984181010.21044, -2773167.15819294]
3798966.5

------------------------------
Perihelion with Doppler

rotation:  3.1415124657977556 rad
        :  179.99540557794776 °
        :  647983.460080612 sec
period  : 43.967 days

[-45984181022.74765, 3687369.952785276]
3798857.0

==============================
Aphelion without Doppler

rotation:  0.00012135589412759604 rad
        :  0.006953180552547704 °
        :  25.031449989171737 sec
period  : 87.939 days

[69817090654.52615, 8472715.503360726]
7597942.75

------------------------------
Aphelion with Doppler

rotation:  0.00020053167402996788 rad
        :  0.011489618580610336 °
        :  41.36262689019721 sec
period  : 87.940 days

[69817089707.9457, 14000538.062702447]
7598085.0
"""

res1 = """

Alpha edt = 3

==============================
Perihelion without Doppler

rotation:  -3.1415672321596144 rad
        :  -179.99854345934156 °
        :  -647994.7564536296 sec
period  : 43.968 days

[-45984151142.16872, -1168982.887840785]
3798910.25

------------------------------
Perihelion with Doppler

rotation:  3.1414787022444246 rad
        :  179.99347106884053 °
        :  647976.4958478259 sec
period  : 43.967 days

[-45984150934.27713, 5239955.887265421]
3798801.625

==============================
Aphelion without Doppler

rotation:  5.054062252750417e-05 rad
        :  0.0028957643647898003 °
        :  10.42475171324328 sec
period  : 87.937 days

[69816779625.62201, 3528583.5081489235]
7597815.5

------------------------------
Aphelion with Doppler

rotation:  0.00015010254981653095 rad
        :  0.008600242598639413 °
        :  30.96087335510189 sec
period  : 87.939 days

[69816778838.53474, 10479676.602345966]
7597994.375
"""



import numpy as np
import matplotlib.pyplot as plt

fname0 = 'mer_alpha_edt_02.npy'
dec0   = 22

fname1 = 'mer_alpha_edt_03.npy'
dec1   = 22

def visu(fname, dec=0):

    a = np.load(fname)
    n = a.shape[1] - dec
    
    std = np.resize(np.array(a[0]), (n, 2))
    dop = np.resize(np.array(a[1]), (n, 2))
    
    # ---------------------------------------------------------------------------
    # Angular location (in degrees)

    ags = (np.degrees(np.arctan2(std[:, 1], std[:, 0])) + 360.) % 360.
    agd = (np.degrees(np.arctan2(dop[:, 1], dop[:, 0])) + 360.) % 360.
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(n), ags, 'black')
    ax.plot(np.arange(n), agd, 'red')
    
    plt.show()

    # ---------------------------------------------------------------------------
    # Angular difference (in sec °)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(n), (agd - ags)*3600, 'black')
    
    plt.show()
    
    # ---------------------------------------------------------------------------
    # Orbit
    
    fig, ax = plt.subplots()
    ax.plot(std[:, 0], std[:, 1], 'black')
    
    mul = 2000
    r0 = np.linalg.norm(std, axis=-1)
    r1= np.linalg.norm(dop, axis=-1)
    q = (1 + (r1 - r0)/r0*mul)
    print(q)
    
    p = np.array(dop) * np.expand_dims(q, axis=-1)
    ax.plot(p[:, 0], p[:, 1], 'red')
    
    plt.show()


#visu(fname0, dec=dec0)
#visu(fname1, dec=dec1)




    


