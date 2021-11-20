#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 08:55:24 2021

@author: alain
"""

# --------------------------------------------------------------------------------
# Given a circular orbit around a planet of mass M at radius r and velocity v,
# we have:
# - v    = rw   where w is the angular velocity
# - rw^2 = GM/r^2
#
# If we multiply values by factors:
# r'  = r.a
# v'  = v.b
# GM' = GM.c
#
# We have:
# w'       = v'/r'   = w.a/b 
# r'w'^2   = v'^2/r' = rw^2.b^2/a
# GM'/r'^2 =           GM/r^2.c/a^2
#
# Hence, the following egality ensures the orbit stays circular
# b^2/a = c/a^2
#
# b = sqrt(c/a)
# 
# The time factor d is given by;
# d = w/w' = b/a



import numpy as np
from number import Number, Vector
import matplotlib.pyplot as plt


Number.add_unit('m',   precision=16, fmt_func=lambda v: v.div(1e6),   fmt_unit='Mkm',     disp_prec=6)
Number.add_unit('m/s', precision=16, fmt_func=lambda v: v.div(1000),  fmt_unit='km/s',    disp_prec=3)
Number.add_unit('s',   precision= 6, fmt_func=lambda v: v.div(86400), fmt_unit='days',    disp_prec=3)
Number.add_unit('°',   precision=16, fmt_func=lambda a: Number(np.degrees(a.value)),      disp_prec=10)
Number.add_unit('sec', precision=16, fmt_func=lambda a: Number(np.degrees(a.value)/3600), disp_prec=10)


G     = Number(6.6743015e-11)
M_sun = Number(1.989e30)
GM    = G.mul(M_sun)
c     = Number(3e8, unit='m/s')

merc_aphelion = 69816900 * 1000
merc_speed    = 38.86 * 1000
merc_period   = 87.969 # days
        


# ---------------------------------------------------------------------------
# A utility buffer

class Buffer():
    def __init__(self, shape, length=1000):
        shape = (length,) + shape if hasattr(length, '__len__') else (length, shape)
        self.a = np.zeros(shape, float)
        self.index = 0
        
    def append(self, value):
        self.a[self.index, ...] = value
        self.index = (self.index + 1) % len(self.a)
        
    def array(self):
        a = np.empty(self.a.shape, float)
        a[:len(a)-self.index] = self.a[self.index:]
        a[len(a)-self.index:] = self.a[:self.index]
        return a
    
    def plot(self):
        a = self.array()
        
        fig, ax = plt.subplots()
        ax.plot(a[:, 0], a[:, 1])
        plt.show()
        
    def plot_comp(self, buf):
        a = self.array()
        b = buf.array()
        
        c = np.array(a)
        for i in range(len(a)):
            c[i] = a[i] + (b[i] - a[i])*1000
        
        fig, ax = plt.subplots()
        ax.plot(a[:, 0], a[:, 1], 'black')
        ax.plot(c[:, 0], c[:, 1], 'red')
        plt.show()
        

# ---------------------------------------------------------------------------
# The structure to store the information on a location

class Instant():
    def __init__(self, r=1, v=1, edt=10):
        
        self.p       = Vector((r, 0), unit='m')
        self.r       = Number(r, unit='m')
        self.v       = Vector((0, v), unit='m/s')
        
        self.edt     = edt
        if edt < 0:
            self.dt = (1 << (-edt))
        else:
            self.dt = 1/(1 << edt)
        self.GMdt    = GM.opposite().mul_pow2(-self.edt-1) # 1/2.GM*dt
        
        self.da2     = self.p.mul(self.GMdt.div(self.p.norm.pow(3)))
        self.da2.unit = 'm/s'
        self.t       = 0.
        
    def __repr__(self):
        s = ""
        s += f"p   : {self.p}\n"
        s += f"r   : {self.r}\n"
        s += f"v   : {self.v}\n"
        s += f"da2 : {self.da2}\n"
        s += f"GMdt: {self.GMdt}\n"
        return s
        
        
    def clone(self):
        
        c = Instant()
        
        c.p    = Vector(self.p)
        c.r    = Number(self.r)
        c.v    = Vector(self.v)
        
        c.edt  = self.edt
        c.dt   = self.dt
        c.GMdt = Number(self.GMdt)
        
        c.da2  = Vector(self.da2)
        c.t    = self.t
        
        return c
    
    def doppler_effect(self, p, da2):
        
        # Normal vector towards the location
        u = p.normalized()
        
        # Speed norm
        beta2 = self.v.squared_norm
        beta  = beta2.sqrt()
        
        # Normalized speed
        v = self.v.div(beta).set_e(20)
        
        # Dot product
        dot = u.dot(v).opposite()
        
        # Beta
        beta.div_eq(c)
        
        # alpha
        #alpha = Number(1).sub(beta2).sqrt()
        
        # Doppler factor
        factor = Number(1).sub(beta.mul(dot)) # *alpha
        
        
        return da2.mul(factor)
        
    
    def leap(self, doppler=False):
        
        # ----- Location variation
        
        dp = self.v.add(self.da2)
        dp.x.mul_pow2_eq(-self.edt, True)
        dp.y.mul_pow2_eq(-self.edt, True)
        dp.set_e(0)
        
        new_p = self.p.add(dp).set_e(0)
        
        # ----- New distance
        
        new_r = self.p.norm.set_e(0)
        
        # ----- New acceleration
        
        new_da2 = new_p.mul(self.GMdt.div(new_r.pow(3))).set_e(10)
        
        # ----- Doppler
        
        if doppler:
            new_da2 = self.doppler_effect(new_p, new_da2).set_e(10)
        
        # ----- New speed
        
        self.v.add_eq(self.da2.add(new_da2)).set_e(10)
        
        # ----- Update 
        
        self.r   = new_r
        self.p   = new_p
        self.da2 = new_da2
        
        self.t  += self.dt 
        
        #self.r.precision_to_unit()
        
    @property
    def angle(self):
        return Number(np.arctan2(self.p.y.value, self.p.x.value), unit='sec')


# ---------------------------------------------------------------------------
# Orbit simulation
    
class Orbit():
    
    def __init__(self, buffer_length=1000):
        self.aphelions       = []
        self.perihelions     = []
        self.dop_aphelions   = []
        self.dop_perihelions = []
        
        self.buffer      = Buffer(2, length=buffer_length)
        self.dop_buffer  = Buffer(2, length=buffer_length)
        
    def leaps(self, instant, duration=100):
        
        dt = instant.dt
        steps = max(10, round(duration /dt))
        
        per_cent  = steps // 10000
        in_buffer = steps // len(self.buffer.a)
        
        self.instant0 = instant.clone()
        dop_instant   = instant.clone()
        
        print(f"\nSimulation with {steps} steps\n")
        
        print("Standard instant")
        print(instant)
        print("Doppler instant")
        print(dop_instant)
                
        self.growing     = False
        self.dop_growing = False
        
        for step in range(steps):
            
            # ----- Normal
            
            r = Number(instant.r)
            instant.leap(doppler = False)
            
            if self.growing:
                if instant.r.less(r):
                    self.growing = False
                    self.aphelions.append(instant.clone())
            else:
                if instant.r.greater(r):
                    self.growing = True
                    self.perihelions.append(instant.clone())
                    
            # ----- Doppler
            
            r = Number(dop_instant.r)
            dop_instant.leap(doppler = True)
            
            if self.dop_growing:
                if dop_instant.r.less(r):
                    self.dop_growing = False
                    self.dop_aphelions.append(instant.clone())
            else:
                if dop_instant.r.greater(r):
                    self.dop_growing = True
                    self.dop_perihelions.append(instant.clone())
                    
            # ----- Tracking
                    
            # in buffer
            if step % in_buffer == 0:
                self.buffer.append(np.array(instant.p.value))
                self.dop_buffer.append(np.array(dop_instant.p.value))

            if step < 10 or (step % per_cent == 0):
                print(f"{step:5d} ({step/steps*100:.2f}%)> r: {instant.r}, v: {instant.v.norm}")
            
            
        print("Done\n")

        self.display()
                
        
    def display(self, full = True):
        
        def disp(aphs):
        
            if len(aphs) == 0:
                print("No aphelion")
                return
            
            for aph in aphs:
                p = aph.p.value
                a = np.arctan2(p[1], p[0])
                n = Number(a, unit='sec')
                print("rotation:", n)
                t = Number(aph.t, unit='s')
                print("period  :", t)
                print()
                print(aph.p)
                print(aph.t)
                
        print("-"*30)
        print("Without Doppler\n")
        disp(self.aphelions)
        print()
        
        print("-"*30)
        print("With Doppler\n")
        disp(self.dop_aphelions)
        print()
        
        self.buffer.plot_comp(self.dop_buffer)
        
        
        return
        
        
        def stats(a):
            ags = np.degrees(np.array(
                [np.arctan2(instant.point[1], instant.point[0]) for instant in a]
                ))
            rs  = np.array([instant.r for instant in a])
            ts  = np.array([instant.t for instant in a])
            
            
            ags = ags[1:] - ags[:-1]
            rs  = (rs[1:]-rs[0]) / rs[0] * 100
            ts  = ts[1:] - ts[:-1]
            
            return ags, rs, ts/86400
        
        
        ags, rs, ts = stats(self.aphelions)
        
        sag = ""
        sr  = ""
        st  = ""
        for i in range(len(ags)):
            sag += f"{ags[i]:.10f}° "
            sr  += f"{rs[i]:.4f}% "
            st  += f"{ts[i]:.0f} "
            if i > 0:
                st += f"({(ts[i]-ts[0])/ts[0]*100:.4f}%) "
                
        if len(self.aphelions) > 0:
            print()
            print(f"Aphelion      : {self.aphelions[0].r*1e-9:.4f} M km")
            print(f"period        : {ts[0]:.3f} days")
            #print('\nAphelions')
            #print('---------')
            print(f"Axis rotation : {ags[0]:.10f}° = {ags[0]*3600:4f} sec")
            #print("Aphelion      :", sr)
            #print("Period (days) :", st)
        
        
        if len(ags) >= 2:
            a = self.aphelions
            ag0 = np.degrees(np.arctan2(a[0].point[1], a[0].point[0]))
            ag1 = np.degrees(np.arctan2(a[-1].point[1], a[-1].point[0]))
            t0 = a[0].t
            t1 = a[-1].t
            dt = (t1-t0)/86400/365.25/100 # Century
            
            if full:
                print(f"Rotation speed :, {(ag1-ag0)/dt:.4f}°/cent", dt, t1-t0)
            print(f"Rotation avg   :, loops: {len(ags)} --> {np.average(ags)/dt*2:.4f}°/cent")
            
        if full:
            fig, ax = plt.subplots()
            ax.plot(ags)
            fig.title = "Aphelion rotation"
            plt.plot()
        
            
    @staticmethod
    def demo():
        
        orbit = Orbit()
        
        instant = Instant(merc_aphelion, merc_speed, edt=-6)
        
        orbit.leaps(instant, 90*86400)
        
        
    @staticmethod
    def mercure():
        
        target = 0.1038 * 2*np.pi/360/3600 #0.1038 sec/rev around 5.03e-7

        orbit = Orbit()
        
        instant = Instant(merc_aphelion, merc_speed, edt=5)
        
        orbit.leaps(instant, 90*86400)
        

#Orbit.demo()
Orbit.mercure()

