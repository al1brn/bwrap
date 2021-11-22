#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 08:55:24 2021

@author: alain
"""

# --------------------------------------------------------------------------------
# Precision
#
# We want to detect a variation of an angle of da = 0.1 sec during an orbit
# This is an angle of:
# 0.1 sec = 4.848 10^-7 radian
# With an aphelion of 69 millions km = 69 10^9 m, this represents an arc of length
# dist = 34000 m = 34km, around 10^4 m
# The order of magnitude is the distance covered by Mercury during one second
#
# Hence the incremental time of the simulation must be a fraction of a second
# Let's use the fraction n for dt = 1O^-n
# With a period of 90 days = 7776000 seconds, around 10^7, this represents a number of steps of
# steps = 10^(7+n)
#
# The distance of 34 km is split in 10^(7+n) which represent a gap of:
# dx = 10^4/10^(7+n) = 10^-(3+n) m
#
# For exemple, with n = 0, we need a precision of 1 mm. The order of magnitude of
# the distances are 10^9 m. Hence we need to keep a unit precision for number
# which an order of magnitude of 10^(12+n). This is huge !
#
# When the distance is squared, we have figures with an order of magnitude 10^(24+2n).
# This is higher than what is managed with standard floats.
#
# To preserve this precision, we need a speed with a precision of:
# (v + o)*dt = v.dt + o.dt






import numpy as np
from number import Number, Vector
import matplotlib.pyplot as plt
import time
import math


Number.add_unit('m',   precision=16, fmt_func=lambda v: v.div(1e6),   fmt_unit='Mkm',     disp_prec=6)
Number.add_unit('m/s', precision=16, fmt_func=lambda v: v.div(1000),  fmt_unit='km/s',    disp_prec=3)
Number.add_unit('s',   precision= 6, fmt_func=lambda v: v.div(86400), fmt_unit='days',    disp_prec=3)
Number.add_unit('°',   precision=16, fmt_func=lambda a: Number(np.degrees(a.value)),      disp_prec=10)
Number.add_unit('sec', precision=16, fmt_func=lambda a: Number(np.degrees(a.value)/3600), disp_prec=10)


G     = 6.6743015e-11
M_sun = 1.989e30
c     = 3e8
c2    = c*c

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
        
        self.p       = Vector((r, 0))
        self.r       = Number(r)
        self.v       = Vector((0, v))
        
        self.edt     = edt
        if edt < 0:
            self.dt = (1 << (-edt))
        else:
            self.dt = 1/(1 << edt)
            
        self.r.precision = max(0, round(math.log(10000/self.dt, 10))+3)
        self.e_m  = self.r.e
        self.e_m2 = 2*self.e_m
        self.p.set_e(self.e_m)
        
        self.v.precision = self.r.precision - 3
        self.e_ms    = self.v.e
            
        self.GMdt    = Number(-G*M_sun*self.dt/2).set_e(0)
        
        r2 = self.p.squared_norm
        self.da2     = self.p.mul(self.GMdt.div(r2).div(r2.sqrt())).set_e(self.e_ms)
        self.t       = 0.
        
    def __repr__(self):
        s = ""
        s += f"p   : {self.p.fmt(unit='m')}\n"
        s += f"r   : {self.r.fmt(unit='m')}\n"
        s += f"v   : {self.v.fmt(unit='m/s')}\n"
        s += f"da2 : {self.da2.fmt(unit='m/s')}\n"
        s += f"GMdt: {self.GMdt.fmt(precision=0)}\n"
        s += f"m prec: {self.e_m}, m/s prec: {self.e_ms}"
        
        return s
        
        
    def clone(self):
        
        c = Instant()
        
        c.e_m  = self.e_m
        c.e_ms = self.e_ms
        
        c.p    = Vector(self.p)
        c.r    = Number(self.r)
        c.v    = Vector(self.v)
        
        c.edt  = self.edt
        c.dt   = self.dt
        c.GMdt = Number(self.GMdt)
        
        c.da2  = Vector(self.da2)
        c.t    = self.t
        
        return c
    
    def doppler_effect(self, u, p, da2):
        
        if True:
            
            # Speed norm
            beta2 = self.v.squared_norm.value
            beta  = math.sqrt(beta2)
            
            # Normalized speed
            v = np.array(self.v.value)/beta
            
            # Dot product
            dot = np.dot(np.array(u.value), v)
            
            # Beta
            beta /= c
            
            # alpha
            
            factor = 1 - beta*dot
            alpha  = math.sqrt(1 - beta*beta)
            
            return da2.mul(factor*alpha)            
            
        
        else:
            
            # Speed norm
            beta2 = self.v.squared_norm
            beta  = beta2.set_e(104).sqrt()
            
            # Normalized speed
            v = self.v.div(beta)
            
            # Dot product
            dot = u.dot(v)
            
            # Beta
            beta.div_eq(c)
            
            # alpha
            #alpha = Number(1).sub(beta2).sqrt()
            
            # Doppler factor
            factor = Number(1).sub(beta.mul(dot)) # *alpha
            
            return da2.mul(factor)
        
    
    def leap(self, doppler=False):
        
        # ----- Location variation
        # (v + da2)/dt = (v + da2) << edt
        
        # v + da2
        dp = self.v.add(self.da2)
        
        # / 2^edt while keeping precision
        dp.x.mul_pow2_eq(-self.edt, True)
        dp.y.mul_pow2_eq(-self.edt, True)
        
        # point += dpoint
        
        new_p = self.p.add(dp).set_e(self.e_m)
        
        # ----- New distance
        
        new_r2 = self.p.squared_norm
        new_r  = new_r2.sqrt().set_e(self.e_m)

        #print(new_p.sub(self.p), new_r.value)
        
        # ----- New acceleration
        
        new_da2 = new_p.mul(self.GMdt.div(new_r2).div(new_r)) #.set_e(self.e_ms*2)
        
        # ----- Doppler
        
        if doppler:
            new_da2 = self.doppler_effect(new_p.div(new_r).set_e(10), new_p, new_da2)
        
        # ----- New speed
        
        self.v.add_eq(self.da2.add(new_da2)).set_e(self.e_ms)
        
        # ----- Update 
        
        self.r   = new_r
        self.p   = new_p
        self.da2 = new_da2
        
        self.t  += self.dt 
        
        
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
        
        #dper = (steps // 100000)
        #per_cent = steps // dper
        per_cent = min(100000, steps // 100)
            
        in_buffer = steps // len(self.buffer.a)
        
        self.instant0 = instant.clone()
        dop_instant   = instant.clone()
        
        print(f"\nSimulation with {steps} steps\n")
        
        print("Departure")
        print(instant)
        print()
                
        self.growing     = False
        self.dop_growing = False
        
        t0 = time.time()
        
        for step in range(steps):
            
            # ----- Normal
            
            r = Number(instant.r)
            instant.leap(doppler = False)
            
            if self.growing:
                if instant.r.less(r):
                    self.growing = False
                    self.aphelions.append(instant.clone())
                    if len(self.dop_aphelions) > 0:
                        break
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
                    if len(self.aphelions) > 0:
                        break
                    
            else:
                if dop_instant.r.greater(r):
                    self.dop_growing = True
                    self.dop_perihelions.append(instant.clone())
                    
            # ----- Tracking
                    
            # in buffer
            if step % in_buffer == 0:
                self.buffer.append(np.array(instant.p.value))
                self.dop_buffer.append(np.array(dop_instant.p.value))

            if step % per_cent == 0:
                elapsed = time.time() - t0
                r_ratio = dop_instant.r.div(instant.r).value*100
                v_ratio = dop_instant.v.norm.div(instant.v.norm).value*100
                
                print(f"{step:9d} ({step/steps*100:.2f}%)> elapsed: {elapsed:.0f} s, rem: {elapsed/per_cent*(steps-step)/60:.0f} min,  r ratio: {r_ratio:.6f} %, v ratio: {v_ratio:.6f} %")
                self.buffer.plot_comp(self.dop_buffer)
                
                t0 = time.time()
            
        print("Done\n")
        
        self.save()

        self.display()
        
    def save(self, fname='toto'):
        a = np.stack((self.buffer.a, self.dop_buffer.a))
        np.save(fname, a)
        
    @staticmethod
    def load_buffers(fname='toto'):
        a = np.load(fname)
        buf0 = Buffer(2, length=a.shape[1])
        buf1 = Buffer(2, length=a.shape[1])
        buf0.a = np.array(a[0])
        buf1.a = np.array(a[1])
        buf0.index = a.shape[1]
        buf1.index = a.shape[1]
        
        buf0.plot_comp(buf1)
        
        return buf0, buf1
        
    def display(self, full = True):
        
        def disp_a(aphs):
        
            if len(aphs) == 0:
                print("No aphelion")
                return
            
            for aph in aphs:
                p = aph.p.value
                a = np.arctan2(p[1], p[0])
                print("rotation: ", a, "rad")
                print("        : ", np.degrees(a), "°")
                print("        : ", np.degrees(a)*3600, "sec")
                
                t = Number(aph.t, unit='s')
                print("period  :", t)
                print()
                print(aph.p.value)
                print(aph.t)

        def disp_p(pers):
        
            if len(pers) == 0:
                print("No perihelion")
                return
            
            for per in pers:
                p = per.p.value
                a = np.arctan2(p[1], p[0])
                print("rotation: ", a, "rad")
                print("        : ", np.degrees(a), "°")
                print("        : ", np.degrees(a)*3600, "sec")
                
                t = Number(per.t, unit='s')
                print("period  :", t)
                print()
                print(per.p.value)
                print(per.t)

        print("="*30)
        print("Perihelion without Doppler\n")
        disp_p(self.perihelions)
        print()
        
        print("-"*30)
        print("Perihelion with Doppler\n")
        disp_p(self.dop_perihelions)
        print()
                
        print("="*30)
        print("Aphelion without Doppler\n")
        disp_a(self.aphelions)
        print()
        
        print("-"*30)
        print("Aphelion with Doppler\n")
        disp_a(self.dop_aphelions)
        print()
        
        self.buffer.plot_comp(self.dop_buffer)
            
    @staticmethod
    def demo():
        
        orbit = Orbit()
        
        instant = Instant(merc_aphelion, merc_speed, edt=-6)
        
        orbit.leaps(instant, 90*86400)
        
        
    @staticmethod
    def mercure():
        
        target = 0.1038 * 2*np.pi/360/3600 #0.1038 sec/rev around 5.03e-7

        orbit = Orbit()
        
        instant = Instant(merc_aphelion, merc_speed, edt=2)
        
        orbit.leaps(instant, 90*86400)
        

#Orbit.demo()
Orbit.mercure()

#Orbit.load_buffers("/Users/alain/Documents/blender/scripts/modules/bwrap/maths/toto.npy", )
Orbit.load_buffers("toto.npy")

