#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:14:47 2021

@author: alain
"""

import numpy as np


def str_v(v, space=""):
    s = space + "["

    if len(np.shape(v)) > 1:
        for vv in v:
            s += "\n" + str_v(vv, space + "    ")
    else:
        for x in v:
            s += f" {x:6.2f}"
            
    return s + "]"

def str_e(e, only_x=False):
    if only_x:
        return f"({e[0]:6.2f} ct={e[3]:6.2f})"
    else:
        return f"({str_v(e[:3])} ct={e[3]:6.2f})"
    
# ---------------------------------------------------------------------------
# Dicchotomy solver    
    
def dichotomy(f, target=0, start=0, prec=0.0001, t0=None, t1=None):
    
    shape = np.shape(target)
    size  = np.size(target)
    
    if np.size(start) > size:
        shape = np.shape(start)
        size  = np.size(start)
        
    v0 = f(start)
    if np.size(v0) > size:
        shape = np.shape(v0)
        size  = np.size(v0)
        
    single = shape == ()
    if single:
        shape = (1,)
        
    # ----- The target
        
    vt = np.empty(shape, float)
    vt[:] = target
    
    v1 = np.array(v0)
    
    # ----- Lowest limit
        
    if t0 is None:
        t0 = np.empty(shape, float)
        t0[:] = start
        
        e = 1
        for i in range(30):
            inds = np.where(v0 > vt)[0]
            if len(inds) == 0:
                break
            t0[inds] -= e
            e *= 2
            v0 = f(t0)
    else:
        t0 = np.resize(t0, shape).astype(float)
        
    # ----- Highest limit
        
    if t1 is None:
        t1 = np.empty(shape, float)
        t1[:] = start
        
        e = 1
        for i in range(30):
            inds = np.where(v1 < vt)[0]
            if len(inds) == 0:
                break
            t1[inds] += e
            e *= 2
            v1 = f(t1)
    else:
        t1 = np.resize(t1, shape).astype(float)

    # ---------------------------------------------------------------------------
    # Dichotomy loop
    
    for i in range(40):
        
        t = (t0 + t1)/2
        
        v = f(t)
        
        if len(np.where(abs(v - vt) > prec)[0]) == 0:
            break
        
        inds = np.where(v < vt)
        t0[inds] = t[inds]
        
        inds = np.where(v > vt)
        t1[inds] = t[inds]
        
    # ---------------------------------------------------------------------------
    # Return the result        
        
    if single:
        return t[0]
    else:
        return t    
    
# ---------------------------------------------------------------------------
# Utility which shapes two arrays into the same global shape

def arrays_shape(a1, a2, item1=(), item2=()):
    
    if len(np.shape(a1)) <= len(item1):
        shape1 = ()
    else:
        shape1 = np.shape(a1)[:len(np.shape(a1))-len(item1)]

    if len(np.shape(a2)) <= len(item2):
        shape2 = ()
    else:
        shape2 = np.shape(a2)[:len(np.shape(a2))-len(item2)]
        
    return shape2 if len(shape2) > len(shape1) else shape1
    
    
# ---------------------------------------------------------------------------
# Initialize an event, or an array of events from location and time
#
# An event if is 4D vector. For consistency, time is place at the fourth location:
#
# (ct, x, y, z) = np.array((x, y, z, ct))
#
# If location is a single evalue, it is interpreted as (x, 0, 0)

def event(location, time):
    
    shape = arrays_shape(location, time, (3,)) + (4,)
        
    e = np.zeros(shape, float)
    if np.shape(location) == ():
        e[..., 0] = location
    else:
        e[..., :3] = location
    e[..., 3] = time
    
    return e

# ---------------------------------------------------------------------------
# Space-time interval
    
def st_interval(e0, e1):
    d = np.array(e1) - np.array(e0)
    d *= d
    
    return np.sum(d[:3], axis=-1) - d[3]

# ---------------------------------------------------------------------------
# Rotation matrix around an axis

def rot_matrix(angle, axis='Z'):
    
    ca = np.cos(angle)
    sa = np.sin(angle)
    
    if axis.upper() == 'X':
        return np.array((
            (1,  0,   0),
            (0,  ca, sa),
            (0, -sa, ca)
            ))
    
    elif axis.upper() == 'Y':
        return np.array((
            ( ca, 0, -sa),
            (  0, 1,   0),
            ( sa, 0,  ca)
            ))
    
    return np.array((
        ( ca, sa, 0),
        (-sa, ca, 0),
        (  0,  0, 1)
        ))


# ---------------------------------------------------------------------------
# Rotation matrix to align an axis along a direction

def orient_matrix(vector, axis='Z'):
    
    # ----- Normalize the direction
    
    v = np.array(vector) / np.linalg.norm(vector)

    # ----- Spherical components (theta, phi)
    
    theta = np.arctan2(v[1], v[0])
    nh    = np.linalg.norm(v[:2])
    phi   = np.arctan2(v[2], nh)   # nh >= 0 --> phi between -pi/2 and pi/2
    
    # ----- Align along the required axis
    
    if axis.upper() == 'X':
        return np.matmul(
                rot_matrix(-phi,  axis='Y'),
                rot_matrix(theta, axis='Z')
                )
    
    elif axis.upper() == 'Y':
        return np.matmul(
                rot_matrix(phi,             axis='X'),
                rot_matrix(theta - np.pi/2, axis='Z')
                )

    return np.matmul(
                rot_matrix(np.pi/2 - phi, axis='Y'),
                rot_matrix(theta,         axis='Z')
                )

# ---------------------------------------------------------------------------
# Some tests

def rm_test():
    for axis in ['X', 'Y', 'Z']:
        for ag in [30, 120]:
            for u in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                print(f"axis: {axis}, angle: {ag:3d}°: u {str_v(u)}", str_v(np.dot(u, rot_matrix(np.radians(ag), axis))))
        print()
        
def om_test():
    
    for axis, a in zip(['X', 'Y', 'Z'], [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        for theta in np.linspace(0, 2*np.pi, 12, False):
            for phi in np.linspace(-np.pi/2, np.pi/2, 6):
                v = (np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi))
                m = orient_matrix(v, axis)
                b = np.dot(a, m)
                diff = np.linalg.norm(v - b)
                
                thd = np.degrees(theta)
                phd = np.degrees(phi)
                print(f"axis={axis}: theta: {thd:4.0f}°, phi: {phd:4.0f}° {str_v(v)} --> {str_v(b)} {diff:.4f}")
        print()
    
# ---------------------------------------------------------------------------
# Lorentz transformation
#
# Standard transformation generally uses:
# - (0, 0, 0, 0) for both origin events
# - Speed along x axis
#
# Compute the general transformation with no constraint
#

class Lorentz():
    
    def __init__(self, speed=0, event0=0, event1=0):
        
        if np.shape(speed) == ():
            self.speed = np.array((speed, 0., 0.))
        else:
            self.speed = np.empty(3, float)
            self.speed[:] = speed
        
        self.beta = np.linalg.norm(self.speed)
        if self.beta >= .99999999:
            raise RuntimeError(f"Special relativity error: speed can't be higher then light velocity: {speed}")
            
        self.alpha = np.sqrt(1 - self.beta*self.beta)
        self.gamma = 1/self.alpha
        
        self.event0 = np.empty(4, float)
        self.event1 = np.empty(4, float)
        
        self.event0[:] = event0
        self.event1[:] = event1
        
        # ---------------------------------------------------------------------------
        # Spacial rotation to orient x along the speed
        
        if self.beta < 1e-6:
            self.x_to_speed  = np.identity(3)
            self.x_to_speed_ = np.identity(3)
        else:
            self.x_to_speed  = orient_matrix(self.speed, axis='X')
            self.x_to_speed_ = np.linalg.inv(self.x_to_speed)
        
        mat  = np.identity(4)
        mat[:3, :3] = self.x_to_speed
        mat_ = np.linalg.inv(mat)
    
        # ---------------------------------------------------------------------------
        # Combine with standard Lorentz transformation
        
        # mat  : X --> speed
        # mat_ : speed --> X 
        
        self.mat = np.matmul(mat_, np.matmul(np.array((
            (           self.gamma, 0, 0, -self.gamma*self.beta),
            (                    0, 1, 0,                     0),
            (                    0, 0, 1,                     0),
            (-self.gamma*self.beta, 0, 0,            self.gamma)
            )), mat))
        
        # And inverse transformation
        
        self.mat_ = np.linalg.inv(self.mat)
        
    def __repr__(self):
        s = f"<Lorentz transformation: beta= {self.beta:.3f}, gamma= {self.gamma:.2f}"
        s += f"\n   event0: {str_e(self.event0)}"
        s += f"\n   event1: {str_e(self.event1)}"
        return s + "\n>"
    
      
    # ---------------------------------------------------------------------------
    # Transformation
    
    def raw_transformation(self, e, inv=False):
        if inv:
            return np.dot(e, self.mat_)
        else:
            return np.dot(e, self.mat)
        
    def __call__(self, e, inv=False):
        if inv:
            return self.event0 + np.dot(e - self.event1, self.mat_)
        else:
            return self.event1 + np.dot(e - self.event0, self.mat)
        
    def inverse(self):
        
        lz = Lorentz()
        
        lz.mat    = np.array(self.mat_)
        lz.mat_   = np.array(self.mat)
        lz.beta   = self.beta
        lz.alpha  = self.alpha
        lz.gamma  = self.gamma
        lz.event0 = np.array(self.event1)
        lz.event1 = np.array(self.event0)
        
        return lz
    
    # ---------------------------------------------------------------------------
    # At a given time in the frame, we want to know where are located
    # points at given locations in the mobile frame
    #
    # - time : time in the "fixed" frame
    # - locs : locations in "mobile" frame
    #
    # e = (x, t) --> e' (x', t')
    #
    # Where t and x' are given. We want x.
    #
    # Let's consider dans the speed is along x and let's consider
    # the event f = (x=0, t)
    #
    # f=(0, t) is transformed in (a, b)
    #
    # Given the length contraction
    # (x, t) is transformed in (a + x*gamma, b')
    #
    # Hence:
    # x' = a + x*gamma ==> x = (x'-a)/gamma
    #
    
    def sim_location(self, global_times, local_points):
        
        if np.shape(local_points) == ():
            locs = np.array((local_points, 0, 0), float)
        else:
            locs = np.array(local_points)
            
        shape = arrays_shape(locs, global_times, (3,))
        
        e = np.zeros(shape + (4,))
        e[..., 3] = global_times 
        
        lcs = np.zeros(shape+(3,), float)
        lcs[:] = locs
        
        ep = self(e)

        lcs = np.matmul(lcs, self.x_to_speed_)
        
        lcs[...] -= ep[..., :3]
        lcs[..., 0] *= self.alpha
        
        # Return gloval locations and local times

        return np.matmul(lcs, self.x_to_speed), ep[..., 3]
        
        
    # ---------------------------------------------------------------------------
    # Twins story
    # 
    # Tells the twins story
    
    @staticmethod
    def twins(beta=.8, distance=8):
        
        # ----- Journey Lorentz transformation
        
        lz = Lorentz(beta)
        e0 = event(0, 0)
        duration = distance/beta
        e1 = event(distance, duration)
        e2 = event(0, 2*duration)
        
        f0 = lz(e0)
        f1 = lz(e1)
        
        # ----- At U-turn event, where is the earth for the Traveler ?
        
        # Traveler Lorentz transformation
        
        lz_ = lz.inverse()
        
        # Earth is at location 0 in the earth frame, with proper time age
        x, age   = lz_.sim_location(f1[3], 0)
        
        # ----- The transformation for the way back
        # the e1 / f1 is chosen as the origin for the frame
        
        wb = Lorentz(-beta, event0=e1, event1=f1)
        
        # We compute the arrival event for the travaler
        
        f2 = wb(e2)
        
        # ----- Let's write the full story
        
        print("Journey")
        print("-------")
        print("    start event : ", str_e(e0, True), str_e(f0, True))
        print("    U-turn event: ", str_e(e1, True), str_e(f1, True))
        print("    Earth location for traveler:", f"{x[0]:6.2f}")
        print("    Earth twin age for traveler:", f"{age:6.2f}")
        print()
        print("Way back")
        print("--------")
        print("    U-turn event :", str_e(e1, True), str_e(wb(e1), True))
        print("    Arrival event:", str_e(e2, True), str_e(f2, True))
        print()
        print("Earth twin age:", f"{e2[3]:6.2f}")
        print("Traveler age  :", f"{f2[3]:6.2f}")
        print()
        
        
# ---------------------------------------------------------------------------
# Add two speeds
# One is expressed in the ref frame of the other
#
# Let's consider a time dt in ref 0
# Speed s1 ins ref 0 means that during dt, location with change of s1.dt
# Two events : (0, 0) --> (s1.dt, dt)
# These two events are transformed 
# - (0, 0)      --> (0, 0)
# - (s1.dt, dt) --> (g.s1.dt + g.b.dt, g.b.s1.dt + g.dt)
#
# Speed is : s = g.s1.dt + g.b.

def speeds_add(speed0, speed1):
    #if np.size(speed0) > np.size(speed1):
    #    shape = np.shape(speed0)
    #else:
    #    shape = np.shape(speed1)
        
    #if shape == ():
    #    shape = (3,)
        
    shape = arrays_shape(speed0, speed1, (3,), (3,)) + (3,)
        
    s0 = np.zeros(shape, float)
    s1 = np.zeros(shape, float)
    
    if np.shape(speed0) == ():
        s0[0] = speed0
    else:
        s0[:] = speed0

    if np.shape(speed1) == ():
        s1[0] = speed1
    else:
        s1[:] = speed1
        
    # ---------------------------------------------------------------------------
    # Motion in the frame moving at speed 0 during dt = 1
    
    e = event(s1, 1)
    
    # ---------------------------------------------------------------------------
    # Referentiel frame 0
    
    f = Lorentz(speed=s0)(e, inv=True)
    
    return f[:3]/f[3]


# ---------------------------------------------------------------------------
# Solid

class Solid():
    
    def __init__(self, location=0, speed=0, omega=None, rot_axis=(0, 1, 0)):
        
        self.lorentz = Lorentz(speed, event0=event(location, 0))
        
        self.omega    = omega
        self.rot_axis = rot_axis
        
    # ---------------------------------------------------------------------------
    # Origin location
    
    def origin(self, t):
        return self.lorentz.event0[:3] + self.lorentz.speed*t
    
    # ---------------------------------------------------------------------------
    # Origin speed
    
    @property
    def speed(self):
        return self.lorentz.speed
    
    # ---------------------------------------------------------------------------
    # Global speed of pts with a local speed
    #
    # We consider points at origin. After a proper time tau, they will be located
    # at (v.tau, tau)
    # Transforming this event will give the speed
    # tau can be chosen to value 1
    #
    # Raw transformation is used since we don't need the origin events event0 and event1
    
    def global_speed(self, speeds):
        e = self.lorentz.raw_transformation(event(speeds, 1), inv=True)
        return e[..., :3]/np.expand_dims(e[..., 3], axis=-1)
    
    # ---------------------------------------------------------------------------
    # Rotation property
    
    @property
    def rot_axis(self):
        return self.rot_axis_
    
    @rot_axis.setter
    def rot_axis(self, value):
        self.rot_axis_    = np.empty(3, float)
        self.rot_axis_[:] = value
        self.rot_axis_    = self.rot_axis_ / np.linalg.norm(self.rot_axis_) 
        self.rot_matrix   = orient_matrix(self.rot_axis_, axis='Z')
        self.rot_matrix_  = np.linalg.inv(self.rot_matrix)
        
    # ---------------------------------------------------------------------------
    # If rotation exists, compute locations and speeds at given times
        
    def rotation(self, tau, locs):
        
        # ---------------------------------------------------------------------------
        # Shape the arrays
        
        shape = arrays_shape(locs, tau, (3,))
        taus = np.zeros(shape, float)
        taus[:] = tau
        lcs = np.zeros(shape+(3,), float)
        lcs[:] = locs
        
        # ---------------------------------------------------------------------------
        # No rotation : nothing to rotate
        
        if self.omega is None:
            return lcs, np.zeros(shape + (3,), float)
        
        # ---------------------------------------------------------------------------
        # Rotation
        
        m = np.zeros(shape + (3, 3), float)
        m[:] = np.identity(3)
        
        ag = self.omega*taus
        
        ca = np.cos(ag)
        sa = np.sin(ag)
        
        m[..., 0, 0] =  ca
        m[..., 1, 1] =  ca
        m[..., 0, 1] =  sa
        m[..., 1, 0] = -sa
        
        # ----- Computation
        
        pts = np.einsum('...ij,...i', # Einsum to rotate locations
              np.matmul(self.rot_matrix_, np.matmul(m, self.rot_matrix)),
              lcs)
        
        return pts, np.cross(self.omega*self.rot_axis, locs)
    
    # ---------------------------------------------------------------------------
    # Transform the solid into the observation frame
    # t is time in the observation frame (global time)
    # t can be shaped as the locations
    
    def transform(self, t, locs):
        
        # ---------------------------------------------------------------------------
        # No rotation : a simple computation
        
        if self.omega is None:
            return self.lorentz.sim_location(t, locs)
        
        # ---------------------------------------------------------------------------
        # Rotation : need to compute the curvatures
        
        # ----- Where is the center at time t
        
        O, tau = self.lorentz.sim_location(t, 0)
        
        # ----- Let's compute the proper time at each location
        # This proper time will change the rotation angle hencer
        # the location. The proper must be such as Lorentz transforms
        # at the global time t
        
        shape = arrays_shape(locs, t, (3,))
            
        # Let's prepare the rotation matrix used in the loop
            
        m = np.zeros(shape + (3, 3), float)
        m[:] = np.identity(3)
        
        # ---------------------------------------------------------------------------
        # The function to solve returns the universe time of each location
        # at their proper time v_tau. 
        # The dichotomy algorithm works only on time but we also need the locations
        # at the end. The supplementary argument events is an object collecting
        # the computed transformed event.
        
        def f(v_tau, events):
            
            # ---- Rotation
            
            ag = self.omega*v_tau
            
            ca = np.cos(ag)
            sa = np.sin(ag)
            
            m[..., 0, 0] =  ca
            m[..., 1, 1] =  ca
            m[..., 0, 1] =  sa
            m[..., 1, 0] = -sa
            
            # ----- Computation
            
            events.e = self.lorentz( # Inverse Lorentz transformation
              event(                 # Event made of rotated locations
                np.einsum(           # Einsum to rotate locations
                  '...ij,...i',
                  np.matmul(self.rot_matrix_, np.matmul(m, self.rot_matrix)),
                  locs),      
                v_tau),              # Time component of the events
              inv=True)              # Inverse transformation
            
            # ----- Return the new taus
                
            return events.e[..., 3]
                
            
        # ---------------------------------------------------------------------------
        # The proper times 
        
        class Events():
            pass
        
        events = Events()
        events.e = None
        
        taus = dichotomy(lambda tau: f(tau, events), target=t, start=np.ones(shape)*tau)
        
        # Returns the universe locations at that times
        
        return events.e[..., :3], taus
    
    # ---------------------------------------------------------------------------
    # Transform the solid as it is perceived by the observer taking into account
    # the time necessary for the light to reachthe observer.
    # Observation point 
    
    def perception(self, t, locs, origin=0):
        
        # ---------------------------------------------------------------------------
        # The arguments into the same shape
        
        shape = arrays_shape(locs, t, (3,))
        
        ts = np.zeros(shape, float)
        ts[:] = t
        lcs = np.zeros(shape + (3,), float)
        lcs[:] = locs
        
        # ---------------------------------------------------------------------------
        # At a given 'global' time w, the locations is transformed at location l
        # The distance is d.
        # The signal will reach the observer at time w + d/c
        # Hence, w is computed such as
        # d(l(w))/c = t-w
        # 
        # The function is solved by dichotomy
        
        def f(w):
            pts, _ = self.transform(w, lcs)
            ds = np.linalg.norm(pts - origin, axis=-1)
            return w + ds
        
        ws = dichotomy(f, target=ts, start=ts, t1=ts)
        
        return self.transform(ws, lcs)
    
    # ---------------------------------------------------------------------------
    # Doppler effect
    
    def doppler(self, tau, locs, origin=0):

        # ----- Speeds at the given proper times
        
        pts, speeds = self.rotation(tau, locs)
        
        # ----- Combine the speeds with the speed of the solid
        
        vs = self.global_speed(speeds)
        
        # ----- Emissions occur at locations locs width speed vs
        # We can compute the dopller factor with the formula
        # - f = sqrt((1+u)/(1-u))
        # where us is the speed along the direction between the observation point
        # and the location
        
        O = np.zeros(3, float)
        O[:] = origin
        
        pts -= O
        
        ns = np.linalg.norm(pts, axis=-1)
        ns[ns==0] = 1.
        
        pts = pts / np.expand_dims(ns, axis=-1)
        
        # ---- Speed along the direction
        
        ux = np.einsum('...i,...i', pts, vs)
        
        # ---- The Doppler factors
        
        return np.sqrt((1+ux)/(1-ux))



def build_wheel(diam_count=10, circ_count=4, count=128):
    radius  = 10
    verts   = np.zeros((diam_count + circ_count, count, 3), float)
    
    for i, ag in enumerate(np.linspace(0, np.pi, diam_count, False)):
        verts[i] = radius * np.transpose(np.stack((
            np.linspace(-np.cos(ag), np.cos(ag), count),
            np.linspace(-np.sin(ag), np.sin(ag), count),
            np.zeros(count, float)
            )))
    
    for i, r in enumerate(np.linspace(radius/(circ_count-1), radius, circ_count)):
        verts[diam_count + i] = r * np.transpose(np.stack((
            np.cos(np.linspace(0, 2*np.pi, count)),
            np.sin(np.linspace(0, 2*np.pi, count)),
            np.zeros(count, float)
            )))
        
    return verts

def test():

    solid = Solid(speed=.8, rot_axis=(0, 0, 1))
    locs = build_wheel(3, 0, 5)
    
    obs_t = -5
    
    solid.omega = -.08
    refs, refs_t = solid.transform(obs_t, locs)
    
    solid.omega = -.08
    trfs, trfs_t = solid.perception(obs_t, locs)
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    for i in range(len(locs)):
        ax.plot(locs[i, :, 0], locs[i, :, 1], 'lightgray')
    
    for i in range(len(locs)):
        ax.plot(refs[i, :, 0], refs[i, :, 1], 'gray')
    
    for i in range(len(locs)):
        ax.plot(trfs[i, :, 0], trfs[i, :, 1], 'r')
    
    
    plt.show()
    
    speeds = solid.doppler(trfs_t, locs)
    print(speeds)
    print(np.linalg.norm(speeds, axis=-1))



