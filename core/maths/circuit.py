#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:33:17 2021

@author: alain
"""

import numpy as np
from .bezier import Beziers, Polynoms



# NOTE : in this module, speeds are expressed in km/h
# Accelerations are expressed in m/s2

# ===========================================================================
# Build an acceleration from
# - Max acceleration
# - Geometric progression with gears
# - Max speed
# - Speeds at which gears are passed

def build_acc_profile(max_acc=12., stages=[60, 100, 140, 180], ratio=0.9,  max_speed=260.):
    
    if True:
    
        v = max_speed
        a = max_acc
        xs = np.array([    0.,   7., 10, 0.5*v, 0.8*v,   v])
        ys = np.array([0.7*a, 0.8*a,  a,     a, 0.6*a, 0.])
        
        return Polynoms(xs, ys)
    
    
    def a_stage(v0, v1, acc):
        dv = v1 - v0
        xs = np.array([v0+3,    v0+5, v0+dv*.6, v0+dv*0.75,    v1-3,  v1])
        ys = np.array([ acc*.7,  acc,      acc,    acc*.92,   acc*.7,  0.])
        
        return xs, ys
    
    # ---------------------------------------------------------------------------
    # First stage
    
    v = stages[0]
    a = max_acc
    xs = np.array([    0.,   7., 10, 0.8*v,  v-3.,  v-2.,  v])
    ys = np.array([0.7*a, 0.8*a,  a,   a,   0.7*a, 0.6*a,  0.])

    # ---------------------------------------------------------------------------
    # Intermediary stages
    
    for i in range(len(stages)-1):
        v0 = stages[i]
        v1 = stages[i+1]
        
        a *= ratio
        x, y = a_stage(v0, v1, a)
        
        xs = np.append(xs, x)
        ys = np.append(ys, y)
        
    # ---------------------------------------------------------------------------
    # Last stage
    
    v0 = stages[-1]
    v1 = max_speed
    dv = v1 - v0
    
    a *= ratio
    
    x = np.array([v0+3,   v0+dv*.5, v0+dv*.75,  v1])
    y = np.array([   a,        a/3,  a/9,       0.])
    
    xs = np.append(xs, x)
    ys = np.append(ys, y)
    
    return Polynoms(xs, ys)


# ===========================================================================
# Gives the maximum speed depending upon the angle of the curve

def max_speed_in_curve(angle, max_speed=260.):
    return max_speed*(1.05 - (1 / (1 + np.power(np.e, -20*(angle-.1)))))


def plot_speed_in_curve(sic):
    
    count = 100
    ag = np.linspace(0, np.pi/2, count)
    deg = np.linspace(0, 90, count)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(deg, sic(ag))
    plt.show()
    

# ===========================================================================
# A circuit

class Circuit():
    
    def __init__(self, track, max_speed=260., resolution=1000):
        
        self.track     = track

        self.max_speed = max_speed
        
        self.d2param, self.length = self.track.distance_to_param(resolution=resolution)
        
        # --------------------------------------------------
        # Points regularly spaced on the track
        
        ts = self.d2param(np.linspace(0, self.length, resolution))
        self.points = self.track(ts)
        
        # --------------------------------------------------
        # Lengths of the segments between the points
        
        segments = self.points[1:] - self.points[:-1]
        
        self.dxs = np.linalg.norm(segments, axis=-1)
        self.distance = np.zeros(resolution, float)
        self.distance[1:] = np.cumsum(self.dxs)
        self.length = self.distance[-1]
        
        # --------------------------------------------------
        # Angles between the segments
        
        segments = segments / np.expand_dims(self.dxs, axis=-1)
        angles = np.arccos(segments[1:, 0] * segments[:-1, 0] + segments[1:, 1] * segments[:-1, 1])
        
        # --------------------------------------------------
        # The curve is on the right or on the left
        
        self.curve_sides = np.ones(resolution, int)
        rights = segments[:-1, 0] * segments[1:, 1] - segments[1:, 0] * segments[:-1, 1] > 0
        self.curve_sides[1:-1] = rights
        self.curve_sides[0]    = self.curve_sides[1]
        self.curve_sides[-1]   = self.curve_sides[-2]
        self.curve_sides[self.curve_sides==0] = -1
        
        self.angles       = np.zeros(resolution, float)
        self.angles[1:-1] = angles
        self.angles[0]    = self.angles[1]
        self.angles[-1]   = self.angles[-2]
        self.angles       = self.angles * self.curve_sides
        
        # --------------------------------------------------
        # Angles allow to compute the curvature radius
        # To avoid infinite radius, we need a minimum angle
        # Not to depend upon the resolution, we take a maxium radius
        
        max_radius = 1000
        min_angle  = np.average(self.dxs)/max_radius 
        
        # ----- Let's compute the inverse of the radius

        self.inv_radius       = np.zeros(resolution, float)
        self.inv_radius[1:-1] = 2*angles/(self.dxs[:-1]+self.dxs[1:])
        self.inv_radius[0]    = self.inv_radius[1]
        self.inv_radius[-1]   = self.inv_radius[-2]
        
        # ----- The radius now
        
        self.radius = 1/self.inv_radius
        self.radius[self.inv_radius < 1/max_radius] = max_radius
        
        if False:

            angles[angles<min_angle] = min_angle
            
            self.radius       = np.empty(resolution, float)
            self.radius[1:-1] = (self.dxs[:-1]+self.dxs[1:])/2/angles
            self.radius[0]    = self.radius[1]
            self.radius[-1]   = self.radius[-2]
        
        # ----- We are also interested with the square root of the radius
        
        self.sqrt_radius = np.sqrt(self.radius)
        
        # ---------------------------------------------------------------------------
        
        if False:
            print('-'*80)
            print("Circuit initialization")
            print(f"Resolution : {resolution} points")
            print(f"Length     : {self.length:.0f} m")
            print()
        
    # ---------------------------------------------------------------------------
    # Compute the track with a start speed, max speed in the curve an acceleration
    # curve and breaking acceleration
    #
    # For an acceleration a on a distance dx, the speed is given by the formula:
    # s1 = sqrt(s0^2 + 2.a.dx)
        
    def compute_track(self, start_speed=0., acc_profile=None, max_acc_curve=8., breaking_acc=-10):
        
        count = len(self.points)
        
        # ---------------------------------------------------------------------------
        # Function giving the max acceleration depending on the speed
        # CAUTION :
        # - speed is expressed in km/h
        # - acceleration is expressed in m/s^2
        
        self.acc_profile = acc_profile
        if self.acc_profile is None:
            self.acc_profile = build_acc_profile(max_speed=self.max_speed)
            
        # Max centripete acceleration in curve 
            
        self.max_acc_curve = max_acc_curve
        
        # Max breaking acceleration
        
        self.breaking_acc  = -abs(breaking_acc)
            
        # --------------------------------------------------
        # Max speed at each point of the circuit
        
        self.speeds = np.clip(np.sqrt(max_acc_curve)*self.sqrt_radius*3.6, 0., self.max_speed)
        
        # Store for graphics
        self.max_speeds = np.array(self.speeds)
        
        # --------------------------------------------------
        # The max speeds are lowered to take into account
        # the breaking capabilities
        
        for i in reversed(range(count-1)):

            s0 = self.speeds[i]/3.6    # in m/s
            s1 = self.speeds[i+1]/3.6
            
            # Speed is higher than next speed
            if s0 > s1:
                sm = np.sqrt(s1*s1 - 2*self.breaking_acc*self.dxs[i])
                self.speeds[i] = min(sm, s0)*3.6
                
        # Store for graphics
        self.breaking_speeds = np.array(self.speeds)
                
        # --------------------------------------------------
        # Now we can accelerate from the start
        
        self.speeds[0] = min(start_speed, self.speeds[0])
        
        for i in range(count-1):

            s0 = self.speeds[i]/3.6    # in m/s
            s1 = self.speeds[i+1]/3.6
            
            if s0 < s1:
                # Compute the max possible next speed
                a = self.acc_profile(self.speeds[i]) # CAUTION = argument is in km/h !!!!
                sm = np.sqrt(s0*s0 + 2*a*self.dxs[i])
                self.speeds[i+1] = min(sm, s1)*3.6
            
        # --------------------------------------------------
        # We can now compute time --> parameter
        
        self.dts = np.zeros(count, float)
        self.dts[1:] = 2*self.dxs/(self.speeds[:-1] + self.speeds[1:])*3.6
        
        self.time = np.cumsum(self.dts)
        self.duration = self.time[-1]
        
        self.time_to_param = Polynoms(self.time, self.d2param(self.distance))
        self.time_to_distance = Polynoms(self.time, self.distance)
        
        # --------------------------------------------------
        # Let's build a Beziers with time --> location
        
        self.time_to_location = Beziers(self.track(self.time_to_param(np.linspace(0, self.duration, count))), t1=self.duration)
        self.time_to_speed    = Polynoms(self.time, self.speeds)
        
        self.accs      = np.zeros(count, float)
        self.accs[:-1] = (self.speeds[1:] - self.speeds[:-1])/self.dts[1:]/3.6
        self.accs[-1]  = self.accs[-2]
        self.time_to_acc = Polynoms(self.time, self.accs, linear=True)
        
        # --------------------------------------------------
        # Centripet acceleration: rw^2 = v^2/r
        
        self.ctr_accs = (((self.speeds*self.speeds)*self.inv_radius)*(1/3.6/3.6))*self.curve_sides
        self.time_to_centripet = Polynoms(self.time, self.ctr_accs)
        
        # --------------------------------------------------
        # Curve angle
        
        self.time_to_curve = Polynoms(self.time, self.angles)
        
        # ---------------------------------------------------------------------------
        
        if False:
            print('-'*80)
            print("Path computation")
            print(f"Resolution : {count} points")
            print(f"Length     : {self.length:.0f} m")
            print(f"Duration   : {self.duration:.0f} s")
            print(f"Speed  avg : {np.average(self.speeds):.0f} km/h, std: {np.std(self.speeds):.0f} km/h  (min: {np.min(self.speeds):.0f} km/h, max: {np.max(self.speeds):.0f} km/h)")
            print(f"Acc    avg : {np.average(self.accs):.2f} m/s² (min: {np.min(self.accs):.2f} m/s², max: {np.max(self.accs):.2f} m/s²)")
        
        
        
    def vect_speed(self, t):
        dt = self.duration/1000
        return (self.time_to_location(t+dt) - self.time_to_location(t-dt))/2/dt*3.6
        
        return self.time_to_location(t, der=1)/3.6 # Conversion to km/h
        
    def vect_acc(self, t):
        dt = self.duration/1000
        return (self.vect_speed(t+dt) - self.vect_speed(t-dt))/2/dt/3.6
        
        return self.time_to_location(t, der=2)
        
            
    # ===========================================================================
    # A circuit to develop
    
    @classmethod
    def Spa(cls, resolution=1000):
        txt_spa = """
        23
        1279.7825927734375; 62.743099212646484; 6.103515625e-05;1258.886962890625; 105.98243713378906; 6.103515625e-05;1343.73974609375; -69.60350036621094; 6.103515625e-05
        1635.4237060546875; -1173.957763671875; 6.103515625e-05;1640.6866455078125; -1071.6505126953125; 6.103515625e-05;1632.9556884765625; -1221.9339599609375; 6.103515625e-05
        1552.1741943359375; -1312.8460693359375; 6.103515625e-05;1556.779296875; -1257.9906005859375; 6.103515625e-05;1547.569091796875; -1367.7015380859375; 6.103515625e-05
        1578.4315185546875; -1516.464599609375; 6.103515625e-05;1613.1929931640625; -1473.588623046875; 6.103515625e-05;1495.239501953125; -1619.076416015625; 6.103515625e-05
        1113.922607421875; -1838.7899169921875; 6.103515625e-05;1204.6162109375; -1831.8707275390625; 6.103515625e-05;1053.111083984375; -1843.4293212890625; 6.103515625e-05
        1080.0650634765625; -1695.129150390625; 6.103515625e-05;1028.8778076171875; -1741.5396728515625; 6.103515625e-05;1131.252197265625; -1648.71875; 6.103515625e-05
        1263.15478515625; -1585.5751953125; 6.103515625e-05;1212.123779296875; -1632.1573486328125; 6.103515625e-05;1314.1856689453125; -1538.9931640625; 6.103515625e-05
        1269.0924072265625; -1436.8741455078125; 6.103515625e-05;1284.7720947265625; -1478.652099609375; 6.103515625e-05;1173.48193359375; -1182.1234130859375; 6.103515625e-05
        1114.731689453125; -791.0829467773438; 6.103515625e-05;1147.1397705078125; -931.210693359375; 6.103515625e-05;1092.0625; -693.0648803710938; 6.103515625e-05
        669.0599975585938; -840.9146118164062; 6.103515625e-05;777.2929077148438; -606.4124755859375; 6.103515625e-05;594.0528564453125; -1003.4283447265625; 6.103515625e-05
        504.4503479003906; -1291.1064453125; 6.103515625e-05;582.0587158203125; -1140.8138427734375; 6.103515625e-05;456.11676025390625; -1384.7069091796875; 6.103515625e-05
        296.194091796875; -1335.090576171875; 6.103515625e-05;334.5047607421875; -1375.4100341796875; 6.103515625e-05;249.28321838378906; -1285.719970703125; 6.103515625e-05
        112.69996643066406; -1355.8363037109375; 6.103515625e-05;154.32362365722656; -1292.2681884765625; 6.103515625e-05;82.82044982910156; -1401.4686279296875; 6.103515625e-05
        -77.65286254882812; -1642.88037109375; 6.103515625e-05;33.37574768066406; -1574.2967529296875; 6.103515625e-05;-124.15057373046875; -1671.6025390625; 6.103515625e-05
        -349.2448425292969; -1418.3226318359375; 6.103515625e-05;-317.71136474609375; -1594.4486083984375; 6.103515625e-05;-376.5520935058594; -1265.8016357421875; 6.103515625e-05
        46.16513442993164; -875.0626220703125; 6.103515625e-05;-91.61497497558594; -941.4180908203125; 6.103515625e-05;183.944580078125; -808.7074584960938; 6.103515625e-05
        445.3338317871094; -613.9816284179688; 6.103515625e-05;360.42138671875; -741.1673583984375; 6.103515625e-05;530.2459716796875; -486.79638671875; 6.103515625e-05
        595.2568359375; -258.5365295410156; 6.103515625e-05;592.461181640625; -396.03631591796875; 6.103515625e-05;598.052490234375; -121.03724670410156; 6.103515625e-05
        476.5167541503906; 106.87821197509766; 6.103515625e-05;493.73876953125; 48.82093811035156; 6.103515625e-05;462.7348327636719; 153.3385467529297; 6.103515625e-05
        416.36322021484375; 578.4976196289062; 6.103515625e-05;358.46484375; 491.8799133300781; 6.103515625e-05;431.1626892089844; 600.6380615234375; 6.103515625e-05
        485.80242919921875; 580.4103393554688; 6.103515625e-05;471.8313293457031; 582.3811645507812; 6.103515625e-05;501.21966552734375; 578.2355346679688; 6.103515625e-05
        504.1620788574219; 613.7923583984375; 6.103515625e-05;510.41253662109375; 601.14306640625; 6.103515625e-05;497.26458740234375; 627.7510986328125; 6.103515625e-05
        81.06096649169922; 1343.52490234375; 6.103515625e-05;88.76667022705078; 1331.705810546875; 6.103515625e-05;64.91004180908203; 1368.29736328125; 6.103515625e-05
        """        
        spa = Beziers.FromText(txt_spa)
        return cls(spa, resolution=resolution)

    # ===========================================================================
    # Debug
    
    def plot_track(self):
        self.track.plot(title="Circuit")
        
    def plot_curves(self):
        
        import matplotlib.pyplot as plt
        
        # ---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        
        x = np.linspace(0, self.time[-1], len(self.time))
        ax.plot(x, self.time_to_param(x))
        ax.set_title("Time to param")
        
        plt.show()
        
        # ---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        
        x = np.linspace(0, self.time[-1], len(self.time))
        f = Polynoms(self.time, self.distance)
        ax.plot(x, f(x))
        ax.set_title("Time to distance")
        
        plt.show()
        
        # ---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        
        x = np.linspace(0, 1, len(self.time))
        f = self.d2param.invert
        ax.plot(x, f(x))
        ax.set_title("Param to distance")
        
        plt.show()
        
        # ---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        
        ax.plot(self.distance, self.radius)
        ax.set_title("Distance to curve radius")
        plt.yscale("log")
        
        plt.show()
        
        # ---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        
        x = np.linspace(0, self.max_speed, 1000)
        ax.plot(x, self.acc_profile(x))
        ax.set_title("Acceleration profile")
        
        plt.show()
        
        # ---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        
        x  = self.time
        ts = np.linspace(0, self.duration, len(self.points))
        
        y0 = self.speeds
        y1 = self.time_to_speed(ts)
        y2 = np.linalg.norm(self.vect_speed(ts), axis=-1)
        y3 = np.zeros(len(self.points), float)
        y3[1:] = (self.dxs/self.dts[1:]*3.6)
        
        i0 = 0
        i1 = 300
        
        ax.plot( x[i0:i1], y3[i0:i1], 'green', label="dx/dt")
        ax.plot( x[i0:i1], y0[i0:i1], 'black', label="(time, speeds)")
        ax.plot(ts[i0:i1], y1[i0:i1], 'blue',  label="time_to_speed()")
        ax.plot(ts[i0:i1], y2[i0:i1], 'red',   label="Bezier")
        ax.set_title("Speed computations")
        ax.legend()
        
        plt.show()
        
        # ---------------------------------------------------------------------------

        fig, ax = plt.subplots()
        
        x = np.linspace(0, self.duration, 1000)
        
        #ax.plot(self.time, self.accs, label="Tangent")
        #ax.plot(self.time, self.ctr_accs, label="Centripet")
        
        ax.plot(x, self.time_to_acc(x),       'blue', label="time_to_acc")
        ax.plot(x, self.time_to_centripet(x), 'red',  label="time_to_centripet")
        
        ax.set_title("Time to acceleration")
        ax.legend()
        
        plt.show()
        
        
    # ===========================================================================
    # Plot the speeds
            
    def plot_speeds(self, count=None):
        
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        
        if count is None:
            count = len(self.points)
        
        x = np.linspace(0, self.time[-1], len(self.time))
        ax.plot(x[:count], self.max_speeds[:count], 'black', label="Max")
        ax.plot(x[:count], self.breaking_speeds[:count], 'blue', label="Breaking")
        ax.plot(x[:count], self.speeds[:count], 'red', label="Speed")
        ax.set_title("Speeds")
        ax.legend()
        
        plt.show()
        
    @staticmethod
    def demo():
        
        spa = Circuit.Spa(1000)
        
        spa.plot_track()
        
        spa.compute_track()
        
        spa.plot_curves()
        spa.plot_speeds(300)


#Circuit.demo()    
    
    
    
        
    
    
    
    
    
    