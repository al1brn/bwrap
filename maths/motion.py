#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:19:56 2021

@author: alain
"""

import numpy as np

# ====================================================================================================
# Random locations

def normalize_ab(a, b):
    
    if np.shape(a) == () and np.shape(b) == ():
        return a, b, ()
    
    if np.shape(a) == ():
        vb, va, shape = normalize_ab(b, a)
        return va, vb, shape
    
    if np.shape(b) == ():
        vb = np.empty(len(a), float)
        vb[:] = b
        return np.array(a), vb, np.shape(a)
    
    ndim = max(len(a), len(b))
    va = np.empty(ndim, float)
    vb = np.empty(ndim, float)
    va[:] = a
    vb[:] = b
    return va, vb, np.shape(va)
    
# ----------------------------------------------------------------------------------------------------
# Uniform in a box

def box_uniform(a, b, shape, seed=0):
    
    if seed is not None:
        np.random.seed(seed)
    
    va, vb, v_shape = normalize_ab(a, b)
    
    a_shape = shape if hasattr(shape, '__len__') else (shape,)
    
    v = np.random.uniform(a, b, a_shape + v_shape)
    return v[np.linalg.norm(v, axis=-1) <= 1, :]
    
    return np.random.uniform(a, b, a_shape + v_shape)

# ----------------------------------------------------------------------------------------------------
# Uniform in a circle

def sphere_uniform(radius, shape, ndim=3, seed=0):
    
    if seed is not None:
        np.random.seed(seed)
        
    a_shape = shape if hasattr(shape, '__len__') else (shape,)
    count = 1 if a_shape == () else np.product(a_shape)
    
    if ndim == 2:
        n = int(count*1.6)  # Greater that 4/pi
        P0 = (-radius, -radius)
        P1 = ( radius,  radius)
    else:
        n = int(count*2.1)  # Greater that 8/(4/3pi)
        P0 = (-radius, -radius, -radius)
        P1 = ( radius,  radius, radius)
        
    # ---------------------------------------------------------------------------
    # Let's generate in a box
    
    v0 = np.random.uniform(P0, P1, (n, ndim))
    v0 = v0[np.linalg.norm(v0, axis=-1) <= radius]
    
    if len(v0) >= count:
        return v0[:count].reshape(a_shape + (ndim,))
    
    # ---------------------------------------------------------------------------
    # Lack of luck, not enough within the disk
    # Let's generate new points
    
    index = len(v0)
    v = np.empty((count, ndim), float)
    v[:index, :] = v0
    
    diff = count - index
    while True: # Let's be bold
    
        n = min(100, diff*3) # A good margin
    
        v0 = np.random.uniform(P0, P1, (n, ndim))
        v0 = v0[np.linalg.norm(v0, axis=-1) <= radius]

        if len(v0) >= diff:
            v[index:count] = v0[:diff]
            return v.reshape(a_shape + (ndim,))
        
        # :-(
        
        v[index:index+len(v0)] = v0
        index += len(v0)
        
        diff = count - index
        
# ----------------------------------------------------------------------------------------------------
# Dispersion around a point

def dispersion(radius, shape, ndim=3, seed=0):
    
    if seed is not None:
        np.random.seed(seed)
        
    a_shape = shape if hasattr(shape, '__len__') else (shape,)
        
    vrad = np.empty(a_shape + (ndim,), float)
    vrad[:] = radius
        
    v = np.empty(a_shape + (ndim,), float)
    for i in range(ndim):
        v[..., i] = np.random.normal(0, vrad[..., i], a_shape)
        
    if ndim == 1:
        return v.reshape(a_shape)
    else:
        return v


# ====================================================================================================
# Build a random shaped array of floats or vectors

def shaped_array(shape, ndim, value, scale=None):
    
    # User shape
    if not hasattr(shape, '__len__'): shape = (shape,)
    if hasattr(ndim, '__len__'):
        shape += ndim
    elif ndim > 1:
        shape += (ndim,)
    
    v = np.zeros(shape, float)
    v[:] = value
    
    if scale is None:
        return v
    
    s = np.zeros(shape, float)
    s[:] = scale

    return np.random.normal(v, s, shape)    


# ====================================================================================================
# Base for Evolution
# Default is a constant value not depending upon time

class Motion():
    
    def __init__(self, value, speed=None, acc=None):
        
        self.shape = np.shape(value)

        if hasattr(value, '__len__'):        
            self.value = np.array(value)
            self.speed = None if speed is None else np.array(speed)
            self.acc   = None if acc is None else np.array(acc)
        else:
            self.value = value
            self.speed = speed
            self.acc   = acc
            
        self.time_functions  = []
        self.value_functions = []
        
        
    @property
    def name(self):
        if self.acc is None:
            if self.speed is None:
                return "Constant"
            else:
                return "Uniform speed"
        else:
            if self.speed is None:
                return "Accelerated from 0"
            else:
                return  "Accelerated"            
        
    def __repr__(self):
        return f"<Motion {self.name}, shape={self.shape}>"
        
    def compute(self, t):
        
        if not hasattr(t, '__len__'):
            return self([t])[0]
        
        t_shape = [1]*(len(self.shape)+1)
        t_shape[0] = len(t)
        ts = np.reshape(t, t_shape)
        
        if self.acc is None:
            if self.speed is None:
                return np.resize(self.value, np.shape(t) + self.shape)
            else:
                return self.value + self.speed*ts
        else:
            if self.speed is None:
                return self.value + self.acc*ts*ts/2
            else:
                return self.value + (self.speed + self.acc*ts/2)*ts

    @classmethod
    def Constant(cls, shape, value, value_scale=None, ndim=3, seed=0):
        
        if seed is not None:
            np.random.seed(seed)
            
        if not hasattr(shape, '__len__'): shape = (shape,)
        
        return cls(shaped_array(shape, ndim, value, scale=value_scale))
            
    
    @classmethod
    def Speed(cls, shape, value, speed, value_scale=None, speed_scale=None, ndim=3, seed=0):
        
        motion = Motion.Constant(shape, value, value_scale, ndim=ndim, seed=seed)
        motion.speed = shaped_array(shape, ndim, speed, scale=speed_scale)
        
        return motion
        
    @classmethod
    def Accelerated(cls, shape, value, speed, acc, value_scale=None, speed_scale=None, acc_scale=None, ndim=3, seed=0):
        
        if speed is None:
            motion = cls.Normal(shape, value, value_scale, ndim=ndim, seed=seed)
        else:
            motion = cls.Speed(shape, value, speed, value_scale, speed_scale, ndim=ndim, seed=seed)
            
        motion.acc = shaped_array(shape, ndim, acc, scale=acc_scale)
        
        return motion
    
    @classmethod
    def FromTo(cls, start, end, duration=1., speed=None):
        
        start_size = np.product(np.shape(start))
        end_size   = np.product(np.shape(end))
        
        if end_size > start_size:
            starts    = np.empty(np.shape(end), float)
            starts[:] = start
            ends      = np.array(end)
        else:
            starts  = np.array(start)
            ends    = np.empty(np.shape(starts), float)
            ends[:] = end
        
        motion = cls(starts)
        
        if speed is None:
            motion.speed = (ends - motion.value)/duration
        else:
            # e = at^2/2 + st + v
            # a = 2(e - v - st)/t^2 = 2((e - v)/t - s)/t
            motion.speed = np.empty(np.shape(motion.value), float)
            motion.speed[:] = speed
            
            motion.acc = 2*( (ends - motion.value)/duration - motion.speed)/duration
            
        return motion
    
        
    def __call__(self, frame):
        
        for f in self.time_functions:
            frame = f(frame)
        
        a = self.compute(frame)
        
        for f in self.value_functions:
            a = f(a)
            
        return a
    
    def add_value_function(self, func):
        self.value_functions.append(func)
        
    def add_time_function(self, func):
        self.time_functions.append(func)
        
    def set_cyclic(self, period):
        self.add_time_function(lambda frame: frame % period)
        #self.add_function(lambda a: np.sin(a)/period*2*np.pi)
        
    def set_sine(self, average=None, amplitude=None):
        if average is None:
            if amplitude is None:
                self.add_value_function(lambda a: np.sin(a))
            else:
                self.add_value_function(lambda a: np.sin(a)*amplitude)
        else:
            if amplitude is None:
                self.add_value_function(lambda a: average + np.sin(a))
            else:
                self.add_value_function(lambda a: average + np.sin(a)*amplitude)
        
    def clip(self, vmin=None, vmax=None):
        self.add_value_function(lambda a: np.clip(a, vmin, vmax))
        
# ====================================================================================================
# Simulation

class Simulation():
    def __init__(self, location, speed, acc_f):
        
        self.location = np.array(location, float)
        self.speed    = np.empty(self.shape, float)
        self.speed[:] = speed
        
        self.acc_f    = acc_f
        
        # Locations and speeds can be clipped
        
        self.location_limit = None
        self.speed_limit    = None
        
        
    @classmethod
    def Gravity(cls, location, speed, masses=1, G=1):
        
        def acc_f(t, loc, speed, GM):
            
            n = len(loc)
            vects = np.zeros((n, n, 3), float)
            for i in range(n-1):
                for j in range(i+1, n):
                    v = loc[i] - loc[j]
                    d = 1/max(.001, np.linalg.norm(v))
                    v3 = v*d*d*d
                    
                    vects[i, j] =   GM[i]*v3
                    vects[j, i] = - GM[j]*v3
                    
            return np.sum(vects, axis=0)
        
        GM = np.empty(np.shape(location)[:-1], float)
        GM[:] = masses
        GM *= G
        
        return cls(location, speed, acc_f=lambda t, l, s: acc_f(t, l, s, GM))
        
        
    @property
    def shape(self):
        return np.shape(self.location)
        
    def compute(self, t0, t1, steps=250, sub_steps=1):

        steps     = max(2, steps)
        sub_steps = max(1, sub_steps)
        
        locs = np.zeros(((steps,) + self.shape), float)
        spds = np.zeros(((steps,) + self.shape), float)
        accs = np.zeros(((steps,) + self.shape), float)
        
        loc = np.array(self.location)
        spd = np.array(self.speed)
        acc = self.acc_f(t0, loc, spd)
        
        locs[0] = loc
        spds[0] = spd
        accs[0] = acc
        
        t = t0
        dt = (t1 - t0)/(steps*sub_steps-1)
        
        for i_main in range(1, steps):
            
            for i_sub in range(sub_steps):
                
                # Speed variation
                ds   = acc*dt
                
                # Limit the speed variation if function exists
                if self.speed_limit is not None:
                    ds = self.speed_limit(spd + ds) - spd
                    
                # New locations
                if self.location_limit is None:
                    loc += (spd + ds/2)*dt
                else:
                    loc = self.location_limit(loc + (spd + ds/2)*dt)
                    
                # Update the speed
                spd += ds
                
                # New acceleration from this step
                t += dt
                acc = self.acc_f(t, loc, spd)
                
            locs[i_main] = loc
            spds[i_main] = spd
            accs[i_main] = acc
                    
        self.locations = locs
        self.speeds    = spds
        self.accs      = accs
        
        
# ====================================================================================================
# Random path

def random_curves(P0, P1, shape=(), steps=0, scale=0, seed=0):
    
    if seed is not None:
        np.random.seed(seed)
        
    if not hasattr(shape, '__len__'): shape = (shape,)
        
    a = np.empty(shape + (steps+2, 3), float)
    a[..., 0, :]  = P0
    a[..., -1, :] = P1
    
    if steps == 0:
        return a

    # Distances
    distances = np.linalg.norm(a[..., -1, :]-a[..., 0, :], axis=-1)
    
    # Fixed intermediary points
    
    arg = np.arange(1, steps+1)
    inc = (a[..., -1, :] - a[..., 0, :])/(steps+1)
    
    a[..., 1:-1, :] = a[..., 0, :].reshape(shape + (1, 3)) + (arg.reshape(steps, 1)) * (inc.reshape(shape + (1, 3)))
    
    # Dispersions around
    radius = distances/(steps+1)*scale
    a[..., 1:-1, :] += dispersion(radius.reshape(shape + (1, 1)), shape + (steps,), seed=None)
        
    return a

    
# ====================================================================================================
# Retime

class Retime():
    
    # ---------------------------------------------------------------------------
    # Initialize with the precomputed parameters
    
    def __init__(self, shape, target, prologs, epilogs, widths, cum_widths=None):
        """Initialize with the arrays of intervals. 
        
        The source time t is retimed into the target interval.
        t is first compared to the interval [prolog, epilog]:
            - Out of this interval return time out of the target interval
            - Within this interval, return a time proportional into the target time interval
            
        Example:
            - target = (100, 200)
            - prolog = 10 epilog = 20
                t =  0 --> 90
                t = 10 --> 100
                t = 15 --> 150
                t = 20 --> 250
                t = 21 --> 251
                
        The prologs and epilogas ca vary for each item
        The (prolog, epilog) interval can be splitten in sub intervals
        The sub intervals can have variable widths
            

        Parameters
        ----------
        shape : tuple of ints
            Shape of the array of the retimed items.
        target : tuple of floats
            target Time interval.
        prologs : array of floats
            Starts of the retiming into the target.
        epilogs : array of floats
            Ends of the retiming.
        widths : array of floats
            If fix widths, one width per item. If widths are variable, the widths is a 2D array
            with a sequence of widths per item. Widths must be calculated in order to fill
            exactly the intervals epilogs - prologs.
        cum_widths : array of floats, optional
            The cumulative sums of the widths when the widths are variable. The default is None.

        Returns
        -------
        None.
        """
        
        self.shape  = shape
        self.count  = 1 if self.shape == () else np.product(self.shape)
        self.target = target
        
        self.prologs = prologs
        self.epilogs = epilogs
        
        self.widths = widths

        self.fix_width = cum_widths is None
        if self.fix_width:
            self.ratios = (self.target[1] - self.target[0])/self.widths
        else:
            self.cum_widths = cum_widths
            self.ratios = ((self.target[1] - self.target[0]) / self.widths).transpose().reshape(np.size(self.widths))

        self.return_ratio = False


    # ---------------------------------------------------------------------------
    # Random mapping

    @classmethod    
    def Random(cls, shape, target=(0, 100), prolog=(0, 0), epilog=(100, 0), intervals=(1, 0), width_scale=None, seed=0):
        
        # CAUTION: in the initialization, count is the average number of intervals
        # per item when self.count is the number of items.
        # shape is memorized as an attribute but the arrays are lineary shapes with self.count

        shape = tuple(shape) if hasattr(shape, '__len__') else (shape,)
        count = 1 if shape == () else np.product(shape)
        
        if not hasattr(prolog, '__len__'): prolog = (prolog, 0)
        if not hasattr(epilog, '__len__'): epilog = (epilog, 0)
        if not hasattr(intervals, '__len__'):  intervals = (intervals, 0)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Target start and end
        prologs = shaped_array(count, (), value = prolog[0], scale = prolog[1])
        epilogs = shaped_array(count, (), value = epilog[0], scale = epilog[1])
        
        # Durations must be positive
        durations = np.clip(epilogs - prologs, (epilog[0] - prolog[0])/intervals[0]/10, None)
        
        # Correct the end
        epilogs = prologs + durations
        
        # Number of intervals
        counts  = np.clip(shaped_array(count, (), value = intervals[0], scale = intervals[1]).astype(int), 1, None)
        
        print("counts", counts)
        
        # And the resulting withs
        widths = durations / counts
        
        # Some variations in the widths
        fix_width = width_scale is None
        if fix_width:
            widths     = widths
            cum_widths = None
            #self.ratios = (self.target[1] - self.target[0])/self.widths
        else:
            max_count = np.max(counts)
            min_width = np.min(widths)/10
            widths = np.clip(np.random.normal(widths, width_scale, (max_count, count)), min_width, None)
            
            # ---- Adjust to fit on epilogs (cum_widths[counts] = durations)
            
            cum_widths = np.zeros((max_count+1, count), float)
            cum_widths[1:] = np.cumsum(widths, axis=0)
            
            inds = np.arange(count)*(max_count+1) + counts
            ends = cum_widths.transpose().reshape((max_count+1)*count)[inds]
            cum_widths *= durations / ends
            widths *= durations / ends
            
            #self.ratios = ((self.target[1] - self.target[0]) / self.widths).transpose().reshape(max_count*self.count)
            
        return cls(shape, target, prologs, epilogs, widths, cum_widths)

        
    @classmethod
    def Split(cls, shape, target=(0, 250), prolog=0, epilog=250, intervals=1, shuffle=False, width=None, seed=0):

        shape = tuple(shape) if hasattr(shape, '__len__') else (shape,)
        count = 1 if shape == () else np.product(shape)
        
        total = epilog - prolog
        if width is None:
            w = total / count
        else:
            w = width 
            
        if count == 1:
            prologs = np.array([prolog], float)
        else:
            prologs = prolog + np.arange(count) * ((total - w) / (count-1))
        
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(prologs)
            
        epilogs = prologs + w
        widths  = np.ones(count, float)*w/intervals
        #retime.ratios[:] = (retime.target[1] - retime.target[0])/w
        
        return cls(shape, target, prologs, epilogs, widths)
    
        
    def __call__(self, t):
        
        if self.return_ratios:
            ret = np.zeros((2, self.count), float) # 0: ratios, 1: % within duration
            ret[1] = np.clip((t-self.prologs)/(self.epilogs-self.prologs), 0, 1)
        
        # ----- Fix widths : a simple modulo
        
        if self.fix_width:
            
            v = self.target[0] + ((t - self.prologs) % self.widths)*self.ratios
            if self.return_ratios:
                ret[0] = self.ratios
            
        # ----- Var widths : must find the right intervals
        else:
            max_count = len(self.widths)
            
            dt    = t - self.prologs
            delta = t - self.prologs - self.cum_widths
            inds = np.clip(np.argmin(delta > 0, axis=0)-1, 0, max_count-1)
            
            i_winds = np.arange(self.count)*max_count + inds
            i_xinds = np.arange(self.count)*(max_count+1) + inds
            
            x = self.cum_widths.transpose().reshape((max_count+1)*self.count)[i_xinds]
            
            if self.return_ratios:
                ret[0] = np.array(self.ratios[i_winds])
            
            v = self.target[0] + (dt - x)*self.ratios[i_winds]
            
        # ----- Before the prologs
        
        bef = t <= self.prologs
        v[bef] = self.target[0] + t - self.prologs[bef]
        
        # ----- After the epilogs
        
        aft = t >= self.epilogs
        v[aft] = self.target[1] + t - self.epilogs[aft]
        
        if self.return_ratios:
            ret[0, bef] = 1
            ret[0, aft] = 1
            
        # ----- Return the shaped values
        
        if self.return_ratios:
            return v.reshape(self.shape), ret.reshape((2,) + self.shape)
        else:
            return v.reshape(self.shape)
        
# ====================================================================================================
# DEBUG / DEMO

def demo_random_curves(steps=3, scale=.6, shape=(), seed=None):

    import matplotlib.pyplot as plt
    from bezier import Beziers
    
    size = int(np.product(shape))
    v0 = random_curves(0, 1, shape=shape, steps=steps, scale=scale, seed=seed)
    
    beziers = Beziers(v0)
    print(beziers)
    beziers.plot(points=False, curves=True)

    #v1 = random_curve(0, 1, steps=steps, scale=scale, in_disk=False, seed=seed)
    
    fig, ax = plt.subplots()
    
    ax.set_aspect('equal')
    
    if shape == ():
        ax.plot(v0[0:, 0], v0[:, 1], '.-')
    else:
        v1 = np.reshape(v0, (size, v0.shape[-2], v0.shape[-1]))
        for i in range(5):
            ax.plot(v1[i, 0:, 0], v1[i, :, 1], '.-')
        
    plt.show()
    
def demo_motion():
    
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 10, 100)
    
    count = 10
    ndim = 2
    val  = np.arange(ndim)
    speed = np.ones(ndim)
    acc = np.ones(ndim)*.4
    end = val + 10
    
    motion = Motion.Constant(count, val, value_scale=.1)
    print(motion, motion(t).shape, motion(1).shape)
    #print(motion(t))
    
    motion = Motion.Speed(count, val, speed, value_scale=.1, speed_scale=.5)
    print(motion, motion(t).shape, motion(1).shape)
    #print(motion(t))
    
    motion = Motion.Accelerated(count, val, speed, value_scale=.1, speed_scale=.5, acc=acc, acc_scale=.5)
    print(motion, motion(t).shape, motion(1).shape)
    #print(motion(t))
    
    motion = Motion.FromTo(val, np.random.normal((10, 10), (5, 5), (3, 2)), duration=10, speed=2)
    print(motion)
    print(motion, motion(t).shape, motion(1).shape)
    
    
    ps = motion(t)
    if len(ps.shape) == 2:
        for i in range(ndim):
            plt.plot(t, ps[:, i])
    else:
        ct = np.product(ps.shape[1:-1])
        ps = np.reshape(ps, (ps.shape[0], ct, ps.shape[-1]))
        for c in range(ct):
            for i in range(ndim):
                plt.plot(t, ps[:, c, i])
    
    plt.show()
    
#demo_motion()

def demo_sim():
    
    locs = np.array(((0, 0, 0), (10, 0, 0), (0, 10, 0)))
    spds = np.array(((0, 1, 0), (0, -1, 0), (1, .5, 0)))
    
    sim = Simulation.Gravity(locs, spds, masses= [1, 1.1, 1.3], G=12)
    
    count = 1000
    sim.compute(0, 45, count, 100)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    ax.plot(sim.locations[:, 0, 0], sim.locations[:, 0, 1])
    ax.plot(sim.locations[:, 1, 0], sim.locations[:, 1, 1])
    ax.plot(sim.locations[:, 2, 0], sim.locations[:, 2, 1])
    
    plt.show()
            
        
#demo_sim()        
        

    

    