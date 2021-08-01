#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:12:35 2021

@author: alain
"""

import numpy as np

import bpy
from ..blender.frames import get_frame
from ..wrappers.wrap_function import wrap
from ..maths.interpolation import BCurve, Easing

from ..core.commons import WError

get_start_end_frame = lambda start: bpy.context.scene.frame_start if start else bpy.context.scene.frame_end

# =============================================================================================================================
# Used to compare the current frame to a given interval
# CAUTION: the interval is closed on left side and open on right side:
# frame < start        --> -1
# start <= frame < end --> 0
# end <= frame         --> 1
# Interval is either a single interval or an array of intervals
# Random operations use a numpy random generator which can be set externally

class Interval():
    """Frame interval or a series of intervals.
    
    An interval is closed on left side and open on right side. This allows
    to have a frame belonging to a unique interval within a series of
    contiguous intervals.
    
    The class offers global methods to create random series of intervals: various durations at random locations
    """

    def __init__(self, start=None, end=None):
        """Interval initialization with two values
        
        If bounds are None, use the start or end of the animation
        
        Parameters
        ----------
        start : float or marker label, default None
            Interval start. Left open if None
        end : float or marker label, default None
            Interval end. Right open if None
        """

        self.frames = np.array([[Interval.get_frame(start, True), Interval.get_frame(end, False)]])
        self.rng_   = None

    @classmethod
    def Intervals(cls, frames):
        """Build an interval from and array of couples.

        Parameters
        ----------
        frames : array of couple of float
            The intervals milestones.

        Returns
        -------
        Interval
            The created Interval instance.
        """
        
        itv = cls()
        itv.frames = Interval.to_frames(frames)
        return itv

    @staticmethod
    def to_frames(values):
        """Transform any array of float in an array of couples.

        Parameters
        ----------
        values : array of float of any shape
            Interpreted as an array of couples of floats.

        Returns
        -------
        array of couple of floats
            A valid series of intervals.
        """
        
        frames = np.array(values, np.float)
        count  = np.size(frames)
        return np.reshape(frames, (count//2, 2))

    @staticmethod
    def get_frame(frame, start=True):
        """Transform a frame specification in a numeric value.        

        Parameters
        ----------
        frame : int, float or marker label
            A frame in the animation.
        start : bool, optional
            If frame is None, returns the start or end of the animation according this argument. The default is True.

        Returns
        -------
        float
            The frame number.
        """
        
        if frame is None:
            return get_start_end_frame(start)
        else:
            return get_frame(frame)

    # ---------------------------------------------------------------------------
    # Random generator
    # Can be initialized externally for reproductibility

    @property
    def rng(self):
        if self.rng_ is None:
            self.rng_ = np.random.default_rng()
        return self.rng_

    @rng.setter
    def rng(self, value):
        self.rng_ = value
        
    # ---------------------------------------------------------------------------
    # Display

    def __repr__(self):
        if self.frames is None:
            return "<Interval with None frames !!!>"

        count = len(self.frames)
        if count == 1:
            return f"<Interval {self.start:6.1f} - {self.end:6.1f} ({(self.end - self.start):6.1f})>"

        s = f"<Intervals: {count:d}\n"
        idx = 0
        max_disp = 5
        for i in range(max_disp*2):
            if idx >= count:
                break

            if (i == max_disp) and (count > 2*max_disp):
                idx = count - max_disp
                s += "   ...\n"

            s += f"   {idx:3d}: {self.frames[idx, 0]:6.1f} - {self.frames[idx, 1]:6.1f} ({(self.frames[idx, 1] - self.frames[idx, 0]):.1f})\n"
            idx += 1

        return s + ">"

    # ---------------------------------------------------------------------------
    # As an array of intervals
    
    @property
    def single(self):
        return len(self.frames) == 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return Interval(self.frames[index, 0], self.frames[index, 1])

    # ---------------------------------------------------------------------------
    # As a single interval

    @property
    def start(self):
        return np.min(self.frames[:, 0])

    @property
    def end(self):
        return np.max(self.frames[:, 1])

    @property
    def duration(self):
        return self.end - self.start

    @property
    def center(self):
        return self.start + self.duration/2

    def factor(self, frame):
        fac = (frame - self.start)/self.duration
        return min(1, max(0, fac))

    # ---------------------------------------------------------------------------
    # As a single interval, int version

    @property
    def istart(self):
        return np.rint(self.start)

    @property
    def iend(self):
        return np.rint(self.end)

    @property
    def iduration(self):
        return self.iend - self.istart

    @property
    def icenter(self):
        return np.reint(self.center)

    # ---------------------------------------------------------------------------
    # As an array of Intervals

    @property
    def starts(self):
        return self.frames[:, 0]

    @property
    def ends(self):
        return self.frames[:, 1]

    @property
    def durations(self):
        return self.ends - self.starts

    @property
    def centers(self):
        return self.starts + self.durations/2

    def factors(self, frame):
        facts = (frame - self.starts)/self.durations
        return np.clip(facts, 0., 1.)

    # ---------------------------------------------------------------------------
    # As an array of Intervals, int version

    @property
    def istarts(self):
        return np.rint(self.frames[:, 0])

    @property
    def iends(self):
        return np.rint(self.frames[:, 1])

    @property
    def idurations(self):
        return self.iends - self.istarts

    @property
    def icenters(self):
        return np.rint(self.centers)

    # ---------------------------------------------------------------------------
    # Apend Intervals

    def append(self, other):
        if issubclass(type(other), Interval):
            fs = other.frames
        else:
            fs = Interval.to_frames(other)
        self.frames = np.concatenate((self.frames, fs))

        return self

    # ---------------------------------------------------------------------------
    # Return indices depending on the frame being before, during or after

    def before(self, frame):
        """Get the intervals before the current frame.
        
        Parameters
        ----------
        frame : float
            The current frame.
            
        Returns
        -------
        Array of int
            The indices of the intervals before the frame.
        """
        
        return np.where(self.frames[:, 0] >  frame)[0]

    def after(self, frame):
        """Get the intervals after the current frame.
        
        Parameters
        ----------
        frame : float
            The current frame.
            
        Returns
        -------
        Array of int
            The indices of the intervals after the frame.
        """
        
        return np.where(self.frames[:, 1] <= frame)[0]

    def during(self, frame):
        """Get the intervals including the current frame.
        
        Parameters
        ----------
        frame : float
            The current frame.
            
        Returns
        -------
        Array of int
            The indices of the intervals including the frame.
        """
        
        return np.where((self.frames[:, 0] > frame) and (self.frames[:, 1] <= frame))[0]
    
    def when(self, frame):
        """Position of intervals relatively to the current frame.
        
        For each interval:
            -1 : the frame is before the interval
             0 : the frame is within the interval
             1 : the frame is after the interval
        
        Parameters
        ----------
        frame : float
            The current frame.
            
        Returns
        -------
        Array of int
            -1, 0 or 1 for each interval.
        """
        
        w = np.zeros(len(self.frames), int)
        w[self.frames[:, 0] >  frame] = -1
        w[self.frames[:, 1] <= frame] =  1
        
        return w

    # ---------------------------------------------------------------------------
    # Split the interval in intervals of equal length

    def split(self, count, duration=None):
        """Split the interval in intervals of equals durations.
        
        Parameters
        ----------
        count : int
            Number of intervals to create
            
        duration : float, optional.
            If None, the current durations is divided. If not None, the final duration
            is duration*count. Default is None.
        
        Returns
        -------
        self
            Return self for chaining purpose.
        """

        lg = len(self.frames)

        durations = self.durations
        if duration is None:
            durs = np.resize(durations / count, lg)
        else:
            durs = np.full(lg, duration, np.float)

        frames = np.zeros((lg, count, 2), np.float)

        frames[:, :, 0] = np.linspace(self.starts, self.ends - durs, count).transpose()
        frames[:, :, 1] = frames[:, :, 0] + durs
        self.frames = np.reshape(frames, (lg*count, 2))

        return self

    # ---------------------------------------------------------------------------
    # Change the durations

    def set_durations(self, durations, keep='CENTER'):
        """Change the durations of the intervals.

        Parameters
        ----------
        durations : array of floats
            The new durtations.
        keep : str in ['START', 'CENTER', 'END'], optional
            How the duration is changed, either by keep one of the bounds, or
            by keeping the center. The default is 'CENTER'.

        Returns
        -------
        self
            Return self for chaining purpose.
        """

        if keep == 'START':
            self.frames[:, 1] = self.frames[:, 0] + durations
        elif keep == 'END':
            self.frames[:, 0] = self.frames[:, 1] - durations
        else:
            centers = self.starts + self.durations/2
            self.frames[:, 0] = centers - durations/2
            self.frames[:, 1] = centers + durations/2

        return self

    # ---------------------------------------------------------------------------
    # Change the starts

    def set_starts(self, starts, keep_durations=True):
        """Change the starts of the intervals. 

        Parameters
        ----------
        starts : array of float
            The new starting frames.
        keep_durations : bool, optional
            Move the interval ends to keep the durations. The default is True.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        if keep_durations:
            durations = self.durations

        self.frames[:, 0] = starts

        if keep_durations:
            self.frames[:, 1] = starts + durations

        return self

    # ---------------------------------------------------------------------------
    # Change the ends

    def set_ends(self, ends, keep_durations=True):
        """Change the ends of the intervals. 

        Parameters
        ----------
        ends : array of float
            The new ending frames.
        keep_durations : bool, optional
            Move the interval starts to keep the durations. The default is True.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        if keep_durations:
            durations = self.durations

        self.frames[:, 1] = ends

        if keep_durations:
            self.frames[:, 0] = ends - durations

        return self

    # ---------------------------------------------------------------------------
    # Change the centers

    def set_centers(self, centers):
        """Change the centers of the intervals. 

        Parameters
        ----------
        centers : array of float
            The new center frames.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        ofs = centers - (self.starts + self.durations/2)
        self.frames[:, 0] += ofs
        self.frames[:, 1] += ofs

        return self

    # ---------------------------------------------------------------------------
    # Clip the intervals
    # Clipping mode can be MOVE or CUT

    def clip(self, start=None, end=None, mode='MOVE'):
        """Clip the intervals with two bounds

        Parameters
        ----------
        start : float, optional
            Minimum values for starts. The default is None.
        end : float, optional
            Maximum values for ends. The default is None.
        mode : str, optional
            The default is 'MOVE'.
            'MOVE': the intervals are moved to be within the bounds.
            'CUT' : The intervals are cut at the bounds

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        start = self.get_frame(start, True)
        end   = self.get_frame(end, False)

        if mode == 'MOVE':
            ofs = np.maximum(start - self.frames[:, 0], 0)
            self.frames[:, 0] += ofs
            self.frames[:, 1] += ofs

            ofs = np.maximum(self.frames[:, 1] - end, 0)
            self.frames[:, 0] -= ofs
            self.frames[:, 1] -= ofs

        # Normally useless in MOVE mode but cut is necessary
        # for intervals longer than the bounds

        self.frames[:, 0] = np.maximum(self.frames[:, 0], start)
        self.frames[:, 1] = np.minimum(self.frames[:, 1], end)

        return self

    # ---------------------------------------------------------------------------
    # Shuffle the intervals

    def shuffle(self):
        """Shuffle the intervals

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        self.rng.shuffle(self.frames)

        return self

    # ---------------------------------------------------------------------------
    # Generate normal durations

    def normal_durations(self, duration, scale, keep='CENTER'):
        """Change the intervals durations following a normal distribution.

        Parameters
        ----------
        duration : float
            Average length of the intervals durations.
        scale : float
            Normal distribution scale.
        keep : str in ['START', 'CENTER', 'END'], optional
            How the duration is changed, either by keep one of the bounds, or
            by keeping the center. The default is 'CENTER'.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        return self.set_durations(
            self.rng.normal(duration, scale, len(self.frames)),
            keep=keep)

    # ---------------------------------------------------------------------------
    # Generate uniform durations

    def uniform_durations(self, min, max, keep='CENTER'):
        """Change the intervals durations following a uniform distribution.

        Parameters
        ----------
        min : float
            Min distribution bounds.
        max : float
            Max distribution bound.
        keep : str in ['START', 'CENTER', 'END'], optional
            How the duration is changed, either by keep one of the bounds, or
            by keeping the center. The default is 'CENTER'.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        return self.set_durations(
            self.rng.uniform(min, max, len(self.frames)),
            keep=keep)

    # ---------------------------------------------------------------------------
    # Generate normal starts

    def normal_starts(self, start, scale, keep_durations=True):
        """Change the starts following a normal distribution. 

        Parameters
        ----------
        start : float
            Center value of the normal distribution.
        scale : float
            Scale value of the normal distribuiton.
        keep_durations : bool, optional
            Move the interval ends to keep the durations. The default is True.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        return self.set_starts(
            self.rng.normal(start, scale, len(self.frames)),
            keep_durations=keep_durations)

    # ---------------------------------------------------------------------------
    # Generate uniform starts

    def uniform_starts(self, min, max, keep_durations=True):
        """Change the starts following a uniform distribution. 

        Parameters
        ----------
        min : float
            Min value of the uniform distribution.
        max : float
            Max value of the uniform distribuiton.
        keep_durations : bool, optional
            Move the interval ends to keep the durations. The default is True.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        return self.set_starts(
            self.rng.uniform(min, max, len(self.frames)),
            keep_durations=keep_durations)

    # ---------------------------------------------------------------------------
    # Generate normal centers

    def normal_centers(self, center, scale):
        """Change the centers following a normal distribution. 

        Parameters
        ----------
        center : float
            Center value of the normal distribution.
        scale : float
            Scale value of the normal distribuiton.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        return self.set_centers(
            self.rng.normal(center, scale, len(self.frames)))

    # ---------------------------------------------------------------------------
    # Generate normal centers

    def uniform_centers(self, min, max):
        """Change the centers following a uniform distribution. 

        Parameters
        ----------
        min : float
            Min value of the normal distribution.
        max : float
            Max value of the normal distribuiton.

        Returns
        -------
        self
            Return self for chaining purpose.
        """
        
        return self.set_centers(
            self.rng.uniform(min, max, len(self.frames)))
    
    # ---------------------------------------------------------------------------
    # Debug
    
    def plot(self, frame=None):
        """Plot the intervals for debug purposes
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        
        cols = ['b' for itv in self.frames]
        if frame is not None:
            ax.vlines(frame, 0, len(self.frames)-1, 'black', ':')
            w = self.when(frame)
            for i in range(len(w)):
                if w[i] < 0:
                    cols[i] = 'r'
                if w[i] > 0:
                    cols[i] = 'g'
        
        for i in range(len(self.frames)):
            ax.plot(self.frames[i], [i, i], cols[i])
            
        plt.show()
        
    
    @classmethod
    def demo(cls):
        """A simple demo of the class.
        """
        
        interval = cls()
        interval.split(10)
        #interval.shuffle()
        interval.normal_centers(125, 50)
        interval.uniform_durations(120, 2)
        interval.plot(125)
        
        print(interval.factors(125))
        



# =============================================================================================================================
# Execution of an action during an interval on a list of objects

class Engine():

    SETUP     = [] # Setup functions (called once)
    FUNCTIONS = [] # Animations (called at frame change)
    VARIABLES = {} # Variables mapped on objects properties
    ANIMATORS = [] # Animator driven animation

    verbose   = False
    scene     = None # Risky algorithm !!!

    # ---------------------------------------------------------------------------
    # Variables

    @staticmethod
    def map_variable(name, object, attribute):
        wo = wrap(object)
        Engine.VARIABLES[name] = (wo.name, attribute)

    @staticmethod
    def variable(name, scene=None):
        woa = Engine.VARIABLES.get(name)
        if woa is None:
            raise WError(f"Engine variable error: the variable named '{name}' is not mapped to an object property!",
                    Class = "Engine",
                    Static_method = "variable",
                    SETUP = Engine.SETUP,
                    FUNCTIONS = Engine.FUNCTIONS,
                    VARIABLES = Engine.VARIABLES,
                    ANIMATORS = Engine.ANIMATORS)

        if scene is None:
            scene = Engine.scene
        if scene is None:
            scene = bpy.context.scene

        wo = wrap(scene.objects[woa[0]])
        return wo.get_attr(woa[1])

    # ---------------------------------------------------------------------------
    # Lists management

    @staticmethod
    def clear():
        Engine.SETUP     = []
        Engine.FUNCTIONS = []
        Engine.ANIMATORS = []

    # Add an animation function

    @staticmethod
    def add(f):
        Engine.FUNCTIONS.append(f)

    # Add a setup function

    @staticmethod
    def add_setup(f):
        Engine.SETUP.append(f)

    # ---------------------------------------------------------------------------
    # Execute setup functions

    @staticmethod
    def setup(self):
        for f in Engine.SETUP:
            f()

    # ---------------------------------------------------------------------------
    # Animate
    # Animation is submitted to global var

    @staticmethod
    def animate(scene):
        frame = scene.frame_current

        if Engine.verbose:
            print(f"Engine animation at frame {frame:6.1f}")
            
        # Current scene as a global variable
        Engine.scene = scene
        
        # Functions
        for f in Engine.FUNCTIONS:
            f(frame)
        
        # Animators
        for a in Engine.ANIMATORS:
            a.animate(frame)
            
        # Reset global scene
        Engine.scene = None

    # ---------------------------------------------------------------------------
    # Is registered
    
    @staticmethod
    def register():
        if not hasattr(bpy.context.scene, "bw_engine_animate"):
            register()

    # ---------------------------------------------------------------------------
    # Set the animation global var

    @staticmethod
    def run(go=True):
        register()
        bpy.context.scene.bw_engine_animate = go
        if go:
            Engine.animate(bpy.context.scene)

    # ---------------------------------------------------------------------------
    # Go directly
    
    @staticmethod
    def go(f):
        Engine.clear()
        Engine.add(f)
        Engine.run(True)
            
            
# =============================================================================================================================
# Execution of an action during an interval on a list of objects
# action template is either f(object, frame) or f(frame) depending upon objects is None

class Animator():
    """Animate objects on intervals.
    
    The animate method is called at each frame. 
    """

    def __init__(self, objects, attribute, bcurve=BCurve(), interval=None):
        """Animate objects on intervals.

        Parameters
        ----------
        action : function of template action(frame, objects=None, xbounds=None, ybounds=None)
            Action to call at each frame. objects, xbounds and ybounds are used if
            not None. This allow to use simpler templates.
        objects : object or array of objects, optional
            The objects to call the action for. The default is None.
        interval : Interval, optional
            Interval per object to use. The default is None.

        Returns
        -------
        None.

        """
        self.objects   = objects
        self.attribute = attribute
        self.bcurve    = bcurve
        self.interval  = interval
        
        # Register the animator
        
        Engine.ANIMATORS.append(self)
        
    def __repr__(self):
        return f"<Animator:\nObjects:{self.objects}\nAttributes: {self.attribute}\nbcurve:{self.bcurve}"
        
    # ---------------------------------------------------------------------------
    # Set the attribute
    
    def set_attribute(self, name, value):
        if hasattr(self.objects, name):
            setattr(self.objects, name, value)
        else:
            for o in self.objects:
                setattr(o, name, value)
            
    # ---------------------------------------------------------------------------
    # Animation

    def animate(self, frame):
        
        xbounds = None
        ybounds = None
            
        if self.interval is not None:
            xbounds = self.interval.starts
            ybounds = self.interval.ends
            
        values = self.bcurve(frame, xbounds, ybounds)
        #print(f"Animate {int(frame):4d}:", values)
        
        if hasattr(self.attribute, '__len__'):
            for attr in self.attribute:
                self.set_attribute(attr, values)
        else:
            self.set_attribute(self.attribute, values)
            
    # ---------------------------------------------------------------------------
    # Add a keyframe
    
    def set_keyframe(self, frame, value, interpolation='BEZIER', ease='AUTO'):
        pt = (get_frame(frame), value)
        return self.bcurve.add(end_point=pt, easing=Easing(interpolation, ease))
        
    # ---------------------------------------------------------------------------
    # Constant (used for bool for instance)
    
    def set_constant(self, frame, value, set_before=False):
        fr = get_frame(frame)
        if set_before:
            self.set_keyframe(fr - 1, 1 - value, interpolation='CONSTANT')
        self.set_keyframe(fr, value, interpolation='CONSTANT')
        
    # ---------------------------------------------------------------------------
    # Hide / Show
    
    @classmethod
    def Hider(cls, objects):
       anim = Animator(objects, ['hide_viewport', 'hide_render'], BCurve())
       return anim
   
    def hide(self, frame, value=True, show_before=False):
        self.set_constant(frame, value, set_before=show_before)
        
    def show(self, frame, value=True, hide_before=False):
        self.set_constant(frame, not value, set_before=hide_before)            

# =============================================================================================================================
# Execution of an action during an interval on a list of objects

def engine_handler(scene):
    if False:
        # DEBUG MODE pour bw_engine_animate
        print('-'*100)
        print("--------- DEBUG IN animation module: engine_handler....")
        print('-'*100)
        
        Engine.animate(scene)
        return
    # END OF DEBUG
    
    if  scene.bw_engine_animate:
        Engine.animate(scene)

# =============================================================================================================================
# Registering the module

def register():
    print("Registering animation")
    if hasattr(bpy.context.scene, "bw_engine_animate"):
        print("Alreay registered...")
        return

    bpy.types.Scene.bw_engine_animate = bpy.props.BoolProperty(description="Animate at frame change")
    bpy.types.Scene.bw_hide_viewport  = bpy.props.BoolProperty(description="Hide in viewport when hiding render")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(engine_handler)
    print("Animation registered.")


def unregister():
    bpy.app.handlers.frame_change_pre.remove(engine_handler)

if __name__ == "__main__":
    register()
