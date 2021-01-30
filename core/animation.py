#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:12:35 2021

@author: alain
"""

import numpy as np

#"""
import bpy

from .blender import get_frame

"""

def get_frame(f):
    return f

#"""

# =============================================================================================================================
# Used to compare the current frame to a given interval
# CAUTION: the interval is closed on left side and open on right side:
# frame < start        --> -1
# start <= frame < end --> 0
# end <= frame         --> 1
# Interval is either a single interval or an array of intervals
# Random operations use a numpy random generator which can be set externally

class Interval():

    def __init__(self, start=None, end=None):
        self.frames = np.array([[Interval.get_frame(start, True), Interval.get_frame(end, False)]])
        self.rng_   = None

    @classmethod
    def Intervals(cls, frames):
        itv = Interval()
        itv.frames = Interval.to_frames(frames)
        return itv

    @staticmethod
    def to_frames(values):
        frames = np.array(values, np.float)
        count  = np.size(frames)
        return np.reshape(frames, (count//2, 2))

    @staticmethod
    def get_frame(frame, start=True):
        if frame is None:
            if start:
                return bpy.context.scene.frame_start
            else:
                return bpy.context.scene.frame_end
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
    # As an array of Intervals, int version

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
    # As an array of Intervals

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
    # Sort indices depending on the frame being before, during or after

    def before(self, frame):
        return np.where(self.frames[:, 0] >  frame)[0]

    def after(self, frame):
        return np.where(self.frames[:, 1] <= frame)[0]

    def during(self, frame):
        return np.where((self.frames[:, 0] > frame) and (frames[:, 1] <= frame))[0]

    # ---------------------------------------------------------------------------
    # Split the interval in intervals of equal length

    def split(self, count, duration=None):

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
        if keep_durations:
            durations = self.durations

        self.frames[:, 0] = starts

        if keep_durations:
            self.frames[:, 1] = starts + durations

        return self

    # ---------------------------------------------------------------------------
    # Change the ends

    def set_ends(self, starts, keep_durations=True):
        if keep_durations:
            durations = self.durations

        self.frames[:, 1] = ends

        if keep_durations:
            self.frames[:, 0] = ends - durations

        return self

    # ---------------------------------------------------------------------------
    # Change the centers

    def set_centers(self, centers):
        ofs = centers - (self.starts + self.durations/2)
        self.frames[:, 0] += ofs
        self.frames[:, 1] += ofs

        return self

    # ---------------------------------------------------------------------------
    # Clip the intervals
    # Clipping mode can be MOVE or CUT

    def clip(self, start=None, end=None, mode='MOVE'):
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
        self.rng.shuffle(self.frames)

        return self

    # ---------------------------------------------------------------------------
    # Generate normal durations

    def normal_durations(self, duration, scale, keep='CENTER'):
        return self.set_durations(
            self.rng.normal(duration, scale, len(self.frames)),
            keep=keep)

    # ---------------------------------------------------------------------------
    # Generate uniform durations

    def uniform_durations(self, min, max, keep='CENTER'):
        return self.set_durations(
            self.rng.uniform(min, max, len(self.frames)),
            keep=keep)

    # ---------------------------------------------------------------------------
    # Generate normal starts

    def normal_starts(self, start, scale, keep_durations=True):
        return self.set_starts(
            self.rng.normal(start, scale, len(self.frames)),
            keep_durations=keep_durations)

    # ---------------------------------------------------------------------------
    # Generate uniform starts

    def uniform_starts(self, min, max, keep_durations=True):
        return self.set_starts(
            self.rng.uniform(min, max, len(self.frames)),
            keep_durations=keep_durations)

    # ---------------------------------------------------------------------------
    # Generate normal centers

    def normal_centers(self, center, scale):
        return self.set_centers(
            self.rng.normal(center, scale, len(self.frames)))

    # ---------------------------------------------------------------------------
    # Generate normal centers

    def uniform_centers(self, min, max):
        return self.set_centers(
            self.rng.uniform(min, max, len(self.frames)))


# =============================================================================================================================
# Execution of an action during an interval on a list of objects
# action template is either f(object, frame) or f(frame) depending upon objects is None

class Animator():

    def __init__(self, action, objects=None, interval=Interval(), before=None, after=None):
        self.interval  = interval
        self.durations = interval.durations

        self.before   = before
        self.action   = action
        self.after    = after

        if objects is None:
            self.objects = None
        else:
            self.objects = np.array(objects)

    def animate(self, frame):

        if self.objects is None:
            w = self.interval.when(frame)

            if (w == -1) and (self.before is not None):
                self.before(frame)

            if (w == 0) and (self.action is not None):
                self.action(frame)

            if (w == 1) and (self.after is not None):
                self.after(frame)

        else:
            bef, dur, aft = self.interval.frame_locations(frame)
            factors = self.interval.factors(frame)

            if self.before is not None:
                for i in bef:
                    self.before(self.objects[i], frame=frame, start=self.interval.frames[i:, 0], end=self.interval.frames[i, 1], factor=factors[i])

            if self.action is not None:
                for i in dur:
                    self.action(self.objects[i], frame=frame, start=self.interval.frames[i:, 0], end=self.interval.frames[i, 1], factor=factors[i])

            if self.after is not None:
                for i in aft:
                    self.after(self.objects[i], frame=frame, start=self.interval.frames[i:, 0], end=self.interval.frames[i, 1], factor=factors[i])


    @staticmethod
    def hide(obj, frame):
        obj.hide_render   = True
        obj.hide_viewport = bpy.context.scene.bw_hide_viewport

    @staticmethod
    def show(obj, frame):
        obj.hide_render   = False
        obj.hide_viewport = False


    def Hide(cls, objects, interval):
        return Animator(cls.hide, objects, interval, cls.show, cls.show)

    def Show(cls, objects, interval):
        return Animator(cls.show, objects, interval, cls.hide, cls.hide)


# =============================================================================================================================
# Execution of an action during an interval on a list of objects

class Engine():

    SETUP     = [] # Setup functions (called once)
    FUNCTIONS = [] # Animations (called at frame change)

    verbose   = False

    # ---------------------------------------------------------------------------
    # Lists management

    @staticmethod
    def clear():
        Engine.SETUP     = []
        Engine.FUNCTIONS = []

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
    def animate():
        frame = bpy.context.scene.frame_current

        if Engine.verbose:
            print(f"Engine animation at frame {frame:6.1f}")

        for f in Engine.FUNCTIONS:
            f(frame)

    # ---------------------------------------------------------------------------
    # Set the animation global var

    @staticmethod
    def run(go=True):
        bpy.context.scene.bw_engine_animate = go
        if go:
            Engine.animate()

# =============================================================================================================================
# Execution of an action during an interval on a list of objects

def engine_handler(scene):
    if  bpy.context.scene.bw_engine_animate:
        Engine.animate()

# =============================================================================================================================
# Registering the module

def register():
    print("Registering animation")

    bpy.types.Scene.bw_engine_animate = bpy.props.BoolProperty(description="Animate at frame change")
    bpy.types.Scene.bw_hide_viewport  = bpy.props.BoolProperty(description="Hide in viewport when hiding render")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(engine_handler)


def unregister():
    bpy.app.handlers.frame_change_pre.remove(engine_handler)

if __name__ == "__main__":
    register()
