#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:12:35 2021

@author: alain
"""

import numpy as np

#"""
import bpy

from .core.blender import get_frame

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

class Interval():
    
    def __init__(self, start=None, end=None):
        self.frames = np.array([[get_frame(start), get_frame(end)]])
        
    @classmethod
    def Intervals(cls, frames):
        itv = Interval()
        itv.frames = np.array(frames)
        count = np.size(itv.frames)
        itv.frames = np.reshape(itv.frames, (count // 2, 2))
        return itv
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        return self.frames[index]
    
    def __repr__(self):
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

    @property
    def start(self):
        return self.frames[0, 0]
    
    @property
    def end(self):
        return self.frames[0, 1]
    
    def when(self, frame, index = 0):
        if frame < self.frames[index, 0]:
            return -1
        elif frame >= self.frames[index, 1]:
            return 1
        return 0
    
    def frame_locations(self, frame):
        before = np.where(self.frames[:, 0] >  frame)[0]
        after  = np.where(self.frames[:, 1] <= frame)[0]
        during = np.delete(np.delete(np.arange(len(self.frames)), before), after)
        
        return before, during, after
    
    
    def duration(self, index = 0):
        return self.frames[index, 1] - self.frames[index, 0]
    
    def split(self, count, duration=None):

        tot_dur = self.duration()
        dur     = tot_dur/count if duration is None else duration

        frames = np.zeros((count, 2), np.float)

        if dur >= tot_dur:
            frames[:, 0] = self.start
            frames[:, 1] = self.end
        else:
            frames[:, 0] = self.start + np.linspace(0., tot_dur - dur, count)
            frames[:, 1] = frames[:, 0] + dur
        return self.Intervals(frames)
    
    def random(self, count, duration, scale, strict=False, seed=None):

        tot_dur = self.duration()
        
        if seed is not None:
            rng     = np.random.default_rng(seed)
        else:
            rng     = np.random.default_rng()
            
        durs    = rng.normal(duration, scale, count)
        starts  = rng.normal(tot_dur/2, max(0, tot_dur - duration*2), count) 

        frames  = np.zeros((count, 2), np.float)
        frames[:, 0] = self.start + starts
        frames[:, 1] = frames[:, 0] + durs
        
        if strict:
            ofs = np.maximum(self.start - frames[:, 0], 0.)
            frames[:, 0] += ofs
            frames[:, 1] += ofs
            
            ofs = np.maximum(frames[:, 1] - self.end, 0.)
            frames[:, 0] -= ofs
            frames[:, 1] -= ofs
            
        return Interval.Intervals(frames)


# =============================================================================================================================
# Execution of an action during an interval on a list of objects
# action template is either f(object, frame) or f(frame) depending upon objects is None

class Animator():
    
    def __init__(self, action, objects=None, interval=Interval(), before=None, after=None):
        self.interval = interval
        
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
            
            if self.before is not None:
                for i in bef:
                    self.before(self.objects[i], frame)
                    
            if self.action is not None:
                for i in dur:
                    self.action(self.objects[i], frame)
                
            if self.after is not None:
                for i in aft:
                    self.after(self.objects[i], frame)
                    
                    
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

    SETUP     = [] # Setup functions
    ANIMATORS = [] # Animations
    
    verbose       = False
    
    # ---------------------------------------------------------------------------
    # Lists management
    
    @staticmethod
    def clear():
        Engine.SETUP     = []
        Engine.ANIMATORS = []
        
    # Add an animator
        
    @staticmethod
    def add(f, animator):
        Engine.ANIMATORS.append(animator)
        
    # Add an action : action(frame)
        
    @staticmethod
    def add_action(action, interval=Interval(), before=None, after=None):
        Engine.add_animator(action, objects=None, interval=interval, before=before, after=after)
        
    # Add an action on objects
        
    @staticmethod
    def add_objects(action, objects, interval=Interval(), before=None, after=None):
        Engine.add_animator(Animator(action, objects, interval, before, after))
        
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
    def animate(self, frame):
        
        if not bpy.context.scene.wa_frame_exec:
            return
            
        frame = bpy.context.scene.frame_current
        
        if Engine.verbose:
            print(f"Engine animation at frame {frame:6.1f}")
            
        for animator in Engine.ANIMATORS:
            animator.animate(frame)


    # ---------------------------------------------------------------------------
    # Set the animation global var

    @staticmethod
    def run(go=True):
        bpy.context.scene.bw_engine_animate = go
        Engine.animate()
        
# =============================================================================================================================
# Execution of an action during an interval on a list of objects
            
def engine_handler(scene):
    if  bpy.context.scene.bw_engine_animate:
        Engine.execute()

# =============================================================================================================================
# Registering the module
        
def register():
    
    bpy.types.Scene.bw_engine_animate = bpy.props.BoolProperty(description="Animate at frame change")
    bpy.types.Scene.bw_hide_viewport = bpy.props.BoolProperty(description="Hide in viewport when hiding render")
   
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(engine_handler)
    

def unregister():
    bpy.app.handlers.frame_change_pre.remove(engine_handler)


if __name__ == "__main__":
    register()


 