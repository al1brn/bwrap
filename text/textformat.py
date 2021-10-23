#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:13:12 2021

@author: alain
"""

import numpy as np

from ..core.commons import WError


# =============================================================================================================================
# Char format
#
# Use to define the char format

class CharFormat():
    def __init__(self, bold=0., shear=0., scale=1., font=0):
        
        self.xscale     = 1.
        self.yscale     = 1.
        self.bold_shift = 0
        
        self.bold       = bold
        self.shear      = shear
        self.scale      = scale
        self.font       = font
        
        self.ch_space   = 0
        
    def __repr__(self):
        return f"<CharFormat bold:{self.bold} (shift:{self.bold_shift}), shear:{self.shear:.1f}, scale:{self.scale}, font:{self.font}>"
    
    @classmethod
    def Copy(cls, other):
        if other is None:
            return cls()
        else:
            copy = cls(bold=other.bold, shear=other.shear, scale=other.scale, font=other.font)
            copy.bold_shift = other.bold_shift
            return copy
    
    def scales(self, ratio=1., dim=2):
        if dim == 3:
            return (self.xscale*ratio, self.yscale*ratio, 1.)
        else:
            return (self.xscale*ratio, self.yscale*ratio)
    
    
    @property
    def scale(self):
        if self.xscale == self.yscale:
            return self.xscale
        else:
            return np.array([self.xscale, self.yscale], float)
        
    @scale.setter
    def scale(self, value):
        if hasattr(value, '__len__'):
            self.xscale = value[0]
            self.yscale = value[1]
        else:
            self.xscale = value
            self.yscale = value
            
            
# =============================================================================================================================
# Text format

class TextFormat():
    
    def __init__(self):
        
        # ------------------------------------------------------------
        # Global location
        
        self.topleft_     = np.zeros(2, float)
        self.width        = None
        self.height       = None
        
        self.align_x      = 'LEFT'
        self.align_y      = 'TOP'
        
        # ------------------------------------------------------------
        # Modifiers
        
        self.scale        = 1.
        self.interlines   = 0.
        self.interchars   = 0.
        
    # ===========================================================================
    # Repr
    
    def __repr__(self):
        sw = " None" if self.width is None else f"{self.width:.2f}"
        #sh = " None" if self.height is None else f"{self.height:.2f}"
        
        s = f"<Text format: top={self.topleft[1]:.2f}, left={self.topleft[0]:.2f}, width={sw}, align_x= {self.align_x}"
        s += f", scale:{self.scale:.2f}, interlines={self.interlines:.2f}, interchars={self.interchars}"
        
        return s

    # ===========================================================================
    # Copy
    
    def copy(self, other):
        
        self.topleft      = other.topleft
        self.width        = other.width
        self.height       = other.height
        
        self.align_x      = other.align_x
        self.align_y      = other.align_y
        
        # ------------------------------------------------------------
        # Modifiers
        
        self.scale        = other.scale
        self.interlines   = other.interlines
        self.interchars   = other.interchars
    
    
    # ===========================================================================
    # Top left property
        
    @property
    def topleft(self):
        return self.topleft_
    
    @topleft.setter
    def topleft(self, value):
        self.topleft_[:] = value
        
        
        
    
        
        
        
        
            
            
            
    
