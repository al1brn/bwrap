#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 16:39:44 2021

@author: alain
"""

import numpy as np

from .glyphe import CharFormat
from .fonts import Font
from .ftext import FText
from ..objects.stacker import Stack, MeshCharStacker, CurveCharStacker
from ..objects.crowd import Crowd
from ..wrappers.wrap_function import wrap
from ..maths import geometry as geo


from ..core.commons import WError

class Chars(Crowd):
    
    def __init__(self, text_object, char_type = 'Curve', name="Chars", var_blocks=True):
        
        if not char_type in ['Mesh', 'Curve']:
            raise WError(f"Chars initialization error: the chars type must be 'Curve' ort 'Mesh', not '{char_type}'",
                         Class = "Chars", Method = "Initialization",
                         text_object = text_object,
                         char_type = char_type)
            
        
        self.wtext = wrap(text_object)
        if self.wtext.object_type != 'TextCurve':
            raise WError(f"Chars initialization error: the object {self.wtext.name} must be a text, not a {self.wtext.object_type}",
                         Class = "Chars", Method = "Initialization",
                         text_object = text_object,
                         char_type = char_type)
            
        fontpath = self.wtext.font_file_path
        if fontpath is None:
            raise WError(f"Chars initialization error: the model text object '{self.wtext.name}' must use a True Type font, no the default Blender font.",
                         Class = "Chars", Method = "Initialization",
                         text_object = text_object,
                         char_type = char_type)
            
        self.font = Font(fontpath)
        if not self.font.loaded:
            raise WError(f"Chars initialization error: impossible to load the font '{fontpath}'. Try another one in text object '{self.wtext.name}'",
                         Class = "Chars", Method = "Initialization",
                         text_object = text_object,
                         char_type = char_type)
            
        super().__init__(char_type, shape=(), name=name, model=text_object, var_blocks=var_blocks)
        
        # ----- Stack of chars and indices in the stack
            
        self.stack = Stack(char_type, var_blocks=self.var_blocks)
        self.chars_index = {}
        
        # ----- Complementary
        
        if char_type == 'Curve':
            
            self.wobject.wdata.dimensions  = '2D'
            self.wobject.wdata.fill_mode   = 'BOTH'
            
            self.wobject.data.resolution_u        = self.wmodel.data.resolution_u
            self.wobject.data.render_resolution_u = self.wmodel.data.render_resolution_u
            self.wobject.data.offset              = self.wmodel.data.offset
            self.wobject.data.extrude             = self.wmodel.data.extrude
            
            self.wobject.data.taper_object        = self.wmodel.data.taper_object
            self.wobject.data.taper_radius_mode   = self.wmodel.data.taper_radius_mode
            
            self.wobject.data.bevel_mode          = self.wmodel.data.bevel_mode
            self.wobject.data.bevel_depth         = self.wmodel.data.bevel_depth
            self.wobject.data.bevel_resolution    = self.wmodel.data.bevel_resolution
            self.wobject.data.use_fill_caps       = self.wmodel.data.use_fill_caps
        
        # ---- The text
        
        self.ftext        = None
        
        self.align_x      = 'LEFT'
        self.align_y      = 'TOP'
        self.align_width  = None
        self.align_height = None
        
        self.text = "Chars"
            
    # ---------------------------------------------------------------------------
    # Reset the chars
            
    def reset(self):
        self.stack.reset()
        self.chars_index = {}
        
    # ---------------------------------------------------------------------------
    # glyphes parameters
        
    @property
    def mesh_delta(self):
        return self.font.font.raster_delta
    
    @mesh_delta.setter
    def mesh_delta(self, value):
        
        self.font.font.raster_delta = max(1, int(value))
        
        text = self.text
        self.reset()
        self.text = text
        
    @property
    def mesh_high_geometry(self):
        return not self.font.font.raster_low
    
    @mesh_high_geometry.setter
    def mesh_high_geometry(self, value):
        
        self.font.font.raster_low = not value
        
        text = self.text
        self.reset()
        self.text = text
        
    # ---------------------------------------------------------------------------
    # Mesh or Curve

    @property
    def char_type(self):
        return self.stack.object_type

    # ---------------------------------------------------------------------------
    # Access to stack by char
    
    def char_index(self, char):
        
        index = self.chars_index.get(char)
        
        if index is None:

            index = len(self.stack)            
            self.chars_index[char] = index

            if self.char_type == 'Mesh':
                self.stack.stack(MeshCharStacker(char, self.font.font.mesh_char(char, return_faces=True)))
            else:
                self.stack.stack(CurveCharStacker(char, self.font.font.curve_char(char)))
                
        return index
    
    # ---------------------------------------------------------------------------
    # Set an array of chars
    # Initialize the geometry from these chars
    
    def set_chars(self, chars):
        self.stack.stack_indices = [self.char_index(c) for c in chars]
        self.init_from_stack()

    # ---------------------------------------------------------------------------
    # Text property
        
    @property
    def text(self):
        if self.ftext is None:
            return ""
        else:
            return self.ftext.text

    # ---------------------------------------------------------------------------
    # Setting the text reset the stack and the text formatter
        
    @text.setter
    def text(self, text):
        
        self.lock()
        
        # ----- Format the raw text
        
        self.ftext = FText(text)
        
        # ----- Create the geometry of the crowd of chars
        
        self.set_chars(self.ftext.array_of_chars)
        
        # ----- Measure the chars
        
        self.set_metrics()
        
        # ----- Alignment
        
        self.align()
        
        # ----- Unlock
        
        self.unlock()
        
    # ---------------------------------------------------------------------------
    # Set the chars metrics to allow the alignment
    # By default, setting the metrics reset the chars because the glyphes
    # can change
        
    def set_metrics(self, use_dirty=False):
        
        if self.ftext is None:
            return
        
        self.ftext.set_metrics(self.font.font, use_dirty=use_dirty)
        
    # ---------------------------------------------------------------------------
    # Compute the chars locations
    # set_metrics() must have been called before
    
    def align(self, width=None, align_x=None):
        
        if self.ftext is None:
            return
        
        if width is None:
            width = self.align_width
        else:
            self.align_width = width
            
        if align_x is None:
            align_x = self.align_x
        else:
            self.align_x = align_x

        self.lock()
        
        self.ftext.align(width=width, align_x = align_x)
        
        loc = np.zeros(3, float)
        for i in range(len(self.ftext)):
            loc[:2] = self.ftext[i].location
            self[i] = geo.tmatrix(location=loc)

        self.unlock()
        
    # ===========================================================================
    # Char size
    
    @property
    def char_scale(self):
        return self.font.font.scale
        
    @char_scale.setter
    def char_scale(self, value):
        self.font.font.scale = value
        self.realign()
        
    # ===========================================================================
    # Format char by indices
    
    def update_glyphes(self, align=False):
        
        self.set_metrics(use_dirty=True)
        
        chars = self.ftext.array_of_chars
        for i in range(len(chars)):
            
            if self.ftext.dirty[i]:
                
                fmt_char = self.ftext.get_char_format(i)
                print("update glyples", fmt_char.bold_shift)
                
                #print("update_glyphes", i, chars[i], self.char_type, len(self.font.font.mesh_char(chars[i], fmt_char)))
                
                if self.char_type == 'Mesh':
                    self.set_block(i, self.font.font.mesh_char(chars[i], fmt_char))
                else:
                    self.set_block(i, self.font.font.curve_char(chars[i], fmt_char))
                    
        if align:
            self.align()
            
        self.ftext.reset_dirty()
                    
    
    def set_bold(self, bold, char_indices=None, align=False):
        self.ftext.format_char(char_indices, 'Bold', bold)
        self.update_glyphes(align=align)

    def set_shear(self, shear, char_indices=None, align=False):
        self.ftext.format_char(char_indices, 'Shear', shear)
        self.update_glyphes(align=align)
        
    def set_scale(self, scale, char_indices=None, align=False):
        self.ftext.format_char(char_indices, 'Scale', scale)
        self.update_glyphes(align=align)
        
    def set_xscale(self, scale, char_indices=None, align=False):
        self.ftext.format_char(char_indices, 'xScale', scale)
        self.update_glyphes(align=align)
        
    def set_yscale(self, scale, char_indices=None, align=False):
        self.ftext.format_char(char_indices, 'yScale', scale)
        self.update_glyphes(align=align)

    # ===========================================================================
    # Access to pieces of text
    
    def word_index(self, word, num=0):
        return self.ftext.word_index(word, num)
    
    def look_up(self, char=None, word=None, index=None, line_index=None, para_index=None, word_index=None, return_all=False, num=0):
        return self.ftext.look_up(char, word, index, line_index, para_index, word_index, return_all, num)

    # ===========================================================================
    # Shape
    
    @property
    def modifiers(self):
        return self.wobject.wrapped.modifiers
    
    def get_modifier(self, modifier_type, create=False):
        for m in self.modifiers:
            if m.type == modifier_type:
                return m
        
        if create:
            return self.modifiers.new(type=modifier_type, name=modifier_type.capitalize())
        
        return None
    
    @property
    def thickness(self):
        if self.char_type == 'Mesh':
            m = self.get_modifier('SOLIDIFY')
            if m is None:
                return 0
            else:
                return m.thickness
        else:
            return self.wobject.data.extrude
        
    @thickness.setter
    def thickness(self, value):
        if self.char_type == 'Mesh':
            m = self.get_modifier('SOLIDIFY', create=True)
            m.thickness = value
        else:
            self.wobject.data.extrude = value
            
            
    def set_bevel(self, bevel=True, width=.1, segments=4):
        if self.char_type == 'Mesh':
            if bevel:
                m = self.get_modifier('BEVEL', create=True)
                m.show_viewport = True
                m.show_render   = True
                m.width         = width
                m.segments      = segments
            else:
                m = self.get_modifier('BEVEL')
                if m is not None:
                    m.show_viewport = False
                    m.show_render   = False
        else:
            if bevel:
                self.wobject.data.bpy.bevel_depth      = width
                self.wobject.data.bpy.bevel_resolution = segments
            else:
                self.wobject.data.bpy.bevel_depth      = 0
                
                
                
            
        
    
    
    
    
    
        
            
            
        
    
        
        
        
        
        
        
    
    
            
            
            
        
            
        
