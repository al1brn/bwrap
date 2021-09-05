#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 16:39:44 2021

@author: alain
"""

import numpy as np

from .fonts import Font
from .aligner import Token
from ..objects.stacker import Stack, MeshCharStacker, CurveCharStacker
from ..objects.crowd import Crowd
from ..wrappers.wrap_function import wrap
from ..maths import geometry as geo


from ..core.commons import WError

class Chars(Crowd):
    
    def __init__(self, text_object, char_type = 'Curve', name="Chars"):
        
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
            
        super().__init__(char_type, shape=(), name=name, model=text_object)
        
        # ----- Stack of chars and indices in the stack
            
        self.stack = Stack(char_type)
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
        
        self.token        = None
        
        self.align_x      = 'LEFT'
        self.align_y      = 'TOP'
        self.align_width  = None
        self.align_height = None
        self.lines_       = None
        
        self.text = "Chars"
        
            
            
    def reset(self):
        self.stack.reset()
        self.chars_index = {}
        
    @property
    def mesh_delta(self):
        return self.font.font.raster_delta
    
    @mesh_delta.setter
    def mesh_delta(self, value):
        self.font.font.raster_delta = max(1, int(value))
        
    @property
    def mesh_high_geometry(self):
        return not self.font.font.raster_low
    
    @mesh_high_geometry.setter
    def mesh_high_geometry(self, value):
        self.font.font.raster_low = not value

    @property
    def char_type(self):
        return self.stack.object_type
    
    def char_index(self, char):
        
        index = self.chars_index.get(char)
        
        if index is None:

            index = len(self.stack)            
            self.chars_index[char] = index

            if self.char_type == 'Mesh':
                self.stack.stack(MeshCharStacker(char, self.font.font.mesh_char(char, return_uvmap=True)))
            else:
                self.stack.stack(CurveCharStacker(char, self.font.font.curve_char(char)))
                
        return index
    
    def set_chars(self, chars):
        self.stack.indices = [self.char_index(c) for c in chars]
        self.init_from_stack()
        
    @property
    def text(self):
        if self.token is None:
            return ""
        else:
            return self.token.text
        
    @text.setter
    def text(self, text):
        
        self.lock()
        
        self.token = Token(text)
        self.token.split()
        
        self.align()
        
        self.unlock()
        
    def align(self, width=None, align_x='LEFT', height=None, align_y='TOP'):

        self.align_x      = align_x
        self.align_y      = align_y
        self.align_width  = width
        self.align_height = height
        
        self.lock()
        
        self.reset()
        
        chars, xyw = self.token.align(metrics=self.font.font,
                            width=self.align_width, align_x=self.align_x,
                            height=self.align_height, align_y=self.align_y)
        
        self.set_chars(chars)
        
        for i in range(len(xyw)):
            loc = (xyw[i, 0], xyw[i, 1], 0)
            self[i] = geo.tmatrix(location=loc)

        self.unlock()
        
        # ----- Store the lines
        
        self.lines_ = [np.where(xyw[:, 1] == y)[0] for y in reversed(np.unique(xyw[:, 1]))]
        
    def realign(self):
        self.align(self.align_width, self.align_x, self.align_height, self.align_y)
        
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
    # Access to pieces of text
    
    @staticmethod
    def token_chars_indices(token):
        return [char.char_index for char in token.tokens(select=Token.CHAR)]
    
    @property
    def chars(self):
        return self.token.chars
    
    @property
    def words(self):
        return self.token.words
    
    @property
    def paras(self):
        return self.token.texts(Token.PARA)
    
    @property
    def chars_indices(self):
        return [token.char_index for token in self.token.tokens()]
    
    @property
    def words_indices(self):
        return [Chars.token_chars_indices(token) for token in self.token.tokens(select=Token.WORD)]
    
    @property
    def paras_indices(self):
        return [Chars.token_chars_indices(token) for token in self.token.tokens(select=Token.PARA)]
    
    @property
    def lines_indices(self):
        if self.lines_ is None:
            return [self.chars_indices]
        else:
            return self.lines_
    
    def chars_in(self, chars):

        if type(chars) is str:
            chars = [chars]
            
        the_chars = self.token.chars
        indices = []
        for i, c in enumerate(the_chars):
            if c in chars:
                indices.append(i)

        return indices
    
    def words_in(self, words):
        if type(words) is str:
            words = [words]
            
        the_words = self.token.tokens(Token.WORD)
        indices = []
        for i, w in enumerate(the_words):
            if w.text in words:
                indices.append(Chars.token_chars_indices(w))
                
        return indices
    
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
                
                
                
            
        
    
    
    
    
    
        
            
            
        
    
        
        
        
        
        
        
    
    
            
            
            
        
            
        
