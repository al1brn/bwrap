#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 16:39:44 2021

@author: alain
"""

import numpy as np

from ..core.text.fonts import Font
from ..core.text.ftext import FText


from ..core.maths import geometry as geo
from .crowd import Crowd

from ..wrappers.geometry import Geometry
from ..wrappers.wrap_function import wrap

from ..core.commons import WError


# =============================================================================================================================
# Chars is a crowd of characters
# The geometries of the characters are stored as parts of a geometry in their order

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
            
        super().__init__(Geometry(char_type), name=name)
        
        # ----- The model
        
        self.wmodel = wrap(text_object)
        
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
        
        self.ftext = None
        
        self.chars_locked = 0
        self.lock_chars()
        
        if len(self.font) == 1:
            self.text = "Chars"
        else:
            s = "Chars"
            for i in range(len(self.font)):
                s += f"\nIndex {i}"
            self.text = s
            for i in range(len(self.font)):
                self.set_font(i, self.look_up(para_index=i+1), align=False)
            self.align()
            
        self.unlock_chars(align=True)
            
    # ---------------------------------------------------------------------------
    # Reset the chars
            
    def reset(self):
        self.rebuild = True
        
    # ---------------------------------------------------------------------------
    # Lock / unlock chars update
    
    def apply(self):
        
        if self.geometry.parts_count == 0:
            return

        super().apply()
        
        if self.rebuild:
            self.rebuild = False
            self.build_object(self.wobject)
        else:
            self.update(self.wobject)
    
    def lock_chars(self):
        self.lock()
        self.chars_locked += 1
        
    def unlock_chars(self, align=False):
        if self.chars_locked == 0:
            raise WError("Chars lock / unlock error. Trying to unlock unlocked state.",
                         Class = "Chars", Method="Unlock")
            
        self.chars_locked -= 1
        if self.chars_locked == 0:
            self.set_chars(align=align)
            
        self.unlock()
        
    # ===========================================================================
    # Text format interface
    
    @property
    def topleft(self):
        return self.ftext.text_format.topleft
    
    @topleft.setter
    def topleft(self, value):
        self.ftext.text_format.topleft = value
        
        self.reset()
        self.set_chars(align=True)
        
    @property
    def align_width(self):
        return self.ftext.text_format.width
    
    @align_width.setter
    def align_width(self, value):
        self.ftext.text_format.width = value
        
        self.reset()
        self.set_chars(align=True)
        
    @property
    def align_x(self):
        return self.ftext.text_format.align_x
    
    @align_x.setter
    def align_x(self, value):
        self.ftext.text_format.align_x = value
        
        self.reset()
        self.set_chars(align=True)
        
    @property
    def font_scale(self):
        return self.ftext.text_format.scale
    
    @font_scale.setter
    def font_scale(self, value):
        self.ftext.text_format.scale = value
        
        self.reset()
        self.set_chars(align=True)
        
    @property
    def interchars(self):
        return self.ftext.text_format.interchars
    
    @interchars.setter
    def interchars(self, value):
        self.ftext.text_format.interchars = value
        
        self.reset()
        self.set_chars(align=True)
        
    @property
    def interlines(self):
        return self.ftext.text_format.interlines
    
    @interlines.setter
    def interlines(self, value):
        self.ftext.text_format.interlines = value
        
        self.reset()
        self.set_chars(align=True)

    @property
    def char_x_base(self):
        return self.ftext.text_format.char_x_base
    
    @char_x_base.setter
    def char_x_base(self, value):
        self.ftext.text_format.char_x_base = value
        
        self.reset()
        self.set_chars(align=True)
    
    # ===========================================================================
    # glyphes parameters
        
    @property
    def mesh_delta(self):
        return self.font.raster_delta
    
    @mesh_delta.setter
    def mesh_delta(self, value):
        
        self.font.raster_delta = max(1, int(value))
        
        self.reset()
        self.set_chars()
        
    @property
    def mesh_high_geometry(self):
        return not self.font.raster_low
    
    @mesh_high_geometry.setter
    def mesh_high_geometry(self, value):
        
        self.font.raster_low = not value
        
        self.reset()
        self.set_chars()
        
    # ===========================================================================
    # Display plane
    
    @property
    def display_plane(self):
        return self.font.display_plane
    
    @display_plane.setter
    def display_plane(self, value):
        self.font.display_plane = value
        
    # ===========================================================================
    # Mesh or Curve

    @property
    def char_type(self):
        return self.geometry.type

    # ===========================================================================
    # Access to stack by couple (char, font index)
    
    def char_index_OLD(self, char, font=None):
        
        if font is None:
            font = self.font.font_index
        
        index = self.chars_index.get((char, font))
        
        if index is None:

            index = len(self.stack)            
            self.chars_index[(char, font)] = index
            
            if self.char_type == 'Mesh':
                self.stack.stack(MeshCharStacker(char, self.font[font].mesh_char(char, text_format=self.ftext.text_format, return_faces=True)))
            else:
                self.stack.stack(CurveCharStacker(char, self.font[font].curve_char(char, text_format=self.ftext.text_format)))

        return index
    
    # ====================================================================================================
    # Set the chars metrics
    # The metrics comes from the font
        
    def set_metrics(self, use_dirty=False):
        
        if self.ftext is None:
            return
        
        self.ftext.set_metrics(self.font, use_dirty=use_dirty)
    
    
    # ====================================================================================================
    # Set the chars from the formatted text in ftext
    
    def set_chars(self, align=False):
        
        if self.chars_locked > 0:
            return
        
        # ----- Set the metrics
        
        self.set_metrics(use_dirty=True)
        
        # ----- Create the geometry
        
        type = self.geometry.type
        geo  = Geometry(self.geometry.type)
            
        # ----- Loop on the chars
        
        for c_i, c in enumerate(self.ftext.array_of_chars):
            fmt_char = self.ftext.get_char_format(c_i)
            geo.join(self.font[fmt_char.font].geometry(c, fmt_char, self.ftext.text_format, type), as_part=True)
            
        # ----- The matrices
        
        del self.geometry
        self.geometry = geo
        
        self.resize((geo.parts_count,))
            
        # ----- Object must be rebuilt
            
        self.rebuild = True
            
        
    # ====================================================================================================
    # Set the chars from the formatted text in ftext
    #
    # The glyphes are stored in a dictionnaty indexed by the couples (char, font index)
    # The stack is initialized with the geometry found in this dictionnary
    # The object geometry (Mesh or Curve) is initialized from the stack without formatting
    # The individual glyphes must be then changed to take the formatting into account.
    # The changes are made in the block (one block per character)
    
    def set_chars_OLD(self, align=False):
        
        if self.chars_locked > 0:
            return
        
        # ----- Set the metrics
        
        self.set_metrics(use_dirty=False)
        
        # ----- Store the (chars, font) in the dictionnary
        
        self.stack.stack_indices = [self.char_index(c, font) for c, font in zip(self.ftext.array_of_chars, self.ftext.fonts)]
        
        # ----- Initialized the geometry
        
        self.init_from_stack()
        
        # ----- Update the glyphes
        
        self.ftext.font_changed = False # Avoiding infinite loop is better
        self.ftext.set_dirty_format()
        
        self.update_glyphes(align=align)
        
        
    # ====================================================================================================
    # Update the glyphes to take the char formatting into account
    
    def update_glyphes(self, align=False):
        
        if self.chars_locked:
            return
        
        # ---------------------------------------------------------------------------
        # If some font indices were changed, we must rebuild the full geometry
        
        if self.ftext.font_changed:
            
            self.set_chars()
            
        # ---------------------------------------------------------------------------
        # Geomtry unchanged, we need only to take the formatt!ng instructions
        # into account
        
        else:
            
            # ----- Recompute metrics of dirty characters
        
            self.set_metrics(use_dirty=True)
            
            # ----- Loop on the dirty chars
            
            for c_i, c in enumerate(self.ftext.array_of_chars):
                
                if self.ftext.dirty[c_i]:
                    
                    fmt_char = self.ftext.get_char_format(c_i)
                    self.font[fmt_char.font].update_geometry(self.geometry, c_i, c, fmt_char, self.ftext.text_format)
                        
        # ---------------------------------------------------------------------------
        # realign if required                        
                    
        if align:
            self.align()
            
        self.ftext.reset_dirty()     
        
    # ====================================================================================================
    # Text property
        
    @property
    def text(self):
        if self.ftext is None:
            return ""
        else:
            return self.ftext.text

    # ---------------------------------------------------------------------------
    # Setting the text
        
    @text.setter
    def text(self, text):
        
        self.lock()
        
        # ----- Format the raw text. Keeps the text_format parameters
        
        old = self.ftext
        self.ftext = FText(text)
        if old is not None:
            self.ftext.text_format.copy(old.text_format)
            del old
        
        # ----- Create the geometry of the crowd of chars
        
        self.set_chars(align=True)
        
        # ----- Unlock
        
        self.unlock()
        
        
    # ====================================================================================================
    # Compute the chars locations to align the text
    
    def align(self, width=None, align_x=None):
        
        if self.chars_locked > 0:
            return
        
        self.lock()
        
        # ----- Set the metrics
        # Normally useless, but doesn't work otherwise
        self.set_metrics(False)
        
        # ----- Compute the alignment
        
        self.ftext.align()
        
        # ----- Transfer to the transformation matrices
        
        plane = 'XY' if self.char_type == 'Curve' else self.display_plane
        
        if plane == 'XZ':
            ix = 0
            iy = 2
        elif plane == 'YZ':
            ix = 1
            iy = 2
        else:
            ix = 0
            iy = 1
            
        loc = np.zeros(3, float)
        for i in range(len(self.ftext)):
            loc_xy = self.ftext[i].location
            loc[ix] = loc_xy[0]
            loc[iy] = loc_xy[1]
            self[i] = geo.tmatrix(location=loc)

        self.unlock()
        
        
    # ===========================================================================
    # Char formatting
    
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
        
    def set_font(self, font_index, char_indices=None, align=False):
        self.ftext.format_char(char_indices, 'font', font_index)
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
                
                
                
            
        
    
    
    
    
    
        
            
            
        
    
        
        
        
        
        
        
    
    
            
            
            
        
            
        
