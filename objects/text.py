#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:27:19 2021

@author: alain
"""

import numpy as np
import bpy

from .crowd import Crowd

from ..core.commons import WError

class Text(Crowd):
    
    def __init__(self, text, model=None, name=None, as_curve=True):
        
        if name is None:
            name = "Wrap text"
        
        if model is None:
            model_name = name + " (Config)"
        elif type(model) is str:
            model_name = model
        elif hasattr(model, 'name'):
            model_name = model.name
        else:
            raise WError(f"The model must be a Blender Text object or the name of the Blender Text object, not {model}")
            
        self.text_model = bpy.data.objects.get(model_name)
        if model is None:
            bpy.ops.object.text_add()
            self.text_model = bpy.context.object
            self.text_model.data.body = ""
            self.text_model.name = model_name
            
        if type(self.text_model.data).__name__ != "TextCurve":
            raise WError(f"The model '{self.text_model}'must be a Blender Text object, not '{type(self.text_model.data).__name__}'")
            
            
        if as_curve:
            
            
            
            