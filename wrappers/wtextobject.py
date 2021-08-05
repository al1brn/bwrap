#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:38:25 2021

@author: alain
"""

from .wtext import WText
from .wobject import WObject

from ..core.class_enhance import expose


class WTextObject(WObject):
    
    def __init__(self, wrapped, is_evaluated=None):
        super().__init__(wrapped, is_evaluated)
        self.wtext = WText(self.name, self.is_evaluated)
        
    def set_evaluated(self, value):
        self.is_evaluated = value
        self.wtext.is_evaluated = value
           
            
        

# Expose wmesh methods and properties  

expose(WTextObject, WText, "wtext", WText.exposed_methods(), WText.exposed_properties())  

    
    
