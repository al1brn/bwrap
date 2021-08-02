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
    
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self.wtext = WText(self.wrapped.data, self.wrapped.is_evaluated)
    
    """
    @property
    def wtext(self):
        wo = self.wrapped
        return WText(wo.data, wo.is_evaluated)
    """

# Expose wmesh methods and properties  

expose(WTextObject, WText, "wtext", WText.exposed_methods(), WText.exposed_properties())  

    
    
