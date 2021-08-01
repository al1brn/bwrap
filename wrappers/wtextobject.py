#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:38:25 2021

@author: alain
"""

from .wtext import WText
from .wobject import WObject


class WTextObject(WObject):
    
    @property
    def wtext(self):
        wo = self.wrapped
        return WText(wo.data, wo.is_evaluated)
    
    @property
    def text(self):
        """The text displayed by the object.

        Returns
        -------
        str
            Text.
        """
        
        return self.wtext.text

    @text.setter
    def text(self, value):
        self.wtext.text = value
        

    
    
