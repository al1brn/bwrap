#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:24:15 2021

@author: alain
"""

from .wcurve import WCurve
from .wobject import WObject

class WCurveObject(WObject):

    # ---------------------------------------------------------------------------
    # Data as a curve
    
    @property
    def wcurve(self):
        wo = self.wrapped
        return WCurve(wo.data, wo.is_evaluated)
    
    # ---------------------------------------------------------------------------
    # Implement directly the array of wsplines
    
    def __len__(self):
        return len(self.wcurve)

    def __getitem__(self, index):
        return self.wcurve[index]
    

