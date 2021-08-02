#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:24:15 2021

@author: alain
"""

from .wcurve import WCurve
from .wobject import WObject

from ..core.class_enhance import expose


class WCurveObject(WObject):
    
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self.wcurve = WCurve(self.wrapped.data, self.wrapped.is_evaluated)

    # ---------------------------------------------------------------------------
    # Data as a curve
    """
    @property
    def wcurve(self):
        wo = self.wrapped
        return WCurve(wo.data, wo.is_evaluated)
    """    

    # ---------------------------------------------------------------------------
    # Implement directly the array of wsplines
    
    def __len__(self):
        return len(self.wcurve)

    def __getitem__(self, index):
        return self.wcurve[index]
    
# ===========================================================================
# Expose wmesh methods and properties  

expose(WCurveObject, WCurve, "wcurve", WCurve.exposed_methods(), WCurve.exposed_properties())  

    
    

