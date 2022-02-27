#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:39:27 2022

@author: alain
"""

import importlib

def import_bwrap_module(file_name, module_name):
    
    bw_path = __file__[:__file__.find("bwrap")+5] + "/"
    
    spec   = importlib.util.spec_from_file_location(module_name, bw_path + file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module



