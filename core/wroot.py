#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:27:03 2022

@author: alain
"""

import numpy as np

# ----------------------------------------------------------------------------------------------------
# Root for BWrapp classes. Used to offer utility methods

class WRoot():
    
    # ---------------------------------------------------------------------------
    # Dump the contant
    
    @staticmethod
    def _dump_title(title, indent):
        return format(title, f"<{indent}s") + ": "
    
    def _dump_yield(self):
        for k in dir(self):
            if k[:1] != "_":
                v = getattr(self, k)
                if type(v).__name__ != 'method':
                    
                    s =self._dump_title(k, 15)
                    if hasattr(self, f"_dump_{k}"):
                        yield eval(f"self._dump_{k}(title='{k}', indent=15)")
                    elif issubclass(type(v), WRoot):
                        yield s + "<WRoot>"
                    else:
                        yield s + f"{v}"
                        
    def _dump(self, title="_dump", indent=""):
        print(indent + title)
        for line in self._dump_yield():
            lines = line.split("\n")
            for l in lines:
                print(indent + l)
    
    # ---------------------------------------------------------------------------
    # Return an array of indices by concatenating the series of successice values
    # [1 2 3 8 10 11 20 21 22 23 24 30] -> [1-3 8 10 11 20-24 30]
    
    @staticmethod
    def _str_indices(inds):
        
        if inds is None:
            return "None"
        
        if len(inds) == 0:
            return "[]"
        
        if len(inds) == 1:
            return f"[{inds[0]}]"
        
        def sint(i0, i1):
            s = f"{i0}"
            if i1 == i0+1:
                s += f" {i1}"
            elif i1 > i0:
                s += f"-{i1}"
            return s
        
        s = "["
        sep = ""
        
        for index, i in enumerate(inds):
            
            if index == 0:
                i0 = i
                i1 = i
                
            elif i != i1+1:
                s += sep + sint(i0, i1)
                sep = " "
                    
                i0 = i
                i1 = i
                
            else:
                i1 = i
                
            if index == len(inds)-1:
                s += sep + sint(i0, i1)
                

        return s + "]"
    
    # ---------------------------------------------------------------------------
    # An array of floats to string
    
    @staticmethod
    def _str_list(values, fmt="7.2f", brackets=True, sepa=""):
        
        count = len(values)
        if count > 10:
            ranges = [range(5), range(count-5, count)]
            range_sepa = f" ...({count})..."
        else:
            ranges = [range(count)]
            range_sepa = ""
            
        s = "[" if brackets else ""
        for rg in ranges:
            for i in rg:
                s += format(values[i], fmt) + sepa
            s += range_sepa
            range_sepa = ""
            
        return s + ("]" if brackets else "")
    
    # ---------------------------------------------------------------------------
    # Print an array of ints
    
    @staticmethod
    def _str_ints(ints, fmt="2d", brackets=True, sepa=""):
        return WRoot._str_list(ints, fmt, brackets, sepa)

    # ---------------------------------------------------------------------------
    # Print an array of floats
    
    @staticmethod
    def _str_floats(floats, fmt="7.2f", brackets=True, sepa=""):
        return WRoot._str_list(floats, fmt, brackets, sepa)
    
    # ---------------------------------------------------------------------------
    # Print a matrix
    
    @staticmethod
    def _str_array(m, fmt="7.2f", title="", indent=0):
        
        shape = np.shape(m)
        
        if title == "":
            prefix = " "*indent
        else:
            indent = max(indent, len(title)+2)
            prefix = format(title+":", f"<{indent}s")
        
        if len(shape) == 0:
            return prefix + "[]"
        
        if len(shape) == 1:
            return prefix + WRoot._str_list(m, fmt, brackets=True, sepa="")
        
        s = prefix + f"[ {np.shape(m)}\n"
        indent += 1
        prefix = " "*indent
        
        count = shape[0]
        if count > 10:
            ranges = [range(5), range(count-5, count)]
            range_sepa = f"\n{prefix}...({count})...\n\n"
        else:
            ranges = [range(count)]
            range_sepa = ""
            
        for rg in ranges:
            for k in rg:
                s += WRoot._str_array(m[k], fmt, title="", indent=indent) + "\n"
            s += range_sepa
            range_sepa = ""
                
        return s + (" "*(indent-1)) + "]"
    
        
