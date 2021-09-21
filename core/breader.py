#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:57:37 2021

@author: alain
"""

# =============================================================================================================================
# The types

class DataType():
    
    INT8    =  0
    INT16   =  1
    INT32   =  2
    INT64   =  3
    
    UINT8   = 10
    UINT16  = 11
    UINT32  = 12
    
    FIXED   = 20
    SFRAC   = 21

# -----------------------------------------------------------------------------------------------------------------------------
# A utility

def defs_str(obj, defs):
    s = ""
    sepa = ""
    for k in defs:
        s += sepa + f"{k}: {getattr(obj, k)}"
        sepa = ", "
    return s

# =============================================================================================================================
# Base structure being read 

class Struct():
    
    def __init__(self, reader, defs):
        
        reader.read_struct(self, defs)
        self._lock = 0
        
    def __repr__(self):
        
        if self._lock > 0:
            return "<self>"
        
        self._lock += 1
            
        s = f"<Struct {type(self).__name__}:\n"
        for k in dir(self):
            if (k[0] != '_') and ((k.upper() != k) or (len(k) >= 8)) and k not in ['ttf']:
                v = getattr(self, k)
                sv = f"{v}"
                if len(sv) > 30:
                    sv = sv[:30] + " ..."
                    if hasattr(v, '__len__'):
                        sv += f" ({len(v)})"
                s += f"     {k:25s}: {sv}\n"
                
        self._lock -= 1
                
        return s + ">"
    
# =============================================================================================================================
# Flags reading
# Flags are individual bits. They are transformed into bool properties

class Flags():
    def __init__(self, dtype, names):
        """Bits in a flag variable are nameds

        Parameters
        ----------
        dtype : int
            A valid type for the reader.
        names : array of string
            The property names to create in the structure. The order corresponds to the location
            of the bit. Use Non to ignore a particular bit/

        Returns
        -------
        None.
        """
        
        self.dtype = dtype
        self.names = names
        
    def read(self, reader, struct):
        
        flags = reader.read(self.dtype)
        
        bit = 1
        for i, nm in enumerate(self.names):
            if nm is not None:
                setattr(struct, nm, flags & bit > 0)
            bit = bit << 1
            
        return flags
    

# =============================================================================================================================
# The binary reader

# -----------------------------------------------------------------------------------------------------------------------------
# The base class and the node class

class BReader():
    
    def __init__(self, owner=None, base_offset=0, name="Binary reader"):
        
        self.owner       = owner
        self.base_offset = base_offset
        self.name        = name
        
        self.breaders    = {}
        self.current     = []
        
        self.opens       = 0
        self.stack       = []
        
        self.open()

    # =============================================================================================================================
    # Nodes management
        
    @property
    def top(self):
        return self if self.owner is None else self.owner
    
    def new_reader(self, name, base_offset):
        self.breaders[name] = BReader(self, base_offset, name)
        return self.breaders[name]
    
    @property
    def file_name(self):
        return self.top.file_name
    
    # =============================================================================================================================
    # Basics
        
    def open(self):
        self.opens += 1
        self.owner.open()
        self.offset = 0
        
    def close(self):
        
        if self.opens == 0:
            raise RuntimeError("BReader close error: impossible to close a closed reader")

        self.opens -= 1
        self.owner.close()
        
    @property
    def offset(self):
        return self.owner.offset - self.base_offset
    
    @offset.setter
    def offset(self, value):
        self.owner.offset = self.base_offset + value
        
    def read_bytes(self, count):
        return self.top.file.read(count)
        
    def push(self):
        self.stack.append(self.offset)
        
    def pop(self):
        if len(self.stack) == 0:
            raise RuntimeError("BReader pop error: impossible to pop an empty stack.")
        self.offset = self.stack.pop()
        
    # =============================================================================================================================
    # Common methods
        
    @staticmethod
    def to_int(bts):
        r = 0
        for b in bts:
            r = (r << 8) + b
        return r
    
    def read(self, dtype):

        if dtype == DataType.UINT8:
            return self.to_int(self.read_bytes(1))
        
        elif dtype == DataType.UINT16:
            return self.to_int(self.read_bytes(2))
        
        elif dtype == DataType.UINT32:
            return self.to_int(self.read_bytes(4))
        
        if dtype == DataType.INT8:
            i = self.to_int(self.read_bytes(1))
            if i > 0x7F:
                return i - 0x100
            else:
                return i
        elif dtype == DataType.INT16:
            i = self.to_int(self.read_bytes(2))
            if i > 0x7FFF:
                return i - 0x10000
            else:
                return i
        elif dtype == DataType.INT32:
            i = self.to_int(self.read_bytes(4))
            if i > 0x7FFFFFFF:
                return i - 0x100000000
            else:
                return i
        elif dtype == DataType.INT64:
            return self.to_int(self.read_bytes(8))
        
        elif issubclass(type(dtype), Flags):
            return dtype.read(self, self.current[-1])
        
        elif dtype == DataType.FIXED:
            return self.read(DataType.INT32)/65536.
        
        elif dtype == DataType.SFRAC:
            return self.read(DataType.INT16)/0x4000
        
        elif type(dtype) is list:
            return [self.read(dtype[1]) for i in range(dtype[0])]
        
        elif type(dtype) is dict:
            return dtype["function"](self.read(dtype["dtype"]))
        
        elif type(dtype).__name__ == 'function':
            return dtype(self)

        elif type(dtype) is str:
            bts = self.read_bytes(int(dtype))
            s = ""
            for b in bts:
                s += chr(b)
            return s

        else:
            raise RuntimeError(f"Unknwon data type {dtype}")
        
    def read_struct(self, obj, defs=None, offset=None):

        if offset is not None:
            self.offset = offset
            
        if defs is None:
            defs = obj._defs

        self.current.append(obj)        
        for k in defs:
            setattr(obj, k, self.read(defs[k]))
        self.current.pop()
            
        return obj
    
    def read_table(self, count, defs):

        table = []
        for i in range(count):
            table.append(Struct(self, defs))
            
        return table
    
    def read_array(self, count, dtype):

        table = []
        for i in range(count):
            table.append(self.read(dtype))
            
        return table        

# -----------------------------------------------------------------------------------------------------------------------------
# Top class which actually read bytes
       
class BFileReader(BReader):
    
    def __init__(self, file_name):
        self.file_name_ = file_name
        self.file = None
        super().__init__()
        
    def open(self):
        self.opens += 1
        if self.opens == 1:
            self.file = open(self.file_name, "br")
        self.offset = 0
        
    def close(self):
        if self.opens == 0:
            raise RuntimeError("BReader close error: impossible to close a closed reader")
        self.opens -= 1
        if self.opens == 0:
            self.file.close()
            self.file = None
            
    @property
    def file_name(self):
        return self.file_name_
        
    @property
    def offset(self):
        return self.file.tell()
    
    @offset.setter
    def offset(self, value):
        self.file.seek(value)



