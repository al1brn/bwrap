#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:35:32 2022

@author: alain
"""

from enum import Enum

SOCKET_TYPES = ['CUSTOM', 'VALUE', 'INT', 'BOOLEAN', 'VECTOR',
                'STRING', 'RGBA', 'SHADER', 'OBJECT', 'IMAGE',
                'GEOMETRY', 'COLLECTION', 'TEXTURE', 'MATERIAL']
    
# ----------------------------------------------------------------------------------------------------
# A socket

class Socket:
    def __init__(self, name, type='VALUE', is_multi_input=False, is_output=False, node=None):
        self.name      = name
        self.type      = type
        self.node      = node
        self.is_output = is_output
        self.links     = []
        self.is_multi_input = is_multi_input
        self.label     = None
        
    @classmethod
    def In(cls, name, is_output=False):
        s = cls(name)
        s.sock_out = sock_out
        return s
    
    @classmethod
    def Out(cls, name, node=None):
        s = cls(name)
        s.node = node
        return s
    
    def check_in(self, message=""):
        if self.is_output:
            raise RuntimeError(f"The socket '{self}' is not in. {message}")
    
    def check_out(self, message=""):
        if not self.is_output:
            raise RuntimeError(f"The socket '{self}' is not out. {message}")
            
    @property
    def is_linked(self):
        return len(self.links) > 0
    
    def __repr__(self):
        s = f"{self.name}"
        if self.is_in:
            if self.in_is_linked:
                return s + f" linked to [{self.sock_out.node.name}.{self.sock_out.name}]"
            else:
                return s + f" not linked ({self.sock_out})" 
        else:
            return s + f" from node '{self.node.name}'"
        
    # ---------------------------------------------------------------------------
    # Ensure a value is a socket
    
    @classmethod
    def socket(cls, other):
        if issubclass(type(other), Socket):
            return other
        elif type(other) in [int, bool, float]:
            return ValueInputNode(other).ovalue
        else:
            raise RuntimeError(f"Socket value error: impossible to transform the value '{other}' in a socket.")
        
    # ---------------------------------------------------------------------------
    # Operations
    
    def __add__(self, other):
        o = Socket.socket(other)
        self.check_out(f"Operation {self} + {other}.")
        o.check_out(f"Operation {self} + {other}.")
        return MathNode('+', value0=self, value1=Socket.socket(other)).ovalue

        
# ----------------------------------------------------------------------------------------------------
# Dictionnary of sockets

class Sockets(dict):
            
    @classmethod
    def In(cls, **kwargs):
        sock = cls()
        for k, v in kwargs.items():
            sock[k] = Socket.In(k, v)
        return sock
    
    @classmethod    
    def Out(cls, node, *names):
        sock = cls()
        for k in names:
            sock[k] = Socket.Out(k, node)
        return sock
        
    
# ----------------------------------------------------------------------------------------------------
# Node made of inputs and outputs

class Node:
    
    nodes = {}
    
    def __init__(self, name, sockets_in={}, sockets_out={}, params={}):
        self.label       = name
        self.name        = Node.unique_name(name)

        self.sockets_in  = sockets_in
        self.sockets_out = sockets_out
        self.params      = params
        
        Node.nodes[self.name] = self
        
        for k, v in self.sockets_in.items():
            setattr(self, f"i{k}", v)

        for k, v in self.sockets_out.items():
            setattr(self, f"o{k}", v)
            
        for k, v in self.params.items():
            setattr(self, f"p{k}", v)
            
        
    @classmethod
    def unique_name(self, name):
        for i in range(10000):
            nm = f"{name} {i:03d}"
            if nm not in list(Node.nodes.keys()):
                return nm
        
    def __repr__(self):
        s = f"Node {type(self).__name__} '{self.name}':\n"
        
        if len(self.sockets_in) > 0:
            s += "sockets in:\n"
            for k, v in self.sockets_in.items():
                s += f"   {k:10} = {v}\n"

        if len(self.sockets_out) > 0:
            s += "sockets out:\n"
            for k, v in self.sockets_out.items():
                s += f"   {k:10} = {v}\n"
                
        if len(self.params) > 0:
            s += "params:\n"
            for k, v in self.params.items():
                s += f"   {k:10} = {v}\n"
                
        return s
    
    # ---------------------------------------------------------------------------
    # List of the links
    
    @classmethod
    def get_links(cls):
        links = []
        for node in cls.nodes.values():
            for name, sock_in in node.sockets_in.items():
                if sock_in.in_is_linked:
                    links.append(sock_in)
        return links
            
    
    # ---------------------------------------------------------------------------
    # List of the nodes
    
    @classmethod
    def dump(cls):
        print()
        print('-'*40)
        print("Nodes")
        print()
        index = 0
        for k, v in cls.nodes.items():
            print(f"{index:3d}>", v)
            print()
            index += 1
            
        print()
        print('-'*40)
        print("Links")
        print()
        for i, link in enumerate(Node.get_links()):
            print(f"{i:3d}> {link}")
    

# ====================================================================================================
# Input nodes

# ----------------------------------------------------------------------------------------------------
# Value

class ValueInputNode(Node):
    def __init__(self, value):
        vtype = type(value)
        if vtype not in [float, int, bool]:
            raise RuntimeError(f"ValueInputNode error: type {type(value)} of {value} not supported!")
            
        super().__init__(
            vtype.__name__,
            sockets_out = Sockets.Out(self, "value"),
            params      = Sockets.In(value=value)
            )


class MathNode(Node):
    def __init__(self, operation, **kwargs):
        sin    = Sockets()
        sout   = Sockets()
        params = Sockets()
        
        if operation in ['+', '-', '*', '/']:
            sin  = Sockets.In(value0=kwargs["value0"], value1=kwargs["value1"])
            sout = Sockets.Out(self, "value")
        else:
            raise RuntimeError(f"MathNode error: operation {operations} not supported!")
                
        super().__init__("add", sockets_in = sin, sockets_out = sout, params = params)
        


a = ValueInputNode(7)
b = ValueInputNode(6)
print(a)
print(b)
c = a.ovalue + b.ovalue
print(c)

Node.dump()


    
    
    
    