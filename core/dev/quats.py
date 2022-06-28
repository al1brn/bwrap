#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:19:11 2022

@author: alain
"""

class Item():
    
    STAR = "."
    
    def __init__(self, value=None, sign=1):
        self.sign  = sign
        self.value = value
        
    def __repr__(self):
        if self.is_none:
            return ""
        
        if self.is_str:
            return self.value
        
        s = ""
        for v in self.value:
            if v.sign < 0:
                s += " - "
            elif s != "":
                s += " + "
            s += v.value
            
        return s
        
    @property
    def is_none(self):
        return self.value is None
    
    @property
    def is_str(self):
        return type(self.value) is str
    
    @property
    def is_comp(self):
        return type(self.value) is list
        
        
    def clone(self, sign=1):
        if self.is_none:
            return Item()
        
        elif self.is_str:
            return Item(self.value, self.sign*sign)

        else:
            return Item([item.clone(sign) for item in self.value])
        
    def __neg__(self):
        return self.clone(-1)
        
    def __add__(self, item):
        
        if self.is_none:
            return item.clone()
        
        if item.is_none:
            return self.clone()
        
        if self.is_str:
            value = [self.clone()]
        else:
            value = self.clone().value
            
        if item.is_str:
            value.append(item.clone())
        else:
            value.extend(item.clone().value)
            
        return Item(value)
            
    def __sub__(self, item):
        
        if self.is_none:
            return item.clone(-1)
        
        if item.is_none:
            return self.clone()
        
        if self.is_str:
            value = [self.clone()]
        else:
            value = self.clone().value
            
        if item.is_str:
            value.append(item.clone(-1))
        else:
            value.extend(item.clone(-1).value)
            
        return Item(value)
    
    def __mul__(self, item):
        if self.is_none or item.is_none:
            return Item()
        
        if self.is_str:
            if item.is_str:
                return Item(self.value + self.STAR + item.value, sign=self.sign * item.sign)
            else:
                return Item([Item(self.value + self.STAR + v.value, self.sign*v.sign) for v in item.value])
            
        if item.is_str:
            return Item([Item(v.value + self.STAR + item.value , self.sign*v.sign) for v in self.value])
        
        value = []
        for v0 in self.value:
            for v1 in item.value:
                value.append(Item(v0.value + self.STAR + v1.value, v0.sign*v1.sign))
                
        return Item(value)



class Quat:
    def __init__(self, name="q"):
        self.name = name
        self.q    = [Item() for i in range(4)]
        
    @property
    def a(self):
        return self.q[0]
    
    @property
    def b(self):
        return self.q[1]
    
    @property
    def c(self):
        return self.q[2]
    
    @property
    def d(self):
        return self.q[3]
    
    @property
    def e(self):
        return self.q[0]
    
    @property
    def f(self):
        return self.q[1]
    
    @property
    def g(self):
        return self.q[2]
    
    @property
    def h(self):
        return self.q[3]
    
    
        
    def __repr__(self):
        s = f"{self.name}\n"
        for i, line in enumerate(self.q):
            s += f"   {i}: {line}\n"
        return s
    
    def __matmul__(self, other):
        
        q = Quat(self.name + " @ " + other.name)
        
        # a*e - b*f - c*g - d*h
        # b*e + a*f + c*h - d*g
        # a*g - b*h + c*e + d*f
        # a*h + b*g - c*f + d*e 
        
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        
        e = other.e
        f = other.f
        g = other.g
        h = other.h
        
        q.q[0] = a*e - b*f - c*g - d*h
        q.q[1] = b*e + a*f + c*h - d*g
        q.q[2] = a*g - b*h + c*e + d*f
        q.q[3] = a*h + b*g - c*f + d*e 
        
        return q
    
    @staticmethod
    def check():
        qx = Quat("qx")
        qy = Quat("qy")
        qx.q = [Item('a'), Item('b'), Item('c'), Item('d')]
        qy.q = [Item('e'), Item('f'), Item('g'), Item('h')]

        print(qx)
        print(qy)
        print(qx @ qy)


Quat.check()        

qx = Quat("qx")
qx.q[0] = Item("cos(X/2)")
qx.q[1] = Item("sin(X/2)")

qy = Quat("qy")
qy.q[0] = Item("cos(Y/2)")
qy.q[2] = Item("sin(Y/2)")

qz = Quat("qz")
qz.q[0] = Item("cos(Z/2)")
qz.q[3] = Item("sin(Z/2)")

q = qz @ qy @ qx
print(q)










