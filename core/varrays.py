#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:59:59 2022

@author: alain
"""

import numpy as np

from .wroot import WRoot

# ----------------------------------------------------------------------------------------------------
# Array of arrays of diffent sizes
#
# Manages two arrays:
# - sizes:  the sizes of each array
# - values: the concatenation of the arrays in one single array
#
# The type and shapes of the values in the values array are initialized when the first array
# is set. For float values, one must ensure that the first setting uses floats and not ints


class VArrays(WRoot):
    
    # ---------------------------------------------------------------------------
    # Initialization can initialize the number of arrays
    # In that case, the initialization must be followed by a setting loop to
    # set the values:
    # 
    # . vas = VArrays(count=10)
    # . for index in range(10):
    # .     vas.append(values, index)
    
    def __init__(self, count=0):
        self.sizes  = np.zeros(count, int)
        self.values = None
        
    # ---------------------------------------------------------------------------
    # Append a new array to the set of arrays
    # Note 1: The values argument is always appened to the values array, even
    # if the index argument is not None
    # Index is used at initialization time to avoid two calls to np.append,
    # one for values, one for the sizes
    # Note 2: The first call to append defines the type and shape of the items
        
    def append(self, values, index=None):
        if index is None:
            index = len(self.sizes)
            self.sizes = np.append(self.sizes, len(values))
        else:
            self.sizes[index] = len(values)
            
        if len(values) > 0:
            if self.values is None:
                self.values = np.array(values)
            else:
                self.values = np.append(self.values, values, axis=0)
                
        return index
                
    # ---------------------------------------------------------------------------
    # Extend with another varrays
    
    def extend(self, other):

        if len(other) == 0:
            return
        
        self.sizes  = np.append(self.sizes,  other.sizes)
        self.values = np.append(self.values, other.values, axis=0)
            
    # ---------------------------------------------------------------------------
    # Content
            
    def __repr__(self):
        s = f"<VArrays of {len(self)} arrays of type '{self.vtype}' shaped {self.value_shape}. Sizes = {self.sizes}, values.shape = {np.shape(self.values)}"
        return s + ">"

    # ---------------------------------------------------------------------------
    # Item type
        
    @property
    def vtype(self):
        if self.values is None:
            return None
        else:
            tp = type(self.values[0])
            if tp is np.ndarray:
                return self.values[0].dtype
            else:
                return tp
            
    @property
    def value_shape(self):
        if self.values is None:
            return (0,)
        else:
            return np.shape(self.values[0])
        

    # ---------------------------------------------------------------------------
    # Implements the array interface
        
    def __len__(self):
        if self.values is None:
            return 0
        else:
            return len(self.sizes)
        
    def __getitem__(self, index):
        
        if hasattr(index, '__len__'):
            
            vas = type(self)(len(index))
            for index, i_array in enumerate(index):
                vas.append(self.values[self.slice(i_array)], index)
            return vas
        
        elif type(index) is slice:
            
            vas = type(self)()
            vas.sizes = self.sizes[index]
            
            i0, i1, _ = index.indices(len(self.sizes))
            vas.values = self.values[sum(self.sizes[:i0]):sum(self.sizes[:i1+1])]
            
            return vas
        
        else:
            return self.values[self.slice(index)]
        
    # ---------------------------------------------------------------------------
    # Return the slice on the values corresponding to one array
    
    def slice(self, index):
        offset = np.sum(self.sizes[:index])
        return slice(offset, offset + self.sizes[index])

    # ---------------------------------------------------------------------------
    # The arrays offsets in the self.values array
    # The array is located at the slice : offset:offset+size
            
    @property
    def offsets(self):
        if len(self.sizes) == 0:
            return np.zeros(0, int)
        else:
            return np.insert(np.cumsum(self.sizes), 0, 0)[:len(self.sizes)]

    # ---------------------------------------------------------------------------
    # Initialize from a python list of lists
    # (or whatever array of arrays)
        
    @classmethod
    def FromList(cls, vlist):
        vas = cls(len(vlist))
        for index, vs in enumerate(vlist):
            vas.append(vs, index)
        return vas
    
    # ---------------------------------------------------------------------------
    # Transform into a python array of arrays
    
    def tolist(self):
        vlist = []
        ofs = 0
        for index in range(len(self.sizes)):
            vlist.append(self.values[ofs:ofs+self.sizes[index]].tolist())
            ofs += self.sizes[index]
        return vlist
    
    # ---------------------------------------------------------------------------
    # From another VArrays
    
    @classmethod
    def FromArrays(cls, sizes, values, copy=False):
        vas = cls()
        if copy:
            vas.sizes  = np.array(sizes)
            vas.values = np.array(values)
        else:
            vas.sizes  = sizes
            vas.values = values
            
        return vas
    
    # ---------------------------------------------------------------------------
    # From another VArrays
    
    @classmethod
    def FromOther(cls, other):
        vas = cls()
        vas.sizes  = np.array(other.sizes)
        vas.values = np.array(other.values)
        return vas
    
    # ---------------------------------------------------------------------------
    # From data
    
    @classmethod
    def FromData(cls, sizes, values, duplicate=False):
        
        vas = cls()
        
        if duplicate:
            vas.sizes  = np.array(sizes)
            vas.values = np.array(values)
        else:
            vas.sizes  = sizes
            vas.values = values
            
        return vas
    
    # ---------------------------------------------------------------------------
    # Some stats
    
    @property
    def same_sizes(self):
        return np.min(self.sizes) == np.max(self.sizes)
    
    @property
    def average_size(self):
        if len(self.sizes) == 0:
            return 0
        else:
            return np.average(self.sizes)
    
    @property
    def min_size(self):
        if len(self.sizes) == 0:
            return 0
        else:
            return np.min(self.sizes)
        
    @property
    def max_size(self):
        if len(self.sizes) == 0:
            return 0
        else:
            return np.max(self.sizes)
        
    def size_equal(self, size):
        if len(self.sizes) == 0:
            return 0
        else:
            return np.sum(self.sizes == size)
        
    def size_at_least(self, size):
        if len(self.sizes) == 0:
            return 0
        else:
            return np.sum(self.sizes >= size)
        
    def size_at_most(self, size):
        if len(self.sizes) == 0:
            return 0
        else:
            return np.sum(self.sizes <= size)
        
    # ---------------------------------------------------------------------------
    # Return an array made of the sequence of arrays indices
    
    def arrays_indices(self):

        if self.values is None:
            return None
        
        count = len(self.sizes)
        
        if np.min(self.sizes) == np.max(self.sizes):
            size = self.sizes[0]
            return (np.arange(count).reshape(count, 1) + np.zeros((1, size), int)).reshape(count*size)
        
        else:
            a = np.empty(len(self.values))
            offset = 0
            for i in range(count):
                a[offset:offset + self.sizes[i]] = i
                offset += self.sizes[i]
            return a
        
    # ---------------------------------------------------------------------------
    # Return the arrays of a given size
    # Return an array of indices twoards the values plus the arrays indices
    
    def arrays_of_size(self, size):

        arrays = np.where(self.sizes == size)[0]
        count = len(arrays)
        
        if count == 0:
            return arrays, np.zeros((0, size), int)
        else:
            return arrays, self.offsets[arrays].reshape(count, 1) + np.arange(size).reshape(1, size)
        
    # ---------------------------------------------------------------------------
    # Unique
    # When double values can exist, the var arrays can be replaced by:
    # - A VArrays using the index towards the array of uniques values
    # - The array of unique values
    
    def unique(self):

        vas = type(self)()
        
        if len(self) == 0:
            return vas, []
        
        vas.sizes = np.array(self.sizes)
        
        values, vas.values = np.unique(self.values, return_inverse=True, axis=0)

        return vas, values
    
    # ---------------------------------------------------------------------------
    # Random selection in each array
    
    def random_values(self):
        size = np.max(self.sizes)*1000
        return self.values[self.offsets + np.random.randint(0, size, len(self.sizes)) % self.sizes]
    
    # ---------------------------------------------------------------------------
    # Test
    
    @staticmethod
    def _demo():
        
        # ----- Arrays of ints
        
        count = 10
        vas = VArrays(count)
        sizes = np.random.randint(3, 9, count)
        for index in range(count):
            vas.append(np.random.randint(0, 100, sizes[index]), index)

        print("-"*30)
        print("Array of arrays of int")
        print()
        print(vas)
        print()
        for index in range(len(vas)):
            print(f"{index:2d}: {vas[index]}")
        print()
        print(f"tris: {vas.size_equal(3)}, quads: {vas.size_equal(4)}, n-gons: {vas.size_at_least(5)}")
        print()
        
        # ----- Unique indices
        
        vuns, unique = vas.unique()
        print("Unique values")
        print(unique)
        print()
        print("Indices to uniques")
        print(vuns)
        print()

        sa = []
        lsa = 0
        sv = []
        lsv = 0
        for i, a in enumerate(vuns):
            sa.append(f"{a}")
            sv.append(f"{unique[a]}")
            
            lsa = max(lsa, len(sa[-1]))
            lsv = max(lsv, len(sv[-1]))
            
        for i in range(len(sa)):
            sa[i] += " " * (lsa - len(sa[i]))
            sv[i] += " " * (lsv - len(sv[i]))

        for i, a in enumerate(vuns):
            print(f"{i:2d}: {sa[i]} -> {sv[i]} = {vas[i]}")
        print()
        
        # ----- As a python list
        print("Python list")
        print(vas.tolist())
        print()

        # ----- Arrays of vectors

        count = 6
        vas = VArrays()
        sizes = np.random.randint(0, 5, count)
        for index in range(count):
            vas.append(np.random.uniform(0, 1., (sizes[index], 3)))
            
        print("-"*30)
        print("Array of arrays of vectors")
        print()
        print(vas)
        print()
        for index in range(len(vas)):
            print(f"{index:2d}: {vas[index]}")
        print()
        
        # ----- As a python list
        print("Python list")
        print(vas.tolist())
        print()
        
        vas._dump()
        
        
