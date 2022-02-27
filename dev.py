#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 08:13:52 2022

@author: alain
"""

import numpy as np

from core.varrays import VArrays
from core.wroot import WRoot
from core.vertparts import VertParts
from core.faces import Faces


cube = VertParts.Default('CUBE', parts=2)
cube.join(VertParts.Default('TETRA'))
print(cube.faces_centers(0))


              