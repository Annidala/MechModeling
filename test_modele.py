# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import class_densities as cde
from importlib import reload
import sympy as sy

P1 = sy.Matrix([[4,-2,-2,0,0,0],
   [-2,1,1,0,0,0],
   [-2,1,1,0,0,0],
   [0,0,0,0,0,0],
   [0,0,0,0,0,0],
   [0,0,0,0,0,0]])*sy.Rational(1,6)
P2 = sy.Matrix([[0,0,0,0,0,0],
            [0,1,-1,0,0,0],
            [0,-1,1,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]])*sy.Rational(1,2)
P3 = sy.Matrix([[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]])
P4 = sy.Matrix([[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,0]])
P5 = sy.Matrix([[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,1]])

Proj = (P1,P2, P3, P4, P5)
params = {'dens':'mooney_rivlin', 'lh':sy.Rational(1), 'l1':1, 'l2':1, 'l3':1, 'l4':1, 'l5':1, 'a':2, 'proj': Proj}
SS = cde.SymbolicDensities(params, sym = False)

