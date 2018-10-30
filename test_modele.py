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
params = {'dens':'mooney_rivlin', 'C1':1, 'C2':5}

SS = cde.SymbolicDensities(params, sym = False)
dens = SS.density()
deriv = sy.derive_by_array(dens, SS.F).tomatrix()

f11 = sy.Symbol('f11')
f12 = sy.Symbol('f12')
f13 = sy.Symbol('f13')
f22 = sy.Symbol('f22')
f21 = sy.Symbol('f21')
f23 = sy.Symbol('f23')
f31 = sy.Symbol('f31')
f32 = sy.Symbol('f32')
f33 = sy.Symbol('f33')
func_eval = sy.lambdify((f11,f22,f33, f13, f23, f12), deriv.subs([(f32, f23), (f31, f13), (f21, f12)]))

C1 = np.linspace(1, 9, 20)
invF =  np.array([[1/C1, np.zeros_like(C1), np.zeros_like(C1)],
                  [np.zeros_like(C1), np.sqrt(C1), np.zeros_like(C1)],
                  [np.zeros_like(C1), np.zeros_like(C1), np.sqrt(C1)]])
test = func_eval(C1, 1/np.sqrt(C1), 1/np.sqrt(C1), np.zeros_like(C1), np.zeros_like(C1),np.zeros_like(C1))
incomp = test[1,1]/np.sqrt(C1)
total = test - incomp*invF