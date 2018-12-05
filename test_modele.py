# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import sympy as sy
import class_densities as cde
import cls_operators as co
import cls_derivation as cder
import scipy.optimize as opt

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

#params = {'dens':{'yeoh'},'C0':3, 'C1':0.1, 'C2':0, 'K':10000}
params = {'dens':{'kelvin_hyperelas_dev'},'l1':1, 'l2':1, 'l3':00, 'l4':0., 'l5':0., 'a':1, 'proj':Proj, 'K':10000}

C1 = np.linspace(1,9, 1000)
SS = cder.MakeMechanicalTensors(params, sym = True)
dens = SS.SD.density
PK2 = np.array(list(SS.func_symb()[1](C1, 1/np.sqrt(C1), 1/np.sqrt(C1), np.zeros_like(C1), np.zeros_like(C1), np.zeros_like(C1))))

PK2 = np.transpose(PK2, (1,2,0))
tensor = np.array([C1, 
                    C1,
                    1/C1**2,
                    np.zeros_like(C1),
                    np.zeros_like(C1),
                    np.zeros_like(C1)])[:,np.newaxis]

inversetens = np.array([1/C1,
                        1/C1,
                        C1**2,
                         np.zeros_like(C1),
                        np.zeros_like(C1),
                        np.zeros_like(C1)
                        ])[:,np.newaxis]

ph =PK2[2,:]/inversetens[2,:]
PK2 -=np.einsum('jk, ijk-> ijk', ph, inversetens)

def funct(c11, a,b,c):
    print(a,b,c)
    c = int(c)
    params2 = {'dens':{'kelvin_hyperelas_dev'}, 'l1':a, 'l2':b, 'l3':0, 'l4':0, 'l5':0, 'a':c, 'proj':Proj}
    A = cder.MakeMechanicalTensors(params2, sym= True)
    #tensor = np.array([[c11, np.zeros_like(c11), np.zeros_like(c11)],
                       #[np.zeros_like(c11), 1/np.sqrt(c11), np.zeros_like(c11)],
                       #[np.zeros_like(c11), np.zeros_like(c11), 1/np.sqrt(c11)]])
    #inversetens = np.array([[1/c11,  np.zeros_like(c11),  np.zeros_like(c11)],
                            #[ np.zeros_like(c11), np.sqrt(c11),  np.zeros_like(c11)],
                            #[ np.zeros_like(c11),  np.zeros_like(c11), np.sqrt(c11)]])
    tensor = np.array([C1, 
                    C1,
                    1/C1**2,
                    np.zeros_like(C1),
                    np.zeros_like(C1),
                    np.zeros_like(C1)])[:,np.newaxis]

    inversetens = np.array([1/C1,
                        1/(C1),
                        C1**2,
                         np.zeros_like(C1),
                        np.zeros_like(C1),
                        np.zeros_like(C1)
                        ])[:,np.newaxis]

    func = np.transpose(np.array(list(A.func_symb()[1](c11, 1/np.sqrt(c11), 1/np.sqrt(c11), np.zeros_like(c11), np.zeros_like(c11), np.zeros_like(c11)))), (1,2,0))
    ph = func[2,:]/inversetens[2,:]
    return (func-np.einsum('jk, ijk-> ijk', ph, inversetens)).ravel()


#def min_func(p, Sobj, c11, c22, c33, c23, c13, c12):
    #return (Sobj[0,0,:] - funct(p, (c11, c22, c33, c23, c13, c12))[0,0,:])**2


##C1 = np.linspace(1,3,12)
sol = opt.curve_fit(funct, C1, PK2.ravel())
soluce = funct(C1, sol[0][0], sol[0][1],sol[0][2]).reshape(6,1,1000)
plt.plot(C1, PK2[0,0,:], 'k', linewidth = 3, label = 'obj - kelvin')
plt.plot(C1, PK2[1,0,:], 'k--', linewidth = 2, label = 'identif')
plt.plot(C1, soluce[0,0,:], 'r-.',linewidth = 2, label = 'identif')
plt.plot(C1, soluce[1,0,:], 'b')
plt.plot(C1, soluce[2,0,:],'g')