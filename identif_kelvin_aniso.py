# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import sympy as sy
import class_densities as cde
import cls_operators as co
import cls_derivation as cder
import cls_tensors as ct
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

Proj = (P1, P2, P3, P4, P5)


datachaine = np.loadtxt('/home/annie/Documents/These/Resultats/2017/Modele/inputdata/cycles/unload/Essil_E1582_Chaine_40mm-min_2/unload_cycle5.txt', delimiter = ' ', skiprows = 1)[::-1]
datatrame = np.loadtxt('/home/annie/Documents/These/Resultats/2017/Modele/inputdata/cycles/unload/Essil_E1582_Trame_40mm-min_2/unload_cycle9.txt', delimiter = ' ', skiprows = 1)[::-1]

nc = int(len(datachaine)/10)
datachaine = datachaine[::nc]
nt = int(len(datachaine)/10)
datatrame = datatrame[::nc]

Fc1 = (datachaine[:,0]/datachaine[0,0])
Fc2 = (datachaine[:,1]/datachaine[0,1])
Ft1 = datatrame[:,0]/datatrame[0,0]
Ft2 = datatrame[:,1]/datatrame[0,1]

dict_fchaine = {'F1': Fc2, 'F2':Fc1, 'F3':1/(Fc1*Fc2), 'F4': np.zeros_like(Fc1), 'F5':np.zeros_like(Fc1), 'F6':np.zeros_like(Fc1)}

dict_ftrame = {'F1': Ft1, 'F2':Ft2, 'F3':1/(Ft1*Ft2), 'F4': np.zeros_like(Ft1), 'F5':np.zeros_like(Ft1), 'F6':np.zeros_like(Ft1)}

dict_schaine = {'F1': np.zeros_like(Fc1), 'F2':(datachaine[:,3]-datachaine[0,3])/Fc1, 'F3':np.zeros_like(Fc1), 'F4': np.zeros_like(Fc1), 'F5':np.zeros_like(Fc1), 'F6':np.zeros_like(Fc1)}
                                                                                                        
dict_strame = {'F1': (datatrame[:,3]-datatrame[0,3])/Ft1, 'F2':np.zeros_like(Ft1), 'F3':np.zeros_like(Ft1), 'F4': np.zeros_like(Ft1), 'F5':np.zeros_like(Ft1), 'F6':np.zeros_like(Ft1)}

schaine = ct.Tensors(dict_schaine)
strame = ct.Tensors(dict_strame)

Fchaine = ct.StrainTensor(dict_fchaine)
Ftrame = ct.StrainTensor(dict_ftrame)

symtenschaine = ct.StrainTensor({'F1': Fc2**2, 'F2':Fc1**2, 'F3':1/(Fc1*Fc2)**2, 'F4': np.zeros_like(Fc1), 'F5':np.zeros_like(Fc1), 'F6':np.zeros_like(Fc1)})

symtenstrame = ct.StrainTensor({'F1': Ft1**2, 'F2':Ft2**2, 'F3':1/(Ft1*Ft2)**2, 'F4': np.zeros_like(Ft1), 'F5':np.zeros_like(Ft1), 'F6':np.zeros_like(Ft1)})


def kelvin_model(p, C):
    a, b, c, d = p 
    c1,c2,c3,c4,c5,c6 = C.mandel
    params1 = {'dens':{'kelvin_hyperelas_dev'}, 'l1':a, 'l2':b, 'l3':0, 'l4':0, 'l5':0, 'a':0, 'proj':Proj}
    cldens1 = cder.MakeMechanicalTensors(params1, sym = True)
    params2 = {'dens':{'kelvin_hyperelas_dev'}, 'l1':c, 'l2':d, 'l3':0, 'l4':0, 'l5':0, 'a':2, 'proj':Proj}
    cldens2 = cder.MakeMechanicalTensors(params2, sym = True)
    evalstress1  = np.transpose(
        np.array(list(cldens1.func_symb()[1](c1,c2,c3,c4,c5,c6))),
        (1,2,0))
    evalstress2 = np.transpose(
        np.array(list(cldens2.func_symb()[1](c1,c2,c3,c4,c5,c6))),
        (1,2,0))
    evalstress = evalstress1 + evalstress2
    ph = evalstress[2,:]*C.mandel[2,:]
    return (evalstress-np.einsum('ik, jk-> jik', ph, C.inversetens))

def min_func(p, Cc, Ct, Sc, St):
    print(p)
    model_c = kelvin_model(p, Cc)
    model_t = kelvin_model(p, Ct)
    errc = Sc[:,np.newaxis,:] - model_c
    errt = St[:,np.newaxis,:] - model_t
    return np.sum(errc[1,0,:]**2) + np.sum(errt[0,0,:]**2) + np.sum(errc[0,0,:]**2) + np.sum(errt[1,0,:]**2)


#def min_func(p, Cc, Sc):
    #print(p)
    #model_c = kelvin_model(p, Cc)
    #errc = Sc[:,np.newaxis,:] - model_c
    #return np.sum(errc[0,0,:]**2) + np.sum(errc[1,0,:]**2)
    
res = opt.minimize(min_func, (1, 1, 50, 200), args = (symtenschaine, symtenstrame, schaine.mandel, strame.mandel), options = {'maxiter':50, 'disp':True})
#res = opt.minimize(min_func, (1, 1, 1), args = (symtenschaine, schaine.mandel), options = {'maxiter':50, 'disp':True})

modelchaine = kelvin_model(res.x, symtenschaine)
modeltrame = kelvin_model(res.x, symtenstrame)

plt.ion()

plt.plot(symtenschaine.mandel[1,:], schaine.mandel[1,:], 'r--', label = 'Warp - Exp')
plt.plot(symtenstrame.mandel[0,:], strame.mandel[0,:], 'b--', label = 'Weft - Exp')

plt.plot(symtenschaine.mandel[1,:], modelchaine[1,0,:], 'r', label = 'Model')
plt.plot(symtenstrame.mandel[0,:], modeltrame[0,0,:], 'b')

plt.plot(symtenschaine.mandel[1,:], modelchaine[0,0,:], 'r-.', label = 'Model -transv')
plt.plot(symtenstrame.mandel[0,:], modeltrame[1,0,:], 'b-.')

plt.xlabel('Stretch')
plt.ylabel('Stress (MPa)')
plt.legend()
plt.grid()
#plt.xlim(xmin = 1);plt.ylim(ymin = 0)



#plt.plot(symtenschaine.mandel[1,:], modelchaine[2,0,:], 'r')
#plt.plot(symtenstrame.mandel[0,:], modeltrame[2,0,:], 'b')
