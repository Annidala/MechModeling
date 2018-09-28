# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cls_operators as co
import cls_tensors as ct
import cls_modele as cmod

from matplotlib import rc, rcParams, gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
from importlib import reload

plt.rc('text', usetex=True)
rc('font', **{'family':'serif','serif': 'Computer Modern', 'weight':'bold', 'size':14})
rcParams.update({'figure.autolayout': True})
plt.ion()

C1 = np.linspace(1, 9, 20)
lim = 5
cmapp = cm.get_cmap('viridis')
norm = colors.Normalize(vmin = 0., vmax =lim)

for i in range(1, lim):
    params = {'model':'St_Venant','lh':1., 'l1':1., 'l2':0.5, 'l3':1., 'l4':1., 'l5':1., 'a':0., 'penal': 1000, 'proj': co.FourthOrderOperators().projs, 'angle':0}
    params['a'] = i
    SS = cmod.IncompressibleModels({'F1':C1, 'F2':C1,}, params)
    #SS = cmod.IncompressibleModels({'F1':C1, 'F2':C1, 'F3':1/(C1**(2./3)), 'F4':np.zeros_like(C1), 'F5':np.zeros_like(C1), 'F6':np.zeros_like(C1)}, params)
    #print (SS.C.mandel[2])
    #cont = SS.solver()
    #cont = SS.model()
    E = SS.C.egreenlagrange[0]

    cont = SS.model()
    #plt.figure(1)
    #plt.plot(E[0], SS.C.invariant[0], color = cmapp(norm(i)), label = 'a = %s'%params['a'])
    ##print (SS.C.mandel[2])
    ##print (params['penal'], cont[0][0])
#plt.grid()
#plt.legend()
#plt.xlim(xmin = 0)
#plt.xlabel('E$_{11}$ - Green Lagrange volumique')
#plt.ylabel('Determinant de C')

    plt.figure(0)
    plt.plot(E[0], cont[0], color = cmapp(norm(i)), label = 'a = %s'%params['a'])
    plt.plot(E[0], cont[1], color = cmapp(norm(i)), linestyle = '--', )
    plt.plot(E[0], cont[2], color = cmapp(norm(i)), linestyle = '-.', )
    print (SS.C.invariant[0])
plt.grid()
plt.legend()
plt.xlim(xmin = 0)
plt.xlabel('E$_{11}$ - Green Lagrange volumique')
plt.ylabel('S - PK2')