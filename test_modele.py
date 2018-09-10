# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cls_operators as co
import cls_tensors as ct
import cls_modele as cmod

from matplotlib import rc, rcParams, gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors

plt.rc('text', usetex=True)
rc('font', **{'family':'serif','serif': 'Computer Modern', 'weight':'bold', 'size':14})
rcParams.update({'figure.autolayout': True})
plt.ion()

C1 = np.linspace(1,5, 10)
lim = 2
cmapp = cm.get_cmap('viridis')
norm = colors.Normalize(vmin = 0., vmax =6)

for i in range(1,lim):
    params = {'model':'kelvin_hencky','lh':1e-2, 'l1':1e-2, 'l2':5e-3, 'l3':1e-2, 'l4':1e-2, 'l5':1e-2, 'a':0, 'penal': 10e6, 'proj': co.KelvinOperators().projs, 'angle':0}
    params['a'] = i
    SS = cmod.IncompressibleModels({'F1':C1, 'F2':C1}, params)
    cont = SS.solver()
    E = SS.C.egreenlagrange
    plt.plot(E[0], cont[0], color = cmapp(norm(i)), label = r'$\alpha$ = %s'%i)
    plt.plot(E[0], cont[1], color = cmapp(norm(i)), linestyle = '--')
    plt.plot(E[0], cont[2], color = cmapp(norm(i)), linestyle = '-.')
    print (E[0]+E[1]+E[2])
plt.grid()
plt.legend()
plt.xlim(xmin = 0)
plt.xlabel('E$_{11}$')
plt.ylabel('S')