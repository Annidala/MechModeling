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

C1 = np.linspace(1, 9, 10)
lim = 5
cmapp = cm.get_cmap('viridis')
norm = colors.Normalize(vmin = 0., vmax =lim)

fig = plt.figure(0)
#gs = gridspec.GridSpec(1,3)
#ax1=fig.add_subplot(gs[:,0:1])
#ax2=fig.add_subplot(gs[:,1:2])
#ax3=fig.add_subplot(gs[:,2:3])
                       
for i in range(1, 2):
    params = {'model':'kelvin_gl','lh':1., 'l1':1., 'l2':0.5, 'l3':1., 'l4':1., 'l5':1., 'a':0., 'penal': 20,'proj': co.FourthOrderOperators().projs, 'angle':0}
    params['a'] = 2
    SS = cmod.IncompressibleModels({'F1':C1, 'F2':C1}, params)
    # the following lines calls the solver since only 2 components of the dilatation tensor are known
    cont = SS.solver()
    E = SS.C.egreenlagrange[0]
    # plot curves
    plt.plot(E[0], cont[0], color = "#df5c5f", label = 'a = %s'%params['a'], linewidth = 2)
    plt.plot(E[0], cont[1], color = "#5b7f90", linestyle = '--', linewidth = 2)
    plt.plot(E[0], cont[2], color = "#3E3d3E",  linestyle = '-.', linewidth = 2)
    print ("Det =", SS.C.invariant[0]) 
    #ax1.plot(E[0], cont[0], color = cmapp(norm(i)), label = 'a = %s'%params['a'])
    #ax2.plot(E[0], cont[1], color = cmapp(norm(i)), linestyle = '--', )
    #ax3.plot(E[0], cont[2], color = cmapp(norm(i)), linestyle = '-.', )
    #print ("Det =", SS.C.invariant[0]) 

plt.grid()
plt.legend()
plt.xlim(xmin = 0)
plt.xlabel('E$_{11}$ - Green Lagrange volumique')
plt.ylabel('S - PK2')
#ax1.legend()
#ax1.set_ylabel(r'$\sigma_{11}$ (MPa)') ; ax1.set_xlabel('E$_{11}$ - Green Lagrange volumique'); ax1.grid()
#ax2.set_ylabel(r'$\sigma_{22}$ (MPa)') ; ax2.set_xlabel('E$_{11}$ - Green Lagrange volumique'); ax2.grid()
#ax3.set_ylabel(r'$\sigma_{33}$ (MPa)') ; ax3.set_xlabel('E$_{11}$ - Green Lagrange volumique'); ax3.grid()
#ax3.set_xlim(xmin = 0, xmax = 4.2)
#ax3.set_ylim(ymin = -2 , ymax = 5)
#ax1.set_xlim(xmin = 0, xmax = 4.2)
#ax1.set_ylim(ymin = -2 , ymax = 5)