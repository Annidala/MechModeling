# -*- coding: utf-8 -*-
import numpy as np
import sympy as sy
import cls_tensors as ct

class SymbolicDensities():
    
    def __init__(self, params, sym = False):
        self.sym = sym
        self.params = params
        self.tens = ct.SymbolicTensors(self.sym)
        self.density = self.make_density()

    def make_density(self):
        dens = 0
        for i in self.params['dens']:
            dens+=getattr(self, i)()
        return dens

    # deviatoric densities
    def yeoh(self):
        C0, C1, C2 = self.params['C0'], self.params['C1'], self.params['C2']
        W = C0*(self.tens.I1() -3) + C1*(self.tens.I1()-3)**2 + C2*(self.tens.I1()-3)**3
        return W
    
    def mooney_rivlin(self):
        C1, C2 = self.params['C1'], self.params['C2']
        W = C1*(self.tens.I1()-3) + C2*(self.tens.I2() - 3)
        return W
    
    def kelvin_hyperelas_dev(self):
        l1, l2, l3, l4, l5 , a = self.params['l1'], self.params['l2'], self.params['l3'], self.params['l4'], self.params['l5'], self.params['a'] # Kelvin modulus and power order of the law
        proj = self.params['proj'] # kelvin projectors - preferably choose only deviatoric projectors
        coeff = np.array([l1, l2, l3, l4, l5])
        ld= coeff/coeff.sum()
        dens = 0
        E = 1/2*(self.tens.deviat_tensor() - sy.Matrix([1, 1, 1, 0, 0, 0]))
        for i, l in enumerate(ld):
            dens+=(l*((E).T)*proj[i]*E)[0]
            #dens+=(l*((self.tens.deviat_tensor()).T)*proj[i]*self.tens.deviat_tensor())[0]

        W = 1/(2*(a+1))*np.sum(coeff)*dens**(a+1)
        return W
        
    def bmw(self):
        pass
        
    def baw(self):
        pass
    
    # volumetric densities
    def quasi_incomp_poly(self):
        K = self.params['K']
        W = K*(self.tens.I3() - 1)**2
        return W
            
