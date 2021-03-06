# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm
from scipy.optimize import fsolve
import cls_tensors as ct

class IncompressibleModels():
    '''
    All models defined in terms of the second Piola Kirchhoff stress 
    dW/dE
    
    The class inherits from the Get_Tensor class. 
    
    Parameters
    ----------
    values: dict to define the tensors
    params: dict of modele parameters. Depending on the model choice, dict keys are different. The params should at least contain the model function name under the key 'model' (e.g.: {'model':'model_rivlin'}) to avoid class error
    '''
    def __init__(self, values, params):
        self.C = ct.StrainTensor(values) #creates the object C - see cls_tensors.py for further detail on the methods of the class
        self.model = getattr(self, params['model']) # calls one of the class method
        self.params = params
        self.angle = params['angle']
        
        
    #------------ Resolution functions ----------------        
    def modele_rivlin(self):
        '''
        Compute isotropic Mooney-Rivlin stress for an incompressible material
        
        Parameters
        ----------
        C0, C1 : model parameters specified when calling the class in the dict params entry
        
        Returns
        -------
        S: the modeled stress tensor, in mandel notation. Shape is (6, n)
        '''
        try: 
            C0, C1 = params['C0'], params['C1'] 
            derivee = (C0 + C1*(self.C.trace -3))*self.C.mandel 
            p = derivee[2,:]/self.C.inverse_tens()[2,:]
            S = derivee - p*self.C.inverse_tens()
            return S
        except:
            print 'Error : class function requires two parameters (C0 and C1) to run'

    def kelvin_hencky(self, E = None):
        """
        Returns the modeled dual stress computed from a hyperelastic anisotropic kelvin approach, using the Hencky deformation measure.
        
        Parameters
        ----------
        params: defined when calling the class. Dict keys are {'lh', 'l1', 'l2', 'l3', 'l4', 'l5', 'a', 'proj', 'penal'}. 'li' are the kelvin modules, 'a' is the power used for the modeling, 'proj' contains the kelvin eigentensors and 'penal' is defined to deals with incompressibility
        
        Returns
        -------
        S: the modeled dual stress tensor written in mandel notation. Shape (6, n) : n is the number of tensors corresponding with the number of deformation tensors in input
        """
        lh, l1, l2, l3, l4, l5 , a = self.params['lh'], self.params['l1'], self.params['l2'], self.params['l3'], self.params['l4'], self.params['l5'], self.params['a'] # Kelvin modulus and power order of the law
        proj = self.params['proj'] # kelvin projectors
        k = self.params['penal'] # penalisation coefficient for incompressibility
        coeff = np.array([l1, l2, l3, l4, l5, lh])
        trC = coeff.sum()
        kd, kh = coeff/trC, lh/trC
        # definition of the deformation measure
        if E is None:
            E = self.C.E_Hencky()
        Ep = proj.dot(E) # projection of the deformation tensor on the Kelvin projectors
        # Computation of the different element of the Kelvin formulation
        sE = np.array([ct.index_to_mandel(np.array([np.linalg.matrix_power(ct.mandel_to_index(tens),2) for tens in Ep[i].T]).T) for i in range(6)]) #computes the square value of the projected deformation
        ePe = np.array([[np.trace(ct.mandel_to_index(tens)) for tens in sE[i].T] for i in range(6)]) # computes the trace of the squared Kelvin deformation (sE)
        dens_E = trC*np.sum(kd[:-1,np.newaxis]*(ePe[:-1,:]), axis = 0)**a # derivation of the deviatoric energy density regarding ePe
        derivEPE = np.sum(kd[:-1,np.newaxis, np.newaxis]*(Ep[:-1,:,:]), axis = 0)  # derivation ePe regarding E
        Sdeviat = (dens_E*derivEPE) # by virtue of the chain rule, the derivation of the strain energy density regardind E
        Shydro = (k*trC*kh*(kh*ePe[-1,:])**a*Ep[-1,:,:]) # the hydrostatic projector is penalised by the coefficient k, chosen >>1 in order to impose incompressibility
        S = (Sdeviat.T + Shydro.T).T # express the modeled stress as the sum over the deviatoric stress and hydrostatic part. 
        return S

    #def kelvin_exp(self, C):
        #lh, l1, l2, l3, l4, l5 , a = self.params['lh'], self.params['l1'], self.params['l2'], self.params['l3'], self.params['l4'], self.params['l5'], self.params['a']
        #proj = self.params['proj']
        #k = self.params['penal']
        #coeff = np.array([l1, l2, l3, l4, l5, lh])
        #kd, kh = coeff, k*lh
        #E = ct.index_to_mandel(np.array([0.5*logm(ct.mandel_to_index(t)) for t in C.T]).T)
        #Ep = proj.dot(E)
        #sE = np.array([ct.index_to_mandel(np.array([np.linalg.matrix_power(ct.mandel_to_index(tens),2) for tens in Ep[i].T]).T) for i in range(6)])
        #ePe = np.array([[np.trace(ct.mandel_to_index(tens)) for tens in sE[i].T] for i in range(6)])
        #dens_E = np.exp(np.sum(kd[:-1,np.newaxis]*(ePe[:-1,:]), axis = 0)**(a+1))
        #derivEPE = np.sum(kd[:,np.newaxis, np.newaxis]*(Ep[:-1,:,:]), axis = 0)
        #Sdeviat = (np.sum(kd[:,np.newaxis]*(ePe[:-1,:]), axis = 0)**a)*dens_E*derivEPE
        #Shydro = k*(kh*ePe[-1,:])**(a)*np.exp((kh*ePe[-1,:])**(a+1))*Ep[-1,:,:]
        #S = (Sdeviat.T + Shydro.T).T
        #return S
    
    #------------ Resolution functions ----------------
    def costfunc(self, x, uni,  i):
        C = self.C.mandel[:, i]
        C[uni] = x
        E = self.C.E_Hencky()[:,i]
        Mod = np.absolute(self.model(E[:,np.newaxis]))
        return Mod[uni].reshape(len(uni),)
        #return E
    
    def solver(self):
        C = self.C.mandel
        col = np.where((C == 1000) | (C == 1000*np.sqrt(2)))
        if col[0].size !=0:
            x = np.zeros((len(np.unique(col[0])), C.shape[1]))
            uni = np.unique(col[0])
            x[:,0] =[1 if i<=2 else 0 for i in np.unique(col[0])]
            for i in range(1,C.shape[1]):
                x[:,i] = fsolve(self.costfunc, (x[:,i-1]), args = (uni, i))
            C[uni,:] = x
        return self.model()
