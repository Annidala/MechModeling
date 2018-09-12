# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm
from scipy.optimize import fsolve, root
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
            derivee = (C0 + C1*(self.C.invariant[1] -3))*self.C.mandel 
            p = derivee[2,:]/self.C.inversetens[2,:]
            S = derivee - p*self.C.inversetens
            return S
        except:
            return ('Error : class function requires two parameters (C0 and C1) to run')
    
    def St_Venant(self, E = None):
        """
        Returns the modeled dual stress computed from a hyperelastic anisotropic kelvin approach, using the Hencky deformation measure.
        
        Parameters
        ----------
        params: defined when calling the class. Dict keys are {'lh', 'l1', 'l2', 'l3', 'l4', 'l5', 'proj', 'penal'}. 'li' are the kelvin modules, 'proj' contains the kelvin eigentensors and 'penal' is defined to deals with incompressibility
        
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

        if E is None:
            E = self.C.egreenlagrange
        Epdev = proj[:].dot(E[1]) # projection of the Green Lagrange deviatoric deformation tensor on the Kelvin deviatoric projectors
        CCt = np.transpose(np.array([1/3.*np.outer(self.C.mandel[:,i], self.C.inversetens[:,i]) for i in range(self.C.mandel.shape[1])]), (1,2,0))
        dEdev = np.einsum('k,ijk->ijk', self.C.invariant[0]**(-1/3.), (np.eye(6) - CCt.T).T)
        dE = np.sum(coeff[:,np.newaxis, np.newaxis]*Epdev, axis = 0)
        Sdeviat = np.array([np.dot(dE[:,i], dEdev[:,:,i]) for i in range(self.C.mandel.shape[1])]).T
        #Shydro = k*lh*proj[-1].dot(E[0])
        Shydro = k*self.C.invariant[0]*(self.C.invariant[0]-1)*self.C.inversetens
        return (Sdeviat.T + Shydro.T).T
    
    
    def kelvin_gl(self, E = None):
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
            E = self.C.egreenlagrange
        Ed = E[1]
        Epdev = proj[:-1].dot(Ed) # projection of the Green Lagrange deviatoric deformation tensor on the Kelvin deviatoric projectors
        # Computation of the different element of the Kelvin formulation
        sEd = np.array([ct.index_to_mandel(np.array([np.linalg.matrix_power(ct.mandel_to_index(tens),2) for tens in Epdev[i].T]).T) for i in range(6)]) #computes the square value of the projected deformation
        ePed = np.array([[np.trace(ct.mandel_to_index(tens)) for tens in sEd[i].T] for i in range(6)]) # computes the trace of the squared Kelvin deformation (sE)
        dens_E = trC*np.sum(kd[:-1,np.newaxis]*(ePed), axis = 0)**a # derivation of the deviatoric energy density regarding ePe
        derivEPE = np.sum(kd[:-1,np.newaxis, np.newaxis]*(Epdev), axis = 0)  # derivation ePe regarding E
        CCt = np.transpose(np.array([np.tensordot(self.C.mandel[:,i], self.C.inversetens[:,i], axes = 0) for i in range(self.C.mandel.shape[1])]), (1,2,0))
        dEdev = np.einsum('k,ijk->ijk', self.C.invariant[0]**(-1/3.),(Id4t - CCt.T).T)        
        Sdeviat = np.einsum('', (dens_E*derivEPE), dEdev) # by virtue of the chain rule, the derivation of the strain energy density regardind E
        
        sEh = np.array([ct.index_to_mandel(np.array([np.linalg.matrix_power(ct.mandel_to_index(tens),2) for tens in (proj[-1].dot(E[0])[i]).T]).T) for i in range(6)])
        ePeh = np.array([[np.trace(ct.mandel_to_index(tens)) for tens in sEh[i].T] for i in range(6)])
        Shydro = (k*trC*kh*(kh*ePed[-1,:])**a*proj[-1].dot(E[0])) # the hydrostatic projector is penalised by the coefficient k, chosen >>1 in order to impose incompressibility
        S = (Sdeviat.T + Shydro.T).T # express the modeled stress as the sum over the deviatoric stress and hydrostatic part. 
        return np.absolute(S)
    
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
            E = self.C.ehencky[0]
        Ep = proj.dot(E) # projection of the deformation tensor on the Kelvin projectors
        # Computation of the different element of the Kelvin formulation
        sE = np.array([ct.index_to_mandel(np.array([np.linalg.matrix_power(ct.mandel_to_index(tens),2) for tens in Ep[i].T]).T) for i in range(6)]) #computes the square value of the projected deformation
        ePe = np.array([[np.trace(ct.mandel_to_index(tens)) for tens in sE[i].T] for i in range(6)]) # computes the trace of the squared Kelvin deformation (sE)
        dens_E = trC*np.sum(kd[:-1,np.newaxis]*(ePe[:-1,:]), axis = 0)**a # derivation of the deviatoric energy density regarding ePe
        derivEPE = np.sum(kd[:-1,np.newaxis, np.newaxis]*(Ep[:-1,:,:]), axis = 0)  # derivation ePe regarding E
        Sdeviat = (dens_E*derivEPE) # by virtue of the chain rule, the derivation of the strain energy density regardind E
        Shydro = (k*trC*kh*(kh*ePe[-1,:])**a*Ep[-1,:,:]) # the hydrostatic projector is penalised by the coefficient k, chosen >>1 in order to impose incompressibility
        S = (Sdeviat.T + Shydro.T).T # express the modeled stress as the sum over the deviatoric stress and hydrostatic part. 
        return np.absolute(S)

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
        tens = self.C.mandel[:, i]
        tens[uni] = x
        self.C.mandel = self.C.mandel  #update the mandel tensors - cf Tensors class and lazyproperty
        Mod = self.model()  # compute the model - model is chosen in the class parameters (params)
        return Mod[uni, i].reshape(len(uni),)  # costfunc returns only the modeled data corresponding to the unknown part of the deformation tensor. It is supposed that the modeled data should there be equal to zero.

    
    def solver(self, e = 0):
        tens = self.C.mandel.copy()
        col = np.where((tens == 1000) | (tens == 1000*np.sqrt(2)))
        if col[0].size !=0:
            y = np.zeros((len(np.unique(col[0])), tens.shape[1]))
            uni = np.unique(col[0])
            y[:,0] =[1 if i<=2 else 0 for i in np.unique(col[0])]
            for i in range(1,tens.shape[1]):
                y[:,i] = fsolve(self.costfunc, (y[:,i-1]), args = (uni, i))
            tens[uni,:] = y
            self.C.mandel = tens
            print('coucou')
        return self.model()
