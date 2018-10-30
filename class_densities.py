# -*- coding: utf-8 -*-
import numpy as np
import sympy as sy
from sympy.interactive import printing
printing.init_printing(use_latex = 'mathjax')
from IPython.display import display

def mandel_to_index(tensor):
    """
    This function transforms a Mandel tensor (symmetric tensor written in a second rank tensor basis of shape (6,1)) into the Cartesian index notation (3,3)[1]
    
    Parameters
    ----------
    tensor: ndarray of shape (6,n). Along axis 0 : components of the tensor. n >=1  with n the number of tensors to transform.
    
    Returns
    -------
    tens: a (3,3, n) shape array with the n cartesian index tensors.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1,9,20)
    >>> mand = np.array([x,x,1/x**2, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)])
    >>> mand.shape
    ... (6,20)
    >>> cart = mandel_to_index(mand)
    >>> cart.shape
    ... (3,3,20)
    
    [1] M. M. Mehrabadi and S. C. Cowin, Eigentensors of linear anisotropic elastic materials, Q. J. Mech. Appl. Math., vol. 44, no. 2, p. 331, 1991.
    """
    tens = np.array([[tensor[0], tensor[5]/np.sqrt(2), tensor[4]/np.sqrt(2)],
                     [tensor[5]/np.sqrt(2), tensor[1], tensor[3]/np.sqrt(2)],
                     [tensor[4]/np.sqrt(2), tensor[3]/np.sqrt(2), tensor[2]]])
    return tens


def index_to_mandel(tensor):
    """
    This function transforms a Cartesian index symmetric second order tensor (3,3) into the same tensor in Mandel notation (6,)[1]
    
    Parameters
    ----------
    tensor: a (3,3,n) array of n  cartesian index second order symmetric tensors
    
    Returns
    -------
    tens: a (6, n) array of the mandel tensors build from 
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1,9,20)
    >>> cart = np.array([x,x,1/x**2, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)])
    >>> cart.shape
    ... (3,3,20)
    >>> mand = index_to_mandel(cart)
    >>> mand.shape
    ... (6,20)
    
    References
    ----------
    [1] M. M. Mehrabadi and S. C. Cowin, Eigentensors of linear anisotropic elastic materials, Q. J. Mech. Appl. Math., vol. 44, no. 2, p. 331, 1991.
    """
    tens = np.array([tensor[0,0],
                     tensor[1,1],
                     tensor[2,2],
                     tensor[2,1]*np.sqrt(2),
                     tensor[2,0]*np.sqrt(2),
                     tensor[1,0]*np.sqrt(2)])
    return tens

class SymbolicDensities():
    
    def __init__(self, params, sym = False):
        self.sym = sym
        self.F = self.symb_tensors()
        self.params = params
        self.density = getattr(self, params['dens'])
        
    def symb_tensors(self):
        f11 = sy.Symbol('f11')
        f12 = sy.Symbol('f12')
        f13 = sy.Symbol('f13')
        f22 = sy.Symbol('f22')
        f21 = sy.Symbol('f21')
        f23 = sy.Symbol('f23')
        f31 = sy.Symbol('f31')
        f32 = sy.Symbol('f32')
        f33 = sy.Symbol('f33')
        if self.sym:
            F = sy.Matrix([f11, f22, f33, f23, f13, f12])
        else:
            F = sy.Matrix([[f11, f12, f13],
              [f21, f22, f23],
              [f31, f32, f33]])
        return F
    
    def invariant_2(self):
        I2 = sy.Rational(1,2)*(sy.trace(self.F)**2 - sy.trace(self.F**2))
        return I2
        
    
    def yeoh(self):
        C0, C1 = self.params['C0'], self.params['C1']
        W = C0*(sy.trace(self.F) -3) + C1*(sy.trace(self.F)-3)**2
        return W
    
    def mooney_rivlin(self):
        C1, C2 = self.params['C1'], self.params['C2']
        W = C1*(sy.trace(self.F)-3) + C2*(self.invariant_2() - 3)
        return W
    
    def kelvin_hyperelas_dev(self):
        l1, l2, l3, l4, l5 , a = self.params['l1'], self.params['l2'], self.params['l3'], self.params['l4'], self.params['l5'], self.params['a'] # Kelvin modulus and power order of the law
        proj = self.params['proj'] # kelvin projectors - preferably choose only deviatoric projectors
        l = np.array([l1, l2, l3, l4, l5])
        ld= l/l.sum()
        dens = 0
        for i, l in enumerate(kd):
            dens+=(l*(self.F.T)*proj[i]*self.F)[0]
        W = coeff.sum()*dens**a
        return W
        