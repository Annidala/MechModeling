# -*- coding: utf-8 -*-
import numpy as np
import sympy as sy
import class_densities as cde
import cls_tensors as ct


def fourthorder_indextomandel(tensor):
    """
    This function transforms a Cartesian index symmetric fourth order tensor (3,3,3,3) into the same tensor in Mandel tensor notation (6,6)[1]
    
    Parameters
    ----------
    tensor: a (3,3,3,3) array
    
    Returns
    -------
    tens: a (6, 6) array of the mandel tensors build from tensor
    
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
    tens  = np.array([[tensor[0,0,0,0],tensor[0,0,1,1], tensor[0,0,2,2], np.sqrt(2)*tensor[1,2,0,0], np.sqrt(2)*tensor[0,2,0,0], np.sqrt(2)*tensor[0,1,0,0]],
                      [0,tensor[1,1,1,1], tensor[1,1,2,2], np.sqrt(2)*tensor[1,2,1,1], np.sqrt(2)*tensor[0,2,1,1], np.sqrt(2)*tensor[0,1,1,1]],
                      [0, 0,tensor[2,2,2,2], np.sqrt(2)*tensor[1,2,2,2], np.sqrt(2)*tensor[0,2,2,2], np.sqrt(2)*tensor[0,1,2,2]],
                      [0, 0, 0, 2*tensor[1,2,1,2],2*tensor[0,2,1,2],2*tensor[0,1,1,2]],
                      [0, 0, 0, 0, 2*tensor[0,2,0,2],2*tensor[0,2,0,1]],
                      [0, 0, 0, 0, 0,2*tensor[0,1,0,1]]
                      ])
    inds = np.triu.indices_from(tens)
    tens[(inds[1], inds[0])] = tens[inds]
    return tens


class MakeMechanicalTensors():
    def __init__(self, params, sym = False):
        self.params = params
        self.sym = sym
        self.SD = cde.SymbolicDensities(self.params, self.sym)
        
    def stress_tensor(self):
        T = sy.derive_by_array(self.SD.density, self.SD.tens.C).tomatrix()
        return T
    
    def tangent_matrix(self):
        Ce = sy.Matrix(sy.derive_by_array(self.stress_tensor(), self.SD.tens.C))
        return Ce        
    
    def func_symb(self):
        sub = [(self.SD.tens.c21, self.SD.tens.c12), (self.SD.tens.c32, self.SD.tens.c23), (self.SD.tens.c31, self.SD.tens.c13)]
        tens = (self.SD.tens.c11, self.SD.tens.c22, self.SD.tens.c33, self.SD.tens.c23, self.SD.tens.c13, self.SD.tens.c12)
        e = np.vectorize(sy.lambdify(tens, (self.SD.density).subs(sub)),otypes = [list])
        #print((self.SD.density).subs(sub))
        f = np.vectorize(sy.lambdify(tens, (self.stress_tensor().subs(sub))), otypes = [np.ndarray])
        #g = np.vectorize(sy.lambdify(tens, (self.tangent_matrix().subs(sub))), otypes = [np.ndarray])
        #return e, f, g
        return e, f