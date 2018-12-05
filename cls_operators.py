# -*- coding: utf-8 -*-
import numpy as np

class FourthOrderOperators():
    """
    The present class returns any 4th rank tensors written in the 6-dimensional second-rank tensor notation [1]
    
    References
    ----------
    [1] M. M. Mehrabadi and S. C. Cowin, Eigentensors of linear anisotropic elastic materials, Q. J. Mech. Appl. Math., vol. 44, no. 2, p. 331, 1991.
    Parameters
    ----------
    C : Default None. When specified C is the fourth rank elasticity tensor written in a 6-dimensional tensor space. It is a (6,6) symmetric array.
    """
    def __init__(self, C = None):
        self.C = C
        self.vap = self.get_projectors()[0]
        self.P1, self.P2, self.P3, self.P4, self.P5, self.Ph = self.get_projectors()[1]
        self.projs = self.get_projectors()
        self.rot = self.get_rot_tens()
    
    def get_projectors(self):
        """
        Class method gives the Kelvin projectors and modules associated with the elasticity tensor C
        If the parameters C is not specified the function only returns isotropic projectors. (this has to be corrected in the future)
        
        Returns
        -------
        vap: eigenvalues of tensor C
        vep: eigenvectors of tensor C
        """
        if self.C == None: 
            P1 = np.array([[4,-2,-2,0,0,0],
               [-2,1,1,0,0,0],
               [-2,1,1,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]])/6.
            P2 = np.array([[0,0,0,0,0,0],
                        [0,1,-1,0,0,0],
                        [0,-1,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])/2.
            P3 = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])
            P4 = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,0]])
            P5 = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,1]])
            Ph = np.array([[1,1,1,0,0,0],
                        [1,1,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])/3.
            vep = np.array([P1, P2, P3, P4, P5, Ph])
            vap = 0
        else:
            vap, vep = np.linalg.eig(C)
        return vap, vep
        
    def get_rot_tens(self, theta = 0., axe = 'z'):
        """
        The function returns the rotation matrix along a precised axis in the 6 dimensional tensor notation [1]
        
        Parameters
        ----------
        theta: float, an angle in radians
        axe: str, the axe of rotation, in the present version, axe is equal to 'x', 'y' or 'z'
        
        Returns
        -------
        R: a (6,6) rotation array. Works only to rotate mandel written 2nd order tensors.
        
        [1] B. A. Auld : Acoustic Fields and Waves in Solids, vol. 2. Robert E.Kreiger Publishing Compagny, Inc., Kriger Drive, Florida 3950, new material édn, 1973. ISBN 0-89874-782-1.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        if axe =='z':
            R = np.array([[c**2, s**2, 0, 0 , 0, np.sqrt(2)*s*c], 
                    [s**2, c**2, 0, 0, 0, -np.sqrt(2)*s*c], 
                    [0, 0, 1, 0, 0, 0], 
                    [0, 0, 0, c, -s, 0], 
                    [0, 0, 0, s, c, 0],
                    [-np.sqrt(2)*s*c, np.sqrt(2)*s*c, 0, 0, 0, c**2 - s**2]])
        elif axe =='y':
            R = np.array([[c**2, 0, s**2, 0, np.sqrt(2)*s*c, 0], 
                    [0, 1, 0, 0, 0, 0], 
                    [s**2, 0, c**2, 0, -np.sqrt(2)*c*s, 0], 
                    [0, 0, 0, c, 0, -s], 
                    [-np.sqrt(2)*c*s, 0, np.sqrt(2)*s*c, 0, c**2-s**2, 0],
                    [0, 0, 0, s, 0, c]])
        elif axe =='x':
            R = np.array([[1, 0, 0, 0, 0, 0], 
                    [0, c**2, s**2, np.sqrt(2)*c*s, 0, 0], 
                    [0, s**2, c**2, -np.sqrt(2)*c*s, 0, 0], 
                    [0, -np.sqrt(2)*c*s, np.sqrt(2)*c*s, c**2 - s**2, 0, 0], 
                    [0, 0, 0, 0, c, -s],
                    [0, 0, 0, 0, s, c]])
        else:
            R = ('error, axe should be x, y or z')
            return R
