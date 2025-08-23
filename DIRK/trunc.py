"""
Truncation Function 

Author: Joost Almekinders
Date: July 4, 2025

"""

import numpy as np
import scipy.linalg
import math 


def trunc(Vx_nn,S_nn,Vy_nn,tol):
    U,S, VT = np.linalg.svd(S_nn, full_matrices=False)
    r = np.sum(S > tol * S[0]) 
    if r == 0:
        r = 1
    

    Vx_nn  = Vx_nn@ U[:,:r]
    Vy_nn = Vy_nn@ VT[:r,:].T
    S_nn = np.diag(S[:r])
    r_nn = r
   

    return Vx_nn, S_nn, Vy_nn,r_nn
