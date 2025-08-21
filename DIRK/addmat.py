"""
Add matrix function 

Author: Joost Almekinders
Date: July 10,2025

"""
from scipy.linalg import block_diag
import numpy as np
import scipy.linalg
from trunc import trunc
from newtrunc import newtrunc
import numpy as np
from scipy.linalg import qr, svd, block_diag


def addmat(Vx_1,S_1,Vy_1,Vx_2,S_2,Vy_2):
    
    # Vxtot = np.hstack((Vx_1, Vx_2))
    # Vytot = np.hstack((Vy_1, Vy_2))

    # Qx,Rx = np.linalg.qr(Vxtot, mode='economic')
    # Qy,Ry = np.linalg.qr(Vytot, mode='economic')

    Qx, Rx = qr(np.hstack([Vx_1, Vx_2]), mode='economic')
    Qy, Ry = qr(np.hstack([Vy_1, Vy_2]), mode='economic')

    Vs_tot = block_diag(S_1, S_2)
    
    val = Rx@Vs_tot@Ry.T

    U,S,V = np.linalg.svd(val, full_matrices=False)

    #r = np.max(np.where(np.diag(S) > 1.0e-12)[0]) + 1 if np.any(np.diag(S) > 1.0e-12) else 0
    tol = 1e-12
    r = np.sum(S > tol)
    if r == 0:
        r = 1

    Vx = Qx@ U[:,:r] 
    S = np.diag(S[:r])
    Vy = Qy@ V[:r, :].T
    #Vy = Qy@ V[:,:r]
    return Vx, S, Vy
