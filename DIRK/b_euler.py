"""
DIRK 1 (B.euler function) Function 
Author: Joost Almekinders
Date: July 3, 2025

"""

import numpy as np

import scipy.linalg

from scipy.linalg import toeplitz
import math 
from scipy.linalg import solve_sylvester

from trunc import trunc

def b_euler(Vx_n,S_n,Vy_n,r_n,dtn,Dxx,Dyy,Nx,Ny,tol,gamma):
    #print("B_euler...")
    #RHS
    RHS = Vx_n@ S_n @Vy_n.T
    
    #equate to augmented 
    Vx_aug = Vx_n 
    Vy_aug = Vy_n 

    # kstep 
    K = solve_sylvester(np.eye(Nx) - gamma * dtn*Dxx, -gamma * dtn *(Dyy @Vy_aug).T@ Vy_aug,  RHS@ Vy_aug)
    Vx_nn, _ = np.linalg.qr(K, mode= 'reduced')

    # L step 
    L = solve_sylvester(np.eye(Ny) - gamma*dtn*Dyy, -gamma* dtn *(Dxx@Vx_aug).T @Vx_aug, RHS.T @ Vx_aug)
    Vy_nn, _ = np.linalg.qr(L, mode='reduced')

    #S step 
    S_nn = solve_sylvester(np.eye(r_n) - gamma* dtn* Vx_nn.T@ (Dxx@ Vx_nn), -gamma* dtn* (Dyy@Vy_nn).T @ Vy_nn, Vx_nn.T @RHS @ Vy_nn )

    #truncate
    Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_nn,S_nn,Vy_nn,tol)

    return Vx_nn, Vy_nn, S_nn, r_nn
