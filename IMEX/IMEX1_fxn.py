"""

IMEX (1,1,1) Functions 
Author: Joost Almekinders
Date: August 5, 2025 

"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg

from scipy.linalg import toeplitz
import math 
from scipy.linalg import solve_sylvester
from addmat import addmat
from trunc import trunc

from red_aug_gen import redaug_g



def IMEX111(Dxx, Dyy, Dx, Dy, tol, dtn,Vx_n,S_n,Vy_n,r_n,Gx,Gxy,Gy,a1,a2,b1,b2,c1,c2,Nx,Ny,tn):
            
            Vx_star, Vy_star, _ = redaug_g([np.empty((Vx_n.shape[0], 0)),Vx_n], [np.empty((Vy_n.shape[0], 0)), Vy_n], 1.0e-12)
            
            tempx, temps, tempy = addmat(Vx_n, S_n, Vy_n, Dx @ (a1[:, np.newaxis] * Vx_n),-dtn*b1(tn) *S_n, c1[:, np.newaxis]*Vy_n)
            tempx2, temps2, tempy2 = addmat(tempx, temps, tempy, a2[:, np.newaxis] * Vx_n,-dtn*b2(tn)*S_n, Dy@(c2[:, np.newaxis]*Vy_n))
            
            Wx, Ws, Wy = addmat(tempx2, temps2, tempy2, Gx, dtn*Gxy(tn + dtn), Gy)

            #k-step
            K = solve_sylvester(np.eye(Nx) - dtn*Dxx, -dtn *(Dyy@Vy_star).T@Vy_star,  Wx@Ws@(Wy.T@Vy_star))
            Vx_nn, _ = np.linalg.qr(K, mode= 'reduced')

            #L-step
            L = solve_sylvester(np.eye(Ny) - dtn*Dyy, -dtn *(Dxx@Vx_star).T @Vx_star, Wy@Ws.T@(Wx.T@ Vx_star))
            Vy_nn, _ = np.linalg.qr(L, mode='reduced')

            #reduced augmentation 
            Vx_nn, Vy_nn, R = redaug_g([Vx_nn,Vx_n], [Vy_nn, Vy_n], 1.0e-12)

            #S-step
            S_nn = solve_sylvester(np.eye(R) - dtn* Vx_nn.T@ (Dxx@ Vx_nn), -dtn* (Dyy@Vy_nn).T @ Vy_nn, (Vx_nn.T@Wx)@Ws@(Wy.T @ Vy_nn ))


            #final truncation 
            Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_nn,S_nn,Vy_nn,tol)

            return Vx_nn, S_nn, Vy_nn, r_nn
 
