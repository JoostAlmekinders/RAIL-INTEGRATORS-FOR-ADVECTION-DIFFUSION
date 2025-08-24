from scipy.linalg import block_diag
import numpy as np
import scipy.linalg
from trunc import trunc
#from newtrunc import newtrunc
import numpy as np
from scipy.linalg import qr, svd, block_diag


def addmat(Vx_1,S_1,Vy_1,Vx_2,S_2,Vy_2):


    Qx, Rx = qr(np.hstack([Vx_1, Vx_2]), mode='economic') # combine Vx 
    Qy, Ry = qr(np.hstack([Vy_1, Vy_2]), mode='economic') # combine Vy and use Qr 

    Vs_tot = block_diag(S_1, S_2)
    
    val = Rx@Vs_tot@Ry.T

    U,S,V = np.linalg.svd(val, full_matrices=False)


    tol = 1e-12 # tolerance level
    r = np.sum(S > tol)
    if r == 0:
        r = 1

    Vx = Qx@ U[:,:r] 
    S = np.diag(S[:r])
    Vy = Qy@ V[:r, :].T

    return Vx, S, Vy
