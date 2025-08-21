
import numpy as np
def redaug_g(Vx_list, Vy_list, tol):
    
    #Vx_comb = np.hstack(Vx_list)
    Qx, Rx = np.linalg.qr(np.hstack(Vx_list), mode='reduced')
    Vx_temp, S_tempx, _ = np.linalg.svd(Rx, full_matrices=False)
    rx = np.sum(S_tempx > tol * S_tempx[0])
    
    #Vy_comb = np.hstack(Vy_list)
    Qy, Ry = np.linalg.qr(np.hstack(Vy_list), mode='reduced')
    Vy_temp, S_tempy, _ = np.linalg.svd(Ry, full_matrices=False)
    ry = np.sum(S_tempy > tol * S_tempy[0])


    r = max(rx,ry)

    Vx_aug = Qx @Vx_temp[:, :r]
    Vy_aug = Qy @Vy_temp[:, :r]

    return Vx_aug, Vy_aug, r 
