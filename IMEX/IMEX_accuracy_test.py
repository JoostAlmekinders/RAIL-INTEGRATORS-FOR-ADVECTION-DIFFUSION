"""

Generalized Code for IMEX methods

Author: Joost Almekinders
Date: August 8, 2025

"""


import numpy as np
from scipy.linalg import qr, svd
from scipy.linalg import qr, svd, block_diag
from scipy.linalg import toeplitz
import math 
from scipy.linalg import solve_sylvester
import time 
from trunc import trunc
from addmat import addmat
import numpy as np
import matplotlib.pyplot as plt
from IMEX1_fxn import IMEX111
from red_aug_gen import redaug_g
from redaug_2 import redaug_2


def main():
    print("Start....")

    #spatial grid 
    Nx = 200 
    Ny = 200 # (200x200)
    L = 4 * np.pi

    x = np.linspace(-L/2, L/2, Nx+1)
    y = np.linspace(-L/2, L/2, Ny+1)

    dx = x[1]-x[0] 
    dy = y[1]-y[0]

    #cell centered
    x = x[1:Nx+1]-(dx/2)
    y = y[1:Ny+1]-(dy/2)

    ## initial condition
    Tf = 0.5
    #Tf = 2.3
    d = 1/5
    d1 = np.sqrt(d)
    d2 = np.sqrt(d)

    u = np.outer(np.exp(-x**2), np.exp(-3*y**2))
    #u = np.outer(np.exp(-x**2), np.exp(-9*y**2))
    exact = np.outer(np.exp(-x**2), np.exp(-3*y**2)) * np.exp(-2*d*Tf)
    
    

    #create Dxx matrix
    k = np.arange(1, Nx)  
    first_entry = -1 / (3 * (2 * dx / L)**2) - 1/6
    off_diagonal = 0.5 * (-1)**(k + 1) / np.sin((2 * np.pi * dx / L) * k / 2)**2
    first_col = np.concatenate(([first_entry], off_diagonal))

    Dxx = (2 * np.pi / L)**2 * toeplitz(first_col)


    #create Dyy matrix 
    k = np.arange(1, Ny)  
    first_entry = -1 / (3 * (2 * dy / L)**2) - 1/6
    off_diagonal = 0.5 * (-1)**(k + 1) / np.sin((2 * np.pi * dy / L) * k / 2)**2
    first_col = np.concatenate(([first_entry], off_diagonal))

    Dyy = (2 * np.pi / L)**2 * toeplitz(first_col)


    Dxx = d * Dxx
    Dyy = d * Dyy
    
    # Construct column for Dx
    n_x = np.arange(1, Nx)
    columnx = np.concatenate(([0], 0.5 * (-1)**n_x * 1 / np.tan(n_x * (np.pi * dx / L))))
    Dx = (2 * np.pi / L) * toeplitz(columnx, columnx[np.r_[0, Nx-1:0:-1]])

    # Construct column for Dy
    n_y = np.arange(1, Ny)
    columny = np.concatenate(([0], 0.5 * (-1)**n_y * 1 / np.tan(n_y * (np.pi * dy / L))))
    Dy = (2 * np.pi / L) * toeplitz(columny, columny[np.r_[0, Ny-1:0:-1]])

    
    flux1 = [np.ones_like(x), lambda t: -1, y]

    flux2 = [x, lambda t: 1, np.ones_like(y)]
    
    
    g = [np.column_stack([np.exp(-x**2), x * np.exp(-x**2), (x**2) * np.exp(-x**2), np.exp(-x**2)]),lambda t: np.diag([6*d*np.exp(-2*d*t),-4*np.exp(-2*d*t),
        -4*d*np.exp(-2*d*t), -36*d*np.exp(-2*d*t)]),
    np.column_stack([ np.exp(-3*y**2), y * np.exp(-3*y**2), np.exp(-3*y**2),(y**2) * np.exp(-3*y**2)])]
    
    #g = [np.column_stack([0*np.exp(-x**2)]),lambda t: np.diag([0*6*d*np.exp(-2*d*t)]), np.column_stack([ 0*np.exp(-3*y**2),])]
    
    
    L1errvals = []
    lambdav = []

############## IMEX (2,2,2) 

    # ######IMEX(2,2,2) Implicit
    # gamma = 1 - np.sqrt(2)/2
    # delta = 1-(1/(2*gamma))
    # cI = np.array([0,gamma,1])
    # bI = np.array([0,1-gamma,gamma])
    
    # aI = np.array([[0,0,0],[0,gamma,0],[0,1-gamma,gamma]])

    # stage = len(bI)
    
    
    # #######IMEX(2,2,2) Explicit

    # cE = np.array([0,gamma,1])
    # bE = np.array([delta,1-delta,0])

    # aE = np.array([[0,0,0],
    #                [gamma,0,0],
    #               [delta,1-delta,0]])
    

############ IMEX (4,4,3)

    # ##### IMEX (4,4,3) Implicit 
    # cI = np.array([0,1/2,2/3,1/2,1])
    # bI = np.array([0,3/2,-3/2,1/2,1/2])
    
    # aI = np.array([[0,0,0,0,0],[0,1/2,0,0,0],[0,1/6,1/2,0,0],[0,-1/2,1/2,1/2,0],[0,3/2,-3/2,1/2,1/2]])

    # stage = len(bI)


    # ##### IMEX (4,4,3) Explicit
    # cE = np.array([0,1/2,2/3,1/2,1])
    # bE = np.array([1/4,7/4,3/4,-7/4])

    # aE = np.array([[0,0,0,0,0],
    #                [1/2,0,0,0,0],
    #               [11/18,1/18,0,0,0],
    #               [5/6,-5/6,1/2,0,0],
    #               [1/4,7/4,3/4,-7/4,0]])






######### IMEX (2,3,3)


    # ###IMEX (2,3,3) Implicit
    # gamma = (3 + np.sqrt(3))/6
    # cI = np.array([0,gamma, 1-gamma])
    # bI = np.array([0,1/2,1/2])
    
    # aI = np.array([[0,0,0],[0,gamma,0],[0,1-(2*gamma),gamma]])

    # stage = len(bI)


    # #####IMEX (2,3,3) Explicit
    # cE = np.array([0,gamma, 1-gamma])
    # bE = np.array([0,1/2,1/2])

    # aE = np.array([[0,0,0],
    #                [gamma,0,0],
    #               [gamma - 1, 2*(1-gamma),0]])



############### IMEX (2,3,2)

    # #### IMEX (2,3,2) Implicit
    # gamma = (2 - np.sqrt(2))/2
    # delta = (-2*(np.sqrt(2)))/3
    # cI = np.array([0,gamma, 1])
    # bI = np.array([0,1-gamma,gamma])
    
    # aI = np.array([[0,0,0],[0,gamma,0],[0,1-gamma,gamma]])

    # stage = len(bI)


    # #### IMEX (2,3,2) Explicit
    # cE = np.array([0,gamma, 1])
    # bE = np.array([0,1-gamma,gamma])

    # aE = np.array([[0,0,0],
    #                [gamma,0,0],
    #               [delta, (1-delta),0]])






########### IMEX (3,4,3)

    # ###IMEX (3,4,3) Implicit
    # cI = np.array([0,0.4358665215, 0.7179332608,1])
    # bI = np.array([0,1.208496649,-0.644363171,0.4358665215])
    
    # aI = np.array([[0,0,0,0],[0,0.4358665215,0,0],[0,0.2820667392,0.4358665215,0],[0,1.208496649,-0.644363171,0.4358665215]])

    # stage = len(bI)


    # # #####IMEX (3,4,3) Explicit
    # cE = np.array([0,0.4358665215, 0.7179332608,1])
    # bE = np.array([0,1.208496649,-0.644363171,0.4358665215])

    # aE = np.array([[0,0,0,0],
    #                [0.4358665215,0,0,0],
    #               [0.3212788860, 0.3966543747,0,0],
    #               [-0.105858296,0.5529291479 , 0.5529291479, 0]])
    

############## IMEX (1,2,2)

    ######IMEX(1,2,2) Implicit
    cI = np.array([0,1/2])
    bI = np.array([0,1])
    
    aI = np.array([[0,0],[0,1/2]])

    stage = len(bI)
    
    
    #######IMEX(1,2,2) Explicit

    cE = np.array([0,1/2])
    bE = np.array([0,1])

    aE = np.array([[0,0],
                  [1/2,0]])





    
    lambdavals = np.arange(0.1,2.1,0.1)
    #lambdavals = np.arange(0.15,0.25,0.1)

    for k in range (len(lambdavals)):
        dt = lambdavals[k]*dx
        
        t = np.arange(0,Tf,dt)

        #make sure Tf is included otherwise add it
        if t[-1] < Tf:
                t = np.append(t, Tf)

        Nt = len(t) # number of steps 

        
        tol = 1.0e-8

        U,S, VT = np.linalg.svd(u, full_matrices=False) # computes reduced svd 
        r0 = math.ceil(Nx/3)
        Vx_n = U[:, : r0]
        S_n = np.diag(S[:r0])
        Vy_n = VT[:r0,:].T
        
        r_n = r0


        #print(lambdavals[k])
        rankvals = [r_n]
        for n in range (1,Nt):
            dtn = t[n] - t[n-1]
            #print(dtn)
            #print(t[n])
            tn = t[n-1]
            
        
            a1 = flux1[0]
            b1 = flux1[1]
            c1 = flux1[2]
            a2 = flux2[0]
            b2 = flux2[1]
            c2 = flux2[2]
            Gx = g[0]
            Gxy = g[1]
            Gy = g[2]
 

            Yhatx = []
            Yhats = []
            Yhaty = []
            Vx = []
            Vs = []
            Vy = []
            Yx = []
            Ys= []
            Yy = []
            W1x = []
            W1s = []
            W1y = []
            W2x = []
            W2y = []
            W2s = []


            for i in range(stage):
                    if i == 0: 
                        Y1hatx, Y1hats, Y1haty = addmat(Dx@(a1[:, np.newaxis]*Vx_n),-(b1(tn)*S_n),(c1[:, np.newaxis]*Vy_n),(a2[:, np.newaxis]*Vx_n), -(b2(tn)*S_n),(Dy@(c2[:, np.newaxis]*Vy_n)))
                        Yhatx.append(Y1hatx)
                        Yhats.append(Y1hats)
                        Yhaty.append(Y1haty)
            
                    elif i == 1: 
                        Vx_1, S_1, Vy_1, _ = IMEX111(Dxx, Dyy, Dx, Dy, tol, cI[i]*dtn,Vx_n,S_n,Vy_n,r_n,Gx,Gxy,Gy,a1,a2,b1,b2,c1,c2,Nx,Ny,tn)
                        Vx.append(Vx_1)
                        Vs.append(S_1)
                        Vy.append(Vy_1)
                    
                        
                        Y1tempx, Y1temps, Y1tempy = addmat(Dxx@Vx_1,S_1,Vy_1,Vx_1,S_1,Dyy@Vy_1)
                        Y1x,Y1s,Y1y = addmat(Y1tempx, Y1temps, Y1tempy,Gx, (Gxy(tn + cI[1]*dtn)),Gy)
                        Yx.append(Y1x)
                        Ys.append(Y1s)
                        Yy.append(Y1y)

                        Y2hatx, Y2hats, Y2haty = addmat(Dx@(a1[:, np.newaxis]*Vx_1), -b1(tn+ cI[1]*dtn)*S_1 ,c1[:, np.newaxis]*Vy_1, a2[:, np.newaxis]*Vx_1,-b2(tn + cI[1]*dtn)*S_1,Dy@(c2[:, np.newaxis]*Vy_1))
                        Yhatx.append(Y2hatx)
                        Yhats.append(Y2hats)
                        Yhaty.append(Y2haty)
                    else: 
                        Vx_dd, _, Vy_dd, _ = IMEX111(Dxx, Dyy, Dx, Dy, tol, cI[i]*dtn,Vx_n,S_n,Vy_n,r_n,Gx,Gxy,Gy,a1,a2,b1,b2,c1,c2,Nx,Ny,tn)
                        
                        xlis = [Vx_dd] + Vx[::-1] + [Vx_n]
                        ylis = [Vy_dd] + Vy[::-1] + [Vy_n]
                        Vx_star, Vy_star, R = redaug_g(xlis, ylis, 1.0e-12)

                        ##W1
                        W11x, W11s, W11y = Yx[0],aI[i,1]*dtn*Ys[0], Yy[0]
                        for j in range(1,i-1):
                            W11x, W11s, W11y = addmat(W11x, W11s, W11y, Yx[j], aI[i,j+1]*dtn*Ys[j], Yy[j])

                        W11x, W11s, W11y = addmat(W11x, W11s, W11y, Gx, aI[i,i]*dtn*(Gxy(tn+ (cI[i])*dtn)),Gy)


                        ##W2
                        W2x, W2s, W2y = Yhatx[0],aE[i,0]*dtn*Yhats[0], Yhaty[0]
                        for j in range(1,i):
                            W2x, W2s, W2y = addmat(W2x, W2s, W2y, Yhatx[j],aE[i,j]*dtn*Yhats[j], Yhaty[j])


                        W1tempx,W1temps,W1tempy = addmat(Vx_n,S_n,Vy_n,W11x, W11s, W11y)
                        W1x,W1s,W1y = addmat(W1tempx, W1temps,W1tempy,W2x,W2s,W2y)


                        K = solve_sylvester(np.eye(Nx) - aI[i,i]*dtn*Dxx, -aI[i,i]*dtn *(Dyy@Vy_star).T@Vy_star,  W1x@W1s@(W1y.T@Vy_star))
                        Vx_2, _ = np.linalg.qr(K, mode= 'reduced')

                        L = solve_sylvester(np.eye(Ny) - aI[i,i]*dtn*Dyy, -aI[i,i]*dtn *(Dxx@Vx_star).T @Vx_star, W1y@W1s.T@(W1x.T@ Vx_star))
                        Vy_2, _ = np.linalg.qr(L, mode='reduced')

                        S_2 = solve_sylvester(np.eye(R) - aI[i,i]*dtn* Vx_2.T@ (Dxx@ Vx_2), -aI[i,i]*dtn* (Dyy@Vy_2).T @ Vy_2, (Vx_2.T@W1x)@W1s@(W1y.T @ Vy_2))

                        
                        if i+1 == stage:
                            #print("hello")
                            Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_2,S_2,Vy_2,tol)
                            Vx.append(Vx_nn)
                            Vs.append(S_nn)
                            Vy.append(Vy_nn)

                            Y2tempx, Y2temps, Y2tempy = addmat(Dxx@Vx_nn,S_nn,Vy_nn,Vx_nn,S_nn,Dyy@Vy_nn)
                            Y2x,Y2s,Y2y = addmat(Y2tempx, Y2temps, Y2tempy,Gx, (Gxy(tn + (cI[i])*dtn)),Gy)
                            Yx.append(Y2x)
                            Ys.append(Y2s)
                            Yy.append(Y2y)

                            Y3hatx, Y3hats, Y3haty = addmat(Dx@(a1[:, np.newaxis]*Vx_nn), -b1(tn+(cI[i])*dtn)*S_nn ,c1[:, np.newaxis]*Vy_nn, a2[:, np.newaxis]*Vx_nn,-b2(tn + (cI[i])*dtn)*S_nn,Dy@(c2[:, np.newaxis]*Vy_nn))
                            Yhatx.append(Y3hatx)
                            Yhats.append(Y3hats)
                            Yhaty.append(Y3haty)

                        else: 
                            Vx_2, S_2, Vy_2, r_2 = trunc(Vx_2,S_2,Vy_2,tol)

                            Vx.append(Vx_2)
                            Vs.append(S_2)
                            Vy.append(Vy_2)

                            
                            Y2tempx, Y2temps, Y2tempy = addmat(Dxx@Vx_2,S_2,Vy_2,Vx_2,S_2,Dyy@Vy_2)
                            Y2x,Y2s,Y2y = addmat(Y2tempx, Y2temps, Y2tempy,Gx, (Gxy(tn + (cI[i])*dtn)),Gy)
                            Yx.append(Y2x)
                            Ys.append(Y2s)
                            Yy.append(Y2y)

                            Y3hatx, Y3hats, Y3haty = addmat(Dx@(a1[:, np.newaxis]*Vx_2), -b1(tn+(cI[i])*dtn)*S_2 ,c1[:, np.newaxis]*Vy_2, a2[:, np.newaxis]*Vx_2,-b2(tn + (cI[i])*dtn)*S_2,Dy@(c2[:, np.newaxis]*Vy_2))
                            Yhatx.append(Y3hatx)
                            Yhats.append(Y3hats)
                            Yhaty.append(Y3haty)
                                    
            if cI[stage-1] == 1 and np.array_equal(aI[-1], bI) == True and cE[stage-1] == 1 and np.array_equal(aE[-1], bE):
                #Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_2,S_2,Vy_2,tol)
                pass
            
            else: 


                ##Implicit 
                
                
                Vx2,s2,Vy2 = np.zeros(Yx[0].shape),bI[0]*dtn*np.zeros(Ys[0].shape),np.zeros(Yy[0].shape)
                for j in range(1,len(bI)):
                    Vx2, s2,Vy2 = addmat(Vx2,s2,Vy2, Yx[j-1],dtn*bI[j]*Ys[j-1],Yy[j-1])
                
                


                ###Explicit 
                Vx3, s3,Vy3 = Yhatx[0], bE[0]*dtn*Yhats[0], Yhaty[0]
                for j in range(1,len(bE)):
                    Vx3, s3,Vy3 = addmat(Vx3, s3,Vy3,  Yhatx[j], bE[j]*dtn*Yhats[j], Yhaty[j])

                
                ##combine implicit and explicit 

                Vxf,Vsf, Vyf = addmat(Vx2,s2,Vy2, Vx3,s3,Vy3)

                ##final step 
                Vx_nn,S_nn,Vy_nn = addmat(Vx_n, S_n, Vy_n,Vxf,Vsf, Vyf)

                #trunc
                Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_nn,S_nn,Vy_nn,tol) 



    
            rankvals.append(r_nn)
            Vx_n = Vx_nn
            S_n = S_nn
            Vy_n = Vy_nn
            r_n = r_nn
            
       
        u_approx = Vx_n@S_n@Vy_n.T
        L1error = dx * dy* np.sum(np.abs(u_approx - exact))

        lambdav.append(lambdavals[k])  
            
        L1errvals.append(L1error)
            
        print(f"λ = {lambdavals[k]:.2f}, error = {L1error:.3e}")


    #figure 1 
    plt.figure()
    plt.loglog(lambdav, L1errvals,'b',linewidth=1.5)
    plt.loglog(lambdav, 0.0001*np.power(lambdav, 2),'k-.',linewidth=1.5)
    plt.xlabel('Lambda')
    plt.ylabel('L1 error')
    plt.title('IMEX General Code')
    plt.show() 

    #figure2
    plt.figure()
    plt.plot(t[1:],rankvals[1:], 'r',linewidth=2.5)
    plt.xlabel('t')
    plt.ylabel('Rank')
    plt.title("Rank plot")
    plt.show()    


    ##Figure 3 
    X, Y = np.meshgrid(x, y) 
    u_plot = u_approx  
    L_s = 4 * np.pi


    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, u_plot, shading='auto', cmap='viridis')
    plt.colorbar(label='u_approx')
    plt.title(f"Final Approximate Solution at t = {Tf}")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.axis([-L_s/2, L_s/2, -L_s/2, L_s/2])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()



main()
