""" 
RAIL for DIRK Methods general code.

Author: Joost Almekinders
Date: July 15, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg

from scipy.linalg import toeplitz
import math 
from scipy.linalg import solve_sylvester
from b_euler import b_euler
from red_aug import redaug
from trunc import trunc
from addmat import addmat
from red_aug_gen import redaug_g
import time 


def main():
    print("Start....")
    start_time = time.time()

    #spatial grid (100x100)
    Nx = 100 
    Ny = 100 
    
    #Length of Domain
    L = 1 

    #create Nx + 1 bins
    x = np.linspace(0,L,Nx+1)
    y = np.linspace(0,L,Ny+1)

    dx = x[1]-x[0] 
    dy = y[1]-y[0]

    #cell centered
    x = x[1:Nx+1]-(dx/2)
    y = y[1:Ny+1]-(dy/2)

    # Final time 
    Tf = 0.3

    ##initial condition 
    d1 = 1/4
    d2 = 1/9
    u = np.outer(np.sin((2*np.pi*x)/L), np.sin((2*np.pi*y)/L))
    exact = np.exp(-(2*np.pi/L)**2*(d1+d2)*Tf) * np.outer(np.sin((2*np.pi*x)/L), np.sin((2*np.pi*y)/L))
    

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

    Dxx = d1 * Dxx
    Dyy = d2 * Dyy

    ### initiate empty list for error values and lamda values 
    L1errvals = []
    lambdav = []


    ### Butcher tableau (DIRK 2)

    # g = 1-(np.sqrt(2)/2) 
    # cvals = np.array([g,1])
    # bvals = np.array([1-g,g])

    # avals = np.array([[g,0],
    #                   [1-g,g]])
    
    # Stage = len(cvals)



    ## Other DIRK third order
    #############g = (3 - np.sqrt(3))/6   DO NOT USE 

    # g = (3 + np.sqrt(3))/6
    
    # cvals = np.array([g,1-g])
    # bvals = np.array([1/2,1/2])

    # avals = np.array([[g,0],
    #                   [1-(2*g),g]])
    # Stage = len(cvals)


    # DIRK 3
    # g = 0.435866521508459
    # beta1 = -(3/2)*g**2 + 4*g - (1/4)
    # beta2 = (3/2)*g**2 - 5*g + (5/4)
    # cvals = np.array([g,(1+g)/2,1])
    # bvals = np.array([beta1,beta2,g])

    # avals = np.array([[g,0, 0],
    #                   [(1-g)/2,g,0],
    #                   [beta1,beta2,g]])
    # Stage = len(cvals)



    ###IMEX (1,2,2) Implicit 1-stge method with second order
    # cvals = np.array([0,(1/2)])
    # bvals = np.array([0,1])

    # avals = np.array([[0,0],
    #                   [0,1/2]])
    # Stage = len(cvals)


        ###DIRK 4 stiffly accurate 
    
    # cvals = np.array([(1/4),(3/4), 11/20,1/2,1])
    # bvals = np.array([25/24,-49/48,125/16,-85/12,1/4])

    # avals = np.array([[1/4,0,0,0,0],
    #                   [1/2,1/4,0,0,0],
    #                   [17/50,-1/25,1/4,0,0],
    #                   [371/1360, -137/2720, 15/544, 1/4,0],
    #                   [25/24,-49/48,125/16,-85/12,1/4]])
    
    # Stage = len(cvals)


    # #4 stage DIRK3 stiffly accurate 
    # cvals = np.array([1/2,1/4,3/2,1])
    # bvals = np.array([-1/12,2/3,-1/12,1/2])

    # avals = np.array([[1/2,0,0,0],
    #                  [-1/4,1/2,0,0],
    #                  [-1,2,1/2,0],
    #                  [-1/12,2/3,-1/12,1/2]
    # ])
    
    # Stage = len(cvals)


    ###4th order 3-stage (NORSETT)
    g = 1.068579021301629
    cvals = np.array([g,1/2,1-g])
    bvals = np.array([1/(6*(1-2*g)**2),(3*(1-2*g)**2-1)/(3*(1-2*g)**2),1/(6*(1-2*g)**2)])

    avals = np.array([[g,0,0],
                      [1/2-g,g,0],
                      [2*g,1-4*g, g]])
    
    Stage = len(cvals)


    ##
    lambdavals = np.arange(0.1, 6.1, 0.1)

    ##Start of main loop going through each lambda value 
    for k in range (len(lambdavals)):
        dt = lambdavals[k]*dx
        
        t = np.arange(0,Tf,dt)

        #make sure Tf is included otherwise add it
        if t[-1] < Tf:
                t = np.append(t, Tf)

        Nt = len(t) # number of steps 

        tol = 1.0e-6

        U,S, VT = np.linalg.svd(u, full_matrices=False) # computes reduced svd 
        r0 = math.ceil(Nx/3)
        Vx_n = U[:, : r0]
        S_n = np.diag(S[:r0])
        Vy_n = VT[:r0,:].T
        r_n = r0

        

        rankvals = [r_n]
        for n in range (1,Nt):
            dtn = t[n] - t[n-1]
            #print(t[n])
            
            Yx = []
            Ys = []
            Yy = []
            Vx_list = []
            Vy_list = []
            Vx_stage = []
            Vy_stage = []
            S_stage = []
            r_stage = []
            

            for i in range(Stage):

                if i == 0: 
                    Vx_1, Vy_1, S_1, r_1 = b_euler(Vx_n,S_n,Vy_n,r_n,dtn,Dxx,Dyy,Nx,Ny,tol,cvals[i])
                    
                    
                    
                    Vx_stage.append(Vx_1)
                    Vy_stage.append(Vy_1)
                    S_stage.append(S_1)
                    r_stage.append(r_1)
                    #print(r_1)

                    Vx_list.append(Vx_1)
                    Vy_list.append(Vy_1)

                    #comrpess
                    Y1x, Y1s, Y1y = addmat(Dxx@Vx_1,S_1,Vy_1,Vx_1,S_1,Dyy@Vy_1)
                    
                    Yx.append(Y1x)
                    Ys.append(Y1s)
                    Yy.append(Y1y)
                    
                else: 
                    Vx_d,Vy_d, _,_ = b_euler(Vx_n,S_n,Vy_n,r_n,dtn,Dxx,Dyy,Nx,Ny,tol,cvals[i])
                    #print(Vx_d.shape)
                    
                    #print(cvals[i])
                    
                    Vx_temp = [Vx_d] + Vx_list[::-1] + [Vx_n]
                    Vy_temp = [Vy_d] + Vy_list[::-1] + [Vy_n]

                    # reduced augmentation
                    Vx_star, Vy_star, r_star = redaug_g(Vx_temp, Vy_temp, 1.0e-12)
                    #Vx_star, Vy_star, r_star= redaug(Vx_d,Vx_1,Vx_n, Vy_d,Vy_1,Vy_n, 1.0e-12)
                    #compress
                    
                    #print(r_star)
                    
                    # print(Vx_star.shape)
                    # print(Vy_star.shape) 
                    # print(r_star)

                    W1x, W1s, W1y = Vx_n, S_n, Vy_n
                    for j in range(i):
                        W1x, W1s, W1y = addmat(W1x, W1s, W1y, Yx[j], avals[i,j]*dtn*Ys[j], Yy[j])
                    #print(W1s.shape)
                    
                     

                    Vx_aug = Vx_star 
                    Vy_aug = Vy_star

                    # kstep 
                    K = solve_sylvester(np.eye(Nx) - avals[i,i] * dtn*Dxx, -avals[i,i] * dtn *(Dyy@Vy_aug).T@Vy_aug,  W1x@W1s@(W1y.T@Vy_aug))
                    Vx_nn, _ = np.linalg.qr(K, mode= 'reduced')
                    

                    # L step 
                    L = solve_sylvester(np.eye(Ny) - avals[i,i]*dtn*Dyy, -avals[i,i]* dtn *(Dxx@Vx_aug).T @Vx_aug, W1y@W1s.T@(W1x.T@ Vx_aug))
                    Vy_nn, _ = np.linalg.qr(L, mode='reduced')
                    
                    #Vx_nn, Vy_nn, r  = redaug_g([Vx_nn,Vx_1,Vx_n],[Vy_nn,Vy_1,Vy_n], 1.0e-12)

                    #S step 
                    S_nn = solve_sylvester(np.eye(r_star) - avals[i,i]* dtn* Vx_nn.T@ (Dxx@ Vx_nn), -avals[i,i]* dtn* (Dyy@Vy_nn).T @ Vy_nn, (Vx_nn.T@W1x)@W1s@(W1y.T @ Vy_nn ))
                    #S_nn = solve_sylvester(np.eye(r) - avals[i,i]* dtn* Vx_nn.T@ (Dxx@ Vx_nn), -avals[i,i]* dtn* (Dyy@Vy_nn).T @ Vy_nn, (Vx_nn.T@W1x)@W1s@(W1y.T @ Vy_nn ))
                    

                    if i + 1 >= Stage: 
                        #Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_nn,S_nn,Vy_nn,tol)
                        #print(r_nn)
                        
                        Vx_stage.append(Vx_nn)
                        Vy_stage.append(Vy_nn)
                        S_stage.append(S_nn)
                        #r_stage.append(r_nn)
                        
                    else: 
                        Vx_2, S_2, Vy_2, r_2 = trunc(Vx_nn,S_nn,Vy_nn,tol)
                        #print(r_2)
                        Y1x, Y1s, Y1y = addmat(Dxx@Vx_2,S_2,Vy_2,Vx_2,S_2,Dyy@Vy_2)
                        Yx.append(Y1x)
                        Ys.append(Y1s)
                        Yy.append(Y1y)

                        Vx_stage.append(Vx_2)
                        Vy_stage.append(Vy_2)
                        S_stage.append(S_2)
                        r_stage.append(r_2)
                    
                    
                    
     
            if cvals[Stage-1] == 1 and np.array_equal(avals[-1], bvals) == True:
                Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_nn,S_nn,Vy_nn,tol)
                #pass   
            else:

                x = []
                s = []
                y = []
                for i in range(Stage): 
                    Vx, s1, Vy = addmat(Dxx@Vx_stage[i],bvals[i]*dtn*S_stage[i], Vy_stage[i], Vx_stage[i],bvals[i]*dtn*S_stage[i],Dyy@Vy_stage[i])
                    x.append(Vx)
                    s.append(s1)
                    y.append(Vy)
           
                Vx2, s2,Vy2 = x[0],s[0],y[0]
                for j in range(0, len(x) - 1, 2): ### THIS FAILS TO HANDLE IF There is an odd number 
                    Vx2, s2,Vy2 = addmat(x[j],s[j],y[j],x[j+1],s[j+1],y[j+1])
                
                #if stage is an odd value 
                if Stage % 2 == 1: 
                    Vx2, s2,Vy2 = addmat(x[-1],s[-1],y[-1],Vx2, s2,Vy2)

                
                Vx_nn,S_nn,Vy_nn = addmat(Vx_n, S_n, Vy_n,Vx2, s2,Vy2)
                # print(Vx_nn)
                # return

                Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_nn,S_nn,Vy_nn,tol)    
                #print(Vx_nn)
                #print(S_nn)
                #print(Vy_nn)
                #print(r_nn)
                #return  

                #print('help')
            #Vx_nn, S_nn, Vy_nn, r_nn = trunc(Vx_nn,S_nn,Vy_nn,tol)

            #print(r_nn)
            
            rankvals.append(r_nn)
            Vx_n = Vx_nn
            S_n = S_nn
            Vy_n = Vy_nn
            r_n = r_nn


        #u_approx = Vx_nn@S_nn@Vy_nn.T
        u_approx = Vx_n @ S_n @ Vy_n.T
        L1error = dx * dy* np.sum(np.abs(u_approx - exact))

        lambdav.append(lambdavals[k])  
            
        L1errvals.append(L1error)
            
        print(f"λ = {lambdavals[k]:.2f}, error = {L1error:.3e}")
    
    #figure 1 
    plt.figure()
    plt.loglog(lambdav, L1errvals,'b',linewidth=1.5)
    #plt.loglog(lambdav, np.power(lambdav, Stage),'k-.',linewidth=1.5)
    plt.loglog(lambdav, 0.00001*np.power(lambdav, 4),'k-.',linewidth=1.5)
    #plt.loglog(lambdav, np.power(lambdav, 2),'k-.',linewidth=1.5)
    plt.xlabel('Lambda')
    plt.ylabel('L1 error')
    plt.title('RAIL General w/ RHScomp')
    plt.show() 

    #figure2
    plt.figure()
    plt.plot(t[1:],rankvals[1:], 'r',linewidth=2.5)
    plt.xlabel('t')
    plt.ylabel('Rank')
    plt.title("Rank plot")
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")




main()
