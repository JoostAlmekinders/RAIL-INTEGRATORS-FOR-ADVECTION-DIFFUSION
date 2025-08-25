# RAIL-INTEGRATORS-FOR-ADVECTION-DIFFUSION

This library provides the general code for the RAIL algorithm from this paper: REDUCED AUGMENTATION IMPLICIT LOW-RANK (RAIL) INTEGRATORS FOR ADVECTION-DIFFUSION AND FOKKER--PLANCK MODELS by JOSEPH NAKAO, JING-MEI QIU, AND LUKAS EINKEMMER. DOI.10.1137/23M1622921(https://doi.org/10.1137/23M1622921). 

**Note**: Any use of this code must cited. 

**Reference**: 
Nakao, Joseph, Jing-Mei Qiu, and Lukas Einkemmer. "Reduced Augmentation Implicit Low-rank (RAIL) integrators for advection-diffusion and Fokker–Planck models." SIAM Journal on Scientific Computing 47.2 (2025): A1145-A1169.

---

The code in this library is a generalization of code necessary to use **DIRK** and **IMEX** methods to solve PDEs.  
Any valid Butcher tableau for DIRK or IMEX methods can be inserted and the code will provide the necessary output.  

---

## 📂 Structure

The library is split into two main parts:  

- **DIRK methods**  
- **IMEX methods**  

Within each folder you will see the possibility to run two different tests:  

- The **accuracy test**  
- The **rank test**  

The code includes a few Butcher tableaus, but **any valid Butcher tableau can be added**.  

---


Example: 
For DIRK 2 we insert the butcher table in the necessary area within the code like so: 
``` </pre>
    g = 1-(np.sqrt(2)/2) 
    cvals = np.array([g,1])
    bvals = np.array([1-g,g])
    avals = np.array([[g,0],
                       [1-g,g]])   
    Stage = len(cvals)
```

We must then edit the order of the accuracy plot at the bottom of the file to ensure that we have second order for this method.
This line must be edited to look like this: 
``` </pre> 
plt.loglog(lambdav, np.power(lambdav, 2),'k-.',linewidth=1.5).
```

The code will print the error value at each lamda value like this: 
``` </pre>
Start....
λ = 0.10, error = 4.898e-06
λ = 0.20, error = 1.979e-05
λ = 0.30, error = 4.418e-05
λ = 0.40, error = 7.912e-05
λ = 0.50, error = 1.236e-04
λ = 0.60, error = 1.761e-04
λ = 0.70, error = 2.369e-04
λ = 0.80, error = 3.166e-04
λ = 0.90, error = 3.921e-04
λ = 1.00, error = 4.950e-04
```


Ultimately, we will see the accuracy plot with second order for DIRK 2. 
<img width="631" height="477" alt="Screenshot 2025-08-24 at 11 23 37 AM" src="https://github.com/user-attachments/assets/c56e02f6-7285-4403-916f-48d4d80c01e9" />

If we run the rank test we get this figure. 


<img width="626" height="472" alt="Screenshot 2025-08-24 at 11 25 55 AM" src="https://github.com/user-attachments/assets/20391cac-760d-4832-bd53-a88cc0894440" />



