# RAIL-INTEGRATORS-FOR-ADVECTION-DIFFUSION

This library provides the general code for the RAIL algorithm from this paper: REDUCED AUGMENTATION IMPLICIT LOW-RANK (RAIL) INTEGRATORS FOR ADVECTION-DIFFUSION AND FOKKER--PLANCK MODELS by JOSEPH NAKAO, JING-MEI QIU, AND LUKAS EINKEMMER. DOI.10.1137/23M1622921(https://doi.org/10.1137/23M1622921). 



The code in this library is a generaliztion of code necessary to use DIRK and IMEX moethods to solve PDE's. Any valid butcher table for DIRK or IMEX methods can be inserted and the code will provide the necessary output. 

#Structure:
The library is split into two parts, one with the code for the DIRK methods and the other for IMEX methods. Within each folder you will see the possibilty to run two different tests: the accuracy test and the rank test. 
