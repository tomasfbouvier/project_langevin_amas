# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:44:52 2020

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import pandas as pd

from numba import njit
 




fDriftVelocity = 0.5;
fTransDiff = 0.0000001; 
fLongDiff  = 0.000001;  
driftTimeStep= 0.01


E_y=100000 


n=0
#plt.plot(x,y, 'r.', markersize=10, label=" $ (x_{0},  y_{0}) $")

@njit
def lange_integrate(x0,y0,z0, vDrift_x,vDrift_y, vDrift_z, sigmaTransvStep, sigmaLongStep):
    x=x0; y=y0; z=z0
    while(y>0.5):
        x = rnd.normal(x+1.E-7*vDrift_x*driftTimeStep,sigmaTransvStep);
        y = rnd.normal(y-1.E-7*vDrift_y*driftTimeStep,sigmaLongStep);
        z = rnd.normal(z+1.E-7*vDrift_z*driftTimeStep,sigmaTransvStep);
        #if(n%1==0):
        #   plt.plot(x,y, 'b.', markersize=0.7)
            #print(y)
    return(x,y,z)
@njit
def lange(x0,y0,z0,B_x,B_y,B_z):    
    xs=[]; zs=[]   
    mu = 1.E+5 * fDriftVelocity/E_y;
    moduleB = np.sqrt(B_x*B_x+B_y*B_y+B_z*B_z); 
    cteMod = 1/(1+ mu*mu*moduleB*moduleB);  
    cteMult = mu*cteMod;  
    productEB = E_y*B_y; 
    
    
    vDrift_x = cteMult * (mu*(E_y*B_z) + mu*mu*productEB*B_x); 
    vDrift_y = cteMult * (E_y + mu*mu*productEB*B_y);        
    vDrift_z = cteMult * (mu*(-E_y*B_x) + mu*mu*productEB*B_z);
    
    sigmaTransvStep = np.sqrt(driftTimeStep*2*fTransDiff*cteMod);
    sigmaLongStep = np.sqrt(driftTimeStep*2*fLongDiff);  

    for j in range(20000):
        
        x0,y0,z0= 0,1,0
        x,y,z=lange_integrate(x0,y0,z0, vDrift_x,vDrift_y, vDrift_z, sigmaTransvStep, sigmaLongStep)
        #if(j%1000==0):
            #print(j)
    
        xs.append(x)
        zs.append(z)
    return xs,zs
#plt.yscale([-0.005, 0.007])
#plt.ylabel("y")
#plt.xlabel("x")
#plt.legend(loc="best", fontsize=9)




def main(n):
    b=[]
    for i in range(n):
        print(i)
        x0, y0, z0= 0.1*(2.*np.random.rand() - 1.),1+0.5*(2.*np.random.rand() - 1.),0.1*(2.*np.random.rand() - 1.)
        B_x, B_y, B_z= 0,0,0.
        
        xs,zs=lange(x0,y0,z0, B_x,B_y,B_z)
        #plt.figure()
        h,_,_,_=plt.hist2d(xs,zs, bins=110, range=np.array([[-0.015, 0.015], [-0.015, 0.015]]) )
        #plt.show()
        a=np.reshape(h, -1); a=np.append(a,x0);a=np.append(a,y0);a=np.append(a,z0); b.append(list(a))
    return(b)

b=main(1)
pd.DataFrame(b).to_csv("foo2.csv", mode='a', index=False, header=False)