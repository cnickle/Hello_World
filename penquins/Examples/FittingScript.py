# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:40:28 2020

@author: Cameron Nickle
@email: camnickle@gmail.com
"""
from scipy.integrate import quad
from scipy.integrate import dblquad
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
import time
start = time.time()

# %% Setting the initial Parameters
eV = 1 #Energy
K  = 1 #Temperature Units
C  = 1 #Coulombs
s  = 1 #seconds
V  = 1 #volts

kb = 8.6173324e-5*eV/K #Boltzmann Constant
q  = 1.6e-19*C
h  = 4.1356e-15*eV*s

initpar = {
        'pos_L'         : -8.976e-01*V,
        'w_L'           :  3.300e-03*V ,
        'pos_H'         : -1.677e-01*V,
        'w_H'           : 4.495e-02*V,
        'p0_H'          : -2.676e-03*V,
        'gammaC1'       : 2.230719e-06*eV,
        'gammaW1'       : 5.388790e-02*eV,
        'deltaE1'       : 3.414365e-01*eV,
        'eta1'          : 0.78,
        'width1'        : 1.445773e-02*eV,
        'gammaC2'       : 1.173901e-05*eV,
        'gammaW2'       : 3.545940e-02*eV,
        'deltaE2'       : 5.594792e-01*eV,
        'eta2'          : 0.78,
        'width2'        : 6.554842e-03*eV,
        'gamC_L'        : 3E-06*5.370e-03*eV,
        'gamW_L'        : 5.370e-03+3E-06*eV,
        'delE_L'        : 0.6*eV,
        'eta_L'         : 5.8e-01,
        'width_L'       : 3.222e-02*eV
        }

# %% Converting from gammaC & gammaW to gammaL and gammaR
gamR_L = .5*(initpar['gammaW2']-np.sqrt(initpar['gammaW2']**2-4*initpar['gammaC2']))
gamR_R = initpar['gammaW2']-gamR_L
print(gamR_L)
print(gamR_R)

# %% Defining the fitting function to be used
@jit
def sigmoid(x,pos,width):
    return 1/(1+np.exp((x-pos)/width))
@jit  
def fermi(E,T):
    return 1/(np.exp((E)/(kb*T))+1)
@jit
def gaussian(x,A, mu,sigma):
    return A*np.exp(-.5*((x-mu)/(sigma))**2)
@jit        
def single_level_tunnel_model_integrand_Alt(E,ep,c,vg,eta,vb,gammaC,gammaW,T):    
    return -gammaC*(fermi(E+vb/2,T)-fermi(E-vb/2,T))/\
        ((E-((ep+c*vg)+(eta-1/2)*vb))**2+(gammaW/2)**2)

def tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T):
    
    if sigma == 0:
        limits = [-1*np.abs(vb),1*np.abs(vb)]
    
        def integrand (E):
            result = single_level_tunnel_model_integrand_Alt(E,deltaE1,c,vg,eta,vb,gammaC,gammaW,T)
            return result
        
        return q/h*quad(integrand,
                           limits[0],
                           limits[1])[0]
    else:
        A = 1
        args = (A,deltaE1,sigma)
        A = 1/quad(gaussian,deltaE1-3*sigma,deltaE1+3*sigma,args=args)[0]
        
        limits = [min([deltaE1-3*sigma,-1*np.abs(vb)]),\
                  max([deltaE1+3*sigma,1*np.abs(vb)])]
        
        def integrand (E,ep):
            result = gaussian(ep,A,deltaE1,sigma)*\
            single_level_tunnel_model_integrand_Alt(E,ep,c,vg,eta,vb,gammaC,gammaW,T)
            return result
        
        return q/h*dblquad(integrand,
                           limits[0],
                           limits[1],
                           lambda x: limits[0],
                           lambda x: limits[1])[0]

def tunnelmodel_2level(vb,c,vg,T,
                       gammaC1, gammaW1, deltaE1, eta1, sigma1,
                       gammaC2, gammaW2, deltaE2, eta2, sigma2):
    
    args1 = (gammaC1, gammaW1, deltaE1, eta1, sigma1,c,vg,T)
    args2 = (gammaC2, gammaW2, deltaE2, eta2, sigma2,c,vg,T)
    
    I1 = tunnelmodel_singleLevel(vb,*args1)
    I2 = tunnelmodel_singleLevel(vb,*args2)
    
    return I1+I2

def model(vb, pos, width,p0,
                   gammaC1_H, gammaW1_H, deltaE1_H, eta1_H,sigma1_H,
                   gammaC2_H, gammaW2_H, deltaE2_H, eta2_H,sigma2_H,
                   gammaL_L, gammaR_L, deltaE_L, eta_L,width_L):
    c = 0
    vg = 0
    T = 300

    args_L = (gammaL_L, gammaR_L, deltaE_L, eta_L,width_L,c,vg,T)
    
    P_H = sigmoid(vb,pos,width)+(vb-1)*p0
    P_L = 1-P_H
    
    I_H = tunnelmodel_2level(vb,c,vg,T,
                                    gammaC1_H,gammaW1_H,deltaE1_H,eta1_H,sigma1_H,
                                    gammaC2_H,gammaW2_H,deltaE2_H,eta2_H,sigma2_H)
    I_L = tunnelmodel_singleLevel(vb,*args_L)
    
    return P_H*I_H+P_L*I_L
model = np.vectorize(model)

# %% Importing the data that was split into high and low conduction states
HCtrack = pd.read_csv('HC.txt', delimiter = '\t', skip_blank_lines=True, header = None)

LCtrack = pd.read_csv('LC.txt', delimiter = '\t', skip_blank_lines=True, header = None)

xtot = np.append(HCtrack[0],LCtrack[0])
yexp = np.append(HCtrack[1],LCtrack[1])

def Everything(initpar):
    #%% Permenant
    args_H = list(initpar.values())[2:]
    mod1=model(HCtrack[0],*args_H)
    
    args_L = list(initpar.values())[0:2]+list(initpar.values())[5:]
    args_L.insert(2,0)
    mod2=model(LCtrack[0],*args_L)
    
    #%% Setting up the Error
    return np.append(mod1,mod2)

ythr = Everything(initpar)
residual = np.subtract(np.log(np.abs(yexp)),np.log(np.abs(ythr)))
Error = np.sqrt(np.sum(residual**2))
print('Error: %.3f'%Error)

#%% Plotting All the things
plt.figure()
plt.scatter(xtot,np.abs(yexp),s=10,color='black')
plt.plot(xtot,abs(ythr),color='red',linewidth=4)
plt.yscale('log')
plt.ylim([1E-15,1E-6])
#
#%% Time 
end = time.time()
print('Total Time (s): %d'%(end-start))