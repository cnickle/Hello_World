# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:46:23 2019
Functions to be used in fitting and calculations
@author: Cameron
"""
import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from numba import jit

eV = 1 #Energy
K  = 1 #Temperature Units
C  = 1 #Coulombs
s  = 1 #seconds
V  = 1 #volts

kb = 8.6173324e-5*eV/K #Boltzmann Constant
q  = 1.6e-19*C
h  = 4.1356e-15*eV*s

# %% Some Basic Mathematical Functions
@jit
def linear(x,m,b):
    return b+x*m

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
def normalized_gaussian(x, mu,sigma):
    A = 1
    def gaus(ep):
        return gaussian(ep,A,mu,sigma)
    
    A = 1/quad(gaus,mu-3*sigma,mu+3*sigma)[0]    
    return gaussian(x, A, mu, sigma)

# %% Some Basic Physics Functions
@jit
def densityOfStates(E,ep,gamma):
    numerator = gamma
    denominator = (E-ep)**2+(gamma/2)**2
    return numerator/denominator#/(2*np.pi)

@jit
def rateRatio(gammaL,gammaR):
    return gammaL*gammaR/(gammaL+gammaR)

# %% The Following functions All Deal with the
# Single level tunnel Model
@jit        
def single_level_tunnel_model_integrand_Alt(E,ep,c,vg,eta,vb,gammaC,gammaW,T):    
    return -gammaC*(fermi(E+vb/2,T)-fermi(E-vb/2,T))/\
        ((E-((ep+c*vg)+(eta-1/2)*vb))**2+(gammaW/2)**2)

# The following function is a 'catch all' function. If sigma != 0 then
# then the function returns the gaussian version.
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

# This is the 2 level version of the Single Level tunnel model.
# This isn't really needed to be a seperate model as you simply need
# To add the contributions from each level.
def tunnelmodel_2level(vb,c,vg,T,
                       gammaC1, gammaW1, deltaE1, eta1, sigma1,
                       gammaC2, gammaW2, deltaE2, eta2, sigma2):
    
    args1 = (gammaC1, gammaW1, deltaE1, eta1, sigma1,c,vg,T)
    args2 = (gammaC2, gammaW2, deltaE2, eta2, sigma2,c,vg,T)
    
    I1 = tunnelmodel_singleLevel(vb,*args1)
    I2 = tunnelmodel_singleLevel(vb,*args2)
    
    return I1+I2

# %%Below are all of the functions deal with Nitzan's Hysteresis Model
@jit
def averageBridgePopulation_integrand(E,ep,c,vg,eta,vb,gammaL,gammaR,T):    
    gammaW = gammaL+gammaR
    
    return ((fermi(E+vb/2,T)*gammaL+gammaR*fermi(E-vb/2,T))/\
        ((E-((ep+c*vg)+(eta-1/2)*vb))**2+(gammaW/2)**2))

def averageBridgePopulation(vb, gammaL, gammaR, deltaE, eta, c, vg, T):
    limits = [-10,10]
    
    def integrand (E):
        result = averageBridgePopulation_integrand(E, deltaE, c, vg, eta, vb, gammaL, gammaR, T)
        return result
    
    return quad(integrand,
                        limits[0],
                        limits[1])[0]/(2*np.pi)

def NitzanSwitchingRate(vb, gammaL, gammaR, deltaE, eta, c, vg, T, R1, R2):
    n = averageBridgePopulation(vb, gammaL, gammaR, deltaE, eta, c, vg, T)
    return (1-n)*R1+n*R2


# %% This is another Nitzan Function that was applied to the BTTF
#    molecule
def E_act_fixedtemp_gatevoltage(Vg,E,l):
    T0=260
    T1=330
    
    def integrandOne(ep):
        num=np.exp(-((E+Vg/2)+ep-l)**2/(4*kb*T0*l))
        denom=1/(np.exp((ep-Vg/2)/(kb*T0))+1)
        return num*denom
    
    def integrandTwo(ep):
        num=np.exp(-((E+Vg/2)+ep+l)**2/(4*kb*T0*l))
        denom=1-1/(np.exp((ep-Vg/2)/(kb*T0))+1)
        return num*denom
    
    def integrandThree(ep):
        num=np.exp(-((E+Vg/2)+ep-l)**2/(4*kb*T1*l))
        denom=1/(np.exp((ep-Vg/2)/(kb*T1))+1)
        return num*denom
    
    def integrandFour(ep):
        num=np.exp(-((E+Vg/2)+ep+l)**2/(4*kb*T1*l))
        denom=1-1/(np.exp((ep-Vg/2)/(kb*T1))+1)
        return num*denom
    
    One = quad(integrandOne, -10, 10)
    Two = quad(integrandTwo, -10, 10)
    Three = quad(integrandThree, -10, 10)
    Four = quad(integrandFour, -10, 10)
    
    leftSide=np.log((One[0]+Two[0])*(1/(np.sqrt(4*np.pi*l*kb*T0))))
    rightSide=np.log((Three[0]+Four[0])*(1/(np.sqrt(4*np.pi*l*kb*T1))))
    
    FinalAns=-1000*kb*T0**2*(leftSide-rightSide)/(T1-T0)
    return FinalAns

def E_act_fixedtemp_biasvoltage(V,E,l,cap,W,A):
    Vg=cap*(1-1/(1+np.exp((V-A)/W)))
    return E_act_fixedtemp_gatevoltage(Vg,E,l)