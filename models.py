# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:46:23 2019
Functions to be used in fitting and calculations
@author: Cameron
"""
import numpy as np
from scipy.integrate import quad

eV = 1 #Energy
K  = 1 #Temperature Units
C  = 1 #Coulombs
s  = 1 #seconds
V  = 1 #volts

kb = 8.6173324e-5*eV/K #Boltzmann Constant
q  = 1.6e-19*C
h  = 4.1356e-15*eV*s

def linear(x,m,b):
    return b+x*m
  
def fermi(E,T):
    return 1/(np.exp((E)/(kb*T))+1)

def gaussian(x,A, mu,sigma):
    return A*np.exp(-.5*((x-mu)/(sigma))**2)

def normalized_gaussian(x, mu,sigma):
    A = 1
    def gaus(ep):
        return gaussian(ep,A,mu,sigma)
    
    A = 1/quad(gaus,mu-3*sigma,mu+3*sigma)[0]    
    return gaussian(x, A, mu, sigma)

def densityOfStates(E,ep,gamma):
    numerator = gamma
    denominator = (E-ep)**2+(gamma/2)**2
    return numerator/denominator#/(2*np.pi)

def rateRatio(gammaL,gammaR):
    return gammaL*gammaR/(gammaL+gammaR)

def single_level_tunnel_model_integrand(E,ep,c,vg,eta,vb,gammaL,gammaR,T):
    gamma = gammaL+gammaR
    return -densityOfStates(E,((ep+c*vg)+(eta-1/2)*vb),gamma)*\
        rateRatio(gammaL,gammaR)*\
        (fermi(E+vb/2,T)-fermi(E-vb/2,T))

def tunnelmodel_1level_nogate_300K(vb, gammaL, gammaR, deltaE1, eta):
    c  = 0*V
    vg = 0*V
    T  = 300*K
    args = (deltaE1,c,vg,eta,vb,gammaL,gammaR,T)
    
    return q/h*quad(single_level_tunnel_model_integrand,-10,10,args = args)[0]    

def tunnelmodel_1level_nogate_300K_gauss(vb, gammaL, gammaR, deltaE1, eta, sigma):
    c  = 0*V
    vg = 0*V
    T = 300*K
    A = 1
    args = (A,deltaE1,sigma)
    A = 1/quad(gaussian,deltaE1-3*sigma,deltaE1+3*sigma,args=args)[0]
    
    def outer(E):
        def inner(ep):
            result = gaussian(ep,A,deltaE1,sigma)*\
            single_level_tunnel_model_integrand(E,ep,c,vg,eta,vb,gammaL,gammaR,T)
            return result
        return quad(inner,deltaE1-3*sigma,deltaE1+3*sigma)[0]
    return q/h*quad(outer,-2.5,2.5)[0]

def tunnelmodel_2level_gated(vb, gammaL, gammaR, deltaE1, deltaE2, Vg, T, eta, c):
    gamma = gammaL+gammaR
    
    def integrand(E):
        left = (gammaL*gammaR)/((E-((deltaE1+c*Vg)+(eta-1/2)*vb))**2+gamma**2/4)
        right = (gammaL*gammaR)/((E-((deltaE2+c*Vg)+(eta-1/2)*vb))**2+gamma**2/4)
        
        return -(left+right)*(fermi(E+vb/2,T)-fermi(E-vb/2,T))
    
    return q/h*quad(integrand,-5,5)[0]

def nitzanmodel_fixedtemp_gatevoltage(Vg,E,l):
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

def nitzanmodel_fixedtemp_biasvoltage(V,E,l,cap,W,A):
    Vg=cap*(1-1/(1+np.exp((V+A)/W)))
    return nitzanmodel_fixedtemp_gatevoltage(Vg,E,l)