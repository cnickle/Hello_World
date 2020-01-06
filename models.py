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

@jit
def densityOfStates(E,ep,gamma):
    numerator = gamma
    denominator = (E-ep)**2+(gamma/2)**2
    return numerator/denominator#/(2*np.pi)

@jit
def rateRatio(gammaL,gammaR):
    return gammaL*gammaR/(gammaL+gammaR)

@jit
def single_level_tunnel_model_integrand(E,ep,c,vg,eta,vb,gammaL,gammaR,T):
    gamma = gammaL+gammaR
    return -densityOfStates(E,((ep+c*vg)+(eta-1/2)*vb),gamma)*\
        rateRatio(gammaL,gammaR)*\
        (fermi(E+vb/2,T)-fermi(E-vb/2,T))
@jit        
def single_level_tunnel_model_integrand_Alt(E,ep,c,vg,eta,vb,gammaC,gammaW,T):    
    return -gammaC*(fermi(E+vb/2,T)-fermi(E-vb/2,T))/\
        ((E-((ep+c*vg)+(eta-1/2)*vb))**2+(gammaW/2)**2)

def tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T):
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

def tunnelmodel_singleLevel_Gauss(vb,gammaC,gammaW, deltaE1,eta,c,vg,T):
    limits = [-1*np.abs(vb),1*np.abs(vb)]
    
    def integrand (E):
        result = single_level_tunnel_model_integrand_Alt(E,deltaE1,c,vg,eta,vb,gammaC,gammaW,T)
        return result
    
    return q/h*quad(integrand,
                       limits[0],
                       limits[1])[0]
    
def tunnelmodel_1level_nogate_300K(vb, gammaL, gammaR, deltaE1, eta):
    c  = 0*V
    vg = 0*V
    T  = 300*K
    args = (deltaE1,c,vg,eta,vb,gammaL,gammaR,T)
    limits = [min([-.1,-1*np.abs(vb)]),\
              max([.1,1*np.abs(vb)])]
    
    return q/h*quad(single_level_tunnel_model_integrand,limits[0],limits[1],
                    args = args)[0]    

def tunnelmodel_1level_nogate_300K_gauss(vb, gammaL, gammaR, deltaE1, eta, sigma):
    c  = 0*V
    vg = 0*V
    T = 300*K
    A = 1
    args = (A,deltaE1,sigma)
    A = 1/quad(gaussian,deltaE1-3*sigma,deltaE1+3*sigma,args=args)[0]
    
    limits = [min([deltaE1-3*sigma,-1*np.abs(vb)]),\
              max([deltaE1+3*sigma,1*np.abs(vb)])]
    
    def integrand (E,ep):
        result = gaussian(ep,A,deltaE1,sigma)*\
        single_level_tunnel_model_integrand(E,ep,c,vg,eta,vb,gammaL,gammaR,T)
        return result
    
    return q/h*dblquad(integrand,
                       limits[0],
                       limits[1],
                       lambda x: limits[0],
                       lambda x: limits[1])[0]

def tunnelmodel_2level_NoGauss(vb, Vg, T, eta, c,
                             gammaL1, gammaR1, deltaE1,
                             gammaL2, gammaR2, deltaE2):
    gamma1 = gammaL1 + gammaR1
    gamma2 = gammaL2 + gammaR2
    
    def integrand(E):
        left  = (gammaL1*gammaR1)/((E-((deltaE1+c*Vg)+(eta-1/2)*vb))**2+gamma1**2/4)
        right = (gammaL2*gammaR2)/((E-((deltaE2+c*Vg)+(eta-1/2)*vb))**2+gamma2**2/4)
        
        return -(left+right)*(fermi(E+vb/2,T)-fermi(E-vb/2,T))
    
    return q/h*quad(integrand,-5,5)[0]

def tunnelmodel_2level(vb,c,vg,T,
                       gammaC1, gammaW1, deltaE1, eta1, sigma1,
                       gammaC2, gammaW2, deltaE2, eta2, sigma2):
    
    args1 = (gammaC1, gammaW1, deltaE1, eta1, sigma1,c,vg,T)
    args2 = (gammaC2, gammaW2, deltaE2, eta2, sigma2,c,vg,T)
    
    I1 = tunnelmodel_singleLevel(vb,*args1)
    I2 = tunnelmodel_singleLevel(vb,*args2)
    
    return I1+I2 

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