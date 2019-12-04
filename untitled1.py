# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:33:31 2019

@author: nickl
"""
import numpy as np
import mpmath as m
import matplotlib.pyplot as plt

kb = 8.6173324e-5
q  = 1.6e-19
h  = 4.1356e-15

def fermi(E,T):
    return 1/(m.exp((E)/(kb*T))+1)

def gaussian(x,A, mu,sigma):
    return A*m.exp(-.5*((x-mu)/(sigma))**2)

def densityOfStates(E,ep,gamma):
    numerator = gamma
    denominator = (E-ep)**2+(gamma/2)**2
    return numerator/denominator

def rateRatio(gammaL,gammaR):
    return gammaL*gammaR/(gammaL+gammaR)

def model(E,ep,vb,gammaL,gammaR,eta):
    T= 300
    gamma = gammaL+gammaR
    return -densityOfStates(E,(ep+(eta-1/2)*vb),gamma)*\
        rateRatio(gammaL,gammaR)*\
        (fermi(E+vb/2,T)-fermi(E-vb/2,T))
        
def func(vb,gammaL,gammaR,deltaE,eta,sigma):
    A=1
    A =1/integrate1D(gaussian,A,deltaE,sigma)
    
    def integrand(x,y):
        return gaussian(y,A,deltaE,sigma)*model(x,y,vb,gammaL,gammaR,eta)
        
    return q/h*m.quad(integrand,[-2,2],[-2,2])
    
def integrate2D(f, *args):
    def integrand(x,y):
        return f(x,y,*args)
    return m.quad(integrand,[-2,2],[-2,2])

def integrate1D(f, *args):
    def integrand(x):
        return f(x,*args)
    return m.quad(integrand,[-2,2])

x = m.arange(-1,1,.01)
y = []
for i in range(len(x)):
    y+=[func(x[i],1.05,.0002,.0004,.699,.13)]
    
plt.figure()
plt.plot(x,y)
