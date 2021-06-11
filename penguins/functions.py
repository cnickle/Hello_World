# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:46:23 2019
Functions to be used in fitting and calculations
@author: Cameron
"""
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate   import quad
from scipy.integrate   import dblquad
from numba             import jit


eV = 1 #Energy
K  = 1 #Temperature Units
C  = 1 #Coulombs
s  = 1 #seconds
V  = 1 #volts

kb = 8.6173324e-5*eV/K #Boltzmann Constant
q  = 1.6e-19*C
h  = 4.1356e-15*eV*s

def interp1D(func, step = .01):
    """
    This function works a bit like numpy.vectorize. You input a function that
    you wish to vectorize and it returns the vectorized version. However, unlike
    numpy, it only cacluates unique values the first input (usually the bias 
    voltage in our case). This means that if the experimentalists do 4 sweeps
    of the voltage i.e. every voltage value is repeated 4 times. It only computes
    it once.
    
    Furthermore, it computes the x values at the 'step' location. So let's say
    you're looking a votlage data and the step is set to 0.01 it only calculates
    the current every 10 mV and interpolates the rest. This value for the step
    can be varied.    

    Parameters
    ----------
    func : function
        Function to be interpolated.
    step : TYPE, optional
        The default is .01.

    Returns
    -------
    function
        Returns interpolated function.

    """
    def wrap(x, *args):
        Xunique = np.arange(np.min(x),np.max(x)+step,step)
        
        vecFun = np.vectorize(func)
        Yunique = vecFun(Xunique,*args)
        
        interpFun = interp1d(Xunique,Yunique)
        return interpFun(x)
    return wrap

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
def tunnelmodel_singleLevel(vb,n, gammaC,gammaW, deltaE1,eta,sigma,c,vg,T):
    
    if sigma == 0:
        limits = [-1*np.abs(vb),1*np.abs(vb)]
    
        def integrand (E):
            result = single_level_tunnel_model_integrand_Alt(E,deltaE1,c,vg,eta,vb,gammaC,gammaW,T)
            return result
        
        return n*q/h*quad(integrand,
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
        
        return n*q/h*dblquad(integrand,
                           limits[0],
                           limits[1],
                           lambda x: limits[0],
                           lambda x: limits[1])[0]

# This is the 2 level version of the Single Level tunnel model.
# This isn't really needed to be a seperate model as you simply need
# To add the contributions from each level.
def tunnelmodel_2level(vb, n, c,vg,T,
                       gammaC1, gammaW1, deltaE1, eta1, sigma1,
                       gammaC2, gammaW2, deltaE2, eta2, sigma2):
    
    args1 = (n, gammaC1, gammaW1, deltaE1, eta1, sigma1,c,vg,T)
    args2 = (n, gammaC2, gammaW2, deltaE2, eta2, sigma2,c,vg,T)
    
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

@jit
def MarcusETRates(vb, gamma, lam, epsilon, T):
    alpha = vb-epsilon
    S = 2*np.sqrt(np.pi*kb*T/lam)
    
    R_plus = (gamma/4)*S*np.exp(-(alpha+lam)**2/(4*lam*kb*T))
    R_minus = (gamma/4)*S*np.exp(-(alpha-lam)**2/(4*lam*kb*T))
    return R_plus,R_minus 

def HysteresisModel_withP(vb, n, gammaL, gammaR, kappa, sigma, E_AB, E_AC, chi, eta,
                  gam, lam, P, u, c, vg, T):

        volts = list(set(np.round(vb,2)))
    
        #%% Calculate all currents:
        calcDB = pd.DataFrame()
        calcDB['V'] = sorted(volts)
        
        eqSTL = interp1D(tunnelmodel_singleLevel)
        calcDB['I_np'] = eqSTL(calcDB['V'], n, gammaL*gammaR, gammaL+gammaR, E_AB,
                                eta, sigma, c, vg, T)
        calcDB['I_p'] = eqSTL(calcDB['V'], n, gammaL*gammaR*kappa**2,
                              (gammaL+gammaR)*kappa, E_AB+chi, eta, sigma, c, vg,
                              T)      
        
        eqETRates = interp1D(MarcusETRates)
        calcDB['R_AC'], calcDB['R_CA'] = eqETRates(calcDB['V'], gam, lam, E_AC, T)
        calcDB['R_BD'], calcDB['R_DB'] = eqETRates(calcDB['V'], gam*kappa, lam,
                                                   E_AC+chi, T)
        
        eqBridge = interp1D(averageBridgePopulation)
        calcDB['n_np'] = eqBridge(calcDB['V'], gammaL, gammaR, E_AB, eta, c, vg, T)
        calcDB['n_p']  = eqBridge(calcDB['V'], gammaL*kappa, gammaR*kappa,
                                  E_AB+chi, eta, c, vg, T)
        
        calcDB['k_S0_S1'] = (1-calcDB['n_np'])*calcDB['R_AC'] + calcDB['n_np']*calcDB['R_BD']
        calcDB['k_S1_S0'] = (1-calcDB['n_p'])*calcDB['R_CA'] + calcDB['n_p']*calcDB['R_DB']
            
        delt = abs(vb[2]-vb[3])/u
        I = []
        Parray = []
        delArray = []
            
        for i,V in enumerate(vb):
            V = np.round(V,2)
            tempDf =calcDB[calcDB['V']==np.round(V,2)].reset_index()
            calcs = dict(tempDf.iloc[0])
            
            Parray += [P]
            I += [((1-P)*calcs['I_np']+P*calcs['I_p'])]
            
            dPdt = calcs['k_S0_S1']-P*(calcs['k_S0_S1']+calcs['k_S1_S0'])
            delArray += [dPdt]
            P = P+dPdt*delt
        
        return I, Parray

def HysteresisModel_witht(vb, t, n, gammaL, gammaR, kappa, sigma, E_AB, E_AC, chi, eta,
                  gam, lam, P, c, vg, T):
        t = np.array(t)
        volts = list(set(np.round(vb,2)))
    
        #%% Calculate all currents:
        calcDB = pd.DataFrame()
        calcDB['V'] = sorted(volts)
        
        eqSTL = interp1D(tunnelmodel_singleLevel)
        calcDB['I_np'] = eqSTL(calcDB['V'], n, gammaL*gammaR, gammaL+gammaR, E_AB,
                                eta, sigma, c, vg, T)
        calcDB['I_p'] = eqSTL(calcDB['V'], n, gammaL*gammaR*kappa**2,
                              (gammaL+gammaR)*kappa, E_AB+chi, eta, sigma, c, vg,
                              T)      
        
        eqETRates = interp1D(MarcusETRates)
        calcDB['R_AC'], calcDB['R_CA'] = eqETRates(calcDB['V'], gam, lam, E_AC, T)
        calcDB['R_BD'], calcDB['R_DB'] = eqETRates(calcDB['V'], gam*kappa, lam,
                                                   E_AC+chi, T)
        
        eqBridge = interp1D(averageBridgePopulation)
        calcDB['n_np'] = eqBridge(calcDB['V'], gammaL, gammaR, E_AB, eta, c, vg, T)
        calcDB['n_p']  = eqBridge(calcDB['V'], gammaL*kappa, gammaR*kappa,
                                  E_AB+chi, eta, c, vg, T)
        
        calcDB['k_S0_S1'] = (1-calcDB['n_np'])*calcDB['R_AC'] + calcDB['n_np']*calcDB['R_BD']
        calcDB['k_S1_S0'] = (1-calcDB['n_p'])*calcDB['R_CA'] + calcDB['n_p']*calcDB['R_DB']
            
        
        I = []
        Parray = []
        delArray = []
        for i,V in enumerate(vb):
            if i == 0:
                delt = t[i]
            else:
                delt = (t[i]-t[i-1])
            
            V = np.round(V,2)
            tempDf =calcDB[calcDB['V']==np.round(V,2)].reset_index()
            calcs = dict(tempDf.iloc[0])
            
            Parray += [P]
            I += [((1-P)*calcs['I_np']+P*calcs['I_p'])]
            
            dPdt = calcs['k_S0_S1']-P*(calcs['k_S0_S1']+calcs['k_S1_S0'])
            delArray += [dPdt]
            P = P+dPdt*delt
        
        return I

def HysteresisModel(vb, n, gammaL, gammaR, kappa, sigma, E_AB, E_AC, chi, eta,
                  gam, lam, P, u, c, vg, T):
    I, __ = HysteresisModel_withP(vb, n, gammaL, gammaR, kappa, sigma, E_AB,
                                  E_AC, chi, eta, gam, lam, P, u, c, vg, T)
    return np.array(I)

# %% This is another Nitzan Function that was applied to the BTTF
#    molecule
def E_act_fixedtemp_gatevoltage(Vg,E,l,T0,T1):
    
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

def chargeN(V,A,W):
    return (1/(1+np.exp((V-A)/W)))

def chargeP(V,A,W):
    return(1-1/(1+np.exp((V-A)/W)))

def E_act_fixedtemp_biasvoltageN(V,E,l,cap,A,W,T0,T1):
    Vg=cap*chargeN(V,A,W)
    return E_act_fixedtemp_gatevoltage(Vg,E,l,T0,T1)
def E_act_fixedtemp_biasvoltageP(V,E,l,cap,A,W,T0,T1):
    Vg=cap*chargeP(V,A,W)
    return E_act_fixedtemp_gatevoltage(Vg,E,l,T0,T1)