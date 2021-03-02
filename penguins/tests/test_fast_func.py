from penguins.functions import tunnelmodel_singleLevel
from penguins.functions import averageBridgePopulation
from penguins.functions import MarcusETRates
from penguins.functions import interp1D
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

def test_SAM():
    v = np.arange(-2,2,.001)
    V=[]
    for i in range(2):
        V = np.append(V,v)
    V = sorted(V)
    
    #         n        gammaW     gammaC  deltaE eta       sigma c  vg T
    args = [1.50e+02,  1.375e-05, 0.0352, 0.75,   5.32e-01, 0,    0, 0, 300]
    
    start = time.time()
    vecCur = np.vectorize(tunnelmodel_singleLevel)
    y1 = vecCur(V, *args)
    time1 = time.time()-start
    
    start = time.time()
    fast = interp1D(tunnelmodel_singleLevel)
    y2 = fast(V, *args)
    time2 = time.time()-start
    
    print('Slow: %.2f\t\tFast: %.2f\t\tSpeed Increase: %.0f%%'%(time1,time2,time1/time2*100))
                                                          
    plt.figure()
    plt.scatter(V,y1, color = 'black')
    plt.plot(V,y2, color = 'red')

def test_SET():
    v = np.arange(-.05,.05,.001)
    V=[]
    for i in range(2):
        V = np.append(V,v)
    V = sorted(V)
    
    #         n        gammaW     gammaC  deltaE eta       sigma c  vg T
    args = [1.50e+02,  1.375e-05, 0.0352, 0.03,   5.32e-01, 0,    0, 0, 300]
    
    start = time.time()
    vecCur = np.vectorize(tunnelmodel_singleLevel)
    y1 = vecCur(V, *args)
    time1 = time.time()-start
    
    start = time.time()
    fast = interp1D(tunnelmodel_singleLevel)
    y2 = fast(V, *args)
    time2 = time.time()-start
    
    print('Slow: %.2f\t\tFast: %.2f\t\tSpeed Increase: %.0f%%'%(time1,time2,time1/time2*100))
                                                          
    plt.figure()
    plt.scatter(V,y1, color = 'black')
    plt.plot(V,y2, color = 'red')

def test_Hysteric():
    def HysteresisModel_Slow(vb, n, gammaL, gammaR, kappa, sigma, E_AB, E_AC, chi, eta,
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
    
    def HysteresisModel_Fast(vb, n, gammaL, gammaR, kappa, sigma, E_AB, E_AC, chi, eta,
                  gam, lam, P, u, c, vg, T):

        volts = list(set(np.round(vb,2)))
    
        #%% Calculate all currents:
        calcDB = pd.DataFrame()
        calcDB['V'] = sorted(volts)
        
        eqSTL = np.vectorize(tunnelmodel_singleLevel)
        calcDB['I_np'] = eqSTL(calcDB['V'], n, gammaL*gammaR, gammaL+gammaR, E_AB,
                                eta, sigma, c, vg, T)
        calcDB['I_p'] = eqSTL(calcDB['V'], n, gammaL*gammaR*kappa**2,
                              (gammaL+gammaR)*kappa, E_AB+chi, eta, sigma, c, vg,
                              T)      
        
        eqETRates = np.vectorize(MarcusETRates)
        calcDB['R_AC'], calcDB['R_CA'] = eqETRates(calcDB['V'], gam, lam, E_AC, T)
        calcDB['R_BD'], calcDB['R_DB'] = eqETRates(calcDB['V'], gam*kappa, lam,
                                                   E_AC+chi, T)
        
        eqBridge = np.vectorize(averageBridgePopulation)
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
    
    initpar = {
            	'n'	:1.50e+02,
            	'gammaL'	:5.52E-04,
            	'gammaR'	:2.03E-02,
            	'kappa'	:2.81,
            	'sigma'	:0.00e+00,
            	'E_AB'	:6.93e-01,
            	'E_AC'	:-7.17e-01,
            	'chi'	:1.58e+00,
            	'eta'	:5.23e-01,
            	'gam'	:7.12e-01,
            	'lam'	:1.21e+00,
            	'P'	:0.00e+00,
            	'u'	:1.00e-02,
            	'c'	:0.00e+00,
            	'vg'	:0.00e+00,
            	'T'	:3.00e+02
            }
    
    DataFile = 'Data\\AsymNeg_cont_Normalized.txt'
    data = pd.read_csv(DataFile, delimiter = '\t')
    colV = '-2.00V_1'
    
    start = time.time()
    y1,_ = HysteresisModel_Slow(data[colV],*list(initpar.values()))
    time1 = time.time()-start
    
    start = time.time()
    y2,_ = HysteresisModel_Fast(data[colV],*list(initpar.values()))
    time2 = time.time()-start
    
    print('Slow: %.2f\t\tFast: %.2f\t\tSpeed Increase: %.0f%%'%(time1,time2,time1/time2*100))
    
    plt.figure()
    plt.scatter(data[colV],np.abs(y1), color = 'black')
    plt.plot(   data[colV], np.abs(y2), color = 'red')
    plt.ylim(7.2e-10,2e-05)
    plt.yscale('log')
    

if __name__ == '__main__':
    test_SAM()
    test_SET()
    test_Hysteric()

