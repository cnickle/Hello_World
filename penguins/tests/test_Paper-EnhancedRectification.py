import pytest
import pandas as pd
import penguins.functions as fun
from penguins.Model import Model as mod
import numpy as np

#%% Alternate Tunnel Model
def TunnelModel_alt(vb, gammaL, gammaR, deltaE1, eta,sigma,c,vg,T):
    """
    This is the same Tunnel Model with gammaC and gammaW replaced with gammaL
    and gammaR
    """
    n=1
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    return fun.tunnelmodel_singleLevel(vb,n, gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

# Test 2H-s-2F Final Values
def test_2Hs2F_FinalValue(fName = 'tests\\Data\\2H-s-2F (amps).txt'):
    """
    This function tests whether or not the tunnel model used in the Enhanced
    Rectification (sigma vs Pi bridges) project hasn't changed.
    """
    Data_2Hs2F = pd.read_csv(fName,delimiter = '\t',header=None)
    Data_2Hs2F.columns = ['V','I']

    initPar={
        'gammaL'  : 0.000060, # eV
        'gammaR'  : 0.006331, # eV
        'epsilon' : 0.708984, # eV
        'eta'     : 0.742364, # Unitless
        'width'   : 0.177987, # eV
        'Cap'     : 0,        # F
        'Vg'      : 0,        # V
        'Temp'    : 300       # K
       }

    SLT = mod(TunnelModel_alt)
    SLT.setParams(initPar,fixed=['Cap','Vg','Temp'])
    Error = SLT.standardError(Data_2Hs2F['V'],Data_2Hs2F['I'], scale='log')
    assert (Error-1.15)/Error*100 < 5
    
# Test the fitting of 2H-s-2F
def test_2Hs2F_Fit(fName = 'tests\\Data\\2H-s-2F (amps).txt'):
    """
    This tests the fitting function. It makes sure that using the same initial
    parameters and bounds as was used for the Enhanced Rectification
    (sigma vs pi bridge) will get the same results. 
    """
    Data_2Hs2F = pd.read_csv(fName,delimiter = '\t',header=None)
    Data_2Hs2F.columns = ['V','I']

    initPar={
        'gammaL'  : 0.000256,
        'gammaR'  : 0.463199,
        'epsilon' : 0.551646,
        'eta'     : 0.594323,
        'width'   : 0,
        'Cap'     : 0,
        'Vg'      : 0,
        'Temp'    : 300
       }
    
    bnds = {
        'gammaL'  : [0,1],
        'gammaR'  : [0,1],
        'epsilon' : [0,1.5],
        'eta'     : [0,1]
        }

    pars = {
        'gammaL'  : 0.000042,
        'gammaR'  : 0.229306,
        'epsilon' : 0.835175,
        'eta'     : 1.000000
        }

    SLT = mod(TunnelModel_alt)
    SLT.setParams(initPar,fixed=['width','Cap','Vg','Temp'],bnds=bnds)
    SLT.fit(Data_2Hs2F['V'],Data_2Hs2F['I'], algorithm = 'LS')
    
    for par in pars:
        assert np.abs((pars[par]-SLT.parameters[par])/pars[par])*100<5

# Test the 2H-c-2F Final Values    
def test_2Hc2F_FinalValue(fName = 'tests\\Data\\2H-c-2F (amps).txt'):
    """
    This function tests whether or not the tunnel model used in the Enhanced
    Rectification (sigma vs Pi bridges) project hasn't changed.
    """
    Data_2Hc2F = pd.read_csv(fName,delimiter = '\t',header=None)
    Data_2Hc2F.columns = ['V','I']

    initPar={
        'gammaL'  : 0.032248,
        'gammaR'  : 0.008489,
        'epsilon' : 0.874857,
        'eta'     : 0.590363,
        'width'   : 0.0087,
        'Cap'     : 0,
        'Vg'      : 0,
        'Temp'    : 300
       }

    SLT = mod(TunnelModel_alt)
    SLT.setParams(initPar,fixed=['width','Cap','Vg','Temp'])
    Error = SLT.standardError(Data_2Hc2F['V'],Data_2Hc2F['I'], scale='log')
    assert (Error-0.33)/Error*100 < 5    
    
# Test the fitting of 2H-s-2F
def test_2Hc2F_Fit(fName = 'tests\\Data\\2H-c-2F (amps).txt'):
    """
    This tests the fitting function. It makes sure that using the same initial
    parameters and bounds as was used for the Enhanced Rectification
    (sigma vs pi bridge) will get the same results. The 'pars' dictionary are 
    the 'correct' values.
    """
    Data_2Hs2F = pd.read_csv(fName,delimiter = '\t',header=None)
    Data_2Hs2F.columns = ['V','I']

    initPar={
        'gammaL'  : 0.000256,
        'gammaR'  : 0.463199,
        'epsilon' : 0.551646,
        'eta'     : 0.594323,
        'width'   : 0,
        'Cap'     : 0,
        'Vg'      : 0,
        'Temp'    : 300
       }
    
    bnds = {
        'gammaL'  : [0,1],
        'gammaR'  : [0,1],
        'epsilon' : [0,1.5],
        'eta'     : [0,1]
        }

    pars = {
        'gammaL'  : 0.000268,
        'gammaR'  : 0.464514,
        'epsilon' : 0.574295,
        'eta'     : 0.582212
        }

    SLT = mod(TunnelModel_alt)
    SLT.setParams(initPar,fixed=['width','Cap','Vg','Temp'],bnds=bnds)
    SLT.fit(Data_2Hs2F['V'],Data_2Hs2F['I'], algorithm = 'LS')
    
    for par in pars:
        assert np.abs((pars[par]-SLT.parameters[par])/pars[par])*100<5    
    
if __name__ == '__main__':
    test_2Hc2F_FinalValue(fName = 'Data\\2H-c-2F (amps).txt')
    test_2Hc2F_Fit(fName = 'Data\\2H-c-2F (amps).txt')
    
    test_2Hs2F_FinalValue(fName = 'Data\\2H-s-2F (amps).txt')
    test_2Hs2F_Fit(fName = 'Data\\2H-s-2F (amps).txt')