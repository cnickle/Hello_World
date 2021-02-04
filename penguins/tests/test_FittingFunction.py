import pytest
import pandas as pd
from penguins.functions import tunnelmodel_singleLevel as tunnelModel
from penguins.Model import Model as mod
import numpy as np
import matplotlib.pyplot as plt  
    
#%% Test the fitting of 2H-s-2F
def test_TestingFit_2Hc2F(fName = 'tests\\Data\\2H-c-2F (amps).txt'):
    """
    INSERT DESCRIPTION
    """
    Data_2Hs2F = pd.read_csv(fName,delimiter = '\t',header=None)
    Data_2Hs2F.columns = ['V','I']
    
    initPar={
        'n'       : 1,
        'gammaC'  : 0.000001,
        'gammaW'  : 0.463199,
        'epsilon' : 0.551646,
        'eta'     : 0.594323,
        'width'   : 0,
        'Cap'     : 0,
        'Vg'      : 0,
        'Temp'    : 300
       }
    
    fixed=['n', 'width','Cap','Vg','Temp']
    
    bnds = {
        'gammaC'  : [1E-6*1E-6,1],
        'gammaW'  : [1E-6+1E-6,1],
        'epsilon' : [0,1.5],
        'eta'     : [0,1]
        }
    
    calculatedParams ={
        'n   '    : lambda args: args['n'],
        'gammaL'  : lambda args: .5*(args['gammaW']-np.sqrt(args['gammaW']**2-4*args['gammaC'])),
        'gammaR'  : lambda args: .5*(args['gammaW']+np.sqrt(args['gammaW']**2-4*args['gammaC'])),
        'Gamma'   : lambda args: args['gammaC']/args['gammaW'],
        'epsilon' : lambda args: args['epsilon']*2,
        'eta '    : lambda args: args['eta'],
        'width'   : lambda args: args['width']
        }
    
    SLT = mod(tunnelModel)
    SLT.setParams(initPar,bnds = bnds,calculatedParams=calculatedParams,
                  fixed=fixed)
    
    SLT.fit(Data_2Hs2F['V'],Data_2Hs2F['I'], mode = 'verbose',
            save = '2Hc2F_AllParams.txt', scale = 'lin')
    SLT.print(Data_2Hs2F['V'],Data_2Hs2F['I'],save = '')
    
    Ythr = SLT.returnThry(Data_2Hs2F['V'])
    plt.figure()
    plt.scatter(Data_2Hs2F['V'],Data_2Hs2F['I'], color = 'black')
    plt.plot(Data_2Hs2F['V'], Ythr, color = 'red')
    
if __name__ == '__main__':
    test_TestingFit_2Hc2F(fName = 'Data\\2H-c-2F (amps).txt')