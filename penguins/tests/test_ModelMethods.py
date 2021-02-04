import pytest
import pandas as pd
from penguins.Model import Model as mod
from penguins.functions import tunnelmodel_singleLevel as tunnelModel
from penguins.functions import tunnelmodel_2level as tunnelModel2
import penguins.functions as fun
import matplotlib.pyplot as plt
import time
import numpy as np

# Test the fitting of 2H-s-2F
def test_compareModels_simple(fName = 'tests\\Data\\2H-c-2F (amps).txt'):
    """
    This compares various fitting methods using one of the simplest data sets
    I've worked on. Its relatively quick, but one can quickly see the benefit
    of using the least squares method as opposed to min or diff.
    """
    Data = pd.read_csv(fName,delimiter = '\t',header=None)
    Data.columns = ['V','I']

    initPar={
        'n'       : 1,
        'gammaC'  : 0.0004*0.5,
        'gammaW'  : 0.0002+0.4,
        'epsilon' : 0.5,
        'eta'     : 0.5,
        'width'   : 0,
        'Cap'     : 0,
        'Vg'      : 0,
        'Temp'    : 300
       }
    
    bnds = {
        'gammaL'  : [1E-12,1],
        'gammaR'  : [1E-6,1],
        'epsilon' : [0,1.5],
        'eta'     : [0,1]
        }

    pars = {
        'gammaL'  : 0.000268,
        'gammaR'  : 0.464514,
        'epsilon' : 0.574295,
        'eta'     : 0.582212
        }

    fmt = ['-','--']
    SLT = mod(tunnelModel)
    plt.figure('Bounded')
    plt.scatter(Data['V'], Data['I'], color = 'black')
    for i,alg in enumerate(['LS','min']):
        if alg == 'LS':
            for j,meth in enumerate(['trf','dogbox']):
                start = time.time()
                SLT.setParams(initPar,fixed=['n','width','Cap','Vg','Temp'], bnds=bnds)
                SLT.fit(Data['V'],Data['I'], algorithm = alg, method=meth)
                Ythr = SLT.returnThry(Data['V'])
                err = np.log10(SLT.standardError(Data['V'],Data['I']))
                end = time.time()
                
                label = '%s-%s->    time: %d    Err: %.2f' %(alg,meth,end-start, err)
                
                plt.plot(Data['V'], Ythr, fmt[i], label=label)
        if alg == 'min':
            for j,meth in enumerate(['Powell','L-BFGS-B', 'TNC', 'SLSQP']):
                try:
                    start = time.time()
                    SLT.setParams(initPar,fixed=['n','width','Cap','Vg','Temp'], bnds=bnds)
                    SLT.fit(Data['V'],Data['I'], algorithm = alg, method=meth)
                    Ythr = SLT.returnThry(Data['V'])
                    err = SLT.standardError(Data['V'],Data['I'])
                    end = time.time()
                    
                    label = '%s-%s->    time: %d    Err: %.2f' %(alg,meth,end-start, err)
                    
                    plt.plot(Data['V'], Ythr, fmt[i], label=label)
                except:
                    print('%s failed'%meth)       
    plt.legend()
    plt.figure('Bounded')
    plt.savefig('Plots//test_ModelMethods_CompareSimple_Bounded')
    
    plt.figure('Unbounded')
    plt.scatter(Data['V'], Data['I'], color = 'black')
    for i,alg in enumerate(['LS','min']):
        if alg == 'LS':
            for j,meth in enumerate(['lm','trf','dogbox']):
                start = time.time()
                SLT.setParams(initPar,fixed=['width','Cap','Vg','Temp'])
                SLT.fit(Data['V'],Data['I'], algorithm = alg, method=meth)
                Ythr = SLT.returnThry(Data['V'])
                err = np.log10(SLT.standardError(Data['V'],Data['I']))
                end = time.time()
                
                label = '%s-%s->    time: %d    Err: %.2f' %(alg,meth,end-start, err)
                
                plt.plot(Data['V'], Ythr, fmt[i], label=label)
        if alg == 'min':
            for j,meth in enumerate(['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                                     'L-BFGS-B', 'TNC','COBYLA',
                                     'SLSQP', 'trust-constr']):
                try:
                    start = time.time()
                    SLT.setParams(initPar,fixed=['width','Cap','Vg','Temp'])
                    SLT.fit(Data['V'],Data['I'], algorithm = alg, method = meth)
                    Ythr = SLT.returnThry(Data['V'])
                    err = SLT.standardError(Data['V'],Data['I'])
                    end = time.time()
                    
                    label = '%s-%s->    time: %d    Err: %.2f' %(alg,meth,end-start, err)
                    
                    plt.plot(Data['V'], Ythr, fmt[i], label=label)
                except:
                    print('%s Failed'%meth)
    plt.legend()
    plt.savefig('Plots//test_ModelMethods_CompareSimple_Unbounded')
    
if __name__ == '__main__':
    test_compareModels_simple(fName = 'Data\\2H-c-2F (amps).txt')