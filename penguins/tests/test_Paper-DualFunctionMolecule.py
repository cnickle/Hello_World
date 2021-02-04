from penguins.functions import tunnelmodel_singleLevel as tunnelModel
from penguins.functions import tunnelmodel_2level as tunnelModel2
from penguins.Model import Model as mod
import penguins.functions as fun
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

def test_memoryMolecule(fNameH = 'tests\\Data\\newhighEnergy.txt',
                        fNameL = 'tests\\Data\\newLowEnergy.txt'):
    """
    Here we're testing the results of the Dual Function molecule. If the function
    still describes the junction in the way that it is presented in Nat. Mat.
    Then this test is passed. 
    
    Further, this tests the fitting function itself using an algorithm and 
    method. Then plots the results of this and saves the file in the 'Plots'
    folder for review by the user. 
    """
    initpar = {
            'pos_L2H'       : -8.976e-01,
            'w_L2H'         :  3.300e-03 ,
            'p0_L2H'        :  0.00,
            'pos_H2L'       : -1.677e-01,
            'w_H2L'         :  4.495e-02,
            'p0_H2L'        : -2.676e-03,
            'gammaC1'       :  2.230719e-06,
            'gammaW1'       :  5.388790e-02,
            'deltaE1'       :  3.414365e-01,
            'eta1'          :  0.78,
            'width1'        :  1.445773e-02,
            'gammaC2'       :  1.173901e-05,
            'gammaW2'       :  3.545940e-02,
            'deltaE2'       :  5.594792e-01,
            'eta2'          :  0.78,
            'width2'        :  6.554842e-03,
            'gamC_L'        :  3E-06*5.370e-03,
            'gamW_L'        :  5.370e-03+3E-06,
            'delE_L'        :  0.6,
            'eta_L'         :  5.8e-01,
            'width_L'       :  3.222e-02,
            'c'             :  0,
            'Vg'            :  0,
            'T'             :  300
            }
    
    DataH = pd.read_csv(fNameH,delimiter = '\t',header=None)
    DataH.columns = ['V','I']
    
    DataL = pd.read_csv(fNameL,delimiter = '\t',header=None)
    DataL.columns = ['V','I']
    
    def memoryMolecule_gauss(vb, pos, width,p0,
                       gammaC1_H, gammaW1_H, deltaE1_H, eta1_H,sigma1_H,
                       gammaC2_H, gammaW2_H, deltaE2_H, eta2_H,sigma2_H,
                       gammaL_L, gammaR_L, deltaE_L, eta_L,width_L,
                       c, vg, T):
        
        P_H = fun.sigmoid(vb,pos,width)+(vb-1)*p0
        P_L = 1-P_H
        
        n=1
        I_H = tunnelModel2(vb,n,c,vg,T,
                                        gammaC1_H,gammaW1_H,deltaE1_H,eta1_H,sigma1_H,
                                        gammaC2_H,gammaW2_H,deltaE2_H,eta2_H,sigma2_H)
        args_L = (n, gammaL_L, gammaR_L, deltaE_L, eta_L,width_L,c,vg,T)
        I_L = tunnelModel(vb,*args_L)
        
        return P_H*I_H+P_L*I_L
    eq = np.vectorize(memoryMolecule_gauss)
    
    def Everything(vb,*args):
        #%% Permenant
        args_H = args[3:]
        mod1=eq(DataH['V'],*args_H)
        
        args_L = args[0:3]+args[6:]
        mod2=eq(DataL['V'],*args_L)
        return np.append(mod1,mod2)
    
    ToyModel = mod(Everything)
    ToyModel.setParams(initpar,fixed=['p0_L2H','c','Vg','T'])
    
    xVals = np.append(DataH['V'],DataL['V'])
    yVals = np.append(DataH['I'],DataL['I'])
    Error = ToyModel.standardError(xVals, yVals, scale = 'lin')
    
    assert (Error+7.18)/Error*100 < 5
    
    ToyModel.fit(xVals, yVals, algorithm = 'LS', method='trf',scale = 'log',
                  mode = 'verbose')
    # Ythr = ToyModel.returnThry(xVals)
    
    # plt.figure()
    # plt.scatter(xVals,np.log10(np.abs(yVals)),color = 'black')
    # plt.plot(xVals,np.log10(np.abs(Ythr)),color = 'red')
    # plt.savefig('Plots//test_Paper-DualFunction')

    

if __name__ == '__main__':
    test_memoryMolecule(fNameH = 'Data\\newhighEnergy.txt', fNameL = 'Data\\newLowEnergy.txt')