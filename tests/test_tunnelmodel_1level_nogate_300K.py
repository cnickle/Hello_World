from class_sciData import sciData
import models
import pytest
import scipy.optimize as sco
import os
directory = '%s\\'%os.getcwd()

initpar = {
        'gammaL'  : 0.000256,
        'gammaR'  : 0.463199,
        'deltaE1' : 0.551646,
        'eta'     : 0.594323
        }

bnds = {
        'gammaL'  : [0,1],
        'gammaR'  : [0,1],
        'deltaE1' : [0,1.5],
        'eta'     : [0,1]
        }
def test_compare_with_mathematica():
    f = models.tunnelmodel_1level_nogate_300K(4,0.5,0.5,1.1,0.5)
    assert abs((f-0.0000476775)/f) <0.05

def test_2Hc2F(fName = 'tests\\2H-c-2F (amps).txt'):
    par = {
        'gammaL'  : 0.000268,
        'gammaR'  : 0.464514,
        'deltaE1' : 0.574295,
        'eta'     : 0.582212
        }
    
    data = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K)
    data.fit(bnds,initpar)
    
    for key in list(par.keys()):
        assert abs((par[key]-data.parameters[key])/par[key]) < 0.05

def test_2Hs2F(fName = 'tests\\2H-s-2F (amps).txt'):
    par = {
        'gammaL'  : 0.000042,
        'gammaR'  : 0.229306,
        'deltaE1' : 0.835175,
        'eta'     : 1.000000
        }
    
    data = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K)
    data.fit(bnds,initpar)
    
    for key in list(par.keys()):
        assert abs((par[key]-data.parameters[key])/par[key]) < 0.05
    
if __name__ == "__main__":
    test_2Hs2F('2H-s-2F (amps).txt')
    test_2Hc2F('2H-c-2F (amps).txt')