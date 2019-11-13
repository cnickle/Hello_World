from class_sciData import sciData
import models
import pytest
import os
import numpy as np
directory = '%s\\tests'%os.getcwd()

initpar = {
        'gammaL'  : 0.5,
        'gammaR'  : 0.5,
        'deltaE1' : 1.1,
        'eta'     : 0.5,
        'W'       : 0.07
        }

bnds = {
        'gammaL'  : [-10,10],
        'gammaR'  : [-10,10],
        'deltaE1' : [-10,10],
        'eta'     : [0.0,1.]
        }

def test_compare_with_mathematica1():
    mathematica = [-2.85537E-8, -1.70412E-8, -8.84325E-9, -3.6813E-9, \
                   3.23947E-9, 6.73937E-9, 1.11846E-8, 1.72905E-8]
    vb = [-1, -.75, -.5, -.25,.25, .50, .75, 1]
    for i in range(len(vb)):
        f = models.tunnelmodel_1level_nogate_300K_gauss(vb[i],.000268, 0.464,\
                                                        0.57, 0.58, 0.07)
        print('%e\t%e\t%.2f'%(f,mathematica[i],100*abs((f-mathematica[i])/f)))
        #assert abs((f-mathematica[i])/f) < 0.05
    
def test_compare_with_mathematica2():
    mathematica = [-8.20749E-8, -3.17403E-8, -5.23501E-9, -5.90173E-10,\
                   2.90532E-10, 6.28546E-10, 1.30025E-9,3.02873E-9]
    vb = [-1, -.75, -.5, -.25,.25, .50, .75, 1]
    for i in range(len(vb)):
        f = models.tunnelmodel_1level_nogate_300K_gauss(vb[i],.0005, 0.02,\
                                                        0.62, 0.7, 0.14)
        assert abs((f-mathematica[i])/f) < 0.05
test_compare_with_mathematica1()
#def test_2Hc2F():
#    par = {
#        'gammaL'  : 0.000268,
#        'gammaR'  : 0.464514,
#        'deltaE1' : 0.574295,
#        'eta'     : 0.582212
#        }
#    
#    fName = '2H-c-2F (amps).txt'
#    data = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K)
#    data.fit(bnds,initpar)
#    
#    for key in list(par.keys()):
#        assert abs((par[key]-data.parameters[key])/par[key]) < 0.05
#
#def test_2Hs2F():
#    par = {
#        'gammaL'  : 0.000042,
#        'gammaR'  : 0.229306,
#        'deltaE1' : 0.835175,
#        'eta'     : 1.000000
#        }
#    
#    fName = '2H-s-2F (amps).txt'
#    data = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K)
#    data.fit(bnds,initpar)
#    
#    for key in list(par.keys()):
#        assert abs((par[key]-data.parameters[key])/par[key]) < 0.05