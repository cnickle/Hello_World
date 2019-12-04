from class_sciData import sciData
import models
import os
import time
directory = '%s'%os.getcwd()

def test_compare_with_mathematica1():
    mathematica = [-4.50152E-9, -3.6177E-9, -2.61204E-9,\
                   -2.1881E-9,1.25685E-9, 4.62326E-9, 8.52859E-9, 1.42613E-8]
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

def test_2Hc2F():
    start = time.time()
    initpar = {
        'gammaL'  : 0.032248,
        'gammaR'  : 0.008489,
        'deltaE1' : 0.874857,
        'eta'     : 0.590363,
        'width'   : 0.0087
        }
    
    fName = 'tests\\2H-c-2F (amps).txt'
    data = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K_gauss)
    SE=data.calcRelativeError(initpar)
    runtime = time.time()-start
    assert SE < 5 and runtime < 20
    
def test_2Hs2F():
    start = time.time()
    initpar = {
        'gammaL'  : 0.000060,
        'gammaR'  : 0.006331,
        'deltaE1' : 0.708984,
        'eta'     : 0.742364,
        'width'   : 0.177987
        }
    
    fName = 'tests\\2H-s-2F (amps).txt'
    data = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K_gauss)
    SE=data.calcRelativeError(initpar)
    runtime = time.time()-start
    assert SE < 32 and runtime < 20

if __name__ == "__main__":
    test_2Hs2F()
    test_2Hc2F()