from penquins.class_sciData import sciData
import penquins.models as models
import time

def reducedTunnelModel(vb, gammaL, gammaR, deltaE1, eta,sigma):
    T = 300
    c = 0
    vg = 0   
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

def reducedTunnelModel_NoGauss(vb, gammaL, gammaR, deltaE1, eta):
    T = 300
    c = 0
    vg = 0
    sigma = 0
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

def test_2Hc2F(fName = 'tests\\2H-c-2F (amps).txt'):
    start = time.time()
    initpar = {
        'gammaL'  : 0.032248,
        'gammaR'  : 0.008489,
        'deltaE1' : 0.874857,
        'eta'     : 0.590363,
        'width'   : 0.0087
        }
    
    data = sciData(fName,reducedTunnelModel)
    SE=data.calcRelativeError(initpar)
    runtime = time.time()-start
    assert SE < 5 and runtime < 25
    
def test_2Hs2F(fName = 'tests\\2H-s-2F (amps).txt'):
    start = time.time()
    initpar = {
        'gammaL'  : 0.000060,
        'gammaR'  : 0.006331,
        'deltaE1' : 0.708984,
        'eta'     : 0.742364,
        'width'   : 0.177987
        }
    
    data = sciData(fName,reducedTunnelModel)
    SE=data.calcRelativeError(initpar)
    runtime = time.time()-start
    assert SE < 32 and runtime < 25

def test_2Hc2F_Nogauss(fName = 'tests\\2H-c-2F (amps).txt'):
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
    
    par = {
        'gammaL'  : 0.000268,
        'gammaR'  : 0.464514,
        'deltaE1' : 0.574295,
        'eta'     : 0.582212
        }
    
    data = sciData(fName,reducedTunnelModel_NoGauss)
    data.fit(bnds,initpar)
    
    for key in list(par.keys()):
        assert abs((par[key]-data.parameters[key])/par[key]) < 0.05

def test_2Hs2F_Nogauss(fName = 'tests\\2H-s-2F (amps).txt'):
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
    
    par = {
        'gammaL'  : 0.000042,
        'gammaR'  : 0.229306,
        'deltaE1' : 0.835175,
        'eta'     : 1.000000
        }
    
    data = sciData(fName,reducedTunnelModel_NoGauss)
    data.fit(bnds,initpar)
    
    for key in list(par.keys()):
        assert abs((par[key]-data.parameters[key])/par[key]) < 0.05