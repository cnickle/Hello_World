from penguins.class_sciData import sciData
import penguins.models as models
import numpy as np

def reducedTunnelModel_NoGauss(vb, gammaL, gammaR, deltaE1, eta):
    T = 300
    c = 0
    vg = 0
    sigma = 0
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

def test_subset():
    dat = {
        'X' : np.arange(0,100,1),
        'Y' : np.arange(0,100,1)
        }
    
    data = sciData('', models.linear, rawdat = dat)
    data.subsetData([0,50])
    print(len(data.workingdat['X']))
    assert len(data.workingdat['X']) < 55

def test_returnDataFunc(fName = 'tests\\2H-c-2F (amps).txt'):
    par = {
        'gammaL'  : 0.0003,
        'gammaR'  : 0.45,
        'deltaE1' : 0.6,
        'eta'     : 0.58
        }
    
    data = sciData(fName, reducedTunnelModel_NoGauss)
    X,Y = data.returnThry(par)
    assert len(X) > 0 and len(Y) >0

def test_returnDataFunc2(fName = 'tests\\2H-c-2F (amps).txt'):
    par = {
        'gammaL'  : 0.000268,
        'gammaR'  : 0.464514,
        'deltaE1' : 0.574295,
        'eta'     : 0.582212
        }
    
    data = sciData(fName, reducedTunnelModel_NoGauss)
    X1,Y1 = data.returnThry(par)
    data.fit({}, par)
    X,Y = data.returnThry()
    diff = np.sum(np.subtract(Y,Y1))
    assert len(X) > 0 and len(Y) >0 and abs(diff) > 0

def test_plot(fName = 'tests\\2H-c-2F (amps).txt'):
    par = {
        'gammaL'  : 0.000268,
        'gammaR'  : 0.464514,
        'deltaE1' : 0.574295,
        'eta'     : 0.582212
        }
    
    data = sciData(fName,reducedTunnelModel_NoGauss)
    data.plot(pars=par,scale = 'nano')

def test_plot_InitParandScale(fName = 'tests\\2H-c-2F (amps).txt'):
    par = {
        'gammaL'  : 0.000268,
        'gammaR'  : 0.464514,
        'deltaE1' : 0.574295,
        'eta'     : 0.582212
        }
    
    data = sciData(fName,reducedTunnelModel_NoGauss)
    data.plot(pars=par,scale = 'nano')

def test_plot_fit(fName = 'tests\\2H-c-2F (amps).txt'):

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
    data.fit(bnds,par)
    data.plot()
    
def test_plot_FitScaleParSave(fName = 'tests\\2H-c-2F (amps).txt'):
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
    data.fit(bnds,par)
    data.plot(pars = par,scale = 'nano',save='tests\\test')

if __name__ == "__main__":
    test_subset()
    test_plot('2H-c-2F (amps).txt')
    test_plot_InitParandScale('2H-c-2F (amps).txt')
    test_plot_fit('2H-c-2F (amps).txt')
    test_plot_FitScaleParSave('2H-c-2F (amps).txt')
    test_returnDataFunc('2H-c-2F (amps).txt')