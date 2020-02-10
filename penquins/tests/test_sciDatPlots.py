from penquins.class_sciData import sciData
import penquins.models as models
import numpy as np

def reducedTunnelModel_NoGauss(vb, gammaL, gammaR, deltaE1, eta):
    T = 300
    c = 0
    vg = 0
    sigma = 0
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

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
    test_plot('2H-c-2F (amps).txt')
    test_plot_InitParandScale('2H-c-2F (amps).txt')
    test_plot_fit('2H-c-2F (amps).txt')
    test_plot_FitScaleParSave('2H-c-2F (amps).txt')