import penquins.models as models
import numpy as np
import matplotlib.pyplot as plt
from penquins.class_sciData import sciData

def reducedTunnelModel(vb, gammaC, gammaW, deltaE1, eta):
    T = 300
    c = 0
    vg = 0
    sigma = 0
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

def test_MinimizeLinear():
    args = {'m' : 1, 'b' : 1}
    
    X = np.array([1,2,3,4,5])
    Y = models.linear(X,*args.values())
    
    init = {
            'm' : 20,
            'b' : 15
            }
    
    def func(Yexp, Ythr):
        return np.sqrt(np.sum((Yexp-Ythr)**2))
    
    data = sciData('',models.linear,rawdat = {'X' : X,'Y' : Y})
    data.customFit(func,init)
    data.printFit()
    
    for key in list(args.keys()):
        assert abs((args[key]-data.parameters[key])/args[key]) < 0.05
    
    plt.figure()
    plt.scatter(X,Y)
    ythr = models.linear(X,*data.parameters.values())
    plt.plot(X,ythr)

#def test_MinimizeLinearWithData(fName = 'tests\\minTest.txt'):
#    init = {
#            'gammaC' : 6.8000e-04,
#            'gammaW' : 5.08e-03,
#            'deltaE' : 1.2,
#            'eta'    : 3.568982e-01
#            }	
#
#    def func(Yexp, Ythr):
#        residual = np.subtract(np.log(np.abs(Yexp)),np.log(np.abs(Ythr)))
#        Error = np.sqrt(np.sum(residual**2))
#        return np.sqrt(np.sum((Yexp-Ythr)**2))
#    
#    data = sciData(fName,reducedTunnelModel)
#    data.customFit(func,init)
##    data.fit({},init)
#    data.printFit(save = 'testparams.txt')
##    if not data.modelDat:
##        data.modelDat['X'] = data.rawdat['X']
##        data.modelDat['Y'] = data.model(data.modelDat['X'],*init.values())
#    
#    plt.figure()
#    plt.scatter(data.rawdat['X'],abs(data.rawdat['Y']),color = 'black')
#    plt.plot(data.rawdat['X'],abs(data.modelDat['Y']), color = 'blue')
#    plt.tight_layout()
#    plt.yscale('log')
#    plt.ylim([.001,1000])
    
if __name__ == "__main__":
    test_MinimizeLinear()
#    test_MinimizeLinearWithData(fName = 'minTest.txt')