import penquins.functions as fun
import numpy as np
import matplotlib.pyplot as plt
from penquins.class_sciData import sciData

def reducedTunnelModel(vb, gammaC, gammaW, deltaE1, eta):
    T = 300
    c = 0
    vg = 0
    sigma = 0
    
    return fun.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

def test_MinimizeLinear():
    args = {'m' : 1, 'b' : 1}
    
    X = np.array([1,2,3,4,5])
    Y = fun.linear(X,*args.values())
    
    init = {
            'm' : 20,
            'b' : 15
            }
    
    def func(Yexp, Ythr):
        return np.sqrt(np.sum((Yexp-Ythr)**2))
    
    data = sciData('',fun.linear,rawdat = {'X' : X,'Y' : Y})
    data.customFit(func,init)
    data.printFit()
    
    for key in list(args.keys()):
        assert abs((args[key]-data.parameters[key])/args[key]) < 0.05
    
    plt.figure()
    plt.scatter(X,Y)
    ythr = fun.linear(X,*data.parameters.values())
    plt.plot(X,ythr)