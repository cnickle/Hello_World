import numpy as np
import matplotlib.pyplot as plt
import penquins.models as models

def reducedBridgePop(vb, gammaL, gammaR, deltaE, eta):
    c = 0
    vg = 0
    T = 300
    return models.averageBridgePopulation(vb, gammaL, gammaR, deltaE, eta, c, vg, T)

def reducedNitzanSwitchingRate(vb, gammaL, gammaR, deltaE, eta, sigma, R1, R2):
    c = 0
    vg = 0
    T = 300
    return models.NitzanSwitchingRate(vb, gammaL, gammaR, deltaE, eta, sigma, c, vg, T, R1, R2)

def test_averageBridgePopulation():
    v = np.arange(-2,1,.1)
    eq = np.vectorize(reducedBridgePop)
    
    #%% High Conduction State
    gammaCH = 6.86E-05
    gammaWH = 2*np.sqrt(gammaCH)
    
    gammaLH = .5*(gammaWH-np.sqrt(gammaWH**2-4*gammaCH))
    gammaRH = gammaWH-gammaLH
    
    parametersH = {
        'gammaL'    :   gammaLH,
        'gammaR'    :   gammaRH,
        'deltaE'    :   0.487,
        'eta'       :   0.556,
        }
    nH = eq(v, *parametersH.values())
    
    #%% Low Conduction State
    gammaCL = 0.000118391
    gammaWL = 0.1474256
    
    gammaLL = .5*(gammaWL-np.sqrt(gammaWL**2-4*gammaCL))
    gammaRL = gammaWL-gammaLL
    
    parametersL = {
        'gammaL'    :   gammaLL,
        'gammaR'    :   gammaRL,
        'deltaE'    :   2.231518,
        'eta'       :   1,
        }
    nL = eq(v, *parametersL.values())
    
    plt.figure()
    plt.plot(v, nH, color = 'red', linewidth=4)
    plt.plot(v, nL, color = 'blue', linewidth=4)
if __name__ == '__main__':
    test_averageBridgePopulation()