import numpy             as     np
import pandas            as     pd
import matplotlib.pyplot as     plt
from glob                import glob
from penguins.functions  import tunnelmodel_singleLevel as tunnelModel
from penguins.functions  import MarcusETRates           as ETRates
from penguins.functions  import averageBridgePopulation as BridgePop
from penguins.Model      import Model                   as mod


def test_Model_Plotted():
    initpar = {
        'n'       : 3.0029e+02,
        'gammaL'  : 1.2366e-02,
        'gammaR'  : 2.5545e-02,
        'kappa'   : 2.3379e-01,
        'sigma'   : 7.6073e-02,
        'E_AB'    : 9.7558e-01,
        'E_AC'    :-3.1251e-01,
        'chi'     : 1.4667e+00,
        'eta'     : 7.6600e-01,
        'gam'     : 1.9967e+01,
        'lam'     : 2.0745e+00,
        'P'       : 0,
        'u'       : 10/1000
    }
    
    plt.figure('All Scan Rate')
    volts = []
    for file in glob('Data\\HATNA_ScanRate\\*'):
        data = pd.read_csv(file, delimiter = '\t', header = None,skiprows =1)
        data.columns = ['V','absJ', 'J', 'V2', 'I']
        
        for V in data['V2']:
            V = np.round(V,2)
            if not V in volts:
                volts += [V]
        
        plt.plot(data['V2'],data['I'])
    plt.savefig('Plots//test_Paper_HATNA_ScanRate_Data')
    
    
    #%% Calculate all currents:
    calcDB = pd.DataFrame()
    calcDB['V'] = sorted(volts)
    
    eqSTL = np.vectorize(tunnelModel)
    calcDB['I_np'] = eqSTL(calcDB['V'], initpar['n'], initpar['gammaL']*initpar['gammaR'],
                           initpar['gammaL']+initpar['gammaR'], initpar['E_AB'],
                           initpar['eta'], initpar['sigma'],0,0,300)
    calcDB['I_p'] = eqSTL(calcDB['V'], initpar['n'],
                          initpar['gammaL']*initpar['gammaR']*initpar['kappa']**2,
                          (initpar['gammaL']+initpar['gammaR'])*initpar['kappa'],
                          initpar['E_AB']+initpar['chi'], initpar['eta'],
                          initpar['sigma'],0,0,300)
    
    eqETRates = np.vectorize(ETRates)
    calcDB['R_AC'], calcDB['R_CA'] = eqETRates(calcDB['V'], initpar['gam'], initpar['lam'], initpar['E_AC'], 300)
    calcDB['R_BD'], calcDB['R_DB'] = eqETRates(calcDB['V'], initpar['gam']*initpar['kappa'], initpar['lam'], initpar['E_AC']+initpar['chi'], 300)
    
    eqBridge = np.vectorize(BridgePop)
    calcDB['n_np'] = eqBridge(calcDB['V'], initpar['gammaL'], initpar['gammaR'], initpar['E_AB'], initpar['eta'], 0, 0, 300)
    calcDB['n_p']  = eqBridge(calcDB['V'], initpar['gammaL']*initpar['kappa'], initpar['gammaR']*initpar['kappa'], initpar['E_AB']+initpar['chi'], initpar['eta'], 0, 0, 300)
    
    calcDB['k_S0_S1'] = (1-calcDB['n_np'])*calcDB['R_AC'] + calcDB['n_np']*calcDB['R_BD']
    calcDB['k_S1_S0'] = (1-calcDB['n_p'])*calcDB['R_CA'] + calcDB['n_p']*calcDB['R_DB']
    
    plt.figure('All Scan Rate')
    plt.plot(calcDB['V'],calcDB['I_np'], color = 'black', linewidth = 4)
    plt.plot(calcDB['V'],calcDB['I_p'], color = 'blue', linewidth = 4)
    plt.savefig('Plots//test_Paper_HATNA_ScanRate_Currents')
    
    
    plt.figure('ET Rates')
    plt.plot(calcDB['V'],calcDB['R_AC'])
    plt.plot(calcDB['V'],calcDB['R_CA'])
    plt.plot(calcDB['V'],calcDB['R_BD'])
    plt.plot(calcDB['V'],calcDB['R_DB'])
    plt.savefig('Plots//test_Paper_HATNA_ETRates')
    
    plt.figure('Average Bridge Population')
    plt.plot(calcDB['V'],calcDB['n_p'])
    plt.plot(calcDB['V'],calcDB['n_np'])
    plt.savefig('Plots//test_Paper_HATNA_Bridge')
    
    plt.figure('K Rates')
    plt.plot(calcDB['V'],calcDB['k_S0_S1'])
    plt.plot(calcDB['V'],calcDB['k_S1_S0'])
    plt.savefig('Plots//test_Paper_HATNA_KRates')
    
    plt.figure('Theory Scan Rate')
    P=0
    diff = 0
    for file in glob('Data\\HATNA_ScanRate\\*'):
        data = pd.read_csv(file, delimiter = '\t', header = None,skiprows =1)
        data.columns = ['V','absJ', 'J', 'V2', 'I']
        scanrate = int(file.split('\\')[2].split('_')[2][4:7])
        
        delt = abs(data['V2'][2]-data['V2'][3])/(scanrate/1000)
        I = []
        Parray = []
        delArray = []
        
        for i,V in enumerate(data['V2']):
            V = np.round(V,2)
            tempDf =calcDB[calcDB['V']==np.round(V,2)].reset_index()
            calcs = dict(tempDf.iloc[0])
            Parray += [P]
            I += [((1-P)*calcs['I_np']+P*calcs['I_p'])]
            
            dPdt = calcs['k_S0_S1']-P*(calcs['k_S0_S1']+calcs['k_S1_S0'])
            delArray += [dPdt]
            P = P+dPdt*delt
        data['thrI'] = I
        plt.plot(data['V2'],data['thrI'], color = 'red')
        plt.plot(data['V2'],data['I'], color = 'black')
        diff += np.sum(np.subtract(data['I'],data['thrI'])**2)
    print(np.log10(np.sqrt(diff)))
    plt.savefig('Plots//test_Paper_HATNA_AllTheory')

def test_Model_Fitted():
    initpar = {
        'n'       : 3.0029e+02,
        'gammaL'  : 1.2366e-02,
        'gammaR'  : 2.5545e-02,
        'kappa'   : 2.3379e-01,
        'sigma'   : 0,
        'E_AB'    : 9.7558e-01,
        'E_AC'    :-3.1251e-01,
        'chi'     : 1.4667e+00,
        'eta'     : 7.6600e-01,
        'gam'     : 1.9967e+01,
        'lam'     : 2.0745e+00,
        'P'       : 0,
        'c'       : 0,
        'vg'      : 0,
        'T'       : 300
    }
    Fixed = ['sigma','P','c','vg','T']
    
    def minfunc(Vb, n, gammaL, gammaR, kappa, sigma, E_AB, E_AC, chi, eta, gam,
                lam, P, c, vg, T):
        
        returnCurrent = []        
        volts = []
        for file in glob('Data\\HATNA_ScanRate\\*'):
            data = pd.read_csv(file, delimiter = '\t', header = None,skiprows =1)
            data.columns = ['V','absJ', 'J', 'V2', 'I']
            
            for V in data['V2']:
                V = np.round(V,2)
                if not V in volts:
                    volts += [V]
        
        
        #%% Calculate all currents:
        calcDB = pd.DataFrame()
        calcDB['V'] = sorted(volts)
        
        eqSTL = np.vectorize(tunnelModel)
        calcDB['I_np'] = eqSTL(calcDB['V'], n, gammaL*gammaR, gammaL+gammaR,
                               E_AB, eta, sigma, c, vg, T)
        calcDB['I_p'] = eqSTL(calcDB['V'], n, gammaL*gammaR*kappa**2,
                              (gammaL+gammaR)*kappa, E_AB+chi, eta, sigma,
                              c, vg, T)
        
        eqETRates = np.vectorize(ETRates)
        calcDB['R_AC'], calcDB['R_CA'] = eqETRates(calcDB['V'], gam, lam,
                                                   E_AC, T)
        calcDB['R_BD'], calcDB['R_DB'] = eqETRates(calcDB['V'],  gam*kappa,
                                                   lam, E_AC+chi, T)
        
        eqBridge = np.vectorize(BridgePop)
        calcDB['n_np'] = eqBridge(calcDB['V'], gammaL, gammaR, E_AB, eta, c,
                                  vg, T)
        calcDB['n_p']  = eqBridge(calcDB['V'], gammaL*kappa, gammaR*kappa,
                                  E_AB+chi, eta, c, vg, T)
        
        calcDB['k_S0_S1'] = (1-calcDB['n_np'])*calcDB['R_AC'] + calcDB['n_np']*calcDB['R_BD']
        calcDB['k_S1_S0'] = (1-calcDB['n_p'])*calcDB['R_CA'] + calcDB['n_p']*calcDB['R_DB']
    
        P=P
        diff = 0
        for file in glob('Data\\HATNA_ScanRate\\*'):
            data = pd.read_csv(file, delimiter = '\t', header = None,skiprows =1)
            data.columns = ['V','absJ', 'J', 'V2', 'I']
            scanrate = int(file.split('\\')[2].split('_')[2][4:7])
            
            delt = abs(data['V2'][2]-data['V2'][3])/(scanrate/1000)
            I = []
            Parray = []
            delArray = []
            
            for i,V in enumerate(data['V2']):
                V = np.round(V,2)
                tempDf =calcDB[calcDB['V']==np.round(V,2)].reset_index()
                calcs = dict(tempDf.iloc[0])
                Parray += [P]
                I += [((1-P)*calcs['I_np']+P*calcs['I_p'])]
                
                dPdt = calcs['k_S0_S1']-P*(calcs['k_S0_S1']+calcs['k_S1_S0'])
                delArray += [dPdt]
                P = P+dPdt*delt
            
            returnCurrent = np.append(returnCurrent,I)
        return returnCurrent
    
    volts = []
    currs = []
    for file in glob('Data\\HATNA_ScanRate\\*'):
        data = pd.read_csv(file, delimiter = '\t', header = None,skiprows =1)
        data.columns = ['V','absJ', 'J', 'V2', 'I']
        volts = np.append(volts, data['V2'])
        currs = np.append(currs, data['I'])
    
    NitzModel = mod(minfunc)
    NitzModel.setParams(initpar,fixed=Fixed)
    NitzModel.fit(volts, currs, algorithm = 'LS', method='dogbox',scale = 'lin',
                  mode = 'verbose', save = 'TotalScanRate')
    NitzModel.print(volts,currs)
    
    
    
if __name__ == '__main__':
    test_Model_Fitted()
#Assert Error ==-3

        
