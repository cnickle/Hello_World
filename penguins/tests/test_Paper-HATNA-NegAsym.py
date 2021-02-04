import numpy             as     np
import pandas            as     pd
import matplotlib.pyplot as     plt
from glob                import glob
from penguins.functions  import HysteresisModel         as HystModel
from penguins.Model      import Model                   as mod

def test_Model_Plotted():
    DataFile = 'Data\\AsymNeg_cont_Normalized.txt'
    data = pd.read_csv(DataFile, delimiter = '\t')
    colV = '-2.00V_1'
    colC = '-2.00C_1'
    
    bnds = {
            'n'       : [0,10E5],
            'gammaL'  : [1E-11, 0.1],
            'gammaR'  : [1E-11, 0.1],
            'kappa'   : [1E-11,200],
            'sigma'   : [1E-6,.5],
            'E_AB'    : [.65,0.90],
            'E_AC'    : [-1.35, 0],
            'chi'     : [0.44, 3.17],
            'eta'     : [0.50, 0.78],
            'gam'     : [0, 20],
            'lam'     : [1.02, 1.34],
            'P'       : [0,.1],
            'u'       : [0,300]
    }

    initpar = {
    	'n':	500000,
    	'gammaL':	7.5238e-05,
    	'gammaR':	5.9848e-05,
    	'kappa':	37.805,
    	'sigma':	0,
    	'E_AB':	0.7241,
    	'E_AC':	-1.1143,
    	'chi':	2.6263,
    	'eta':	0.67,
    	'gam':	3.0255,
    	'lam':	1.1208,
    	'P':	2.0167e-04,
    	'u':	0.01,
        'c'       : 0,
        'vg'      : 0,
        'T'       :300
    }
    Fixed = ['n','sigma','P','u','c','vg','T']
    
    plt.figure('currents')
    data = data[abs(data[colV]) > 0]
    
    plt.plot(data[colV],np.abs(data[colC]))
    NitzModel = mod(HystModel)
    NitzModel.setParams(initpar,fixed=Fixed,bnds=bnds)
    NitzModel.fit(np.array(data[colV]), data[colC], algorithm = 'LS',
                   scale = 'log', mode = 'verbose', save = 'AsymNeg',
                   method = 'trf')
    NitzModel.print(data[colV], data[colC])
    Ythr = NitzModel.returnThry(data[colV])
    
    plt.plot(data[colV],np.abs(Ythr), color = 'red', linewidth = 4)
    plt.yscale('log')
    
    
if __name__ == '__main__':
    test_Model_Plotted()
#Assert Error ==-3

        
