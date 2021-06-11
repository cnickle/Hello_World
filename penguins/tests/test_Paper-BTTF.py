from penguins.functions import E_act_fixedtemp_biasvoltage  as E_act
from penguins.Model import Model as mod
import matplotlib.pyplot as plt
import pandas as pd

def test_Eac_20171024(fileName = 'tests\\Data\\BTTF Data\\20171024.txt'):
    initpar = {
        'Energy' : -0.85,
        'lambda' : 0.82,
        'cap'    : 1.78,
        'AWidth' : -0.85,
        'QWidth' : 0.53,
        'T0'     : 298,
        'T1'     : 300
        }
    Fixed = {
        'T0' : 298,
        'T1' : 300,
        }
    
    data = pd.read_csv(fileName,delimiter = '\t',header=None)
    data.columns = ['V', 'Ea']
       
    EaModel = mod(E_act)
    EaModel.setParams(initpar, Fixed = Fixed)
    EaModel.fit(data['V'],data['Ea'], save = 'BTTF_20171024', algorithm = 'LS', mode = 'verbose')
    EaModel.print(data['V'],data['Ea'],save = 'BTTF_20171024')
    Err= EaModel.standardError(data['V'],data['Ea'])
    data['thr'] = EaModel.returnThry(data['V'])
    
    plt.figure('BTTF_20171024')
    plt.scatter(data['V'],data['Ea'], color = 'black')
    plt.plot(data['V'],data['thr'], color='red')
    assert Err < 2.20

def test_Eac_20180203_1(fileName = 'tests\\Data\\BTTF Data\\20180203-1.txt'):
    initpar = {
        'Energy' : -0.89,
        'lambda' : 1.22,
        'cap'    : 1.45,
        'AWidth' : -0.51,
        'QWidth' : 0.30,
        'T0'     : 298,
        'T1'     : 300
        }
    Fixed = {
        'T0' : 298,
        'T1' : 300,
        }
        
    bnds = {
        'Energy' : [-1,0],
        'lambda' : [0,2],
        'cap'    : [0,2],
        'AWidth' : [-1,1],
        'QWidth' : [.01,1]
        }
    
    data = pd.read_csv(fileName,delimiter = '\t',header=None)
    data.columns = ['V', 'Ea']
       
    EaModel = mod(E_act)
    EaModel.setParams(initpar, Fixed = Fixed,bnds = bnds)
    EaModel.fit(data['V'],data['Ea'], save = 'BTTF_20180203-1', algorithm = 'LS', mode = 'verbose')
    EaModel.print(data['V'],data['Ea'],save = 'BTTF_20180203-1')
    Err= EaModel.standardError(data['V'],data['Ea'])
    data['thr'] = EaModel.returnThry(data['V'])
    
    plt.figure('BTTF_20180203-1')
    plt.scatter(data['V'],data['Ea'], color = 'black')
    plt.plot(data['V'],data['thr'], color='red')
    assert Err < 1.75
    
def test_Eac_20180203_2(fileName = 'tests\\Data\\BTTF Data\\20180203-2.txt'):
    initpar = {
        'Energy' : -0.89,
        'lambda' : 1.22,
        'cap'    : 1.45,
        'AWidth' : -0.51,
        'QWidth' : 0.30,
        'T0'     : 298,
        'T1'     : 300
        }
    Fixed = {
        'T0' : 298,
        'T1' : 300,
        }
    bnds = {
        'Energy' : [-1,0],
        'lambda' : [0,2],
        'cap'    : [0,2],
        'AWidth' : [-1,1],
        'QWidth' : [.01,1]
        }
    
    data = pd.read_csv(fileName,delimiter = '\t',header=None)
    data.columns = ['V', 'Ea']
       
    EaModel = mod(E_act)
    EaModel.setParams(initpar,Fixed = Fixed, bnds = bnds)
    EaModel.fit(data['V'],data['Ea'], save = 'BTTF_20180203-2', algorithm = 'LS')
    EaModel.print(data['V'],data['Ea'],save = 'BTTF_20180203-2')
    Err= EaModel.standardError(data['V'],data['Ea'])
    data['thr'] = EaModel.returnThry(data['V'])
    
    plt.figure('BTTF_20180203-2')
    plt.scatter(data['V'],data['Ea'], color = 'black')
    plt.plot(data['V'],data['thr'], color='red')
    assert Err < 2.60

def test_Eac_20180406(fileName = 'tests\\Data\\BTTF Data\\20180406.txt'):
    initpar = {
        'Energy' : -2.78e-01,
        'lambda' : 8.32e-01,
        'cap'    : 6.76e-01,
        'AWidth' : -8.77e-01,
        'QWidth' : 1.67e-01,
        'T0'     : 298,
        'T1'     : 300
        }
    Fixed = {
        'T0' : 298,
        'T1' : 300,
        }
    bnds = {
        'Energy' : [-1,0],
        'lambda' : [0,2],
        'cap'    : [0,2],
        'AWidth' : [-1,1],
        'QWidth' : [.01,1]
        }
    
    data = pd.read_csv(fileName,delimiter = '\t',header=None)
    data.columns = ['V', 'Ea']
       
    EaModel = mod(E_act)
    EaModel.setParams(initpar, Fixed = Fixed,bnds = bnds)
    EaModel.fit(data['V'],data['Ea'], save = 'BTTF_20180406', algorithm = 'LS', mode = 'verbose')
    EaModel.print(data['V'],data['Ea'],save = 'BTTF_20180406')
    Err= EaModel.standardError(data['V'],data['Ea'])
    data['thr'] = EaModel.returnThry(data['V'])
    
    plt.figure('BTTF_20180406')
    plt.scatter(data['V'],data['Ea'], color = 'black')
    plt.plot(data['V'],data['thr'], color='red')
    assert Err < 1.86
    
def test_Eac_20180426(fileName = 'tests\\Data\\BTTF Data\\20180426.txt'):
    initpar = {
        'Energy' : -0.89,
        'lambda' : 1.22,
        'cap'    : 1.45,
        'AWidth' : -0.51,
        'QWidth' : 0.30,
        'T0'     : 298,
        'T1'     : 300
        }
    Fixed = {
        'T0' : 298,
        'T1' : 300,
        }
    bnds = {
        'Energy' : [-1,0],
        'lambda' : [0,2],
        'cap'    : [0,2],
        'AWidth' : [-1,1],
        'QWidth' : [.01,1]
        }
    
    data = pd.read_csv(fileName,delimiter = '\t',header=None)
    data.columns = ['V', 'Ea']
       
    EaModel = mod(E_act)
    EaModel.setParams(initpar, Fixed = Fixed,bnds = bnds)
    EaModel.fit(data['V'],data['Ea'], save = 'BTTF_20180426', algorithm = 'LS', mode = 'verbose')
    EaModel.print(data['V'],data['Ea'],save = 'BTTF_20180426')
    Err= EaModel.standardError(data['V'],data['Ea'])
    data['thr'] = EaModel.returnThry(data['V'])
    
    plt.figure('BTTF_20180426')
    plt.scatter(data['V'],data['Ea'], color = 'black')
    plt.plot(data['V'],data['thr'], color='red')
    assert Err < 2

if __name__ == '__main__':
    test_Eac_20171024(fileName   = 'Data\\BTTF Data\\20171024.txt')
    # test_Eac_20180203_1(fileName = 'Data\\BTTF Data\\20180203-1.txt')
    # test_Eac_20180203_2(fileName = 'Data\\BTTF Data\\20180203-2.txt')
    # test_Eac_20180406(fileName   = 'Data\\BTTF Data\\20180406.txt')
    # test_Eac_20180426(fileName   = 'Data\\BTTF Data\\20180426.txt')