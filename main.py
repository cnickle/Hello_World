from Class_sciData import sciData
import Models
import os

os.chdir('..\..')

initpar = {
        'gammaL'  : 0.000240,
        'gammaR'  : 0.463199,
        'deltaE1' : 0.551646,
        'eta'     : 0.594323#,
        #'width'   : 0.041789
        }

bnds = {
        'gammaL'  : [-10,10],
        'gammaR'  : [-10,10],
        'deltaE1' : [-10,10],
        'eta'     : [0.0,1.]#,
        #'width'   : [0.0,1.]
        }

directory = '%s\\Rh2 Tet1'%os.getcwd()
fName = 'Data.txt'
Rh2Tet1 = sciData(fName,directory,Models.tunnelmodel_1level_nogate_300K)
Rh2Tet1.randomized_fit(bnds,initpar)
Rh2Tet1.printFit()
Rh2Tet1.plot()

directory = '%s\\Rh2 Tet2'%os.getcwd()
Rh2Tet2 = sciData(fName,directory,Models.tunnelmodel_1level_nogate_300K)
Rh2Tet2.fit(bnds,initpar)
Rh2Tet2.printFit()
Rh2Tet2.plot()