from class_sciData import sciData
import time
start = time.time()
import models
import os
import numpy
import matplotlib.pyplot as plt
import cProfile

os.chdir('..')
initpar = {
        'gammaL'  : 0.032248,
        'gammaR'  : 0.008489,
        'deltaE1' : 0.874857,
        'eta'     : 0.590363#,
        #'width'   : 0.002
        }

bnds = {
        'gammaL'  : [0,.5],
        'gammaR'  : [0,.5],
        'deltaE1' : [0,1],
        'eta'     : [0.0,1.]#,
        #'width'   : [0.0,1.]
        }

directory = '%s\\Rh2 Tet1'%os.getcwd()
fName = '2H-c-2F (amps).txt'
Rh2Tet1 = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K)
Rh2Tet1.fit(bnds,initpar)
Rh2Tet1.printFit()
save1 = Rh2Tet1.modelDat
#Rh2Tet1.saveModel(fname)
#Rh2Tet1.plot('I',initpar)

#bnds = {
#        'gammaL'  : [0,.5],
#        'gammaR'  : [0,.5],
#        'deltaE1' : [0,1],
#        'eta'     : [0.0,1.],
#        'width'   : [0.0,1.]
#        }
#
#initpar = {
#        'gammaL'  : 0.000060,
#        'gammaR'  : 0.006331,
#        'deltaE1' : 0.708984,
#        'eta'     : 0.742364,
#        'width'   : 0.177987
#        }
#
#directory = '%s\\Rh2 Tet2'%os.getcwd()
#fName = '2H-s-2F (amps).txt'
#Rh2Tet2 = sciData(fName,directory,models.tunnelmodel_1level_nogate_300K_gauss)
#Rh2Tet2.fit(bnds,initpar)
#Rh2Tet2.printFit()
#fname = '%s\\thrymodel' %directory
#save2 = Rh2Tet2.modelDat
#Rh2Tet2.plot()

end = time.time()
print(end-start)