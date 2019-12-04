# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:21:27 2019

This file will simply be used to provide functions for manipulation of the data
files given to us via our Singapore Collaboration.

@author: Cam
"""
import csv
import matplotlib.pyplot as plt
import scipy.optimize as sco
import numpy as np
import random
import math
from class_gen import gen as genAlg

class sciData:
# %% Fitting functions
    def __fit(self,model,parBnds,parInitial):        
        p0=[]
        bnds = []
        lower = []
        upper = []
        for name in list(parBnds.keys()):
            p0 += [parInitial[name]]
            lower +=[parBnds[name][0]]
            upper +=[parBnds[name][1]]
       
        bnds = [lower,upper]
        
        X  = self.workingdat['X']
        Y  = self.workingdat['Y']
        
        results,covar=sco.curve_fit(model,X,Y,p0=p0,bounds=bnds)
        return results,covar
    
    def fit(self,parBnds, parInitial):
        self.fitbool = True
        results,covar  = self.__fit(self.model,parBnds,parInitial)
        
        self.modelDat['X']=self.workingdat['X']
        self.modelDat['Y']=self.model(self.workingdat['X'], *results)
        
        cnt = 0
        for name in list(parBnds.keys()):
            self.parameters[name] = results[cnt]
            self.errors[name]  = np.sqrt(np.diag(covar))[cnt]
            cnt +=1
    
#    def randomized_fit(self,parBnds, parInitial):
#        parInitial = {}
#        for name in list(parBnds.keys()):
#            parInitial[name] = random.uniform(parBnds[name][0],parBnds[name][1])
#        results,__  = self.__fit(self.model,parBnds,parInitial)
#        standarderror = np.sum(np.subtract(
#                self.workingdat['Y'],
#                self.model(self.workingdat['X'], *results
#                           ))**2)
#        print(standarderror)
    
    def evolutionary(self, parBnds, popSize, genSize):
        popObj = genAlg(parBnds,popSize)
        
        for gen in range(genSize):
            weights = []
            pop = popObj.nextgen(weights)
            for count in pop.keys():
                w = self.calcRelativeError(pop[count])
                print('\t%.3f'%w)
                weights += [w]
            
            print('Gen %d Avg: %.3f'%(gen,np.mean(weights)))
            np.reciprocal(weights)
            if gen > 0:
                for name in popObj.parameters.keys():
                    avg = np.mean(popObj.parameters[name])
                    std = np.std(popObj.parameters[name])
                    lower = avg-3*std
                    higher = avg+3*std
                    print('%s\t[%f,%f]'%(name,lower,higher))
            
            

# %% Output Data and Plots
    def parRange(self, initpar, par, ran, size = 10):        
        pars = initpar.copy()
        step = (ran[1]-ran[0])/size
        t = np.arange(ran[0],ran[1],step)
        
        output = ''
        for i in t:
            pars[par]=i
            Err = self.calcRelativeError(pars)
            for name in pars.keys():
                output = output + '%f\t'%pars[name]
            output = output + '%.3f\n'%Err
            f= open('Test.txt',"a")
            f.write(output)
            f.close()
            print('%.6f\t%.3f'%(i,Err))
            
    def returnThry(self, xvals, initpar=[]):
        if not initpar:
            initpar = self.parameters.copy()
        yThr = self.model(xvals, *initpar.values())
        dat = {'X': xvals,
               'Y': yThr}
        return dat
    
    def calcRelativeError(self, initpar):
        X = self.workingdat['X']
        Y = self.workingdat['Y']
        Ythr = self.model(X,*initpar.values())
        residual = np.subtract(np.log(np.abs(Y)),np.log(np.abs(Ythr)))
        Error = np.sqrt(np.sum(residual**2))
        return Error
    
    def printFit(self,*args):
        Err = self.calcRelativeError(self.parameters)
        output = "Fit Report:\tError:\t%.5f\n" % Err
        output = output + "\tPar:\tVal\tErr\n"
        
        for name in list(self.parameters.keys()):
            output = output + "\t%s\t%e\t%e\n" %(name,
                                               self.parameters[name],
                                               self.errors[name])
        print(output)
        if args:
            if args[0] == 'L':
                output = ''
                for name in self.parameters.keys():
                    output = output + '%f\t'%self.parameters[name]
                output = output + '%.3f\n'%Err
                f= open(args[1],"a")
                f.write(output)
                f.close()
                
    def plot(self,*args):
        plt.figure()
        plt.scatter(self.rawdat['X'],self.rawdat['Y'],s=10,color='black')
        plt.autoscale(False)
        
        # If a fit has been done it will plot the model on top
        if self.fitbool:
            XThr = self.workingdat['X']
            YThr = self.model(self.workingdat['X'], *self.parameters.values())
            plt.plot(XThr,YThr)
        
        if args:
        #If par = I, plot the data with the initial parameters
            if args[0] == 'I':
                Y = self.workingdat['Y']
                XThr = self.workingdat['X']
                YThr = self.model(self.workingdat['X'],*args[1].values())
                plt.plot(XThr,YThr)
                print(np.mean(np.abs(np.subtract(Y,YThr)/(np.max(Y)-np.min(Y))))*100)
            
            
            
        #If the user as specified a name for the plot, then the plotwill be saved.
            if args[0] == 's':
                PlotN = '%s\\%s.png' %(self.directory,args[1])
                plt.savefig(PlotN)
                plt.close()

# %%Utility Functions           
    def __init__(self,fName, directory,equation, rawdat = {},scale = []):
        self.fitbool = False
        self.rawdat = {}
        self.parameters = {}
        self.errors = {}
        self.scale = scale
        self.modelDat = {}
        
        self.directory  = directory
        self.fileName = fName
        if not bool(rawdat):
            self.__readData()
        else:
            self.rawdat = rawdat
        self.workingdat = self.rawdat.copy()
        self.model = np.vectorize(
                lambda x,*args: equation(x,*args)*self.scale[1])
    
    def subsetData(self, xRange = []):
        X = self.rawdat['X']
        Y = self.rawdat['Y']
        newX=[]
        newY=[]
        for i in range(len(X)):
            if xRange[0]>X[i]:
                continue
            if xRange[1]<X[i]:
                continue
            
            newX+=[X[i]]
            newY+=[Y[i]]
        
        self.workingdat['X'] = newX
        self.workingdat['Y'] = newY
        
        return newX,newY

# %% Private Functions
    def __readData(self):
        DataFileN='%s\\%s' %(self.directory,self.fileName)

        X=[]
        Y=[]
        with open(DataFileN,newline='') as f:
            reader=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONNUMERIC)
            for V,E in reader:
                X+=[V]
                Y+=[E]
        if not self.scale:
            self.__calculatescale(Y)
        self.rawdat['X'] = np.float64(X)
        self.rawdat['Y'] = np.float64(Y)*self.scale[1]
        
        return X,Y   

    def __calculatescale(self, x):
        scaleparm = int(np.floor(math.log10(np.abs(np.mean(x)))/3))
        multiplier = 10**(-3*scaleparm)
        
        switchCases={
                -6  :'atto',
                -5  :'fempto',
                -4  :'pico',
                -3  :'nano',
                -2  :'micro',
                -1  :'milli',
                0   :'',
                1   :'kilo',
                2   :'mega',
                3   :'giga',
                4   :'tera',
                5   :'peta',
                6   :'exa',
                }
        
        self.scale = [switchCases[scaleparm],multiplier]
        return self.scale