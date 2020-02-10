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
import warnings
import random
import math
from penquins.class_gen import gen as genAlg

class sciData:
# %% Fitting functions    
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
    def parRange(self, initpar, par, ran, size = 10, method = 'relative'):        
        pars = initpar.copy()
        step = (ran[1]-ran[0])/size
        t = np.arange(ran[0],ran[1],step)
        
        output = ''
        for i in t:
            pars[par]=i
            
            if method == 'relative':
                Err = self.calcRelativeError(pars)
            else:
                Err = self.calcLinearError(pars)
                
            for name in pars.keys():
                output = output + '%e\t'%pars[name]
            output = output + '%.3e\n'%Err
            f= open('Test.txt',"a")
            f.write(output)
            f.close()
            if Err < 0.01:
                print('%.3e\t%.3e'%(i,Err))
            else:
                print('%.3e\t%.3f'%(i,Err))
            
    def returnThry(self, xvals, initpar=[]):
        if not initpar:
            initpar = self.parameters.copy()
        yThr = self.model(xvals, *initpar.values())
        dat = {'X': xvals,
               'Y': yThr}
        return dat
    
    def subsetData(self, xRange = []):
        X = list(self.rawdat['X'])
        Y = list(self.rawdat['Y'])
        
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

# %% Tested and Reliable Functions
    def __init__(self,fName, equation, rawdat = {}):
        self.fitbool = False
        self.scaling = False
        self.multiplier = 1
        self.rawdat = {}
        self.parameters = {}
        self.errors = {}
        self.modelDat = {}
        self.fileName = fName
        
        if not bool(rawdat):
            self.__readData()
        else:
            self.rawdat = rawdat
        
        minval = min(abs(self.rawdat['Y']))
        if minval < np.sqrt(np.finfo(float).eps):
            # Due to curve_fit and machine precision limitations the data and
            # model are being scaled into the nano range. This should not 
            # effect the fitting parameters
            warnings.warn("Scaling data and equation due to floating point percision")
            self.scaling = True
            self.multiplier = 1E9
            self.rawdat['Y'] = self.rawdat['Y']*self.multiplier
        
        self.workingdat = self.rawdat.copy()
        self.model = np.vectorize(lambda x,*args: equation(x,*args)*\
                                  self.multiplier)    
    
    def __readData(self):
        DataFileN=self.fileName

        X=[]
        Y=[]
        with open(DataFileN,newline='') as f:
            reader=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONNUMERIC)
            for V,E in reader:
                X+=[V]
                Y+=[E]

        self.rawdat['X'] = np.float64(X)
        self.rawdat['Y'] = np.float64(Y)
        
        return X,Y
    
    def __fit(self,model,parBnds,parInitial):        
        p0=[]
        bnds = []
        lower = []
        upper = []
        for name in list(parInitial.keys()):
            p0 += [parInitial[name]]
            if parBnds:
                lower +=[parBnds[name][0]]
                upper +=[parBnds[name][1]]            
       
        bnds = [lower,upper]

        X  = self.workingdat['X']
        Y  = self.workingdat['Y']
        
        if parBnds:
            results,covar=sco.curve_fit(model,X,Y,p0=p0,bounds=bnds)
        else:
            results,covar=sco.curve_fit(model,X,Y,p0=p0)
        return results,covar
    
    def fit(self,parBnds, parInitial):
        self.fitbool = True
        results,covar  = self.__fit(self.model,parBnds,parInitial)
        
        self.modelDat['X']=self.workingdat['X']
        self.modelDat['Y']=self.model(self.workingdat['X'], *results)
        
        cnt = 0
        for name in list(parInitial.keys()):
            self.parameters[name] = results[cnt]
            self.errors[name]  = np.sqrt(np.diag(covar))[cnt]
            cnt +=1

    def __rangeOrder(self, order):
        switchCases={
                'atto'      :   18,
                'fempto'    :   15,
                'pico'      :   12,
                'nano'      :   9,
                'micro'     :   6,
                'milli'     :   3,
                ''          :   0,
                'kilo'      :   -3,
                'mega'      :   -6,
                'giga'      :   -9,
                'tera'      :   -12,
                'peta'      :   -15,
                'exa'       :   -18
                }
        
        multiplier = 10**(switchCases[order])
        return multiplier
    
    def calcLinearError(self, initpar):
        X = self.workingdat['X']
        Y = self.workingdat['Y']

        if self.modelDat:
            Ythr = self.modelDat['Y']
        else:
            Ythr = self.model(X,*initpar.values())
            
        residual = np.subtract(Y,Ythr)
        Error = np.sqrt(np.sum(residual**2))
        return Error
    
    def calcRelativeError(self, initpar):
        X = self.workingdat['X']
        Y = self.workingdat['Y']

        if self.modelDat:
            Ythr = self.modelDat['Y']
        else:
            Ythr = self.model(X,*initpar.values())
            
        residual = np.subtract(np.log(np.abs(Y)),np.log(np.abs(Ythr)))
        Error = np.sqrt(np.sum(residual**2))
        return Error
    
    def printFit(self,save = ''):
        Err = self.calcRelativeError(self.parameters)
        output = "Fit Report:\tError:\t%.2f\n" % Err
        output = output + "\tPar:\tVal\tErr\n"
        
        for name in list(self.parameters.keys()):
            output = output + "\t%s\t%e\t%e\n" %(name,
                                               self.parameters[name],
                                               self.errors[name])
        print(output)
        if save:
            output = ''
            for name in self.parameters.keys():
                output = output + '%e\t'%self.parameters[name]
            output = output + '%.3f\n'%Err
            f= open(save,"a")
            f.write(output)
            f.close()
               
    def plot(self,pars=[],save = '',scale = ''):
        if self.scaling:
            scale = 'nano'
        mult = self.__rangeOrder(scale)        
        
        plt.figure()
        plt.scatter(self.rawdat['X'],self.rawdat['Y']*mult,s=10,color='black')
        plt.autoscale(False)
        
        # If a fit has been done it will plot the model on top
        if self.fitbool:
            XThr = self.workingdat['X']
            YThr = self.model(self.workingdat['X'], *self.parameters.values())*mult
            plt.plot(XThr,YThr)
        
        if pars:
        #If pars, plot will plot the data with pars given
            XThr = self.workingdat['X']
            YThr = self.model(self.workingdat['X'],*pars.values())*mult
            plt.plot(XThr,YThr)
        
        if save:
        #If the user as specified a name for the plot, then the plotwill be saved.   
            plt.savefig(save)