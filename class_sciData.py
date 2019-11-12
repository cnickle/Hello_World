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

class sciData:
    fitbool = False
    rawdat = {}
    parameters = {}
    errors = {}

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
        
        cnt = 0
        for name in list(parBnds.keys()):
            self.parameters[name] = results[cnt]
            self.errors[name]  = np.sqrt(np.diag(covar))[cnt]
            cnt +=1
    
    def randomized_fit(self,parBnds, parInitial):
        parInitial = {}
        for name in list(parBnds.keys()):
            parInitial[name] = random.uniform(parBnds[name][0],parBnds[name][1])
        results,__  = self.__fit(self.model,parBnds,parInitial)
        standarderror = np.sum(np.subtract(
                self.workingdat['Y'],
                self.model(self.workingdat['X'], *results
                           ))**2)
        print(standarderror)

# %% Output Data and Plots
    def saveModel(self,fName):
        LogFileN='%s\\%s.txt' % (self.directory,fName)
        LogFile = open(LogFileN, 'w')
        for index in range(len(self.workingdat)):
            LogFile.write('%f\t%f\n' %(
                    self.workingdat['X'][index],
                    self.workingdat['Y'][index]))
        LogFile.close()
    
    def printFit(self):
        output = "Fit Report:\n"
        output = output + "\tPar:\tVal\tErr\n"
        
        for name in list(self.parameters.keys()):
            output = output + "\t%s\t%f\t%f\n" %(name,
                                               self.parameters[name],
                                               self.errors[name])
        print(output)
        
    def plot(self,saveAs = ''):
        plt.figure()
        plt.scatter(self.rawdat['X'],self.rawdat['Y'],s=10,color='black')
        
        # If a fit has been done it will plot the model on top
        if self.fitbool:
            XThr = self.workingdat['X']
            YThr = self.model(self.workingdat['X'], *self.parameters.values())
            plt.plot(XThr,YThr)
        
        # If the user as specified a name for the plot, then the plot
        # will be saved.
        if saveAs:
            PlotN = '%s\\%s.png' %(self.directory,saveAs)
            plt.savefig(PlotN)
            plt.close()

# %%Utility Functions           
    def __init__(self,fName, directory,equation, rawdat = {}):
        self.directory  = directory
        self.fileName = fName
        self.model = np.vectorize(
                lambda x,*args: equation(x,*args)*self.scale[1])
        if not bool(rawdat):
            self.__readData()
        else:
            self.rawdat = rawdat
        self.workingdat = self.rawdat.copy()
    
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