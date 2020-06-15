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
from numpy.random import random
import math
import time

class Model:
    _counter = 0
    # def returnThry(self, initpar=[]):
    #     if self.modelDat:
    #         return self.modelDat['X'], self.modelDat['Y']
    #     else:
    #         return self.workingdat['X'], self.model(self.workingdat['X'], *initpar.values())    
    def setparameters(self, params):
        self.parameters = params.copy()
    
    def returnThry(self, x, params = {}):
        temp = {}
        if params:
            temp = params.copy()
        elif self.parameters:
            temp = self.parameters.copy()
        else:
            print('No parameters given')
            return 0
        
        y = self.function(x,*temp.values())
        return y
    
    def plot(self, x, params = {},fignm = '', color = 'red', label = 'Model'):
        temp = {}
        if not fignm:
            fignm = self.ID
        if params:
            temp = params.copy()
        elif self.parameters:
            temp = self.parameters.copy()
        else:
            print('No parameters given')
            return 0
        
        y = self.function(x,*temp.values())
        plt.figure(fignm)
        plt.plot(x,y,color=color,label=label)
    
    def __init__(self, function, method = 'curve_fit'):
        self.ID = Model._counter
        Model._counter +=1
        if method == 'curve_fit':
            self.function = np.vectorize(function)
            self.Vectorized = True
        else:
            self.function = function
            self.Vectorized = False
        self.parameters = {}
        self.data = []
    #     self.fitbool = False
    #     self.scaling = False
    #     self.multiplier = 1
    #     self.rawdat = {}
    #     self.parameters = {}
    #     self.errors = {}
    #     self.modelDat = {}
    #     self.fileName = fName
        
    #     if not bool(rawdat):
    #         self.__readData()
    #     else:
    #         self.rawdat = rawdat
        
    #     minval = min(abs(self.rawdat['Y']))
    #     if minval <np.sqrt(np.finfo(float).eps):
    #         # Due to curve_fit and machine precision limitations the data and
    #         # model are being scaled into the nano range. This should not 
    #         # effect the fitting parameters
    #         # warnings.warn("Scaling data and equation due to floating point percision")
    #         self.scaling = True
    #         self.multiplier = 1E9
    #         self.rawdat['Y'] = self.rawdat['Y']*self.multiplier
        
    #     self.workingdat = self.rawdat.copy()
    #     self.model = np.vectorize(lambda x,*args: equation(x,*args)*\
    #                               self.multiplier)    
        
    # def __fit(self,model,parBnds,parInitial):        
    #     p0=[]
    #     bnds = []
    #     lower = []
    #     upper = []
    #     for name in list(parInitial.keys()):
    #         p0 += [parInitial[name]]
    #         if parBnds:
    #             lower +=[parBnds[name][0]]
    #             upper +=[parBnds[name][1]]            
       
    #     bnds = [lower,upper]

    #     X  = self.workingdat['X']
    #     Y  = self.workingdat['Y']
        
    #     if parBnds:
    #         results,covar=sco.curve_fit(model,X,Y,p0=p0,bounds=bnds)
    #     else:
    #         results,covar=sco.curve_fit(model,X,Y,p0=p0)
    #     return results,covar
    
    # def fit(self,parBnds, parInitial):
    #     self.fitbool = True
        
    #     if parBnds:
    #         for parm in parBnds.keys():
    #             if not (parInitial[parm] < parBnds[parm][1] and parInitial[parm] >parBnds[parm][0]):
    #                 print('%s is infeasible. Choosing random number'%parm)
    #                 parInitial[parm] = parBnds[parm][0] + (random()*(parBnds[parm][1]-parBnds[parm][0]))
        
    #     results,covar  = self.__fit(self.model,parBnds,parInitial)
        
    #     self.modelDat['X']=self.workingdat['X']
    #     self.modelDat['Y']=self.model(self.workingdat['X'], *results)
        
    #     cnt = 0
    #     for name in list(parInitial.keys()):
    #         self.parameters[name] = results[cnt]
    #         self.errors[name]  = np.sqrt(np.diag(covar))[cnt]
    #         cnt +=1
            
    # def customFit(self,func,parInitial):
    #     self.fitbool = True
        
    #     def minfunc(x):
    #         yThr = self.model(self.workingdat['X'],*x)
    #         return func(self.workingdat['Y'],yThr)
        
    #     res = sco.minimize(minfunc,x0 = list(parInitial.values()))
        
    #     self.modelDat['X']=self.workingdat['X']
    #     self.modelDat['Y']=self.model(self.workingdat['X'], *res.x)
        
    #     cnt = 0
    #     for name in list(parInitial.keys()):
    #         self.parameters[name] = res.x[cnt]
    #         self.errors[name]  = 0
    #         cnt +=1
    
    # def calcLinearError(self, initpar):
    #     X = self.workingdat['X']
    #     Y = self.workingdat['Y']

    #     if not self.modelDat:
    #         self.modelDat['X'] = self.workingdat['X']
    #         self.modelDat['Y'] = self.model(X,*initpar.values())
            
    #     residual = np.subtract(Y,self.modelDat['Y'])
    #     Error = np.sqrt(np.sum(residual**2))
    #     return Error/len(X)
    
    # def calcRelativeError(self, initpar):
    #     X = self.workingdat['X']
    #     Y = self.workingdat['Y']

    #     if self.modelDat:
    #         Ythr = self.modelDat['Y']
    #     else:
    #         Ythr = self.model(X,*initpar.values())
            
    #     residual = np.subtract(np.log(np.abs(Y)),np.log(np.abs(Ythr)))
    #     Error = np.sqrt(np.sum(residual**2))
    #     return Error
    
    # def printFit(self,save = '', initpar = {}, relError = False):
    #     if not self.parameters:
    #         self.parameters = initpar.copy()
    #         self.errors = initpar.copy()
        
    #     if not relError:
    #         Err = self.calcLinearError(self.parameters)
    #     else:
    #         Err = self.calcRelativeError(self.parameters)
    #     output = "Fit Report:\tError:\t%.2f\n" % Err
    #     output = output + "\tPar:\tVal\tErr\n"
        
    #     for name in list(self.parameters.keys()):
    #         if initpar:
    #             self.errors[name] = np.inf
    #         output = output + "\t%s\t%e\t%e\n" %(name, self.parameters[name], self.errors[name])
    #     print(output)
    #     if save:
    #         output = ''
    #         for name in self.parameters.keys():
    #             output = output + '%e\t'%self.parameters[name]
    #         output = output + '%.5f\n'%Err
    #         f= open(save,"a")
    #         f.write(output)
    #         f.close()
               
    # def plot(self,pars=[],save = '',scale = ''):
    #     if self.scaling:
    #         scale = 'nano'
    #     mult = self.__rangeOrder(scale)        
        
    #     plt.figure()
    #     plt.scatter(self.rawdat['X'],self.rawdat['Y']*mult,s=10,color='black')
    #     plt.autoscale(False)
        
    #     # If a fit has been done it will plot the model on top
    #     if self.fitbool:
    #         XThr = self.workingdat['X']
    #         YThr = self.model(self.workingdat['X'], *self.parameters.values())*mult
    #         plt.plot(XThr,YThr)
        
    #     if pars:
    #     #If pars, plot will plot the data with pars given
    #         XThr = self.workingdat['X']
    #         YThr = self.model(self.workingdat['X'],*pars.values())*mult
    #         plt.plot(XThr,YThr)
        
    #     if save:
    #     #If the user as specified a name for the plot, then the plotwill be saved.   
    #         plt.savefig(save)