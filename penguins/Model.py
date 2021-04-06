# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:21:27 2019

This file will simply be used to provide functions for manipulation of the data
files given to us via our Singapore Collaboration.

@author: Cam
"""
import numpy as np
import scipy.optimize as sco
import scipy.stats as scat
import os
import time

class Model:
    #%% Initialization
    def __init__(self, function):
        """
        This initializes the model. It saves the model function in two ways
        As self.function and self.originalFunction. This is because
        self.function will eventually be altered when parameters are fixed

        Parameters
        ----------
        function : function
            This is the function that will be used to fit the data.

        Returns
        -------
        None.

        """
        self.originalFunction = function
        self.function = function
        self.time = 0
        self.alg  = 'Manual'
        self.method  = 'None'
    
    #%% Setting Parameters
    def setParams(self, parameters, bnds = None, calculatedParams={}, Fixed = None):
        """
        Here we set the initial parameters and fixed parameters. This function
        creates:
            self.parameters (dictionary of all parameters)
            self.initialParameters (dictionary of parameters to be used in
                                    fitting)
            self.fixed (list of strings of parameters that should remain fixed)
        
        It changes:
            self.function (setParams changes the self.function so that only
                           parameters that are not fixed are taken as arguments)

        Parameters
        ----------
        parameters : dict
            Dictionary of parameters used in the model
        fixed : TYPE, optional
            List of dictionary keys of parameters that will be fixed
            during fitting. The default is None.

        Returns
        -------
        None
        """
        if Fixed == None:
            fixed = []
        else:
            if type(Fixed) == dict:
                fixed = Fixed.keys()
            if type(Fixed) == list:
                fixed = Fixed
        
                #Check to make sure that the initial parameters lie inside the bounds
        if not bnds == None:
            for par in bnds.keys():
                if par in fixed: continue
                if parameters[par]<bnds[par][0] or parameters[par]>bnds[par][1]:
                    print('The initial value for %s is outside the bounds given. A number between the bounds is being choosen.'%par)
                    parameters[par] = (bnds[par][0]+bnds[par][1])/2
        self.calculatedParams = calculatedParams
        self.bounds = bnds
        self.initialParameters = parameters.copy()
        self.parameters = parameters.copy()
        
        if not fixed == None:
            self.fixed = fixed
            
            for par in fixed:
                del self.initialParameters[par]
                if type(Fixed) == dict:
                    self.parameters[par] = Fixed[par]
                
                if self.bounds == None: continue
                if par in self.bounds.keys():
                    del self.bounds[par]
            
            def reducedfunc(Xvals,*args):
                for i,par in enumerate(self.initialParameters.keys()):
                    self.initialParameters[par] = args[i]
                    self.parameters[par] = args[i]
                
                return self.originalFunction(Xvals,* self.parameters.values())            
            self.function = reducedfunc
            

    #%% Calculate Function
    def returnThry(self,Xvals, pars=[]):
        """
        Uses the model to cacluate the corresponding Y values for the given
        Xvals and parameters.

        Parameters
        ----------
        Xvals : Array
            The xvals that the Model uses
        pars : Array, optional
            An array of parameters used in the fitting. The default is None.

        Returns
        -------
        Yvals : Array
            The correpsonding Yvals calculated using the model
        """
        if len(pars) == 0:
            pars =self.initialParameters.values()
        try:
            Yvals = self.function(Xvals,*pars)
        except:
            func = np.vectorize(self.function)
            Yvals = func(Xvals,*pars)
        return np.array(Yvals)
    
    #%% Calculate Residual
    def residual(self, Xvals, Yvals, pars=[], scale = 'lin', fit=False,
                 mode = 'quiet', save = ''):
        """
        This calculates the residutal between the theory and experiment
        
        Parameters
        ----------
        Xvals : Array
            X values used in model.
        Yvals : Array
            Y values from experiment.
        pars : Array, optional
            Parameters to be used in the model. The default is None.
        scale : String, optional
            This can either be 'lin' for calculate the linear difference
            between the two. Or 'log' for calculating the log difference.
            The default is 'lin'.

        Raises
        ------
        ValueError
            If the scale is something other than 'lin' or 'log'.

        Returns
        -------
        Array
            Residual between Yvals (experiment) and Ythr (model).
        """
        Xvals = np.array(Xvals, dtype=np.float64)
        Yvals = np.array(Yvals, dtype=np.float64)
        
        X = Xvals
        if scale == 'lin':
            Y = Yvals
            Ythr = self.returnThry(Xvals, pars)
        elif scale == 'log':
            Y = np.log10(np.abs(Yvals))
            Ythr = np.log10(np.abs(self.returnThry(Xvals, pars)))
        else:
            raise ValueError('Not an appropirate scale')
        
        Err = np.log10(np.sqrt(np.sum(np.subtract(Y,Ythr)**2 )))
        if save:
            self.saveParams(pars, Err, save)    
        if mode == 'verbose':
            if self.counter % 50 == 0:
                totTime = time.gmtime(time.time()-self.time)
                print('Calls: %d\tErr: %.2f\tTime: %s'%(self.counter,Err,time.strftime("%H:%M:%S",totTime)))
            self.counter += 1
        
        minval = np.median(abs(Y))
        if fit and minval < np.sqrt(np.finfo(float).eps):
            Y = Y*1E9
            try:
                Ythr = Ythr*1E9
            except:
                Ythr = Ythr*1E9
        return np.subtract(Y,Ythr)
    
    #%% Calculate StandardError
    def standardError(self, Xvals, Yvals, pars=[], scale = 'lin', fit=False,
                      mode = 'quiet', save = ''):
        """
        Returns the standard error between the model and Yvals. 

        Parameters
        ----------
        Xvals : Array
            All x values used in model.
        Yvals : Array
            Observed y values.
        pars : Array, optional
            Parameters used in the model. The default is None.
        scale : TYPE, optional
            Whether the model should be calculated on linear or log scale.
            The default is 'lin'.

        Returns
        -------
        float
            Standard Error sqrt(sum(exp^2-thry^2)).
        """
        res = self.residual(Xvals, Yvals, pars, scale, fit, mode = mode,
                            save = save)
        return np.log10(np.sqrt(np.sum(res**2)))
    
    #%% Calculating chi2
    def chi2(self, Xvals, Yvals, scale = 'lin'):
        """
        Simply uses scipy.stats.chisquare() to calculate the chi^2 value.

        Parameters
        ----------
        Xvals : array of floats
            Experimental X values.
        Yvals : array of floats
            Experimental Y values.
        scale : string, optional
            lin or log. The default is 'lin'.

        """
        
        if scale == 'lin':
            Ythr = self.returnThry(Xvals)
            return scat.chisquare(Yvals,Ythr, len(self.parameters))[0]
        elif scale == 'log':
            Yvals = np.log10(np.abs(Yvals))
            Ythr = np.log10(np.abs(self.returnThry(Xvals)))
            return scat.chisquare(Yvals,Ythr, len(self.parameters))[0]
        else:
            raise ValueError('Not a recognized scale')
            
    #%% The Fitting Function
    def fit(self, Xvals, Yvals, scale = 'lin', algorithm = 'LS',
            method = None, mode = 'quiet', save = ''):
        """
        So this is where all the magic happens. Calling this function starts
        the fitting. This function was made to be versatile. One can use three
        different fitting methods provided by scipy.optimize (sco). These can
        be choosen by passing one of the values below to algorithm. Each of 
        these methods have different requirements are on the format of the 
        arguments. Therefore this function does a lot to make sure everything
        is formatted correctly so that the user only needs to specify the
        method.
        
        1) 'LS'
            - Fits the data using sco.least_squares(). This is the simplest 
            and fastest method to fit. There are three methods it uses. Check
            documentation for details
        2) 'min'
            - Fits using sco.minimize(). This takes a bit longer, but uses a 
            non-linear algorithm that allows for constraints. Check documentation
            for details
        3) 'diff'
            - Fits using sco.differential_evolution(). This is takes a long time
            however in some cases it can be very beneficial. The algorithm 
            chooses random values for the parameters and determines which of 
            these random choices is best. It over time 'evolves' the random
            choices and eventualy comes to an optimal soluation. Pass 'diff' 
            when 'LS' and 'min' aren't quite working for you and you really
            don't know what good initial parameters should be.

        Parameters
        ----------
        Xvals : array of floats
            Experimental X values
        Yvals : array of floats
            Experimental Y values
        scale : string, optional
            Can take on two values 'lin' and 'log'.
            Often we are asked to fit data so that it looks good on a log
            scale. It can be tedius switching back and forth so this option
            does it for you. The default is 'lin'.
        algorithm : string, optional
            Three allowed values, 'LS', 'min', 'diff'. The default is 'LS'.
        method : string, optional
            This is passed to the fitting algorithm and specifies the method
            that the algorithm uses. The default is None.
        mode : string, optional
            'Quiet' or 'Verbose' This determines how the fitting function
            will behave. Will it periodically update the user on how the
            fitting will go? Or will it be totally silent. The default is 'quiet'.
        save : string, optional
            If you choose 'Verbose' then you can also specify a location to
            save all of the results from the fit. The default is ''.

        """        
        self.time = time.time()
        self.alg = algorithm
        self.method  = method
        #%% How will the fit progress? Give updates on the fitting process
        # or be silent? If verbose this is what you get:
        if mode == 'verbose':
            Output = "Fitting with Verbose means that the Error will be"
            Output = Output + " printed to console every 50 calls of the"
            Output = Output + " fitting function, and the data will be saved"
            Output = Output + " to a text file."
            self.counter = 0
            if not save:
                Output = Output + " Since no save location was given. The data"
                Output = Output + " will be saved to  Results//FitResults.txt"
                save = 'FitResults.txt'
            print(Output)
        
        # The three different algorithms each prefer the bounds in a different
        # fromat. Here is where that formating is done.
        bounds = []
        if algorithm == 'LS':
            minfunc = lambda args: self.residual(Xvals, Yvals, args, scale,
                                                 fit = True, mode = mode,
                                                 save = save)
            if not self.bounds:
                bounds=(-np.inf,np.inf)
            else:
                lower = []
                upper = []
                for key in self.bounds.keys():
                    lower += [self.bounds[key][0]]
                    upper += [self.bounds[key][1]]
                bounds = [lower,upper]
        elif algorithm in ['min','diff','basin']:
            minfunc = lambda args: self.standardError(Xvals, Yvals, args, scale,
                                                      fit = True, mode = mode,
                                                      save = save)
            if not self.bounds:
                bounds = None
            else:
                bounds = list(self.bounds.values())
        else:
            raise ValueError('Not a recognized Algorithm')
        
        # Each of these has a default method, if no method is given then the
        # default is used. Otherwise the specified method is used
        pars = list(self.initialParameters.values())
        if not method:
            if algorithm == 'LS':
                result = sco.least_squares(minfunc,x0=pars, bounds = bounds)
            elif algorithm == 'min':
                result = sco.minimize(minfunc,x0=pars, bounds = bounds)
            elif algorithm == 'diff':
                result = sco.differential_evolution(minfunc,bounds = bounds)
            elif algorithm == 'basin':
                result = sco.basinhopping(minfunc,x0=pars)
        if method:
            if algorithm == 'LS':
                result = sco.least_squares(minfunc,x0=pars, bounds=bounds, method = method)
            elif algorithm == 'min':
                result = sco.minimize(minfunc,x0=pars,bounds = bounds, method = method)
            elif algorithm == 'diff':
                result = sco.differential_evolution(minfunc,bounds = bounds)
            elif algorithm == 'basin':
                result = sco.basinhopping(minfunc,x0=pars)
        
        for i,par in enumerate(self.initialParameters.keys()):
            self.initialParameters[par] = result.x[i]
            self.parameters[par] = result.x[i]

    #%% Print fit Results
    def print(self, Xvals, Yvals, save = '', scale = 'lin'):
        """
        Prints the values of the parameters stored in self.parameters and 
        calculates the chi^2 using scipy.stats.chisquare() and the standard 
        error using self.standardError(). In this outputs all of that data
        and gives the option to save the results to a file.

        Parameters
        ----------
        Xvals : TYPE
            Experimental X values.
        Yvals : TYPE
            Experimental Y values.
        save : TYPE, optional
            Save name of file. The default is ''.
        scale : TYPE, optional
            Error calculated on log or lin scale. The default is 'lin'.
        """
        Err  = self.standardError(Xvals, Yvals, scale=scale)
        
        chi2 = self.chi2(Xvals, Yvals, scale = scale)
        
        output = "\n\033[4mFIT REPORT" + ' '*32+'\033[0m\n'
        if self.time:
            totTime = time.gmtime(time.time()-self.time)
            output = output + "Total Fit Time: %s\n"%(time.strftime("%H:%M:%S",totTime))
        output = output + "Fitting Method: %s %s\n\n" %(self.alg, self.method)
        output = output + "\033[4mParameter\033[0m:\t\t\033[4mValue\033[0m:\n"
        for name in list(self.parameters.keys()):
            if name in self.fixed:
                output = output + "\t%s*\t\t\t%.2e\n" %(name, self.parameters[name])
            else:
                output = output + "\t%s\t\t\t%.2e\n" %(name, self.parameters[name])
        output = output + '*Fixed\n\n'
        output = output + '\033[4mError\033[0m:\n'
        output = output + '\tStandard:\t\t%.2f\n' %Err
        output = output + '\tchi sq  :\t\t%.2e\n\n' % chi2
        
        if not len(self.calculatedParams.keys()) == 0:
            output = output + "\033[4mCalculated Param\033[0m:\t\t\033[4mValue\033[0m:\n"
            for name in list(self.calculatedParams.keys()):
                val = self.calculatedParams[name](self.parameters)
                output = output + "\t%s\t\t\t\t%.2e\n" %(name, val)
        if save:
            output = output + 'Save Location: \'Results\\\%s\'\n'%save
        output = output + '\033[4m_'*42+'\033[0m\n'
        print(output)
        if save:
            self.saveParams(list(self.initialParameters.values()), Err, save)
    
    def saveParams(self, params, Err, loc):
        """
        This takes an array of values, an error and a location, and saves the
        results of the fit.

        Parameters
        ----------
        params : array of floats
            This is an array of values that should be the same length of the
            parameters.
        Err : float
            Error in calculation.
        loc : TYPE
            Save location.
        """
        Temp = self.parameters.copy()
        for i,name in enumerate(self.initialParameters.keys()):
            Temp[name] = params[i]
        
        filepath = 'Results\\%s'%loc
        output = ''
        isDir = os.path.isdir('Results')
        if not isDir:
            os.mkdir('Results')
        
        isFile = os.path.isfile(filepath)
        if not isFile:
            for name in Temp.keys():
                output = output + name +'\t'
            output = output +'error'+ '\n'
        for val in Temp.values():
            output = output + '%e\t'%val
        output = output + '%.5f\n'%Err
        f= open(filepath,"a")
        f.write(output)
        f.close()
