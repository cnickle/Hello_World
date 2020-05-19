import pandas as pd
from penquins.newModel import newModel
import penquins.functions as fun
import numpy as np

def test_BTTF():
    # First we need to create a model so we'll use the
    # 'newModel' module to create a new model for us
    # using one of the premade functions in the
    # 'functions' module you can always create your own
    # if you desire.
    
    # The newModel module as multiple methods you can
    # use, by default is uses scipy.optimize.curve_fit
    # function.
    line = newModel(fun.linear)
    
    # Now one thing that you might want to do first is
    # to plot your new model. But first you're going to
    # need to give it some parameters.
    
    # The newModel module takes the parameters in the
    # form of a dictionary with the names being the
    # keys and the values being the parameters.
    params = {
        'm' : 1,
        'b' : 1
        }
    line.setparameters(params)
    line.plot(x = np.linspace(-1,1,100))
    
    params = {
        'm' : -1,
        'b' : 1
        } 
    line.plot(x = np.linspace(-1,1,100),params = params)
    
    line = newModel(fun.linear)
    line.plot(x = np.linspace(-1,1,100),params = params)
    
    # The most important thing is to grab the data via a
    # pandas dataframe. The below function grabs the
    # data from a txt file that in the Data folder and
    # is tab delimited and doesn't have any 'headers'.
    # It then puts this data in a  dataframe called
    # 'dataDF'
    dataDF = pd.read_csv('Data\\BTTF.txt',delimiter = '\t', header = None)
    

if __name__ == '__main__':
    test_BTTF()