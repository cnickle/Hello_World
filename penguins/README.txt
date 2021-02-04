# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:09:34 2019

@author: nickl
"""

Penquins is meant to be a package designed to assist in using scipy's data fitting functions. Earlier versions attempted to be a one stop shop of fitting needs. Essentially taking the place of pandas as a dataframe package in addition to being a fitting package.

Updates in Version 2.0:

Two projects highlighted the limitations of Version 1.0. It became clear that a couple of things were needed.

A) More flexibility in data selection. In Version 1.0. The data input was all taken care of within the package itself. Since this was done mostly behind the scenes in pre-processing of the raw data was made more difficult. Therefore, that has been removed and a standard pandas method will be used hence forth. No sense in creating the wheel.'

B) More flexibility is needed with regards to plotting options. I won't to be able to edit the plots

C) I want the fitting function to have multiple options. I want to be able to fit, with scipy's curve_fit, minimize, or differiential_evolution algorithms at the choice of the user. These have different requirements so that will have to be addressed.

D) I want the fitting function to be able to have multiple modes, a 'quiet' mode in which very little is printed to the console, a 'normal' mode in which the fitting parameters and the error are printed to console, and finally a 'verbose' mode in which not only are all the parameters printed to console, but a folders are created that contain a file that shows all of the parameters with the error, and plots showing the model.

E) I want a an easy way to 'fix' parameters!