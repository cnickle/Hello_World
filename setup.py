# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:58:49 2019

@author: nickl
"""

from setuptools import setup

with open("README",'r') as f:
    long_description = f.read()
    
setup(
      name = 'penguins',
      version = '0.1',
      description = 'A module for fitting data',
      author = 'Cameron Nickle',
      author_email = 'camnickle@gmail.com',
      packages = ['penquins'],
      install_requires = ['random',
                          'numpy',
                          'csv',
                          'matplotlib.pyplot',
                          'scipy.optimize',
                          'math',
                          'scipy.integrate'
                          ])
