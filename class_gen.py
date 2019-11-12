# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:21:27 2019

This file is used for Genetic Algorithms

@author: Cam
"""
import random
import numpy as np
from numpy.random import choice
class gen:
    genID=0
    size = 20
    ranges = {}
    population = {}
    parameters = {}
    
    def __init__(self,ranges,size):
        self.ranges = ranges
        self.size = size
        for name in list(self.ranges.keys()):
            self.parameters[name] = np.zeros(size)
    
    def __normalize_weights(self,weights):
        if not weights:
            return weights
        else:
            total = np.sum(weights)
            weights = weights/total
            return weights
    
    def nextgen(self, weights=[]):
        print('Gen %d' %self.genID)
        self.genID +=1
        
        if not bool(self.population):
            for index in range(self.size):
                pop = {}
                for name in list(self.ranges.keys()):
                    pop[name] = random.uniform(
                            self.ranges[name][0],
                            self.ranges[name][1])
                self.population[index] = pop
            return self.population
        else:
            for index in list(self.population.keys()):
                for name in list(self.ranges.keys()):
                    self.parameters[name][index] = self.population[index][name]
        
            for index in range(self.size):
                for name in list(self.ranges.keys()):
                    self.population[index][name] = choice(
                            self.parameters[name],
                            self.__normalize_weights(weights))             