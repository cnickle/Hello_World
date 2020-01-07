# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:48:47 2019

@author: nickl
"""
import models
import numpy as np
import time
    
def test_compare_with_mathematica2():
    mathematica = [-8.20749E-8, -3.17403E-8, -5.23501E-9, -5.90173E-10,\
                   2.90532E-10, 6.28546E-10, 1.30025E-9,3.02873E-9]
    vb = [-1, -.75, -.5, -.25,.25, .50, .75, 1]
    for i in range(len(vb)):
        f = models.tunnelmodel_1level_nogate_300K_gauss(vb[i],.0005, 0.02,\
                                                        0.62, 0.7, 0.14)
        assert abs((f-mathematica[i])/f) < 0.05

def equation1(vb, gammaL, gammaR, deltaE1, eta,sigma):
    T = 300
    c = 0
    vg = 0   
    gammaC = gammaL*gammaR
    gammaW = gammaL+gammaR
    
    return models.tunnelmodel_singleLevel(vb,gammaC,gammaW, deltaE1,eta,sigma,c,vg,T)

def test_compareTime_tunnelModeltoAlt():
    x  = np.arange(-1,1,.01)
    args = (.0005, 0.02, 0.62, 0.7, 0.14)
    
    start = time.time()
    f1 = np.vectorize(equation1)
    f1(x,*args)
    r1 = time.time()-start
    print(r1)
    
    start = time.time()
    f2 = np.vectorize(models.tunnelmodel_1level_nogate_300K_gauss)
    f2(x,*args)
    r2 = time.time()-start
    print(r2)
    
    assert r1-r2 < 1

def test_compareValues_tunnelModeltoAlt():
    x  = np.arange(-1,1,.01)
    args = (.0005, 0.02, 0.62, 0.7, 0.14)
    
    f1 = np.vectorize(equation1)
    y1 = f1(x,*args)
    
    f2 = np.vectorize(models.tunnelmodel_1level_nogate_300K_gauss)
    y2 = f2(x,*args)
    
    SE = np.sum((np.subtract(y1,y2))**2)
    assert SE < .25

def test_compare_with_mathematica():
    f = models.tunnelmodel_1level_nogate_300K(4,0.5,0.5,1.1,0.5)
    assert abs((f-0.0000476775)/f) <0.05