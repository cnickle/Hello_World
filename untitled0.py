# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:32:06 2019

@author: nickl
"""

import numpy as np
import matplotlib.pyplot as plt 

# generate some data
x = np.arange(0, 10, 0.2)
y = np.sin(x)

# plot it
f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
a0.plot(x, y)
a1.plot(y, x)