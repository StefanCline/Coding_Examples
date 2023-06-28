# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:30:09 2023

@author: Sclin
"""

#testing out the interp nonsense!

import numpy as np
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

x = [-4,-3,-2,-1,0,1,2,3,4]
y = [0,0,-2,-1,0,1,2,0,0]

f = interp1d(x,y,kind='linear',fill_value="extrapolate")

X = np.linspace(-10,10,101)
plt.plot(X,f(X))
plt.grid()