# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:30:59 2023

@author: Sclin
"""

# Turning a PDF into a randomly sampled dataset
# Note, this function assumes denseness and therefore only
#   linearly interpolates
# Here we utilize: adaptive rejection sampling
# https://www.youtube.com/watch?v=OXDqjdVVePY&t=311s&ab_channel=ritvikmath

def pdf_to_samples(x,fx,density,sample_total):

    from scipy.interpolate import PchipInterpolator
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    # Needed for sampling random points
    def gauss_curve_M(point,M,sig):
      output = M*(1/(sig*math.sqrt(2*math.pi)))*np.exp(-point**2/(2*sig**2))
      return output
    
    # this is usually sufficiently dense
    # leaving the option open for it to be adjusted if needed
    if density == 1:
        density = len(x)
    # First, turning the data of a curve, into a continuously callable function
    pdf_curve = PchipInterpolator(x,fx)
    x_new = np.linspace(x[0], x[-1], density)
    plt.plot(x_new, pdf_curve(x_new))
    plt.plot(x, fx, 'o', label='data')
    plt.title('Turning f(x) into Continuous Curve')
    plt.grid()
    plt.show()
    
    # Making sure that PDF < Gaussian Utilizer
    # here, selecting the mu based off of the max point of the function
    mu=0
    peak_old = 0
    for ii in range(len(x)):
        peak_new = pdf_curve(x[ii])
        if peak_new > peak_old:
            mu = x[ii]
            peak_old = peak_new
 
    M = 2.3
    sig = 1.2
    ii = 0
    while ii < 1:
        test_gauss = M*(1/(sig*math.sqrt(2*math.pi)))*np.exp(-(x-mu)**2/(2*sig**2))
        abs_greater_test = test_gauss-pdf_curve(x)
        min_test = np.min(abs_greater_test)
        if min_test <= 0:
            sig = sig + 0.1
            M = sig**2
        else: 
            print('Success. Gaussian Curve is strictly greater than FFT data curve.')
            ii = 2
    # plot the resulting successful distribution curve
    plt.plot(x_new, pdf_curve(x_new))
    plt.plot(x,test_gauss)
    plt.title('Sampling Gaussian above Skew PDF')
    plt.grid()
    plt.show()
    
    # Generating the Points
    
    # Creating a sampling now for the above pdf
    ps = []
    accept = 0
    reject = 0
    L = x[0]
    R = x[-1]
    while accept < sample_total:
        rand_g = random.gauss(mu,sig)
        if L <= rand_g and rand_g <= R:
            prob = pdf_curve(rand_g)/(gauss_curve_M(rand_g,M,sig))
            test = random.uniform(0,1)
            if test <= prob:
              ps.append(rand_g)
              accept = accept + 1
            else:
              reject = reject + 1
        else:
            reject = reject + 1

    print('accepted total: ', accept)
    print('rejected total: ', reject)
    
    # Show the Output
    
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.subplot(1,2,1)
    plt.hist(ps,bins=int(np.round(sample_total/100)))
    plt.xlim([x[0],x[-1]])
    plt.subplot(1,2,2)
    plt.plot(x,fx)
    plt.show()
    
    return np.asarray(ps)