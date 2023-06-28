# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:49:23 2023

@author: Sclin
"""

# Performing Otsu on a Gaussian

import numpy as np
import matplotlib.pyplot as plt

def custom_otsu(nparray,dplace):

    # nparray = dataset to be analyzed
    # dplace = how many decimal places will each bin size be? 
    # dplace must be a positive integer 1:0.1, 2:0.01, 3:0.001, etc. 
    # From Otsu's original paper on foreground and background pixel classification
    # The data to be processed
    # the data is placed neatly into bins for easier processing by the algorithm
    place = dplace
    P = np.round(nparray,place)
    
    # making the bin sizes sufficiently small
    bw = 1.0/(10**place) # bin width 
    k = np.arange(np.min(P),np.max(P),bw)
    k = np.round(k,place)
    
    # Plotting the histogram for the original data
    plt.subplot(1,2,1)
    plt.hist(P,bins=(len(k)+1))
    plt.title('Otsu: Input Data Histogram')
    plt.xlabel('Data Values')
    plt.ylabel('Counts')
    
    inter_CV = np.zeros((len(k)-1))
    mut = 0
    for kk in range(len(k)):
        mutp = np.count_nonzero(P == k[kk])/(P.size)
        mut = mut + k[kk]*mutp
    
    # Finding the threshold
    for kk in range((len(k)-1)): #stop at k-1 because omega1 = 0 if k=L
        # O0,O1,mu0,m1 and mu(k)
        omega0 = 0
        for jj in range(kk+1):
            omega0p = np.count_nonzero((P == k[jj]))/(P.size)
            omega0 = omega0+omega0p
        omega1 = 1-omega0
        
        muk = 0
        for jj in range(kk+1):
            mukp = np.count_nonzero(P == k[jj])/(P.size)
            muk = muk + k[jj]*mukp
        inter_CV[kk] = (mut*omega0-muk)**2.0/(omega0*omega1)
    
    x = np.delete(k,-1)
    #x = np.delete(k,0)
    plt.subplot(1,2,2)
    plt.plot(x,inter_CV)
    plt.title('Otsu: k* Curve')
    plt.ylabel('Inter Class Variance')
    max_index_val = (np.where(inter_CV == np.max(inter_CV)))[0]
    kstar = k[max_index_val]
    plt.axvline(x = kstar)
    kstar_plot_val = kstar.item()
    pltstr = 'kstar = '+str(kstar_plot_val)
    plt.xlabel(pltstr)
    plt.show()

    print('')
    print('')
    print('Done with Otsu Thresholding')
    print('')
    print('')
    return kstar

# use if you want to test the function in house quickly

# P = np.random.normal(0,1,1000)
# place = 1
# kstar = custom_otsu(P,place)










############ DUMP ##########

# # Finding interclass variance 
# CV0sq = 0
# CV1sq = 0
# # for jj in range((kk+1)):
# #     p = np.where(P<= k[jj])
# #     p = len(p)*1.0
# #     CV0sq = (k[jj]-mean0)**2.0*p/omega0 + CV0sq
# # for jj in range((kk+2),(len(k)-1)):
# #     p = np.where(P >= k[jj])
# #     p = len(p)*1.0
# #     CV1sq = (k[jj]-mean1)**2.0*p/omega1 + CV1sq

# mu = 0
# sig = .7
# P = np.zeros(N)
# Creating gaussian histogram
# P = np.random.normal(-2.5,sig,N)
# P1 = np.random.normal(2.5,sig,N)
# P = np.concatenate((P,P1))
# P = np.random.normal(mu,1,N)

# Rounding to make even histable bins  
#place = 2 # here must be +int, i.e. 1: .1, 2: .01, 3: .001 etc.

# k_ind = np.arange(1,(len(k)+1))

    # eps = bw/10

    # print('')
    # print('') 
    # print('Threshold k, k*: ',kstar)

