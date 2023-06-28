# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 13:10:31 2023

@author: Sclin
"""

def load_params(sig_name,axis): #axis is only for VDP
    # basic run params
    if sig_name == 'harmonic':
        t0,tf,sr = 0,3,101
        sval,mu,alpha = 0,0,0
        N = int(5e6) # number of samples, recommend e6
        M = 28000 # height multipler 
        sig = 70
    elif sig_name == 'harmonic_full':
        t0,tf,sr = 0,6,201 #tf was 3 and sr was 101
        sval,mu,alpha = 0,0,0
        N = int(5e6) # number of samples, recommend e6
        M = 1938.738 # height multipler 
        sig = 1.289
    elif sig_name == 'sto_linear':
        t0,tf,sr = 0,6,0
        sval,mu,alpha = 0.05, 0, 20.0
        N = int(5e4) # number of samples, recommend e6
        M = 185674.527 #8511.066 # height multipler 
        sig = 134.679 #188.635
    elif sig_name == 'VDP':
        t0,tf,sr = 0,6,0
        sval,mu,alpha = 2.0, 1.0, 20.0
        N = int(5e5) # number of samples
        #THE BELOW BLOCK IS FOR HIGH_SIG 
        # if axis == 'x1':
        #     # for x1 of the two choices, x1 and x2
        #     M = 353.005
        #     sig = 0.879
        # elif axis == 'x2':
        #     # for x2 of the two choices, x1 and x2
        #     M = 48.881 # height multipler 
        #     sig = 1.043
        # THE BELOW BLOCK IS FOR LOW_SIG
        if axis == 'x1':
            # for x1 of the two choices, x1 and x2
            M = 47.047
            sig = 0.853
            N = int(1e5)
        elif axis == 'x2':
            # for x2 of the two choices, x1 and x2
            M = 44.05 # height multipler 
            sig = 0.849
            N = int(1e5)
    elif sig_name == 'fsig2':
        t0,tf,sr= 0,1,0 
        sval,mu,alpha = 0,0,0
        N = int(1e5) # number of samples
        M = 12127.784 # height multipler 
        sig = .806
        load_file = 1 # 0 means you're not loading in the file and dong the whole thing from scratch again
    elif sig_name == 'ECG':
        t0,tf,sr= 0,1,0 
        sval,mu,alpha = 0,0,0
        N = int(1e5) # number of samples
        M = 607.789 # height multipler 
        sig = .872
    return t0,tf,sr,sval,mu,alpha,N,M,sig
    

def load_params_DHG():
    N = int(5e5) # must be even
    left_gauss = -7.5
    right_gauss = 7.5
    mu1,mu2,s1,s2 = -2.0, 2.0, 1.0, 1.0
    return N, left_gauss, right_gauss, mu1, mu2, s1, s2


def maximum_plot_y(sig_name):
    #y_max per cycle
    if sig_name == 'harmonic':
        y_max = 0.035
    if sig_name == 'harmonic_full':
        y_max = 0.7
    elif sig_name == 'sto_linear':
        y_max = 0.035
    elif sig_name == 'VDP':
        y_max = 1.0
    elif sig_name == 'DHG':
        y_max = 0.21
    elif sig_name == 'fsig2':
        y_max = .7
    elif sig_name == 'ECG':
        y_max = 1.0
    return y_max