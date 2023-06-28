# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:57:50 2023

@author: Sclin
"""

# this file will make the VDP Oscillator x1 and x2 datasets that will then be imported into 
#   the main executed Norm Flows 1D file
#   Both x1 and x2 will be run independently but will need to be from the same run, and not disjoint


# Package and File Imports
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scipy
import sys
from scipy.integrate import quad
from scipy.fft import fft, ifft, fftshift
from signals import *

# Parameters:
sig_name = 'VDP'
folder = file_save_name(sig_name)
t0,tf,sr = 0,20,0              #t0,tf,sr = 0,6,0 original settings (used for more crazy run)
sval,mu,alpha = 0.05, 1.1, 1.0 #sval,mu,alpha = 2.0, 1.0, 20.0 original settings (used for more crazy run)

# THE BELOW LINE IS THE LINE THAT WE'LL BE REPLACING
#    omega, S_original, time_axis = signal_type(sig_name,t0,tf,sr,sval,mu,alpha,view_other)
Pi = np.pi
omd = 2.*Pi
Nstps = int(2**9) #curtis had this at 2^11
tvals = np.linspace(0,tf,Nstps+1)
vdposc = lambda x: vanderpol(x, mu)
nltseries = random_integrator(omd,tf,Nstps,sval,alpha,vdposc)
fnltseries_x1 = fft(nltseries[0,:]) # this will grab specifically the 'x1' variable
fnltseries_x2 = fft(nltseries[1,:]) # this will grab specifically the 'x2' variable
nlredseries_x1 = fftshift(np.abs(fnltseries_x1))/np.sqrt(Nstps)
nlredseries_x2 = fftshift(np.abs(fnltseries_x2))/np.sqrt(Nstps)

# plots
axis_val_x1 = '1'
axis_val_x2 = '2'
stochastic_VDP_plot(nltseries)
stochastic_VDP_plot_time(tvals,nltseries[0,:],axis_val_x1)
stochastic_VDP_plot_time(tvals,nltseries[1,:],axis_val_x2)
stochastic_VDP_plot_FFT(nlredseries_x1,axis_val_x1)
stochastic_VDP_plot_FFT(nlredseries_x2,axis_val_x2)

if len(nlredseries_x1) % 2 == 0:
    nlredseries_x1 = nlredseries_x1[:-1]
    nlredseries_x2 = nlredseries_x2[:-1]
    tvals = tvals[:-1]
    #nlredseries = nlredseries[:-1]
omega = np.arange(-int(np.floor(len(nlredseries_x1)/2)),int(np.floor(len(nlredseries_x1)/2)+1))*math.pi/(len(nlredseries_x1)/2) 

pickle_file_name = folder+sig_name+"_Time_Series_Data_Pickle.pkl"
with open(pickle_file_name,'wb') as import_x1_x2_pickle:
    pickle.dump(nlredseries_x1,import_x1_x2_pickle)
    pickle.dump(nlredseries_x2,import_x1_x2_pickle)
    pickle.dump(omega,import_x1_x2_pickle)
    pickle.dump(tvals,import_x1_x2_pickle)
    
pickle_file_name = folder+sig_name+"_Time_Series_Data_Pickle_x1_fft.pkl"
with open(pickle_file_name,'wb') as import_x1_x2_pickle:
    pickle.dump(nlredseries_x1,import_x1_x2_pickle)

pickle_file_name = folder+sig_name+"_Time_Series_Data_Pickle_x2_fft.pkl"
with open(pickle_file_name,'wb') as import_x1_x2_pickle:
    pickle.dump(nlredseries_x2,import_x1_x2_pickle)

pickle_file_name = folder+sig_name+"_Time_Series_Data_Pickle_omega.pkl"
with open(pickle_file_name,'wb') as import_x1_x2_pickle:
    pickle.dump(omega,import_x1_x2_pickle)

pickle_file_name = folder+sig_name+"_Time_Series_Data_Pickle_tvals.pkl"
with open(pickle_file_name,'wb') as import_x1_x2_pickle:
    pickle.dump(tvals,import_x1_x2_pickle)

pickle_file_name = folder+sig_name+"_Time_Series_Data_Pickle_x1_ts.pkl"
with open(pickle_file_name,'wb') as import_x1_x2_pickle:
    pickle.dump(nltseries[0,:],import_x1_x2_pickle)
    
pickle_file_name = folder+sig_name+"_Time_Series_Data_Pickle_x2_ts.pkl"
with open(pickle_file_name,'wb') as import_x1_x2_pickle:
    pickle.dump(nltseries[1,:],import_x1_x2_pickle)

# nlredseries = np.abs(fnltseries[:int(Nstps/2)])/np.sqrt(Nstps)
# S_new = nlredseries


