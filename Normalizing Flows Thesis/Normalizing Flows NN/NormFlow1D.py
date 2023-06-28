# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:02:12 2023

@author: Stefan Cline
"""

## Normalizing Flows using Pyro: 1D Signal Processing Application

print('')
print('')
print('Beginning Program: 1D Norm Flows')
print('')
print('')


########################################################################################################
#                          Required Packages and Baseline Imports                                      #
########################################################################################################

# %reset -s -f
import logging
import os
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter('always', category=UserWarning) # comment this out if plots stop working
warnings.filterwarnings("ignore") # turns off all warnings, comment out if troubleshooting

import torch
from torch import nn
import pickle

################################ Create device agnostic code ################################
class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

if torch.cuda.is_available():
    device = "cuda"
    print('')
    print(f"{bcolors.OKGREEN}GPU Detected. Using GPU for faster computations!"+ bcolors.ENDC)
    print('')
else:
    device = "cpu"
    print('')
    print(f"{bcolors.FAIL}WARNING, NO GPU DETECTED! Switching to running on CPU."+ bcolors.ENDC)
    print('')
##############################################################################################

# REMOVE IF NOT REQUIRED LATER, WILL CAUSE TO FAIL IF NO GPU AVAILABLE
# device = "cuda:0"

import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

# NOTE! The first time you download these files they may not work and require
# changing things inside of the spline, householder, etc. Pyro code to get 
# everything to run off of the GPU (this is for
# !pip3 install pyro-ppl)

#!pip3 install pyro-ppl
import pyro
import math


smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')

pyro.enable_validation(True)
# pyro.set_rng_seed(1) # uncomment if you want to control the randomness of pyro
# logging.basicConfig(format='%(message)s', level=logging.INFO)

# Set matplotlib settings
#%matplotlib inline
plt.style.use('default')

import pyro.distributions as dist
import pyro.distributions.transforms as T

plt.rcParams['figure.figsize'] = [5, 5]

# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
from pyro.nn.dense_nn import DenseNN
from torch.distributions import constraints
import torch.nn.functional as F

from torch.distributions import Transform
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.nn import AutoRegressiveNN

from numpy import random
from numpy import diff
import random
from numpy.lib.histograms import histogram

import scipy
from scipy import interpolate
from scipy.interpolate import PchipInterpolator
from scipy import optimize
from scipy.integrate import quad
from scipy.fft import fft, ifft
from scipy.misc import derivative
import scipy.io

# import sys
from otsu_on_gaussian import custom_otsu
from curtis_osc_and_otsu import *
from signals import *
from norm_flow_model import *
from training_loop_funcs import *
from otsu_on_gaussian import *
from k_star_bank_generator import *
from breaks_and_plot import *
from loading_parameters import *
from forward_backward_point_verify import *

#import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512Mb"

print('')
print('Done with: package installs and function definitions')
print('')

########################################################################################################
#             Turning Signal Curve into FFT and Sampling from it as if it's a PDF                      #
########################################################################################################

# current signal name choices: 
    # 'harmonic', 'harmonic_full', 'sto_linear', 'VDP', 'DHG', 'fsig2', 'ECG'

for cb_val in range(512,513):
    # each iteration adjusts the number of splines in the ML model

    # 1. Getting signal sample, doing plots and doing FFT
    # VDP: Van der Pol, stochastic linear, DHG: double hump gaussian
    # ECG and fsig2: see Empirical Wavelet Transforms by Dr. Jerome Gilles
    sig_name = 'VDP'
    axis = 'x2' #used only for VDP, make sure to adjust loading_params file
    png = '.png'
    view_TL = True                      #true shows plots during training, false will turn them off
    view_other = True                   #true shows plots of the signal and setup, false will turn them off
    view_passes = False                 #true shows plots of passing datapoints through the NN, false will turn them off
    view_bij_f = True                   #true shows plots of f and f', false will turn them off
    view_approx_target_compare = True   #true shows plots of the approximated PDF of the target, false will turn them off
    load_file = False                   #true loads in precomputed histogram, otherwise will calculate from scratch
    do_the_breaks = True                #true will show break points via multi-otsu, false will skip it
    do_the_pickle = True                #true will save the data to a pickle file, false will not
    
    #File save names
    #folder = file_save_name_vscode(sig_name) # uncomment if using vscode
    folder = file_save_name(sig_name) #uncommment if using spyder
    #loading all of the parameters required for making the histogram from the signal
    t0,tf,sr,sval,mu,alpha,N,M,sig = load_params(sig_name,axis)
    
    print('')
    print('Making the pseudo samples')
    print('')

    if sig_name != 'DHG':
        if sig_name != 'VDP':
            omega, S_original, time_axis = signal_type(sig_name,t0,tf,sr,sval,mu,alpha,view_other)
            S_new = S_original
        if sig_name == 'VDP': #for higher sig, remove the "_low_sig" portino from the x1 x2 data folder name
            with open ('F:\\Research_Thesis\\NormFlows\\Image Runs\\VDP\\x1_x2_data_folder_low_sig\\Im_VDP_Time_Series_Data_Pickle.pkl','rb') as import_x1_x2_pickle:
                if axis == 'x1':
                    S_new = pickle.load(import_x1_x2_pickle) 
                    discard = pickle.load(import_x1_x2_pickle)
                    omega = pickle.load(import_x1_x2_pickle)
                    time_axis = pickle.load(import_x1_x2_pickle)
                    del discard
                elif axis == 'x2':
                    discard = pickle.load(import_x1_x2_pickle) 
                    S_new = pickle.load(import_x1_x2_pickle)
                    omega = pickle.load(import_x1_x2_pickle)
                    time_axis = pickle.load(import_x1_x2_pickle)
                    del discard
        omega_original = omega
        # 2. Interpolation of the Manipulated FFT
        omeg_new, pdf_curve = signal_interpolation(omega,S_new,folder,png,view_other)
        # 3. Turn into a PDF and get a dataset of randomly sampled points
        test_gauss = gauss_over_fft(omega,omeg_new,pdf_curve,M,sig,folder,png,sig_name,view_other)
        if load_file == False:
            ps, accept = signal_to_samples(omega,N,M,sig,pdf_curve,folder,png,sig_name)
            if sig_name == 'VDP':
                if axis == 'x1':
                    pickle_file_name = folder+sig_name+"_x1_Pickle.pkl"
                    with open(pickle_file_name,'wb') as FFF:
                        pickle.dump(ps,FFF)
                elif axis == 'x2':
                    pickle_file_name = folder+sig_name+"_x2_Pickle.pkl"
                    with open(pickle_file_name,'wb') as FFF:
                        pickle.dump(ps,FFF)
        elif load_file == True: 
            if sig_name == 'fsig2':
                with open ('F:\\Research_Thesis\\NormFlows\\Image Runs\\fsig2\\ps_folder\\Im_fsig2_Pickle.pkl','rb') as FFF:
                    ps = pickle.load(FFF) 
            if sig_name == 'ECG':
                # still needs to be saved and built, this line not ready yet
                with open ('F:\\Research_Thesis\\NormFlows\\Image Runs\\fsig2\\ps_folder\\Im_ECG_Pickle.pkl','rb') as FFF:
                    ps = pickle.load(FFF) 
            if sig_name == 'VDP': 
                if axis == 'x1':
                    with open ('F:\\Research_Thesis\\NormFlows\\Image Runs\\VDP\\ps_folder\\Im_VDP_x1_Pickle.pkl','rb') as FFF:
                        ps = pickle.load(FFF) 
                elif axis == 'x2':
                    with open ('F:\\Research_Thesis\\NormFlows\\Image Runs\\VDP\\ps_folder\\Im_VDP_x2_Pickle.pkl','rb') as FFF:
                        ps = pickle.load(FFF) 
            accept = N
    elif sig_name == 'DHG': #specific toy case of the double humped gaussian
        N, left_gauss, right_gauss, mu1, mu2, s1, s2 = load_params_DHG()
        ps, accept = DHG_samples(N,mu1,mu2,s1,s2)
        omega = np.linspace(left_gauss,right_gauss,3000)
        S_new = ((1.0/(s1*np.sqrt(2.0*math.pi))*np.exp(-(omega-mu1)**2./(2.*(s1**2.))) + (1.0/(s2*np.sqrt(2.0*math.pi))*np.exp(-(omega-mu2)**2./(2.*(s2**2.))))))*0.5
        omeg_new, pdf_curve = signal_interpolation(omega,S_new,folder,png)
    
    print('')
    print('Done Making the pseudo samples')
    print('')
    
    # 4. Turn the np array into a torch tensor
    signal_data_points = torch.from_numpy(np.array(ps))
    signal_data_points = signal_data_points.unsqueeze(1)
    signal_data_points = signal_data_points.to(device)
    #omega = omega-np.mean(omega)
    omega = torch.from_numpy(np.array(omega)).to(device)
    
    print('')
    print('Done with: generating pseudo sample data')
    print('')

    
    ########################################################################################################
    #         Setup of Shell Model and Base Distribution (Base is normalized Gaussian)                     #
    ########################################################################################################
    
    ## Combo of splines (w/ disc. cosine to avoid numerical instability) and householder transformations
    
    #NormFlowModel, transforms, inv_transforms, base_dist, flow_dist = Norm_Flow_Model(omega,device)
    #NormFlowModel, transforms, inv_transforms, base_dist, flow_dist = Norm_Flow_Model_small(omega,device)
    #NormFlowModel, transforms, inv_transforms, base_dist, flow_dist = Norm_Flow_Model_splineonly(omega,device)
    NormFlowModel, transforms, inv_transforms, base_dist, flow_dist = Norm_Flow_Model_bad(omega,device,cb_val)
    
    
    ########################################################################################################
    #                                           Training the Model                                         #
    ########################################################################################################
    
    ### Training the Model
    y_max = maximum_plot_y(sig_name)
    ## %%time
    loss_old = 10000000
    left_vis =  omega[0].cpu().detach().numpy()
    right_vis = omega[-1].cpu().detach().numpy()
    total_steps = 701
    total_plots = 2
    outer_loops_LR = 3
    
    for ii in range(outer_loops_LR): # here the value inside range() dictates how many levels of decreased LR is used
        print('Outer Loop is',ii,' of ',outer_loops_LR)
        ## For the visual outputs below
        plt.rcParams['figure.figsize'] = [20, 5]
        samp = torch.Size([signal_data_points.shape[0],])
        #vis = 10.0  
        LR = training_loop_LR(ii,total_steps,total_plots)
        # Specifying step total as well as how often to plot
        total_disp = math.ceil(total_steps/total_plots)
        STEP = []  # shells for the output plots
        LOSS = []  # shells for the output plots
        steps = 1 if smoke_test else (total_steps+1)
        if ii == 2: 
            steps = 701
        ## Pulling in the target distribution
        # dataset = torch.tensor(pseudo_data, dtype=torch.float).to(device)
        dataset = torch.tensor(signal_data_points, dtype=torch.float).to(device)
        ## Setting up the optimizer as splines and spicy Adam SGD
        # optimizer = torch.optim.Adam(spline_transform.parameters(), lr=LR) # how to do it w/ 1 spline layer
        optimizer = torch.optim.Adam(NormFlowModel.parameters(), lr=LR)  # multiple layered feller
        for step in range(steps):
            print('Starting step: ',step, ' of ',(steps-1))
            optimizer.zero_grad()
            loss = -flow_dist.log_prob(dataset).mean() # minus is supposed to be here or identity map: X --> F(X) --> X
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            # Purely for visualization of how the NN is doing
            STEP.append(step)
            LOSS.append(loss.item())
            if (step % total_disp == 0 or step == (steps-1)) and step != 0 and view_TL == True:
                training_loop_visuals(samp,ps,flow_dist,loss,step,total_steps,left_vis,right_vis,STEP,LOSS,LR,y_max,folder,png)
            # Printing the loss 
            loss_str = str(loss.item())
            loss_new = loss.item()
            # if loss_new < loss_old:
            #     best_flow_dist = flow_dist
            flow_dist.clear_cache()
    
    
    print('')
    print('Done with: Training Loop')
    print('')
    
    del dataset
    
    ########################################################################################################
    #                     Verifying Forward and Backwards Pass are sucessful                               #
    ########################################################################################################
    
    forward_backward_point_verification(signal_data_points,device,transforms,inv_transforms,view_passes,accept,ps,folder,png)
    
    
    ########################################################################################################
    #           Observing what the analytical function F looks like as a bijective 1D Plot                 #
    ########################################################################################################
    
    # We know a single gaussian hump's greatest interclass variance is at mu
    #    in this case mu=0
    if sig_name == 'DHG':
        Npts = 3001 #make odd to ensure zero is included in the domain
    else:
        Npts = 3001 # len(S_new) #always odd already
    X_dom_G = torch.linspace(omega[0].item(),omega[-1].item(),Npts).to(device) 
    TWO = torch.tensor([2.0]).to(device)
    ONE = torch.tensor([1.0]).to(device)
    PI = torch.acos(torch.zeros(1)).to(device) * TWO
    X_rng_G = ONE/torch.sqrt(PI*TWO)*torch.exp(-torch.square(X_dom_G)/TWO)
    X_dom_G = X_dom_G.cpu().detach().numpy()
    
    # verify the single hump gaussian shape
    # plt.plot(X_dom_G,X_rng_G.cpu().detach().numpy())
    # plt.grid()
    # plt.show()
    
    # Showing a computationally created funciton, f: gauss--> f --> signal
    #      this is x, and f(x)
    gauss_in_vec = torch.linspace(omega[0].item(),omega[-1].item(),Npts).to(device)
    ForwardData_mu0 = []
    for ii in range(len(transforms)):
        if ii == 0: 
            ForwardData_mu0 = transforms[ii](gauss_in_vec.unsqueeze(1))
        else:
            ForwardData_mu0 = transforms[ii](ForwardData_mu0)
    ForwardData_mu0 = ForwardData_mu0.squeeze(1)
    
    if view_bij_f == True: 
        plt.plot(gauss_in_vec.cpu().detach().numpy(),ForwardData_mu0.cpu().detach().numpy())
        plt.title("Approximation of $f(z)$",fontsize=36)
        plt.xlabel('$z$',fontsize=32)
        plt.ylabel('$f(z)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        pic_file_name = folder+ "Approximation of Bijective F" + png
        plt.savefig(pic_file_name)
        plt.show()
    
    # Viewing the target PDF
    #     note this will return the x, f(x) as well, same thing as above
    F_bij = interpolate.interp1d(gauss_in_vec.cpu().detach().numpy(),ForwardData_mu0.cpu().detach().numpy())
    xtest = gauss_in_vec.cpu().detach().numpy()
    ytest = F_bij(xtest)
    dx = xtest[1]-xtest[0]
    Fprime = np.gradient(F_bij(xtest),dx) #diff(ytest)/dx
    # for jj in range(1001):
    #     Fprime[0][jj] = derivative(F, xdom[jj])
    if view_bij_f == True: 
        plt.plot(xtest,Fprime) #[0][:]
        plt.title("Derivative of $f$",fontsize=36)
        plt.xlabel('$z$',fontsize=32)
        plt.ylabel("$f'(z)$",fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
    
    # Target Distribution Approximation
    X_rng_G = X_rng_G.cpu().detach().numpy()
    PZ = X_rng_G/Fprime #[0][:]
    if view_approx_target_compare == True: 
        plt.plot(F_bij(xtest),PZ,label="Approximation") 
        plt.title("Target Distribution",fontsize=36)
        plt.xlabel("$z$",fontsize=32)
        plt.ylabel("$p_Z(z)$",fontsize=32)
    if sig_name == 'DHG':
        target = 0.5*(1.0/(s1*np.sqrt(2*math.pi))*np.exp(-(xtest-mu1)**2.0/(2*s1**2))+1.0/(s1*np.sqrt(2*math.pi))*np.exp(-(xtest-mu2)**2.0/(2*s2**2)))
    else: 
        #normalizing the target distribution to be (integral_R target dx = 1)
        target = pdf_curve(xtest)/np.trapz(pdf_curve(xtest),dx=(xtest[1]-xtest[0]))
    if view_approx_target_compare == True: 
        plt.plot(xtest,target,linestyle='dashed',label="Exact")
        if sig_name == 'DHG':
            plt.xlim([-6.0,6.0])
            plt.ylim([-0.01,.21])
        elif sig_name == 'harmonic':
            plt.xlim([-76.0,-50])
            plt.ylim([-0.01,0.2])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.grid()
        plt.show()
    
    # No zoom on the approximation and target
    if view_approx_target_compare == True: 
        plt.plot(F_bij(xtest),PZ,label="Approximation") 
        plt.title("Target Distribution",fontsize=36)
        plt.xlabel("$z$",fontsize=32)
        plt.ylabel("$p_Z(z)$",fontsize=32)
        plt.plot(xtest,target,linestyle='dashed',label="Exact")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        if sig_name == 'harmonic' or sig_name == 'harmonic_full':
            print('im in here like an asshole')
            plt.xlim([-0.3,0.3])
            plt.ylim([-0.01,8.0])
        elif sig_name == 'sto_linear':
            plt.xlim([-50,50])
            #plt.ylim([-0.01,8.0])
        elif sig_name == 'VDP':
            plt.ylim([-0.001,max(target)+.05*max(target)])
            plt.xlim([-.5,0.5])
        plt.grid()
        plt.show()
    
    del gauss_in_vec, ForwardData_mu0
    
    
    print('')
    print('')
    print("Done with: Approximations to PDF and bijective $f$ and $f'$")
    print('')
    print('')
    
    ########################################################################################################
    #                               Pause Before Finding Breakpoints                                       #
    ########################################################################################################
    
    cont_run = input("Kill this now if it looks bad, otherwise, enter any number and hit enter. ")
    
    ########################################################################################################
    #                                      Finding Breakpoints                                             #
    ########################################################################################################
    
    if do_the_breaks == True:
        if sig_name == 'fsig2':
            #cb_val is the total number of count_bins in the model
            #    this is the number of spline segments made
            # note currently doing cubic spline interp, may need to do quadratic or linear
            start_perc = 60
            final_add_perc = 20
            kbreaks_mat_3 = np.zeros((final_add_perc,3))
            kbreaks_mat_4 = np.zeros((final_add_perc,4))
            kbreaks_mat_5 = np.zeros((final_add_perc,5))
            
            interp_types = ["linear","quadratic","cubic"]
            for kind_name in [interp_types[0]]:
                for ii in range(3,6):
                    for perc in range(start_perc,start_perc+final_add_perc):
                        kbreakvals = breaks_and_plot(sig_name,ii,F_bij(xtest),PZ,kind_name,xtest,target,cb_val,perc)
                        if ii == 3: 
                            kbreaks_mat_3[(perc-start_perc),:] = kbreakvals
                        elif ii == 4: 
                            kbreaks_mat_4[(perc-start_perc),:] = kbreakvals
                        elif ii == 5: 
                            kbreaks_mat_5[(perc-start_perc),:] = kbreakvals
                        print("For ",ii," breaks, k_breaks = ",kbreakvals," at percent: ",perc," with type: ",kind_name)
                        
        if sig_name == 'ECG':
            #cb_val is the total number of count_bins in the model
            #    this is the number of spline segments made
            # note currently doing cubic spline interp, may need to do quadratic or linear
            start_perc = 90
            final_add_perc = 9
            interp_types = ["linear","quadratic","cubic"]
            for kind_name in interp_types:# [interp_types[0]]:
                for ii in range(46,47):
                    for perc in range(start_perc,start_perc+final_add_perc):
                        kbreakvals = breaks_and_plot(sig_name,ii,F_bij(xtest),PZ,kind_name,xtest,target,cb_val,perc)
                        print("For ",ii," breaks, k_breaks = ",kbreakvals," at percent: ",perc," with type: ",kind_name)
                        
        if sig_name == 'VDP':
            #cb_val is the total number of count_bins in the model
            #    this is the number of spline segments made
            # note currently doing cubic spline interp, may need to do quadratic or linear
            # the last number in the array keeps track overall of which kbreaks val is which
            start_perc = 60
            final_add_perc = 40
            end_mode = 8
            #all_kbreaks_mat = np.zeros(end_mode)
            ticker = 0
            interp_types = ["linear","quadratic","cubic"]
            for kind_name in interp_types:# [interp_types[0]]:
                for ii in range(2,end_mode):
                    zero_in = end_mode-ii
                    zero_mat = np.zeros(zero_in)
                    for perc in range(start_perc,start_perc+final_add_perc):
                        kbreakvals = breaks_and_plot(sig_name,ii,F_bij(xtest),PZ,kind_name,xtest,target,cb_val,perc)
                        print("For ",ii," breaks, k_breaks = ",kbreakvals," at percent: ",perc," with type: ",kind_name,", axis: ",axis,", ticker: ",ticker)
                        if ticker > 0:
                            add_mat = np.hstack((kbreakvals,zero_mat))
                            all_kbreaks_mat = np.append(all_kbreaks_mat,[add_mat],axis=0)
                            all_kbreaks_mat[ticker,-1] = ticker
                        elif ticker == 0:
                            first_mat = np.hstack((kbreakvals,zero_mat))
                            all_kbreaks_mat = [first_mat]
                        ticker = ticker + 1
                        
                

########################################################################################################
#                                            Pickling Data                                             #
########################################################################################################


if do_the_pickle == True:
    X_x =   F_bij(xtest)
    P_X_x = PZ
    
    if axis == 'x1' and sig_name == 'VDP':
        pickle_file_name = folder+sig_name+"_VDP_kbreaks_pickle_Pickle_x1_low_sig.pkl"
        with open(pickle_file_name,'wb') as kbreaks_pickle:
            pickle.dump(all_kbreaks_mat,kbreaks_pickle)
    if axis == 'x2' and sig_name == 'VDP':
        pickle_file_name = folder+sig_name+"_VDP_kbreaks_pickle_Pickle_x2_low_sig.pkl"
        with open(pickle_file_name,'wb') as kbreaks_pickle:
            pickle.dump(all_kbreaks_mat,kbreaks_pickle)
        


########################################################################################################
#  Completion Notice                                                                                   #
########################################################################################################

print('')
print('')
print('ALL COMPUTATIONS COMPLETE - CODE FINISHED')
print('')
print('')


