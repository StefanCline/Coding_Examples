# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:53:15 2023

@author: Sclin
"""

#forward and backwards pass verification for individual datapoints
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import math
import seaborn as sns

def forward_backward_point_verification(signal_data_points,device,transforms,inv_transforms,view_passes,accept,ps,folder,png):

    # 0. Building out some pseudo data points to check the forward pass
    plt.rcParams['figure.figsize'] = [10, 5]
    
    mu0 = torch.tensor([0]).to(device)
    sig = torch.tensor([1]).to(device)
    
    XS = signal_data_points 
    XG = torch.tensor([]).to(device)  # gaussian data points
    N = accept
    
    for ii in range(N):
        # first gaussian random point
        R1 = random.gauss(mu0, sig).unsqueeze(0)
        R1 = R1.to(device)
        XG = torch.cat((XG,R1),0)
    
    XG = XG.squeeze(1)
    
    if view_passes == True:
        plt.hist(XG.cpu(),bins=300)
        plt.title('Random Gaussian: Data to be put into Forward Pass')
        plt.show()
        
        plt.hist(ps,bins=300)
        plt.title("Original Data Points from the Signal's Histogram")
        plt.show()
    
    # 1. Forward Pass
    
    XG = XG.float()
    ForwardData = []
    for ii in range(len(transforms)):
        if ii == 0: 
            ForwardData = transforms[ii](XG.unsqueeze(1))
        else:
            ForwardData = transforms[ii](ForwardData)
    
    ForwardData = ForwardData.squeeze(1)
    
    if view_passes == True: 
        plt.hist(ForwardData.cpu().detach().numpy(),bins=300);
        plt.title('Gaussian through $\mathcal{F}^{-1}$',fontsize=36)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        pic_file_name = folder+ "Output of Gaussian through F" + png
        plt.savefig(pic_file_name)
        plt.show()
    
    del XG
    
    #print(torch.cuda.memory_summary(device=None, abbreviated=False)) #testing to see what's happening with the memory
    
    # 2. Plotting backward call
    
    XS = XS.float()
    BackwardData = torch.tensor([]).to(device)
    for ii in range(len(inv_transforms)):
        if ii == 0: 
            BackwardData = inv_transforms[ii](XS.unsqueeze(1))
        else:
            BackwardData = inv_transforms[ii](BackwardData)
    BackwardData = BackwardData.squeeze(1)
    
    # simple gaussian
    X  = np.linspace(-5,5,300)
    fX = 1/(np.sqrt(2*math.pi))*np.exp(-np.square(X)/2)
    
    if view_passes == True: 
        plt.subplot(1,2,1)
        plt.hist(BackwardData.cpu().detach().numpy(),bins=300);
        plt.xlim([-5,5])
        plt.title('Signal Data through $\mathcal{F}$',fontsize=36)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    
    
    BackwardData = BackwardData.cpu()
    if view_passes == True: 
        plt.subplot(1,2,2)
        sns.distplot(np.asarray(BackwardData.detach().numpy()), hist=True, kde=True, bins=None, color='blue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, label='flow')
        plt.plot(X,fX,color='green')
        plt.xlim(-5,5)
        plt.title('KDE of Signal through F^-1 to Gaussian')
        pic_file_name = folder+ "Output of Gaussian through F inv" + png
        plt.savefig(pic_file_name)
        plt.show()
    
    print('')
    print('')
    print('Done with: Forward and Backward Verification')
    print('')
    print('')