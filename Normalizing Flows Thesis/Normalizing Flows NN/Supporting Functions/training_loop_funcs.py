# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:36:57 2023

@author: Sclin

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def training_loop_LR(ii,ts,tp):
    # Learning Rates per iteration loop
    if ii == 0:
      LR = 0.1
    elif ii == 1:
      LR = 0.01
    elif ii == 2:
      LR = 0.001
    else: 
      LR = 0.0001  
    return LR


def training_loop_visuals(samp,ps,flow_dist,loss,step,total_steps,left_vis,right_vis,STEP,LOSS,LR,y_max,folder,png):
    # Visualizing the Loss
    # Plotting the estimation and the target
    plt.subplot(1, 2, 1)    
    forward_flow = flow_dist.sample(samp).cpu() #.detach().numpy()
    #forward_flow = forward_flow.cpu()
    #plt.plot(omega.cpu(),S_new) #lets see what just the dude is doin...
    sns.distplot(forward_flow.squeeze(), hist=True, kde=True, bins=None, color='firebrick', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, label='flow')     # estimation
    sns.distplot(np.asarray(ps), hist=True, kde=True, bins=None, color='blue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, label='flow')
    #sns.distplot(px_plot, hist=False, kde=True, bins=None, color='black', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, label='flow')   # actual target (double hump)
    loss_str = str(loss.item())
    plt.title('Target (blue), $\mathcal{F}(G_{N})$ (red)',fontsize=28)
    if y_max != 0:
        plt.ylim([-.001,y_max])
    plt.xlim([left_vis,right_vis])
    plt.ylabel('Density',fontsize=22)
    plt.xlabel('$z$',fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    #plt.xlim([(omega[0].item()-1),(omega[-1].item()+1)])

    # Plotting the loss function 
    plt.subplot(1,2,2)
    loss_str = str(loss.item())
    plt.loglog(STEP,LOSS)
    plt.title('Loss Curve, $l_r$= '+str(LR)+', $\mathcal{L}$ = '+loss_str[:5],fontsize=28)
    plt.xlabel('Step: '+str(step)+'/'+str(total_steps-1),fontsize=22)
    plt.ylabel('Loss',fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    pic_file_name = folder+ "LR="+str(LR)+' '+str(step)+' of '+ str(total_steps) + png
    plt.savefig(pic_file_name)
    plt.show()

    
