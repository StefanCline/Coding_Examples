# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:04:47 2023

@author: Sclin
"""
import numpy as np
import numpy as np
from scipy.integrate import quad
from scipy import optimize
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, ifft
from scipy import interpolate
import random
from scipy.interpolate import splrep, splev

def breaks_and_plot(sig_name,break_total,XI,rawfun,kind_name,XTEST,TARGET,cb_val,perc):
    """

    Parameters
    ----------
    sig_name : Name of the function type, e.g. DHG or fsig2.
    break_total : Total number of k star values.
    XI : Frequency Domain 'x-axis'.
    rawfun : Function approximation output by the NF.
    kind_name : name of interp type, either linear, quadratic or cubic
    XTEST : exact x domain (freqeuncy space)
    TARGET : exact y values (freqeuncy space)

    Returns
    -------
    The numerical values of the determined k-star values.

    """
    
    def jac_maker(kvec, pdist, avgs, probs):
        jacmat = np.zeros((kvec.size, kvec.size), dtype=np.float64)
        for jj in range(kvec.size):
            if jj == 0:
                jacmat[0, 0] = .5*pdist(kvec[jj])*((kvec[jj]-avgs[jj])/probs[jj]  +  (avgs[jj+1]-kvec[jj])/probs[jj+1]) - 1.
                jacmat[0, 1] = .5*pdist(kvec[jj+1])*((kvec[jj+1]-avgs[jj+1])/probs[jj+1])
            elif jj == kvec.size -1:
                jacmat[kvec.size - 1, kvec.size - 2] = .5*pdist(kvec[jj-1])*((avgs[jj]-kvec[jj-1])/probs[jj])
                jacmat[kvec.size - 1, kvec.size - 1] = .5*pdist(kvec[jj])*((kvec[jj]-avgs[jj])/probs[jj]  +  (avgs[jj+1]-kvec[jj])/probs[jj+1]) - 1.
            else:
                jacmat[jj,jj-1] = .5*pdist(kvec[jj-1])*((avgs[jj]-kvec[jj-1])/probs[jj]) 
                jacmat[jj,jj] = .5*pdist(kvec[jj])*((kvec[jj]-avgs[jj])/probs[jj]  +  (avgs[jj+1]-kvec[jj])/probs[jj+1]) - 1.
                jacmat[jj,jj+1] = .5*pdist(kvec[jj+1])*((kvec[jj+1]-avgs[jj+1])/probs[jj+1])
        return jacmat
    
    
    def objective_fun_and_jac(kvec, pdist):
        probs = np.zeros(kvec.size+1, dtype=np.float64)
        avgs = np.zeros(kvec.size+1, dtype=np.float64)
        gfun = np.zeros(kvec.size, dtype=np.float64)
        avgfun = lambda x: x*pdist(x)
        #farleft = 0.
        farleft =  0 # XI[0] #L # was -np.inf
        farright = right_cutoff #0.25 #XI[-1] #R # was np.inf
        for jj in range(kvec.size+1):
            if jj == 0:
                probs[jj] = quad(pdist, farleft, kvec[0])[0]
                avgs[jj] = quad(avgfun, farleft, kvec[0])[0]/probs[jj]
            elif jj == kvec.size:
                probs[jj] = quad(pdist, kvec[-1], farright)[0]
                avgs[jj] = quad(avgfun, kvec[-1], farright)[0]/probs[jj]
            else:
                probs[jj] = quad(pdist, kvec[jj-1], kvec[jj])[0]
                avgs[jj] = quad(avgfun, kvec[jj-1], kvec[jj])[0]/probs[jj]
        jacmat = jac_maker(kvec, pdist, avgs, probs)
        #print(jacmat)
        for jj in range(kvec.size):
            gfun[jj] = .5*(avgs[jj] + avgs[jj+1]) - kvec[jj]
            
        return gfun, jacmat
    
        
    # parameters
    samps = 2001
    freq = 1/samps
    # XI = 2*math.pi*np.arange(0,samps)/samps
    # XI = XI-XI[(int(samps/2-0.5))]
    # normalizing the function
    delXI = XI[1]-XI[0]
    XI = np.pad(XI,(1,1),'constant',constant_values=(XI[0]-delXI,XI[-1]+delXI))
    XI = np.pad(XI,(1,1),'constant',constant_values=(XI[0]-delXI,XI[-1]+delXI))
    rawfun = np.pad(rawfun,(2,2),'constant',constant_values=(0,0))
    expdist = interpolate.interp1d(XI,rawfun,kind=kind_name,bounds_error=False,fill_value=0)
    # plot normalized function
    # These plots aren't really needed, just help verify that the plot is doing what it's supposed to
    # plt.plot(XI,expdist(XI))
    # plt.grid()
    # plt.xlim([0,0.15])
    # plt.title('plot 1')
    # plt.show()
    
    # plt.plot(XI,expdist(XI))
    # plt.grid()
    # plt.title('plot 2')
    # plt.show()
    
    #determining the cutoffpoint to not destroy multi-otsu
    start = np.argmax(rawfun)
    total_area = np.trapz(rawfun[start:],XI[start:])
    count, crit_area = 2, 0.0
    while crit_area < perc/100:
        crit_area = np.trapz(rawfun[start:start+count],XI[start:start+count])
        crit_area = crit_area/total_area
        #print(crit_area)
        count = count + 1
    crit_index = start+count-2
    right_cutoff = XI[crit_index]
    rco_str = str(float(f'{right_cutoff:.3f}'))
    
    #performing multi-otsu
    kvec0 = np.random.uniform(0,XI[-3],break_total)
    sol = optimize.root(objective_fun_and_jac, kvec0, args=(expdist), jac=True, method='hybr')
    kbreaks = sol.x
    finalgval, finaljacmat = objective_fun_and_jac(kbreaks, expdist)
    
    #making final plots
    #Full Plot of interpolated function with k breaks
    kvals = XI
    TARGET = TARGET/max(TARGET)*max(expdist(kvals)) #scale target to view easier
    
    if sig_name == 'VDP':
        xlim_left = 0 
        xlim_rite = 150
        ylim_bot  = -0.001
    elif sig_name == 'fsig2':
        xlim_left = 0 
        xlim_rite = math.pi
        ylim_bot  = -0.1
    
    plt.subplot(1,3,1)
    plt.plot(XTEST,TARGET,color="orange")
    plt.plot(kvals, expdist(kvals),color='k')
    #plt.xlabel(r"$\xi$")
    plt.xlabel("interp type: "+kind_name)
    plt.ylabel(r"$p_{Z}(z)$")
    #plt.plot([0., 0.], [0., expdist(0.)],color='r',ls='--')
    for jj in range(kbreaks.size):
        #plt.plot([kbreaks[jj], kbreaks[jj]], [0., expdist(kbreaks[jj])],color='r',ls='--')
        plt.plot([kbreaks[jj], kbreaks[jj]], [0,max(expdist(kvals))],color='r',ls='--')
    plt.xlim([xlim_left,xlim_rite])
    plt.ylim([ylim_bot,max(expdist(kvals))+max(expdist(kvals))*.1])
    plt.title("Full $[0,\pi]$. Break Total: " + str(len(kbreaks)))
    plt.grid()
    #plt.show()
    
    plt.subplot(1,3,2)
    plt.plot(XTEST,TARGET,color="orange")
    # more zoomed in plot of interp func with k breaks
    plt.plot(kvals, expdist(kvals),color='k')
    plt.xlabel(r"$\xi$, Perc: "+str(perc))
    #plt.ylabel(r"$p_{Z}(z)$")
    #plt.plot([0., 0.], [0., expdist(0.)],color='r',ls='--')
    for jj in range(kbreaks.size):
        #plt.plot([kbreaks[jj], kbreaks[jj]], [0., expdist(kbreaks[jj])],color='r',ls='--')
        plt.plot([kbreaks[jj], kbreaks[jj]], [0,max(expdist(kvals))],color='r',ls='--')
    plt.xlim([xlim_left,right_cutoff]) #see fhat_sig2 in paper
    plt.grid()
    plt.ylim([ylim_bot,max(expdist(kvals))+max(expdist(kvals))*.1])
    plt.title("Zoomed [0,"+rco_str+"]. Break Total: " + str(len(kbreaks)))
    #plt.show()
    
    # log plot of interp function with kbreaks
    plt.subplot(1,3,3)
    plt.plot(XTEST,TARGET,color="orange")
    plt.loglog(kvals, expdist(kvals),color='k')
    plt.xlabel("cb_val = "+str(cb_val))
    #plt.ylabel(r"$p_{Z}(z)$")
    #plt.plot([0., 0.], [0., expdist(0.)],color='r',ls='--')
    for jj in range(kbreaks.size):
        #plt.plot([kbreaks[jj], kbreaks[jj]], [0., expdist(kbreaks[jj])],color='r',ls='--')
        plt.plot([kbreaks[jj], kbreaks[jj]], [0,max(expdist(kvals))],color='r',ls='--')
    plt.grid()
    plt.ylim([0,max(expdist(kvals))+max(expdist(kvals))*.1])
    plt.title("Log $[0,\pi]$. Break Total: " + str(len(kbreaks)))
    plt.show()
    
    print("Cut off: ",right_cutoff)
    
    return kbreaks
    
    
    
    
    
    
    
    