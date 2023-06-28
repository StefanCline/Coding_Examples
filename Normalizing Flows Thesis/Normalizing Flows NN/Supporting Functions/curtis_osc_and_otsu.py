# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:35:14 2023

@author: Sclin
"""

## Function Holder for all Oscillator Data


# needed packages for these functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

######################################################################
#                                                                    #
# ## Stochastically Perturbed, Forced Linear Oscillator ##############
#                                                                    #
######################################################################

# COMMENTED OUT!!!!!!!!!

# If this needs to work go to the signals.py file

# def lin_osc(lhs, om):
#     rhs = np.zeros(lhs.size)
#     rhs[0] = lhs[1]
#     rhs[1] = -om**2. * lhs[0]
#     return rhs

# def vanderpol(lhs,mu):
#     rhs = np.zeros(lhs.size)
#     rhs[0] = lhs[1]
#     rhs[1] = mu*(1.-lhs[0]**2.)*lhs[1] - lhs[0]
#     return rhs

# # this is Heun's stochastic method (see wikipedia page for now)
# def random_integrator(omd,tf,Nstps,sval,alpha,ffun):
#     xvec = np.zeros((2,Nstps+1),dtype=np.float64)
#     xvec[0,0] = 1.
#     dt = tf/Nstps
    
#     for jj in range(1,Nstps+1):
#         uval = np.random.rand(1)
#         if uval > .5:
#             sk = 1.
#         else:
#             sk = -1.
#         normrv = np.random.randn(1)
#         bvec = sval*np.ones(2)*np.sqrt(dt)*(normrv - sk)
#         bvecp = sval*np.ones(2)*np.sqrt(dt)*(normrv + sk)
        
#         frc = np.array([0., alpha*np.sin(omd*(jj-1)*dt)])
#         frcp = np.array([0., alpha*np.sin(omd*jj*dt)])
#         k1 = dt*(ffun(xvec[:,jj-1]) + frc) + bvec
#         k2 = dt*(ffun(xvec[:,jj-1]+k1)+frcp) + bvecp
#         xvec[:,jj] = xvec[:,jj-1] + .5*(k1 + k2)
#     return xvec

# ########################## Plots

# def stochastic_linear_plot(tseries):
#     plt.plot(tseries[0,:],tseries[1,:],color='k')
#     plt.xlabel(r"$x_{1}$")
#     plt.ylabel(r"$x_{2}$")
#     plt.title("Phase Plot of Stochastic Linear Oscillator")
    

# def stochastic_linear_plot_time(tvals,tseries,ref):
#     plt.plot(tvals,tseries[0,:],color='r')
#     plt.plot(tvals,ref,color='k')
#     plt.ylabel(r"$x_{1}(t)$")
#     plt.xlabel(r"$t$")


# def stochastic_linear_plot_FFT(redseries):
#     plt.plot(np.ma.log10(redseries),color='k')
#     plt.xlabel(r"$\omega$")
#     plt.ylabel(r"$\log_{10}(\hat{S}(\omega))$")

# ######################################################################
# #                                                                    #
# ####### Stochastically Perturbed, Forced Van der Pol Oscillator ######
# #                                                                    #
# ######################################################################


# def stochastic_VDP_plot(nltseries):
#     plt.plot(nltseries[0,:],nltseries[1,:],color='k')
#     plt.xlabel(r"$x_{1}$")
#     plt.ylabel(r"$x_{2}$")
#     plt.title("Phase Plot of Stochastic Van der Pol Oscillator")


# def stochastic_VDP_plot_time(tvals,nltseries):
#     plt.plot(tvals,nltseries[0,:],color='k')
#     plt.ylabel(r"$x_{1}(t)$")
#     plt.xlabel(r"$t$")


# def stochastic_VDP_plot_FFT(nlredseries):
#     plt.plot(np.ma.log10(nlredseries),color='k')
#     plt.xlabel(r"$\omega$")
#     plt.ylabel(r"$\log_{10}(\hat{S}(\omega))$")



######################################################################
#                                                                    #
#######               Multi_Breaks_Otsu_Method                  ######
#                                                                    #
######################################################################

# Note, will not handle the simple case of 1 break point

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

def objective_fun_and_jac(kvec, pdist,L,R):
    print(kvec)
    probs = np.zeros(kvec.size+1, dtype=np.float64)
    avgs = np.zeros(kvec.size+1, dtype=np.float64)
    gfun = np.zeros(kvec.size, dtype=np.float64)
    avgfun = lambda x: x*pdist(x)
    #farleft = 0.
    #farleft = -np.inf
    #farright = np.inf
    farleft = L
    farright = R
    
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
    for jj in range(kvec.size):
        gfun[jj] = .5*(avgs[jj] + avgs[jj+1]) - kvec[jj]
        
    return gfun, jacmat


def otsu_plot(expdist,kbreaks,L,R,folder,png):
    print("kbreaks=",kbreaks)
    kvals = np.linspace(L,R,(int(1e3)+1))
    plt.plot(kvals, expdist(kvals),color='k')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$p_{u}(x)$")
    #plt.plot([0., 0.], [0., expdist(0.)],color='r',ls='--') #artifact of picking zero as the left most point
    for jj in range(kbreaks.size):
        plt.plot([kbreaks[jj], kbreaks[jj]], [0., expdist(kbreaks[jj])],color='r',ls='--')
    #plt.xlim([-230,-12])
    pic_file_name = folder+ "Multi-Otsu Breaks" + png
    plt.savefig(pic_file_name)
    plt.show()







