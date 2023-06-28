# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:44:22 2023

@author: Sclin
"""

# K_Star_Bank_Generator

def k_star_Bank_Generator(crit_vals,B):

    import numpy as np
    from scipy.integrate import quad
    from scipy import optimize
    import matplotlib.pyplot as plt
    #%matplotlib inline
    
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
        farleft = -np.inf
        farright = np.inf
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
    
    
    ##################################################################################################################################################
    
    # Defining the Normal Gaussian to take critical points from
    mu1 = 0.
    s1 = 1.
    left_lim = -6.0
    right_lim = 6.0
    
    #rawfun = lambda x: (np.exp(-(x-mu1)**2./(2.*(s1**2.))) + np.exp(-(x-mu2)**2./(2.*(s2**2.)))) # defines double hump gaussian shape
    rawfun = lambda x: (np.exp(-(x-mu1)**2./(2.*(s1**2.))))                                      # defines single hump gaussian shape
    scfac = quad(rawfun,-np.inf,np.inf)[0] # scaling factor that normalizes gaussian
    expdist = lambda x: rawfun(x)/scfac # redefines the rawfunc above to be the normalized double hump gaussian 
    
    
    if crit_vals == 0:
        kbreaks = [0.0]
        print("kbreaks= ",0)
    else:
        kvec0 = np.linspace(-B,B,crit_vals)
        sol = optimize.root(objective_fun_and_jac, kvec0, args=(expdist), jac=True, method='hybr')
        kbreaks = sol.x
        finalgval, finaljacmat = objective_fun_and_jac(kbreaks, expdist)
        
        
        kvals = np.linspace(left_lim,right_lim,int(1e3)+1)
        plt.plot(kvals, expdist(kvals),color='k')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$p_{u}(x)$")
        plt.title("Ticker Count: "+str(crit_vals))
        #plt.plot([0., 0.], [0., expdist(0.)],color='r',ls='--')
        # for jj in range(kbreaks.size):
        #     plt.plot([kbreaks[jj], kbreaks[jj]], [0., expdist(kbreaks[jj])],color='r',ls='--')
        # plt.show()
        
        print("kvec0= ",kvec0)
        print("kbreaks",crit_vals,"=",kbreaks)
    return kbreaks

