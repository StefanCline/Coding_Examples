# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:39:49 2023

@author: Sclin
"""

# Signal Type

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scipy
import sys
from scipy.integrate import quad
from scipy.fft import fft, ifft, fftshift

######################################################################
#                                                                    #
# ## Stochastically Perturbed, Forced Linear Oscillator ##############
#                                                                    #
######################################################################

def lin_osc(lhs, om):
    rhs = np.zeros(lhs.size)
    rhs[0] = lhs[1]
    rhs[1] = -om**2. * lhs[0]
    return rhs

def vanderpol(lhs,mu):
    rhs = np.zeros(lhs.size)
    rhs[0] = lhs[1]
    rhs[1] = mu*(1.-lhs[0]**2.)*lhs[1] - lhs[0]
    return rhs

# this is Heun's stochastic method (see wikipedia page for now)
def random_integrator(omd,tf,Nstps,sval,alpha,ffun):
    xvec = np.zeros((2,Nstps+1),dtype=np.float64)
    xvec[0,0] = 1.
    dt = tf/Nstps
    
    for jj in range(1,Nstps+1):
        uval = np.random.rand(1)
        if uval > .5:
            sk = 1.
        else:
            sk = -1.
        normrv = np.random.randn(1)
        bvec = sval*np.ones(2)*np.sqrt(dt)*(normrv - sk)
        bvecp = sval*np.ones(2)*np.sqrt(dt)*(normrv + sk)
        
        frc = np.array([0., alpha*np.sin(omd*(jj-1)*dt)])
        frcp = np.array([0., alpha*np.sin(omd*jj*dt)])
        k1 = dt*(ffun(xvec[:,jj-1]) + frc) + bvec
        k2 = dt*(ffun(xvec[:,jj-1]+k1)+frcp) + bvecp
        xvec[:,jj] = xvec[:,jj-1] + .5*(k1 + k2)
    return xvec

########################## Plots

def stochastic_linear_plot(tseries):
    plt.plot(tseries[0,:],tseries[1,:],color='k')
    plt.xlabel(r"$x_{1}$")
    plt.ylabel(r"$x_{2}$")
    plt.title("Phase Plot of Stochastic Linear Oscillator")
    plt.show()
    

def stochastic_linear_plot_time(tvals,tseries,ref):
    plt.plot(tvals,tseries[0,:],color='r')
    plt.plot(tvals,ref,color='k')
    plt.ylabel(r"$x_{1}(t)$")
    plt.xlabel(r"$t$")
    plt.show()


def stochastic_linear_plot_FFT(redseries):
    plt.plot(np.ma.log10(redseries),color='k')
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\log_{10}(\hat{S}(\omega))$")
    plt.show()
    

######################################################################
#                                                                    #
####### Stochastically Perturbed, Forced Van der Pol Oscillator ######
#                                                                    #
######################################################################


def stochastic_VDP_plot(nltseries):
    plt.plot(nltseries[0,:],nltseries[1,:],color='k')
    plt.xlabel(r"$x_{1}$",fontsize=32)
    plt.ylabel(r"$x_{2}$",fontsize=32)
    plt.title("Stochastic VDP Oscillator",fontsize=36)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.show()


def stochastic_VDP_plot_time(tvals,nltseries,axis_val):
    plt.plot(tvals,nltseries,color='k')
    plt.title("Time Series for $x_n(t)$, $n=$"+axis_val,fontsize=36)
    plt.ylabel(r"$x_{n}(t)$",fontsize=32)
    plt.xlabel(r"$t$",fontsize=32)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.show()


def stochastic_VDP_plot_FFT(nlredseries,axis_val):
    plt.plot(np.ma.log10(nlredseries),color='k')
    plt.title("Magnitude FFT for $x_n(t)$, $n=$"+axis_val,fontsize=36)
    plt.xlabel(r"$\omega$",fontsize=32)
    plt.ylabel(r"$\log_{10}(\hat{S}(\omega))$",fontsize=32)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.show()


######################################################################
#                                                                    #
####                 Actual Signal Manipulations                 #####
#                                                                    #
######################################################################


def signal_type(sig_name,t0,tf,sr,sval,mu,alpha,view_other):
    """
    sig_name: e.g. 'harmonic', 'sto_linear', 'VDP'
    
    t0,tf: initial and final times
    
    sr: scale rate, multiplied by (tf-t0), increasing point total
    
    sval: sigma for stochastic noise, i.e. higher sigma = more noise
    
    mu: for VDP parameter
    
    alpha: for VDP parameter
    
    view_other: this will show the plots here
    """
    
    if t0>tf:
        sys.exit("Final time, tf, needs to be greater than initial time, t0")
    
    if sig_name != "not_used": # makes seeing errors easier by tricking the program into thinking the packages will be installed
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        import random
        import scipy
        
    plt.rcParams['figure.figsize'] = [10, 10] #defaulting to larger signal size
    
    if sig_name == 'harmonic':
        # t0 = 0
        # tf = 3
        # sr = 101
        density = ((tf-t0)*sr)
        if density % 2 == 0: # making sure density is odd for the fft stuff below
          density = density + 1
        
        t = np.linspace(t0,tf,density) #samp rate of 60kHz
        S = np.sin(2*math.pi*t)+np.sin(5*math.pi*t)+np.sin(7*math.pi*t)
        
        plt.subplot(1,3,1)
        plt.plot(t,S)
        plt.title('Signal',fontsize=36)
        plt.xlabel('$t$',fontsize=32)
        plt.ylabel('$R(t)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
        
        # 2. Adding some noise
        
        # Adding some noise to the psuedo sample
        random.seed(3)
        for ii in range(len(S)):
          S[ii] = S[ii]+random.uniform(0,2)
        
        plt.subplot(1,3,2)
        plt.plot(t,S)
        plt.title('w/ Noise',fontsize=36)
        plt.xlabel('$t$',fontsize=32)
        plt.ylabel('$R(t)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
        
        # 3. FFT and Manipulations
        
        S_fft = scipy.fft.fftshift(scipy.fft.fft(S))
        S_fft = abs(S_fft)
        
        # Uncomment to see the magnitude FFT as is
        plt.plot(t,S_fft)
        plt.title('FFT of $R(t)$',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('$\hat{R}(\omega)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
        
        # Removing half of the signal
        start = math.floor(len(S)/2) # index of the first value to pick up
        S_new = np.zeros((start+1))
        omega = np.arange(0.0,(start+1))
        
        
        for ii in range(len(omega)):
          S_new[ii] = S_fft[(start+ii)]
        
        plt.rcParams['figure.figsize'] = [10, 10]
        omega = omega[:-1] # removing the last omega element (these are dumb I know but i'm not changing them...)
        omega = omega-np.mean(omega).item()
        S_new = S_new[1:]  # removing the first S_new element (these are dumb I know but i'm not changing them...)
        
        plt.subplot(1,3,3)
        plt.plot(omega,S_new)
        plt.title('Shifted FFT',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('$\hat{R}(w)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
        
        TA = 0 #used for the other calls, made zero here because i'm lazy and thats easier...
        
        return omega, S_new, TA
    
    if sig_name == 'harmonic_full':
        density = ((tf-t0)*sr)
        if density % 2 == 0: # making sure density is odd for the fft stuff below
          density = density + 1
        
        t = np.linspace(t0,tf,density) #samp rate of 60kHz
        S = np.sin(2*math.pi*t)+np.sin(5*math.pi*t)+np.sin(7*math.pi*t)
        
        plt.subplot(1,2,1)
        plt.plot(t,S)
        plt.title('Signal',fontsize=36)
        plt.xlabel('$t$',fontsize=32)
        plt.ylabel('$R(t)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
        
        # 2. Adding some noise
        
        # Adding some noise to the psuedo sample
        random.seed(3)
        for ii in range(len(S)):
          S[ii] = S[ii]+random.uniform(-1,1)
        
        plt.subplot(1,2,2)
        plt.plot(t,S)
        plt.title('Noise Added',fontsize=36)
        plt.xlabel('$t$',fontsize=32)
        plt.ylabel('$R_N(t)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
        
        # 3. FFT and Manipulations
        
        S_fft = scipy.fft.fftshift(scipy.fft.fft(S))
        S_fft = abs(S_fft)
        
        # Uncomment to see the magnitude FFT as is
        omega = t-np.mean(t)
        plt.plot(omega,S_fft)
        plt.title('FFT of $R(t)$',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('$\hat{R}(\omega)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
        
        # Removing half of the signal
        # start = math.floor(len(S)/2) # index of the first value to pick up
        # S_new = np.zeros((start+1))
        # omega = np.arange(0.0,(start+1))
        # for ii in range(len(omega)):
        #   S_new[ii] = S_fft[(start+ii)]
        
        plt.rcParams['figure.figsize'] = [10, 10]
        #omega = omega[:-1] # removing the last omega element (these are dumb I know but i'm not changing them...)
        #omega = omega-np.mean(omega).item()
        #S_new = S_new[1:]  # removing the first S_new element (these are dumb I know but i'm not changing them...)
        
        # padding with zeros
        domega = omega[1]-omega[0]
        omega = np.pad(omega,(1,1),'constant',constant_values=(omega[0]-domega,omega[-1]+domega))
        #omega = np.pad(omega,(1,1),'constant',constant_values=(omega[0]-domega,omega[-1]+domega))
        S_fft = np.pad(S_fft,(1,1),'constant',constant_values=(0,0))
        
        #plt.subplot(1,2,2)
        plt.plot(omega,S_fft)
        plt.title('Full FFT',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('$\hat{R}(w)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
        
        TA = 0 #used for the other calls, made zero here because i'm lazy and thats easier...
        return omega, S_fft, TA
    
    
    if sig_name == 'sto_linear':
        Pi = np.pi
        om = 2.*Pi
        omd = 8.*Pi
        #alpha = 20. # magnitude of forcing creates more pronounced beating pattern
        # tf = 6.
        Nstps = int(2**10)
        #sval = .05 # this parameter controls the volume of the noise.  can produce dramatic differences in outcome.  
        if t0 != 0:
            print("Warning, suggest looking at impact of not using t0=0")
        tvals = np.linspace(t0,tf,Nstps+1)
        ref = np.cos(om*tvals) + alpha/( om**2. - omd**2. )*(np.sin(omd * tvals) - omd/om * np.sin(om * tvals))
        
        linosc = lambda x: lin_osc(x, om)
        tseries = random_integrator(omd,tf,Nstps,sval,alpha,linosc)
        ftseries = fft(tseries[0,:])
        redseries = np.abs(np.fft.fftshift(ftseries)) #np.abs(ftseries[:int(Nstps/2)])/np.sqrt(Nstps)
        S_new = redseries
        if len(S_new) % 2 == 0:
            S_new = S_new[:-1]
        # plots
        stochastic_linear_plot(tseries)
        stochastic_linear_plot_time(tvals,tseries,ref)
        stochastic_linear_plot_FFT(redseries)
        omega = np.arange(-int(np.floor(len(S_new)/2)),int(np.floor(len(S_new)/2)+1))
        TA = 0  #used for the other calls, made zero here because i'm lazy and thats easier...
        return omega, S_new, TA 
    
    
    if sig_name == 'VDP':
        Pi = np.pi
        omd = 2.*Pi
        #alpha = 2. # magnitude of forcing creates more complex dynamics
        #tf = 20.
        Nstps = int(2**9) #curtis had this at 2^11
        #sval = .05 # this parameter controls the volume of the noise.  can produce dramatic differences in outcome.  
        #mu = 1.
        
        tvals = np.linspace(0,tf,Nstps+1)
        vdposc = lambda x: vanderpol(x, mu)
        nltseries = random_integrator(omd,tf,Nstps,sval,alpha,vdposc)
        #fnltseries = fft(nltseries[0,:]) # this will grab specifically the 'x1' variable
        fnltseries = fft(nltseries[1,:]) # this will grab specifically the 'x2' variable
        #nlredseries = np.abs(fnltseries[:int(Nstps/2)])/np.sqrt(Nstps)
        nlredseries = fftshift(np.abs(fnltseries))/np.sqrt(Nstps)
        S_new = nlredseries
        # plots
        stochastic_VDP_plot(nltseries)
        stochastic_VDP_plot_time(tvals,nltseries)
        stochastic_VDP_plot_FFT(nlredseries)
        if len(S_new) % 2 == 0:
            S_new = S_new[:-1]
            tvals = tvals[:-1]
            #nlredseries = nlredseries[:-1]
        omega = np.arange(-int(np.floor(len(S_new)/2)),int(np.floor(len(S_new)/2)+1))*math.pi/(len(S_new)/2) #get rid of *math.pi/...
        #temp = np.linspace(tvals[0],tvals[-1],len(nlredseries))
        #omega = temp-np.mean(temp)
        return omega, S_new, tvals
    
    
    if sig_name == 'fsig2':
        samps = 2001
        freq = 1/samps
        t = np.arange(0,1,freq) 
        S = np.zeros(len(t))
        for T in range(len(t)):
          if t[T] > 0.5:
            S[T] = 6*t[T]+np.cos(10*math.pi*t[T]+10*math.pi*t[T]**2)+np.cos(80*math.pi*t[T]-15*math.pi)
          else:
            S[T] = 6*t[T]+np.cos(10*math.pi*t[T]+10*math.pi*t[T]**2)+np.cos(60*math.pi*t[T])
        
        if view_other == True:
            # Plot of the signal
            plt.plot(t,S)
            plt.title('Signal',fontsize=36)
            plt.xlabel('$t$',fontsize=32)
            plt.ylabel('$f_{sig2}(t)$',fontsize=32)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid()
            plt.show()
        
        # 3. FFT and Manipulations
        S_fft = scipy.fft.fftshift(scipy.fft.fft(S))
        S_fft = abs(S_fft)
        
        # Uncomment to see the magnitude FFT as is
        omega = np.linspace(-math.pi,math.pi,samps)
        if view_other == True:
            plt.plot(omega,S_fft)
            plt.title('FFT of $f_{sig2}(t)$',fontsize=36)
            plt.xlabel('$\omega$',fontsize=32)
            plt.ylabel('$\hat{f}_{sig2}(t)(\omega)$',fontsize=32)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid()
            plt.show()
        
        # Uncomment to see the magnitude FFT as is
        omega = np.linspace(-math.pi,math.pi,samps)
        if view_other == True:
            plt.plot(omega,S_fft)
            plt.title('FFT of $f_{sig2}(t)$',fontsize=36)
            plt.xlabel('$\omega$',fontsize=32)
            plt.ylabel('$\hat{f}_{sig2}(t)(\omega)$',fontsize=32)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlim([-.5,.5])
            plt.grid()
            plt.show()
        
        plt.rcParams['figure.figsize'] = [10, 10]
        # padding with zeros
        domega = omega[1]-omega[0]
        omega = np.pad(omega,(1,1),'constant',constant_values=(omega[0]-domega,omega[-1]+domega))
        #omega = np.pad(omega,(1,1),'constant',constant_values=(omega[0]-domega,omega[-1]+domega))
        S_fft = np.pad(S_fft,(1,1),'constant',constant_values=(0,0))
        
        #plt.subplot(1,2,2)
        if view_other == True:
            plt.plot(omega,S_fft)
            plt.title('Full FFT',fontsize=36)
            plt.xlabel('$\omega$',fontsize=32)
            plt.ylabel('$\hat{f}_{sig2}(\omega)$',fontsize=32)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid()
            plt.show()
        
        TA = 0 #used for the other calls, made zero here because i'm lazy and thats easier...
        return omega, S_fft, TA
    

    if sig_name == 'ECG':
        samps = 4170
        freq = 1/samps
        omega = np.linspace(-math.pi,math.pi,samps) 
        mat = scipy.io.loadmat('C:\\Users\\Sclin\\MATLAB Drive\\Research\\Normalize_Flow\\ECG\\fft_of_ECG_signal.mat')
        S_fft = mat['ECG_signal'].squeeze()
        plt.rcParams['figure.figsize'] = [10, 10]
        
        # image of the FFT of the signal
        if view_other == True:
            plt.plot(omega,S_fft)
            plt.title('FFT of ECG Signal',fontsize=36)
            plt.xlabel('$\omega$',fontsize=32)
            plt.ylabel('$\hat{f}(\omega)$',fontsize=32)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid()
            plt.show()
        
        # padding with zeros
        domega = omega[1]-omega[0]
        omega = np.pad(omega,(1,1),'constant',constant_values=(omega[0]-domega,omega[-1]+domega))
        S_fft = np.pad(S_fft,(1,1),'constant',constant_values=(0,0))
        
        #plt.subplot(1,2,2)
        if view_other == True:
            plt.plot(omega[int(samps/2):],S_fft[int(samps/2):])
            plt.title('Half FFT of ECG Signal',fontsize=36)
            plt.xlabel('$\omega$',fontsize=32)
            plt.ylabel('$\hat{f}(\omega)$',fontsize=32)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid()
            plt.show()
        
        TA = 0 #used for the other calls, made zero here because i'm lazy and thats easier...
        return omega, S_fft, TA 
    
    
def signal_interpolation(omega_in,S_in,folder,png,view_other):
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator
    pdf_curve = PchipInterpolator(omega_in, S_in)
    omeg_output = np.linspace(omega_in[0], omega_in[-1], 3000)
    if view_other == True:
        plt.plot(omeg_output, pdf_curve(omeg_output))
        plt.plot(omega_in, S_in, 'o', label='data')
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('$\hat{R}(\omega)$',fontsize=32)
        plt.title('Interpolation of Input Data',fontsize=36)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        pic_file_name = folder+ "Interpolated Signal" + png
        plt.savefig(pic_file_name)
        plt.show()
    return omeg_output, pdf_curve
    
    
def gauss_over_fft(omega,omeg_new,pdf_curve,M,sig,folder,png,signame,view_other):
    # NOTE - This is the weak link in the chain at the moment. I don't know how to automate this quite yet
    # I'm sure there's a simple enough way but I don't have it yet so this is guess and check until it works out
    # note that the gaussian curve must be stirctly greater than the fft interpolated curve (orange > blue)
    # mu_old = 0.0
    # for ii in range(len(omega)):
    #     mu_new = pdf_curve(omega[ii])
    #     if mu_old < mu_new:
    #         mu = omega[ii]
    #         mu_old = mu_new
    mu = 0
    #M = 28000
    #sig = 70
    test_gauss = M*(1/(sig*math.sqrt(2*math.pi)))*np.exp(-(omega-mu)**2/(2*sig**2))

    plt.rcParams['figure.figsize'] = [20, 10]
    if view_other == True:
        plt.subplot(1,2,1)
        plt.loglog(omeg_new, pdf_curve(omeg_new))
        plt.plot(omega,test_gauss)
        plt.title('Gaussian above $\hat{R}(\omega)$',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('$\hat{R}(\omega)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        pic_file_name = folder+ "Gauss over FFT" + png
        plt.savefig(pic_file_name)
        
        plt.subplot(1,2,2)
        plt.plot(omeg_new, pdf_curve(omeg_new))
        plt.plot(omega,test_gauss)
        plt.title('Gaussian above $\hat{R}(\omega)$',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('$\hat{R}(\omega)$',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        pic_file_name = folder+ "Gauss over FFT" + png
        plt.savefig(pic_file_name)
        plt.show()

    # Make sure the gaussian is larger than the signal everywhere
    # if not, it will halt the program and create an error

    test_gauss = M*(1/(sig*math.sqrt(2*math.pi)))*np.exp(-(omeg_new-mu)**2/(2*sig**2))
    abs_greater_test = test_gauss-pdf_curve(omeg_new)
    min_test = np.min(abs_greater_test)
    if min_test <= 0:
        print('max of S_new: ',max(pdf_curve(omega)))
        print('omega limit: ',omega[-1])
        print('Max of S_new([0:10]): ',max(pdf_curve(omega[0:10])))
        sys.exit('Error: Need to make sure the Gaussian Curve is strictly greater than the FFT data curve. Do this by increasing M and sig.')
    else: 
        print('Success. Gaussian Curve is strictly greater than FFT data curve.')
    return test_gauss


def signal_to_samples(omega,N,M,sig,pdf_curve,folder,png,sig_name): 
    print('')
    print('Beginning Curve to Samples Process. This may take a minute...')
    
    if sig_name != 'sto_linear' and sig_name != 'fsig2' and sig_name != 'ECG' and sig_name != 'VDP':
        def gauss_curve_M(point,M,sig):
          output = M*(1/(sig*math.sqrt(2*math.pi)))*np.exp(-point**2/(2*sig**2))
          return output    
        # Generating the Points
        # Creating a sampling now for the above pdf
        #N number of sample data points attempts
        P = np.zeros(N)
        random.seed(3)
        for ii in range(N):
          P[ii] = random.gauss(0,sig)
        ps = []
        accept = 0
        reject = 0
        L = omega[0]
        R = omega[-1]
        for ii in range(N):
          if P[ii] >= L and P[ii] <= R:
            prob = pdf_curve(P[ii])/(gauss_curve_M(P[ii],M,sig))
            random.seed(ii)
            test = random.uniform(0,1)
            if test <= prob:
              ps.append(P[ii])
              accept = accept + 1
            else:
              reject = reject + 1
          else: 
            reject = reject + 1
        print('accepted total: ', accept)
        print('rejected total: ', reject)
        # Show the Output
        plt.subplot(1,2,2)
        plt.title('Histogram of FFT Signal',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('Counts',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.hist(ps,bins=300)
        pic_file_name = folder+ "Histogram of FFT" + png
        plt.savefig(pic_file_name)
        plt.show()
    
    if sig_name == 'sto_linear' or sig_name == 'fsig2' or sig_name == 'ECG' or sig_name == 'VDP':
        def gauss_curve_M(point,M,sig):
          output = M*(1/(sig*math.sqrt(2*math.pi)))*np.exp(-point**2/(2*sig**2))
          return output    
        # Generating the Points
        # Creating a sampling now for the above pdf
        #N number of sample data points attempts
        #random.seed(3)
        ps = np.array([])
        accept = 0
        reject = 0
        L = 0
        R = omega[-1]
        P = np.random.normal(0,sig,int(N/2))
        ticker = 0
        while accept < int(N/2):
            pt = np.random.normal(0,sig)
            if pt >= L and pt <= R:
                prob = pdf_curve(pt)/(gauss_curve_M(pt,M,sig))
                test = random.uniform(0,1)
                if test <= prob:
                    ps = np.append(ps,pt)
                    accept = accept + 1
                    ticker = ticker + 1
                else:
                    reject = reject + 1
                if accept % 100 == 0: 
                    print("accepted: ",accept," of ",int(N/2))
        
        print('accepted total: ', accept)
        print('rejected total: ', reject)
        # Show the Output
        ps_neg = (ps*(-1)).tolist()
        ps_pos = ps.tolist()
        ps = ps_neg+ps_pos
        plt.subplot(1,2,2)
        plt.hist(ps,bins=300)
        plt.title('Histogram of FFT Signal',fontsize=36)
        plt.xlabel('$\omega$',fontsize=32)
        plt.ylabel('Counts',fontsize=32)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        pic_file_name = folder+ "Histogram of FFT" + png
        plt.savefig(pic_file_name)
        plt.show()
        ps = np.array(ps)
        
    return ps, accept


def signal_to_samples_symmetric(omega,N,M,sig,pdf_curve,folder,png,sig_name): 
    print('')
    print('Beginning Curve to Samples Process. This may take a minute...')
    def gauss_curve_M(point,M,sig):
      output = M*(1/(sig*math.sqrt(2*math.pi)))*np.exp(-point**2/(2*sig**2))
      return output
    # shows those last points have a gap and test gauss is higher
    # print(test_gauss[-1])           #  16.064432176361244
    # print(pdf_curve(omeg_new[-1]))  #  15.511804033581058
    
    # Generating the Points
    # Creating a sampling now for the above sin pdf
    #N = 5000000 # number of sample data points attempts
    P = np.zeros(N)
    random.seed(3)
    for ii in range(N):
      P[ii] = random.gauss(0,sig)
    ps = []
    accept = 0
    reject = 0
    L = omega[0]
    R = omega[-1]

    for ii in range(N):
      if P[ii] >= L and P[ii] <= R:
        prob = pdf_curve(P[ii])/(gauss_curve_M(P[ii],M,sig))
        random.seed(ii)
        test = random.uniform(0,1)
        if test <= prob:
          ps.append(P[ii])
          accept = accept + 1
        else:
          reject = reject + 1
      else: 
        reject = reject + 1
    print('accepted total: ', accept)
    print('rejected total: ', reject)

    # Show the Output
    plt.subplot(1,2,2)
    plt.title('Histogram of FFT Signal',fontsize=36)
    plt.xlabel('$\omega$',fontsize=32)
    plt.ylabel('Counts',fontsize=32)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.hist(ps,bins=300)
    pic_file_name = folder+ "Histogram of FFT" + png
    plt.savefig(pic_file_name)
    plt.show()
    
    # plt.subplot(1,3,3)
    # plt.plot(omega,S_new)
    # plt.title('Original FFTd signal')
    # plt.xlabel('w')
    # plt.ylabel('S(w)')
    # plt.show()
    return ps, accept


def DHG_samples(N,mu1,mu2,s1,s2):
    N = int(np.round(N/2))
    if s1 == 0 or s2 == 0:
        print('Error! Must have sigma values be greater than zero')
    temp1 = np.random.normal(mu1,s1,(1,N))
    temp2 = np.random.normal(mu2,s2,(1,N))
    temp  = np.concatenate((temp1,temp2),axis=None)
    return temp, N


def signal_interpolation_otsu(omega_in,S_in):
    from scipy.interpolate import interp1d
    pdf_curve = interp1d(omega_in, S_in,kind='linear',fill_value="extrapolate")
    omeg_output = np.linspace(omega_in[0], omega_in[-1], 3000)
    plt.plot(omeg_output, pdf_curve(omeg_output))
    plt.plot(omega_in, S_in, 'o', label='data')
    plt.xlabel('$\omega$',fontsize=32)
    plt.ylabel('$\hat{R}(\omega)$',fontsize=32)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Interpolation of Input Function/Data',fontsize=36)
    plt.grid()
    plt.show()
    return omeg_output, pdf_curve


def file_save_name(sig_name):
    #sypder
    if sig_name == 'harmonic':
        folder = "F:\\Research_Thesis\\NormFlows\\Image Runs\\harmonic\\Im_"
    if sig_name == 'harmonic_full':
        folder = "F:\\Research_Thesis\\NormFlows\\Image Runs\\harmonic_full\\Im_"
    elif sig_name == 'sto_linear':
        folder = "F:\\Research_Thesis\\NormFlows\\Image Runs\\sto_linear\\Im_"
    elif sig_name == 'VDP':
        folder = "F:\\Research_Thesis\\NormFlows\\Image Runs\\VDP\\Im_"
    elif sig_name == 'DHG':
        folder = "F:\\Research_Thesis\\NormFlows\\Image Runs\\DHG\\Im_"
    elif sig_name == 'fsig2':
        folder = "F:\\Research_Thesis\\NormFlows\\Image Runs\\fsig2\\Im_"
    elif sig_name == 'ECG':
        folder = "F:\\Research_Thesis\\NormFlows\\Image Runs\\ECG\\Im_"
    return folder


def file_save_name_vscode(sig_name):
    #vscode
    if sig_name == 'harmonic':
        folder = "F:/Research_Thesis/NormFlows/Image Runs/harmonic/Im_"
    if sig_name == 'harmonic_full':
        folder = "F:/Research_Thesis/NormFlows/Image Runs/harmonic_full/Im_"
    elif sig_name == 'sto_linear':
        folder = "F:/Research_Thesis/NormFlows/Image Runs/sto_linear/Im_"
    elif sig_name == 'VDP':
        folder = "F:/Research_Thesis/NormFlows/Image Runs/VDP/Im_"
    elif sig_name == 'DHG':
        folder = "F:/Research_Thesis/NormFlows/Image Runs/DHG/test/Im_"
    elif sig_name == 'fsig2':
        folder = "F:/Research_Thesis/NormFlows/Image Runs/fsig2/test/Im_"
    elif sig_name == 'ECG':
        folder = "F:/Research_Thesis/NormFlows/Image Runs/ECG/test/Im_"
    return folder