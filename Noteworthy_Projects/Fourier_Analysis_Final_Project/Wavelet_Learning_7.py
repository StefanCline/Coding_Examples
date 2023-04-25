# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:22:46 2023

@author: Sclin
"""

for BIG_OUTER_LOOP in range(28,32):
    print('')
    print("Starting Outer Loop: ",BIG_OUTER_LOOP)
    save_folder_0 = "F:\Research_Thesis\Fourier_Final_Project\Runs_23APR_B\Loop_"
    save_folder_1 = str(BIG_OUTER_LOOP)
    save_folder   = save_folder_0+save_folder_1 
    
    import numpy as np
    #import torch
    import matplotlib.pyplot as plt
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}
    plt.rc('font', **font)
    plt.style.use('default')
    import math
    import random
    import pickle
    import scipy
    #from scipy import signal
    import scipy.integrate as integrate
    
    
    ##########################################################################################
    #############         EARLY FUNCTIONS REQUIRED FOR PLOTS         #########################
    ##########################################################################################
    
    # using only numpy to do a forward pass as proof of concept/code working correctly
    def phi_j_np(a_to_j,t,two,one,oh,eps,A,B,tau):
      denominator = np.exp(-(t-tau)**two/(a_to_j**two))*(eps*(t-tau)/a_to_j+A*np.sin(B*(t-tau)/a_to_j))**two
      return one/np.sqrt(a_to_j)*(np.exp(-oh*(t-tau)**two/a_to_j**two)*A*np.sin(B*(t-tau)/a_to_j))/(np.sqrt(integrate.simpson(denominator,dx=delta_t)))
    
    
    # used to do the circular convolution (look up the theory in jerome gilles' lecture notes for Fourier Analysis)
    def conv_circ(signal,ker):
      return np.real(scipy.fft.ifft(np.fft.fft(signal)*scipy.fft.fft(ker)))
    
    
    def mother_wavelet(eps,t,A,B,delta_t,two): 
      # wavelet general shape exp*A*sin(Bt) divided by L2 Norm
      denominator = np.exp(-t**two)*(eps*t+A*np.sin(B*t))**two
      return np.exp(-t**two)*(eps*t+A*np.sin(B*t))/(np.sqrt(integrate.simpson(denominator,dx=delta_t)))
    
    
    ##########################################################################################
    #############         GRABBING SAMPLE OF THE SEISMIC DATA        #########################
    ##########################################################################################
    

    filename = 'C:/Users/Sclin/OneDrive/Documents/School/06_SDSU_Grad_School_Docs/04_Research with Curtis/PythonCode/Fourier_Final_Project/seismic.pkl'
    infile = open(filename, 'rb')
    seismic_data = pickle.load(infile)
    seismic_data = seismic_data[0][:]
    tt = np.linspace(0,1,len(seismic_data))
    plt.plot(tt,seismic_data)
    plt.title("Full Seismic Data Import")
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.show(); 
    del tt
    
    
    ##########################################################################################
    #############         CRITICAL VALUES AND HYPERPARAMETERS        #########################
    ##########################################################################################
    
    # CRITICAL PARAMS FOR SPEED BELOW! ###################################################################
    # the jdxd density size of the wavelet transform box 
    # d = len(f_t)  # time length of the image (horizontal axis)
    d = 351 # make this odd for simpson's rule
    if BIG_OUTER_LOOP % 4 == 0:
        start = random.randint(0,30000)
        f_t = seismic_data[start:(start+d)]
    jd = 40 # scalespace axis (horizontal)
    eps = 0
    a = 1.20
    plotcolor = 'seismic' # 'magma' #
    ts = 0.0
    tf = 2.0*math.pi
    t = np.linspace(ts,tf,d)
    m = t
    osc = 4.0
    #f_t = np.sin(osc*t**(2.0)) # remove this to go back to the seismic example
    delta_t = abs(t[0]-t[1])
    
    js = 0
    jf = jd
    j = np.linspace(js,(jd-1),jd)
    # CRITICAL PARAMS FOR SPEED ABOVE! ##################################################################
    
    beta = 12000.0 # error you want the function to have for picking A or B = 0
    limit = 1.0/0.5 # point where it starts accruing error for giving smaller numbers for A and B
                    # e.g. .5 means that A and B will likely remain outside of [-0.5,0.5]
    
    
    ##########################################################################################
    #############         PLOTTING SEISMIC SNIP AND TEST WAVELET      ########################
    ##########################################################################################
    
    
    # plot of the function
    plt.plot(t,f_t)
    plt.title("Input Signal")
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    sample_save_name =  save_folder_0+save_folder_1+"\seismic_sample.png"
    plt.savefig(sample_save_name)
    plt.show();
    del sample_save_name

    # seeing what our wavelet looks like with the given values below of A and B
    A = 1.0
    B = 50
    two = 2.0
    
    phi = mother_wavelet(eps,t,A,B,delta_t,two)
    plt.plot(t,phi)
    plt.grid()
    plt.title("Wavelet Example: A="+str(A)+", B="+str(B))
    plt.xlabel("Time (t)")
    plt.ylabel("Phi(t)")
    plt.show();
    del phi
    
    
    ##########################################################################################
    #############                TESTING OUT A DWT                    ########################
    ##########################################################################################
    
    
    one = 1.0
    two = 2.0
    oh = 0.5
    neg_two = -2.0
    thd2 = 1.5
    neg_oh = -0.5
    
    DWT = np.zeros((jd,d))
    for JJ in range(len(j)):
      a_to_j = a**JJ
      for tau in range(len(t)):
        phi_j = phi_j_np(a_to_j,t[tau],two,one,oh,eps,A,B,t)
        DWT[JJ][:] = conv_circ(phi_j,f_t)
    
    #plt.imshow(DWT,extent=[tt[0],tt[-1],0,jj[-1]],aspect='auto')
    plt.imshow(DWT)
    plt.xlabel('Time (t)')
    plt.ylabel('Scale (j)')
    plt.title('Example Forward Pass of the Field')
    plt.show()
    
    
    ##########################################################################################
    #############         RANDOM A and B with SHELL GRAD_GAMMA        ########################
    ##########################################################################################
    
    
    # Random starts between values in random.uniform(a,b)
    # here we want to ensure we're on all parts of the large divide when it comes to 
    #     parameterized A/B space (the Q_Gamma functions will prevent crossover)
    if BIG_OUTER_LOOP % 4 == 0: # both positive
        A = np.random.uniform(1.0,3.0)
        B = np.random.uniform(1.0,50)
    elif BIG_OUTER_LOOP % 4 == 1: #A pos B neg
        A = np.random.uniform(1.0,3)
        B = np.random.uniform(-50,-1.0)
    elif BIG_OUTER_LOOP % 4 == 2: #A neg B pos
        A = np.random.uniform(-3.0,-1.0)
        B = np.random.uniform(1.0,50)
    else: # both neg
        A = np.random.uniform(-3.0,-1.0)
        B = np.random.uniform(-50,-1.0)
    
    print('')
    print("Initial A = ", A)
    print("Initial B = ", B)
    
    print('')
    initial_params = [A,B]
    
    # Back prop shells for nabla C
    A_grad = np.zeros((jd,d))
    B_grad = A_grad
    
    
    ##########################################################################################
    #############                       SUPPORTING FUNCTIONS          ########################
    ##########################################################################################
    
    ############# Essential Forward Calls ###################################

    # using only numpy to do a forward pass as proof of concept/code working correctly
    def phi_j_np(a_to_j,t,two,one,oh,eps,A,B,tau):
      denominator = np.exp(-(t-tau)**two/(a_to_j**two))*(eps*(t-tau)/a_to_j+A*np.sin(B*(t-tau)/a_to_j))**two
      return one/np.sqrt(a_to_j)*(np.exp(-oh*(t-tau)**two/a_to_j**two)*A*np.sin(B*(t-tau)/a_to_j))/(np.sqrt(integrate.simpson(denominator,dx=delta_t)))
    
    
    # used to do the circular convolution (look up the theory in jerome gilles' lecture notes for Fourier Analysis)
    # def conv_circ(signal,ker):
    #   return np.real(scipy.fft.ifft(np.fft.fft(signal)*scipy.fft.fft(ker)))
    
    
    def mother_wavelet(eps,t,A,B,delta_t,two): 
      # wavelet general shape exp*A*sin(Bt) divided by L2 Norm
      denominator = np.exp(-t**two)*(eps*t+A*np.sin(B*t))**two
      return np.exp(-t**two)*(eps*t+A*np.sin(B*t))/(np.sqrt(integrate.simpson(denominator,dx=delta_t)))
    
        
    ############# Back Prop Function ########################################
    
    # make sure limit is positive
    # make sure limit is equal to 1/limit 
    def d_mesa_gamma(Gamma,beta,limit,neg_two,two): 
      if Gamma <= -limit or Gamma >= limit:
        return 0.0
      else:
        return neg_two*(limit**two)*beta*Gamma
    
    
    # num = numerator, dem = denominator
    def EQA(neg_oh,t,tau,two,thd2,a_to_j,A,B,f_t,delta_t,eps):
      Q1 = (t-tau)/a_to_j
      a_to_j2 = a_to_j**two
      S1 = A*np.sin(B*Q1)
      # usin em
      num1 = np.exp(neg_oh*(t-tau)**two/(a_to_j2))*np.sin(B*Q1)*f_t
      dem1_intg = np.exp(-(t-tau)**two/a_to_j2)*(eps*Q1+S1)**two
      dem1_integral = integrate.simpson(dem1_intg,dx=delta_t)
      dem1 = np.sqrt(a_to_j)*np.sqrt(dem1_integral)
      T1 = num1/dem1
      LT = np.exp(neg_oh*(t-tau)**two/(two*a_to_j2))*(eps*Q1+S1)*f_t
      RT_intg = two*np.exp(-(t-tau)**two/a_to_j2)*(eps*Q1+S1)*np.sin(B*Q1)
      RT = integrate.simpson(RT_intg,dx=delta_t)
      dem2_intg = np.exp(-(t-tau)**two/a_to_j2)*(eps*Q1+S1)**two
      dem2 = two*np.sqrt(a_to_j)*(integrate.simpson(dem2_intg,dx=delta_t))**(thd2)
      A_curve = T1-LT*RT/dem2
      return integrate.simpson(A_curve,dx=delta_t)
    
    
    def EQB(neg_oh,t,tau,two,thd2,a_to_j,A,B,f_t,delta_t,eps):
      Q1 = B*(t-tau)/a_to_j
      Q2 = eps*(t-tau)/a_to_j
      Trigs = A*np.sin(Q1)
      Trigc = A*(t-tau)*np.cos(Q1)
      # now to use em
      num1 = np.exp(neg_oh*(t-tau)**two)*Trigc*f_t
      dem1_intg = np.exp(-(t-tau)**two/a_to_j**two)*(Q2+Trigs)**two
      dem1 = a_to_j**thd2*np.sqrt(integrate.simpson(dem1_intg,dx=delta_t))
      LT = np.exp(neg_oh*(t-tau)**two/a_to_j**two)*(Q2+Trigs)*f_t
      RT_intg = two*np.exp(-(t-tau)/a_to_j)*(Q2+Trigs*Trigc)
      RT = integrate.simpson(RT_intg,dx=delta_t)
      dem2_intg = np.exp(-(t-tau)**two/a_to_j**two)*(Q2+Trigs)**two
      dem2 = two*np.sqrt(a_to_j)*(integrate.simpson(dem2_intg,dx=delta_t))**thd2
      B_curve = num1/dem1-LT*RT/dem2
      return integrate.simpson(B_curve,dx=delta_t)
    
    
    ############# Function for Determining Loss ###############################
    
    # make sure limit is positive
    # make sure limit is equal to 1/limit 
    def mesa_gamma(Gamma,beta,limit,two):
      if Gamma <= -limit or Gamma >= limit:
        return 0.0
      else:
        return -(limit*np.sqrt(beta)*Gamma)**two + beta
    
    
    ################# Other random useful functions calls ####################
    def A_to_J(aa,J):
      return np.power(aa,J)
    
    
    def keep_best(A,B,loss_old,loss_new,best_params):
      if abs(loss_new) < abs(loss_old):
        best_params = [A,B]
        return best_params
      else:
        return best_params
        
    
    def lr_reduce(params_mat,ii,lr,ticker):
      if ii > 2:
        if np.array_equal((params_mat[(ii+1)][:]), (params_mat[(ii-1)][:])):
          print("LR reduced at = ",ii)
          zz = 1
        elif np.array_equal((params_mat[(ii+1)][:]), (params_mat[(ii-2)][:])):
          print("LR reduced at = ",ii)
          zz = 1
        elif np.array_equal((params_mat[(ii+1)][:]), (params_mat[(ii-3)][:])):
          print("LR reduced at = ",ii)
          zz = 1
        else:
          zz = 0
        if zz == 1:
            lr = lr/2.0
            ticker = ticker + 1
            print("Ticker at ",ticker," of ", ticker_max, "and LR = ", lr) 
      return lr, ticker
    
    
    def view_grads(A_grad,B_grad,ii,save_folder,save_folder_1):
      names = ["Gradient Map: A","Gradient Map: B"]
      grads = [A_grad,B_grad]
      grads_name = save_folder+str("\Loop_")+save_folder_1+"_GradientView_"+str(ii)+".png"
      for pp in range(2):
        plt.subplot(1,2,(pp+1))
        plt.imshow(grads[pp],cmap=plotcolor)
        plt.title(names[pp])
        plt.xlabel("Time (t)")
        plt.ylabel("Scale (j)")
        plt.colorbar(fraction=0.030,pad=.05)
        plt.tight_layout()
        plt.savefig(grads_name)
      plt.show()
    
    
    ##########################################################################################
    #############                       VISUAL FUNCTIONS              ########################
    ##########################################################################################
    
    # plotting the field generated by the forward pass
    def all_plots(field,t,ii,A,B,loss_vec,params_mat):
      fig, ax = plt.subplots(2, 2)
      plot_field = field
      plot_field = plot_field + abs(plot_field.min())
      plot_field = plot_field/plot_field.max() # norms to 1
      im = ax[0][0].imshow(plot_field,cmap = plotcolor)
      #plt.colorbar(location="left",pad = 0.5,fraction=0.047*im_ratio)
      ax[0][0].set_title("Wavelet Convolution")
      ax[0][0].set_xlabel("Time (t): ["+str(np.round(t[0],2))+","+str(np.round(t[-1],2))+"]")
      ax[0][0].set_ylabel("Scale (j)")
      #fig.colorbar(im, cax=cax, orientation='vertical')
      fig.colorbar(im,fraction=0.028)
    
      #plt.subplot(1,3,2)
      TTT = np.arange(0,(ii+1))
      param_names = ["A","B"]
      for kk in range(2):
        ax[0][1].plot(TTT,params_mat[0:(ii+1),kk],label=param_names[kk])
      ax[0][1].grid()
      ax[0][1].legend(loc="upper left")
      
      #plt.subplot(1,3,3)
      ax[1][0].plot(t,mother_wavelet(eps,t,A,B,delta_t,two))
      ax[1][0].set_title(("Wavelet at Iteration: "+str(ii)))
      ax[1][0].set_xlabel("Time (t)")
      ax[1][0].set_ylabel("Psi(t)")
      ax[1][0].grid()
      
      ax[1][1].plot(np.linspace(1,len(loss_vec),len(loss_vec)),loss_vec)
      ax[1][1].set_title("Loss Curve for j "+str(np.round(jf,2)))
      ax[1][1].set_xlabel("Iterations")
      ax[1][1].set_ylabel("Loss Value")
      ax[1][1].grid()
      
      fig.tight_layout()
      fig.set_figwidth(15)
      #saving picture
      pic_file_name = save_folder+str("\Loop_")+save_folder_1+"_Image_"+str(ii)+".png"
      plt.savefig(pic_file_name)
      plt.show()
    
    ##########################################################################################
    #############                         MAIN FUNCTIONS              ########################
    ##########################################################################################
    
    
    # this is the entire field generated by a forward pass
    # DWT = Digital Wavelet Transform
    def forward_pass(jd,d,j,A,B,t,a,one,two,eps):
      DWT = np.zeros((jd,d))
      for JJ in range(len(j)):
        a_to_j = a**JJ
        for tau in range(len(t)):
          phi_j = phi_j_np(a_to_j,t[tau],two,one,oh,eps,A,B,t)
          DWT[JJ][:] = conv_circ(phi_j,f_t)
      return DWT
    
    
    def loss_calculation(field,A,B,two,neg_two,beta):
      return np.log10(np.power(np.sum(field),two)+mesa_gamma(A,beta,limit,two)+mesa_gamma(B,beta,limit,two))
    
    
    def back_prop(field,one,two,thd2,neg_oh,A,B,f_t,eps,delta_t,m,t,beta,limit):
      for JJ in range(len(j)):
        a_to_j = A_to_J(a,j[JJ])
        for mm in range(len(m)):
          A_grad[JJ][mm] = two*field[JJ][mm]*EQA(neg_oh,m[mm],t,two,thd2,a_to_j,A,B,f_t,delta_t,eps)+d_mesa_gamma(A,beta,limit,neg_two,two)
          B_grad[JJ][mm] = two*field[JJ][mm]*EQB(neg_oh,m[mm],t,two,thd2,a_to_j,A,B,f_t,delta_t,eps)+d_mesa_gamma(B,beta,limit,neg_two,two)
      return A_grad, B_grad
    
    
    def update_parameters(A,B,A_grad,B_grad,lr):
      if np.sum(A_grad) < 0: 
        A = A + lr
      elif np.sum(A_grad) > 0:
        A = A - lr
      if np.sum(B_grad) < 0: 
        B = B + lr
      elif np.sum(B_grad) > 0:
        B = B - lr
      return A, B 
    
    
    def clear_grads(jd,d):
      A_grad = np.zeros((jd,d))
      B_grad = A_grad
      return A_grad, B_grad 
    
    
    ##########################################################################################
    #############                        TRAINING LOOP                ########################
    ##########################################################################################
    
    # Loss vector for the loss curve
    loss_vec = []
    loss_old = 1e40 #arbitrarily huge number so that it gets replaced
    best_params = [A,B]
    
    # Loops and Plots count
    total_loops = 200
    total_images = 40
    display_val = np.ceil(total_loops/total_images)
    
    #Learning Rate and stop point for local minima
    lr = 0.9
    ticker = 0
    ticker_max = 6
    
    # Setting up a tracker for the parameters
    params_mat = np.zeros((1,2)) # 4 for A,B
    params_mat[0][:] = [A,B]
    
    # Ensuring zeroed out gradients to start with
    A_grad, B_grad = clear_grads(jd,d)
    
    # Controlling the while loop generally
    ii = 0
    loss_cap = 30000  #this is log scaled 
    loss_new = loss_cap-1
    iter_cap = 30
    iter_buffer = 10
    
    while ticker < ticker_max and ii < total_loops: #((abs(loss_new)<loss_cap) or ii < iter_buffer) and ii < iter_cap:
      print("Starting Iteration ", ii)
      print("A: ",A)
      print("B: ",B)
      field = forward_pass(jd,d,j,A,B,t,a,one,two,eps)
      loss_new = loss_calculation(field,A,B,two,neg_two,beta)
      A_grad, B_grad = back_prop(field,one,two,thd2,neg_oh,A,B,f_t,eps,delta_t,m,t,beta,limit)
      if (ii % display_val) == 0 or ii == 0 or ii == (total_loops-1):
        view_grads(A_grad,B_grad,ii,save_folder,save_folder_1)
      A,B = update_parameters(A,B,A_grad,B_grad,lr) #add subtract learning rate
      params_mat = np.insert(params_mat,(ii+1),[A,B],axis=0)
      loss_vec = np.append(loss_vec,loss_new) # Adding the loss to a vector for plotting
      if (ii % display_val) == 0 or ii == 0 or ii == (total_loops-1): #
        all_plots(field,t,ii,A,B,loss_vec,params_mat)
      A_grad, B_grad = A_grad*0, B_grad*0
      last_field = field
      field = field*0
      best_params = keep_best(A,B,loss_old,loss_new,best_params) #Saving best loss value
      loss_old = loss_new
      lr, ticker = lr_reduce(params_mat,ii,lr,ticker) #Testing to see if values are stagnant and need to be "kicked" 
      ii = ii + 1
    
    
    ##########################################################################################
    #############                        FINAL PLOTS                  ########################
    ##########################################################################################
    
    print("A = ", A)
    print("B = ", B)
    
    # PARAMETER PLOTS parameter plot over the whole training loop
    T = np.arange(0,(ii+1))
    T_np = t
    param_names = ["A","B"]
    for kk in range(2):
      plt.plot(T,params_mat[:,(kk)],label=param_names[kk])
    plt.grid()
    plt.legend(loc="upper left")
    plt.xlabel("Total Number of Iterations")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Values while Training")
    plt.show();
    
    # WAVELET PLOTS Subplot of initial, best, final wavelets
    fig, ax = plt.subplots(1, 3)
    I = initial_params 
    BP = best_params
    T_np_np = np.linspace(-T_np[-1],T_np[-1],2000)
    ax[0].plot(T_np_np,mother_wavelet(eps,T_np_np,I[0],I[1],delta_t,two)) 
    ax[0].set_ylabel("Mother Wavelet")
    ax[0].set_title("Initial Wavelet")
    ax[0].set_xlabel("Time(t)")
    
    ax[1].plot(T_np_np,mother_wavelet(eps,T_np_np,BP[0],BP[1],delta_t,two)) 
    ax[1].set_title("Best Wavelet")
    ax[1].set_xlabel("Time(t)")
    
    ax[2].plot(T_np_np,mother_wavelet(eps,T_np_np,A,B,delta_t,two)) 
    ax[2].set_title("Final Wavelet")
    ax[2].set_xlabel("Time(t)")
    
    fig.tight_layout()
    fig.set_figwidth(15)
    pic_file_name = save_folder+str("\Loop_")+save_folder_1+"_MotherWavelets.png"
    plt.savefig(pic_file_name)
    plt.show();
    
    ##############################################################
    # WAVETLET TRANSFORM PLOTS Subplot of initial, best, final WTs
    I_P = initial_params 
    I_F = forward_pass(jd,d,j,I_P[0],I_P[1],t,a,one,two,eps) 
    B_P = best_params
    B_F = forward_pass(jd,d,j,B_P[0],B_P[1],t,a,one,two,eps)
    
    fig, ax = plt.subplots(2, 2)
    
    im0 = ax[0][0].imshow(I_F,cmap = plotcolor)
    ax[0][0].set_ylabel("Scale (j)")
    ax[0][0].set_xlabel("Time(t)")
    ax[0][0].set_title("Initial DWT")
    
    im1 = ax[0][1].imshow(B_F,cmap = plotcolor)
    ax[0][1].set_xlabel(" ")
    ax[0][1].set_title("Best DWT")
    
    im2 = ax[1][0].imshow(last_field,cmap = plotcolor)
    ax[1][0].set_xlabel("Time(t)")
    ax[1][0].set_title("Final DWT")
    ax[1][0].set_ylabel("Scale (j)")
    
    im3 = ax[1][1].plot(t,f_t)
    ax[1][1].set_xlabel("Time(t)")
    ax[1][1].set_title("Initial Signal")
    ax[1][1].set_ylabel("Amplitude f(t)")
    
    fig.tight_layout()
    fig.set_figwidth(20)
    fig.colorbar(im0,fraction=0.028,pad=.1,location='right')
    fig.colorbar(im1,fraction=0.028,pad=.1,location='right')
    fig.colorbar(im2,fraction=0.028,pad=.1,location='right')
    pic_file_name = save_folder+str("\Loop_")+save_folder_1+"_WaveletTransforms.png"
    plt.savefig(pic_file_name)
    plt.show();
    del I_F, B_F, I_P, B_P, I
    
    # Save all of the data that can reproduce everything
    pickle_file_name = save_folder+str("\Loop_")+save_folder_1+"_Pickle.pkl"
    with open(pickle_file_name,'wb') as FFF:
      pickle.dump(initial_params,FFF)
      pickle.dump(params_mat,FFF)
      pickle.dump(loss_old,FFF)
      pickle.dump(loss_vec,FFF)
      pickle.dump(best_params,FFF)
      pickle.dump(BIG_OUTER_LOOP,FFF)
    
    # Finally, clearing out variables to make sure they don't interact
    # with the next loop of the code
    del initial_params, params_mat, best_params, A, B
    

print('')
print('')
print("Program Complete")
print('')
print('')
    

    