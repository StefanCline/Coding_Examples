# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:41:35 2023

@author: Sclin
"""

# dump for Norm Flows project




# DUMP 

# INFO ON BIJECTIVE PYRO TRANSFORMS

# 1. AffineAutoregressive
# 2. AffineCoupling
# 3. BatchNorm (used with `log_prob`, helps with numerical instability)
# 4. BlockAutoregressive
# 5. ConditionalAffineAutoregressive
# 6. ConditionalAffineCoupling
# 7. ConditionalGeneralizedChannelPermute
# 8. ConditionalHouseholder
# 9. ConditionalMatrixExponential
# 10. ConditionalNeuralAutoregressive
# 11. ConditionalPlanar (use with conditional transformed distribution)
# 12. ConditionalRadial (no analytical inverse but done computationally?)
# 13. ConditionalSpline
# 14. ConditionalSplineAutoregressive
# 15. GeneralizedChannelPermute (do not use, 2D only)
# 16. Householder*
# 17. MatrixExponential* 
# 18. NeuralAutoregressive
# 19. Planar*
# 20. Polynomial* (use: `from pyro.nn import AutoRegressiveNN`)
# 21. Radial* 
# 22. Spline*
# 23. SplineAutoregressive
# 24. SplineCoupling
# 25. Sylvester*


# MU_0 = torch.tensor([0.0]).to(device)
# ForwardData_mu0 = []
# for ii in range(len(transforms)):
#   if ii == 0: 
#     ForwardData_mu0 = transforms[ii](MU_0.unsqueeze(1))
#   else:
#     ForwardData_mu0 = transforms[ii](ForwardData_mu0)

# kstar_forward = ForwardData_mu0

# print('')
# print('')
# print('kstar from Gaussian through F. Otsu with one k is at mu=0. Here, mu=0, so kstar= ',kstar_forward)
# print('')
# print('')

# sigdata = signal_data_points.squeeze(1).cpu().detach().numpy()
# kstar = custom_otsu(sigdata,1) #recommend 1 for speed, 2 is a lot, 3 is crazy

# print('')
# print('')
# print('kstar determined by doing Otsu on the signal data: ',kstar)
# print('')
# print('')

# print('')
# print('')
# print('Difference between bijective kstar and actual kstar: ',abs(kstar.item()-kstar_forward.item()))
# print('')
# print('')

# # Gaussian as a test of the Otsu's method you wrote
# XG_np = XG.cpu().detach().numpy()
# kstar_gauss = custom_otsu(XG_np,1)


# first need to turn gauss_in_vec into a function
# omega = gauss_in_vec.cpu().detach().numpy()
# S_new = ForwardData_mu0.cpu().detach().numpy()
# omega, S_new = signal_interpolation(omega,S_new)

########################################################################################################
#           Seeing what the PDF of the target function looks like through the NN                       #
########################################################################################################


# X_dom_G = torch.linspace(-5,5,1001).to(device)
# TWO = torch.tensor([2.0]).to(device)
# ONE = torch.tensor([1.0]).to(device)
# PI = torch.acos(torch.zeros(1)).to(device) * TWO
# X_rng_G = ONE/torch.sqrt(PI*TWO)*torch.exp(-torch.square(X_dom_G)/TWO)

# ForwardData = []
# for ii in range(len(transforms)):
#   if ii == 0: 
#     ForwardData = transforms[ii](X_rng_G.unsqueeze(1))
#   else:
#     ForwardData = transforms[ii](ForwardData)

# X_rng_G = X_rng_G.cpu().detach().numpy()
# X_dom_G = X_dom_G.cpu().detach().numpy()
# ForwardData = ForwardData.cpu().detach().numpy()

# plt.plot(X_dom_G,ForwardData)
# plt.show()
# plt.plot(X_rng_G,ForwardData)
# plt.show()
# plt.plot(X_dom_G,X_rng_G)
# plt.show()

# max_str = str(torch.max(ForwardData_mu0).item())
# min_str = str(torch.min(ForwardData_mu0).item())
# plt.title('Approximation of Bijective F, min: '+min_str+' max: '+max_str)

# plt.plot(xtest,ytest)
# plt.title("Should be bijective AF")
# plt.show()

# if sig_name == 'dhg':
#     target = 0.5*(1.0/(s1*np.sqrt(2*math.pi))*np.exp(-(xtest-mu1)**2.0/(2*s1**2))+1.0/(s1*np.sqrt(2*math.pi))*np.exp(-(xtest-mu2)**2.0/(2*s2**2)))
# else:
#     target = S_new/norm(S_new,2)


######## looked like a duplicate of the approximation of target distribution plot ############

# plt.plot(F_bij(xtest),PZ,label="Approximation") 
# plt.title("Approximation of Target Distribution",fontsize=36)
# plt.xlabel("$z$",fontsize=32)
# plt.ylabel("$p_Z(z)$",fontsize=32)
# plt.plot(xtest,target,linestyle='dashed',label="Exact")
# plt.legend(fontsize=16)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.grid()
# plt.show()



#PZnorm = PZ/np.trapz(PZ,dx=(xtest[1]-xtest[0]))









# # picking how many threshold values we want
# if sig_name == 'DHG':
#     rawfun = lambda x: (np.exp(-(x-mu1)**2./(2.*(s1**2.))) + np.exp(-(x-mu2)**2./(2.*(s2**2.)))) # defines double hump gaussian shape
#     scfac = quad(rawfun,-np.inf,np.inf)[0] # scaling factor that normalizes gaussian
#     expdist = lambda x: rawfun(x)/scfac # redefines the rawfunc above to be the normalized double hump gaussian 
#     time_axis = omega
#     omega_original = np.array([omeg_new[0],omeg_new[-1]]) 
# else:
#     delomg = omega_original[1]-omega_original[0] # for adding new point to the start and end of omega_original
#     omega_original = np.insert(omega_original,0,(omega_original[0]-delomg))
#     omega_original = np.insert(omega_original,0,(omega_original[0]-delomg))
#     omega_original = np.append(omega_original,(omega_original[-1]+delomg))
#     omega_original = np.append(omega_original,(omega_original[-1]+delomg))
#     S_original = np.insert(S_original,0,0)
#     S_original = np.insert(S_original,0,0)
#     S_original = np.append(S_original,0)
#     S_original = np.append(S_original,0)
#     omeg_final, S_curve = signal_interpolation_otsu(omega_original,S_original)
#     temp_curve = S_curve(omeg_final)
#     rawfun = lambda x: np.interp(x,omeg_final,temp_curve)
#     scfac = quad(rawfun,-np.inf,np.inf)[0]
#     expdist = lambda x: rawfun(x)/scfac #note here a better name might be pdist
    
# total_k_vals = 2 # note must be >1
# #kvec0 = np.linspace(omega_original[10],omega_original[-10],total_k_vals)
# kvec0 = np.arange(0,total_k_vals)
# #kvec0 = np.linspace(omega_original[200],omega_original[-200],total_k_vals)
# print("kvec0=",kvec0)
# L = omega_original[0]
# R = omega_original[-1]
# sol = optimize.root(objective_fun_and_jac, kvec0, args=(expdist,L,R), jac=True, method='hybr')
# kbreaks = sol.x
# finalgval, finaljacmat = objective_fun_and_jac(kbreaks, expdist,L,R)
# otsu_plot(expdist,kbreaks,omega_original[0],omega_original[-1],folder,png)


########################################################################################################
#                             Getting Kvals for a forward pass through                                 #
########################################################################################################

# Note that this code works as intended. It just turned out to be a touch more useless than 
#   anticipated. 

# for kk in range(2,100):
#     desired_k_stars = kk
#     omega_ends = np.abs([omega[0].item(),omega[-1].item()])
#     kbreaks = k_star_Bank_Generator(desired_k_stars,max(omega_ends))
#     kstars_in = torch.from_numpy(kbreaks).to(device)
#     for ii in range(len(transforms)):
#       if ii == 0: 
#         Gauss_to_sig = transforms[ii](kstars_in.unsqueeze(1))
#       else:
#         Gauss_to_sig = transforms[ii](Gauss_to_sig)
#     kbreaks = Gauss_to_sig.detach().cpu().numpy()

#     # Zoomed in Plot 
#     plt.plot(omega.detach().cpu().numpy(),S_new)
#     for jj in range(desired_k_stars):
#         plt.plot([kbreaks[jj].item(), kbreaks[jj].item()], [0., max(S_new)],color='r',ls='--')
#     plt.xlim([-0.2,0.2])
#     plt.show()
#     # Full plot
#     plt.plot(omega.detach().cpu().numpy(),S_new)
#     for jj in range(desired_k_stars):
#         plt.plot([kbreaks[jj].item(), kbreaks[jj].item()], [0., max(S_new)],color='r',ls='--')
#         plt.title("Ticker Count: "+str(kk))
#     plt.show()

########################################################################################################
#                             Getting Kvals for a forward pass through                                 #
########################################################################################################