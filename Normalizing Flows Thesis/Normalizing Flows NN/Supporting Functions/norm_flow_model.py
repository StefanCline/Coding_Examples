# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:53:35 2023

@author: Sclin
"""
import torch
from torch import nn
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T


def Norm_Flow_Model(omega,device):
    class NFModel(nn.Module):
      def __init__(self):
        super().__init__()
        ## Forward calls, i.e. mapping the NN in the 'F' direction X -- F --> Z
        self.sq0 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'quadratic')
        self.sq1 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'quadratic')
        self.sq2 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'quadratic')
        self.sq3 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'quadratic')
        self.sq4 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'quadratic')
        # self.sq5 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'quadratic')

        self.sL0 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'linear')
        self.sL1 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'linear')
        self.sL2 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'linear')
        self.sL3 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'linear')
        self.sL4 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'linear')
        # self.sL5 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'linear')

        self.hh0 = T.Householder(input_dim=1, count_transforms=1)
        self.hh1 = T.Householder(input_dim=1, count_transforms=1)
        self.hh2 = T.Householder(input_dim=1, count_transforms=1)
        self.hh3 = T.Householder(input_dim=1, count_transforms=1)
        self.hh4 = T.Householder(input_dim=1, count_transforms=1)
        self.hh5 = T.Householder(input_dim=1, count_transforms=1)
        self.hh6 = T.Householder(input_dim=1, count_transforms=1)
        self.hh7 = T.Householder(input_dim=1, count_transforms=1)
        self.hh8 = T.Householder(input_dim=1, count_transforms=1)
        self.hh9 = T.Householder(input_dim=1, count_transforms=1)

        self.dc0 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc1 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc2 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc3 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc4 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc5 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc6 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc7 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc8 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc9 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dcA = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        #self.dcB = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        #self.dcC = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)

        #Worth testing
        # self.pL0 = T.Planar(input_dim=1)
        # self.pn0 = T.Polynomial()
        # self.ra0 = T.Radial(input_dim=1)
        # self.sy0 = T.Sylvester(input_dim=1,count_transforms=1)
        # self.me0 = T.MatrixExponential(input_dim=1,iterations=1,normalization='none',bound=None)
        # self.bn0 = T.BatchNorm()


        ###########################
        ###########################
        # Inverse calls, i.e. mapping the NN in the 'F^-1' direction, Z -- F^-1 --> X
        ###########################
        ###########################

        self.sq0_inv = self.sq0.inv
        self.sq1_inv = self.sq1.inv
        self.sq2_inv = self.sq2.inv
        self.sq3_inv = self.sq3.inv
        self.sq4_inv = self.sq4.inv
        # self.sq5_inv = self.sq5.inv

        self.sL0_inv = self.sL0.inv
        self.sL1_inv = self.sL1.inv
        self.sL2_inv = self.sL2.inv
        self.sL3_inv = self.sL3.inv
        self.sL4_inv = self.sL4.inv
        # self.sL5_inv = self.sL5.inv

        self.hh0_inv = self.hh0.inv
        self.hh1_inv = self.hh1.inv
        self.hh2_inv = self.hh2.inv
        self.hh3_inv = self.hh3.inv
        self.hh4_inv = self.hh4.inv
        self.hh5_inv = self.hh5.inv
        self.hh6_inv = self.hh2.inv
        self.hh7_inv = self.hh3.inv
        self.hh8_inv = self.hh4.inv
        self.hh9_inv = self.hh5.inv

        self.dc0_inv = self.dc0.inv
        self.dc1_inv = self.dc1.inv
        self.dc2_inv = self.dc2.inv
        self.dc3_inv = self.dc3.inv
        self.dc4_inv = self.dc4.inv
        self.dc5_inv = self.dc5.inv
        self.dc6_inv = self.dc6.inv
        self.dc7_inv = self.dc7.inv
        self.dc8_inv = self.dc8.inv
        self.dc9_inv = self.dc9.inv
        self.dcA_inv = self.dcA.inv
        # self.dcB_inv = self.dcB.inv
        # self.dcC_inv = self.dcC.inv

        # Worth testing
        # self.pL0_inv = self.pL0.inv
        # self.ra0_inv = self.ra0.inv
        # self.sy0_inv = self.sy0.inv
        # self.me0_inv = self.me0.inv

    # Creating an instance of the above model to be able to grab the parameters
    NormFlowModel = NFModel()
    NormFlowModel = NormFlowModel.to(device)

    # Forward direction: 
    transforms =      [NormFlowModel.hh0,
                       NormFlowModel.sL0,
                       NormFlowModel.dc0,
                       NormFlowModel.hh1,
                       NormFlowModel.sq0,
                       NormFlowModel.dc1,
                       NormFlowModel.hh2,
                       NormFlowModel.sL1,
                       NormFlowModel.dc2,
                       NormFlowModel.hh3,
                       NormFlowModel.sq1,
                       NormFlowModel.dc3,
                       NormFlowModel.hh4,
                       NormFlowModel.sL2,
                       NormFlowModel.dc4,
                       NormFlowModel.hh5,
                       NormFlowModel.sq2,
                       NormFlowModel.dc5,
                       NormFlowModel.hh6,
                       NormFlowModel.sL3,
                       NormFlowModel.dc6,
                       NormFlowModel.hh7,
                       NormFlowModel.sq3,
                       NormFlowModel.dc7,
                       NormFlowModel.hh8,
                       NormFlowModel.sL4,
                       NormFlowModel.dc8,
                       NormFlowModel.hh9,
                       NormFlowModel.sq4]

    # Inverse Direction
    inv_transforms =  [NormFlowModel.sq4_inv,
                       NormFlowModel.hh9_inv,
                       NormFlowModel.dc8_inv,
                       NormFlowModel.sL4_inv,
                       NormFlowModel.hh8_inv,
                       NormFlowModel.dc7_inv,
                       NormFlowModel.sq3_inv,
                       NormFlowModel.hh7_inv,
                       NormFlowModel.dc6_inv,
                       NormFlowModel.sL3_inv,
                       NormFlowModel.hh6_inv,
                       NormFlowModel.dc5_inv,
                       NormFlowModel.sq2_inv,
                       NormFlowModel.hh5_inv,
                       NormFlowModel.dc4_inv,
                       NormFlowModel.sL2_inv,
                       NormFlowModel.hh4_inv,
                       NormFlowModel.dc3_inv,
                       NormFlowModel.sq1_inv,
                       NormFlowModel.hh3_inv,
                       NormFlowModel.dc2_inv,
                       NormFlowModel.sL1_inv,
                       NormFlowModel.hh2_inv,
                       NormFlowModel.dc1_inv,
                       NormFlowModel.sq0_inv,
                       NormFlowModel.hh1_inv,
                       NormFlowModel.dc0_inv,
                       NormFlowModel.sL0_inv,
                       NormFlowModel.hh0_inv]
                       
    # Base of Gaussian and flow that internalizes gaussian as the base and the transforms
    # as listed in transforms as the path to get there. 
    base_dist = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    # flow_dist calls later will accept target data points and work from the gaussian to
    # the target 
    flow_dist = dist.TransformedDistribution(base_dist, transforms)
    
    print('')
    print('')
    print('Done with: generating NormFlowModel')
    print('')
    print('')
    
    return  NormFlowModel, transforms, inv_transforms, base_dist, flow_dist



#########################################################################
#       CREATING SMALLER NORM FLOW MODEL SIZE FOR MEMORY'S SAKE         #
#########################################################################


def Norm_Flow_Model_small(omega,device):
    class NFModel(nn.Module):
      def __init__(self):
        super().__init__()
        ## Forward calls, i.e. mapping the NN in the 'F' direction X -- F --> Z
        # original is count_bins=16
        self.sq0 = T.Spline(input_dim=1, count_bins=256, bound = omega[-1].item(), order = 'quadratic')
        self.sq1 = T.Spline(input_dim=1, count_bins=256, bound = omega[-1].item(), order = 'quadratic')
    
        # self.sq5 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'quadratic')
    
        self.sL0 = T.Spline(input_dim=1, count_bins=256, bound = omega[-1].item(), order = 'linear')
        self.sL1 = T.Spline(input_dim=1, count_bins=256, bound = omega[-1].item(), order = 'linear')
    
        # self.sL5 = T.Spline(input_dim=1, count_bins=16, bound = omega[-1].item(), order = 'linear')
    
        self.hh0 = T.Householder(input_dim=1, count_transforms=5) # originally set to 1
        self.hh1 = T.Householder(input_dim=1, count_transforms=5)
        self.hh2 = T.Householder(input_dim=1, count_transforms=5)
        self.hh3 = T.Householder(input_dim=1, count_transforms=5)
    
        self.dc0 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc1 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc2 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
        self.dc3 = T.DiscreteCosineTransform(dim=-1,smooth=0.0,cache_size=0)
    
        ###########################
        ###########################
        # Inverse calls, i.e. mapping the NN in the 'F^-1' direction, Z -- F^-1 --> X
        ###########################
        ###########################
    
        self.sq0_inv = self.sq0.inv
        self.sq1_inv = self.sq1.inv
    
    
        self.sL0_inv = self.sL0.inv
        self.sL1_inv = self.sL1.inv
    
    
        self.hh0_inv = self.hh0.inv
        self.hh1_inv = self.hh1.inv
        self.hh2_inv = self.hh2.inv
        self.hh3_inv = self.hh3.inv
    
    
        self.dc0_inv = self.dc0.inv
        self.dc1_inv = self.dc1.inv
        self.dc2_inv = self.dc2.inv
        self.dc3_inv = self.dc3.inv

    # Creating an instance of the above model to be able to grab the parameters
    NormFlowModel = NFModel()
    NormFlowModel = NormFlowModel.to(device)

    # Forward direction: 
    transforms =      [NormFlowModel.hh0,
                       NormFlowModel.sL0,
                       NormFlowModel.dc0,
                       NormFlowModel.hh1,
                       NormFlowModel.sq0,
                       NormFlowModel.dc1,
                       NormFlowModel.hh2,
                       NormFlowModel.sL1,
                       NormFlowModel.dc2,
                       NormFlowModel.hh3,
                       NormFlowModel.sq1,
                       NormFlowModel.dc3]

    # Inverse Direction
    inv_transforms =  [NormFlowModel.dc3_inv,
                       NormFlowModel.sq1_inv,
                       NormFlowModel.hh3_inv,
                       NormFlowModel.dc2_inv,
                       NormFlowModel.sL1_inv,
                       NormFlowModel.hh2_inv,
                       NormFlowModel.dc1_inv,
                       NormFlowModel.sq0_inv,
                       NormFlowModel.hh1_inv,
                       NormFlowModel.dc0_inv,
                       NormFlowModel.sL0_inv,
                       NormFlowModel.hh0_inv]
                       
    # Base of Gaussian and flow that internalizes gaussian as the base and the transforms
    # as listed in transforms as the path to get there. 
    base_dist = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    # flow_dist calls later will accept target data points and work from the gaussian to
    # the target 
    flow_dist = dist.TransformedDistribution(base_dist, transforms)
    
    print('')
    print('')
    print('Done with: generating NormFlowModel')
    print('')
    print('')
    
    return  NormFlowModel, transforms, inv_transforms, base_dist, flow_dist


#########################################################################
#       CREATING SPLINE ONLY NORM FLOW MODEL FOR SIMPLICITY'S SAKE      #
#########################################################################


def Norm_Flow_Model_splineonly(omega,device):
    class NFModel(nn.Module):
      def __init__(self):
        super().__init__()
        ## Forward calls, i.e. mapping the NN in the 'F' direction X -- F --> Z
        # original is count_bins=16
        self.sq0 = T.Spline(input_dim=1, count_bins=178, bound = omega[-1].item(), order = 'quadratic')
        self.sq1 = T.Spline(input_dim=1, count_bins=178, bound = omega[-1].item(), order = 'quadratic')

        self.sL0 = T.Spline(input_dim=1, count_bins=178, bound = omega[-1].item(), order = 'linear')
        self.sL1 = T.Spline(input_dim=1, count_bins=178, bound = omega[-1].item(), order = 'linear')

        ###########################
        ###########################
        # Inverse calls, i.e. mapping the NN in the 'F^-1' direction, Z -- F^-1 --> X
        ###########################
        ###########################

        self.sq0_inv = self.sq0.inv
        self.sq1_inv = self.sq1.inv

        self.sL0_inv = self.sL0.inv
        self.sL1_inv = self.sL1.inv

    # Creating an instance of the above model to be able to grab the parameters
    NormFlowModel = NFModel()
    NormFlowModel = NormFlowModel.to(device)

    # Forward direction: 
    transforms =      [NormFlowModel.sL0,
                       NormFlowModel.sq0,
                       NormFlowModel.sL1,
                       NormFlowModel.sq1]

    # Inverse Direction
    inv_transforms =  [NormFlowModel.sq1_inv,
                       NormFlowModel.sL1_inv,
                       NormFlowModel.sq0_inv,
                       NormFlowModel.sL0_inv]
                       
    # Base of Gaussian and flow that internalizes gaussian as the base and the transforms
    # as listed in transforms as the path to get there. 
    base_dist = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    # flow_dist calls later will accept target data points and work from the gaussian to
    # the target 
    flow_dist = dist.TransformedDistribution(base_dist, transforms)
    
    print('')
    print('')
    print('Done with: generating NormFlowModel')
    print('')
    print('')
    
    return  NormFlowModel, transforms, inv_transforms, base_dist, flow_dist


#########################################################################
#       CREATING BAD SINGLE TRANSFORMATION NORM FLOW MODEL              #
#########################################################################


def Norm_Flow_Model_bad(omega,device,cb_val):
    class NFModel(nn.Module):
      def __init__(self):
        super().__init__()
        ## Forward calls, i.e. mapping the NN in the 'F' direction X -- F --> Z
        # original is count_bins=16
        self.sq0 = T.Spline(input_dim=1, count_bins=cb_val, bound = omega[-1].item(), order = 'quadratic')

        ###########################
        ###########################
        # Inverse calls, i.e. mapping the NN in the 'F^-1' direction, Z -- F^-1 --> X
        ###########################
        ###########################

        self.sq0_inv = self.sq0.inv

    # Creating an instance of the above model to be able to grab the parameters
    NormFlowModel = NFModel()
    NormFlowModel = NormFlowModel.to(device)

    # Forward direction: 
    transforms =      [NormFlowModel.sq0]

    # Inverse Direction
    inv_transforms =  [NormFlowModel.sq0_inv]
                       
    # Base of Gaussian and flow that internalizes gaussian as the base and the transforms
    # as listed in transforms as the path to get there. 
    base_dist = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    # flow_dist calls later will accept target data points and work from the gaussian to
    # the target 
    flow_dist = dist.TransformedDistribution(base_dist, transforms)
    
    print('')
    print('')
    print('Done with: generating NormFlowModel')
    print('')
    print('')
    
    return  NormFlowModel, transforms, inv_transforms, base_dist, flow_dist