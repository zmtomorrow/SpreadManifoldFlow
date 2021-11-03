import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal,Bernoulli,Normal

from nflib.nets import *


    
class IncompressibleCNN(nn.Module):
    def __init__(self, in_channels, nh=16):
        super().__init__()
        self.dim = in_channels

        self.net = CNN(in_channels//2,in_channels,nh)        
        #self.net = norm_CNN(in_channels//2,in_channels,nh)

        
    def forward(self, x):
        x0, x1 = x[:, :self.dim // 2,:,:], x[:,self.dim // 2:,:,:]
        st = torch.tanh(self.net(x0))
        s,t=st[:, :self.dim // 2,:,:], st[:,self.dim // 2:,:,:]
        s=s-s.mean(1,keepdim=True)

        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=(1,2,3))
        return z, log_det
    
    def backward(self, z):
        z0, z1 = z[:,:self.dim // 2,:,:], z[:,self.dim // 2:,:,:]
        st = torch.tanh(self.net(z0))
        s,t=st[:, :self.dim // 2,:,:], st[:,self.dim // 2:,:,:]
        s=s-s.mean(1,keepdim=True)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=(1,2,3))
        return x, log_det
    
class Incompressibleflow(nn.Module):
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2-1)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2-1 , nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
        
    def forward(self, x):
        x0, x1 = x[:,:self.dim // 2], x[:,self.dim // 2:]
        if self.parity:
            x0, x1 = x1, x0
        s_free = self.s_cond(x0)
        s_final=-torch.sum(s_free,1).view(-1,1)
        s=torch.cat((s_free,s_final),1)
        t = self.t_cond(x0)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        z0, z1 = z[:,:self.dim // 2], z[:,self.dim // 2:]
        if self.parity:
            z0, z1 = z1, z0
        s_free = self.s_cond(z0)
        s_final=-torch.sum(s_free,1).view(-1,1)
        s=torch.cat((s_free,s_final),1)
        t = self.t_cond(z0)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det
    


  

class AffineHalfFlow(nn.Module):
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
        
    def forward(self, x):
        x0, x1 = x[:,:self.dim // 2], x[:,self.dim // 2:]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        s=s-s.mean(1,keepdim=True)
        t = self.t_cond(x0)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        z0, z1 = z[:,:self.dim // 2], z[:,self.dim // 2:]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        s=s-s.mean(1,keepdim=True)
        t = self.t_cond(z0)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det







