import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal,Bernoulli,Normal


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m = x.shape[0]
        log_det = torch.zeros(m).to(x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m = z.shape[0]
        log_det = torch.zeros(m).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs

    


  
    
class FASpreadNormalizingFlowModel(nn.Module):
    def __init__(self, flows, dim):
        super().__init__()
        self.dim=dim
        self.A_init=nn.Parameter(torch.diag(torch.ones(self.dim)))
        self.A=torch.diag(torch.ones(self.dim))
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x, s_std):
        self.A=torch.tril(self.A_init)
        spread_cov=self.A@(self.A.t())+torch.diag(torch.ones(self.dim)).to(x.device)*s_std**2
        prior = MultivariateNormal(torch.zeros(self.dim).to(x.device), spread_cov)
        zs, log_det = self.flow.forward(x)
        zs_spread=zs[-1]+torch.ones_like(zs[-1])*s_std
        prior_logprob = prior.log_prob(zs_spread.to(x.device)).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = (torch.randn((num_samples,self.dim)).to(self.A.device))@(self.A.t())
        xs, _ = self.flow.backward(z)
        return xs
  

