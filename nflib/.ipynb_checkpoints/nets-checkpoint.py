"""
Various helper network modules
"""

import torch
import torch.nn.functional as F
from torch import nn





class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)
    
class norm_CNN(nn.Module):
    def __init__(self, nin, nout, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(torch.nn.BatchNorm2d(nin),
                                 nn.Conv2d(nin, hidden_channels, 3, padding=1),
                                 nn.ReLU(),
                                 torch.nn.BatchNorm2d(hidden_channels),
                                 nn.Conv2d(hidden_channels, hidden_channels, 1, padding=0), 
                                 nn.ReLU(),
                                 torch.nn.BatchNorm2d(hidden_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(hidden_channels, nout, 3, padding=1))
    def forward(self, x):
        return self.net(x)
    
    
class CNN(nn.Module):
    def __init__(self, nin, nout, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(nin, hidden_channels, 3, padding=1),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(hidden_channels, hidden_channels, 1, padding=0), 
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(hidden_channels, nout, 3, padding=1))
    def forward(self, x):
        return self.net(x)

