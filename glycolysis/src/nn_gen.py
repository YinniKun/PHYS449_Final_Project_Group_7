## Neutral Network for the glycolysis model

import math
import numpy as np
import data_gen as data 

from src.data_gen import *

import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    '''
    4-layer fully connected neural network
    Activation functions are SiLU (Swish)
    parameters:
        n_output = the number of chemical species generated
        ode_mean = the mean value of the ode solution used in the output
            scaling layer
        t_max = the max time point t in the input scaling layer
    '''

    def __init__(self, n_output, data):
        super(Net, self).__init__()
        self.fc1= nn.Linear(7, 128)
        self.fc2= nn.Linear(128, 128)
        self.fc3= nn.Linear(128, 128)
        self.fc4= nn.Linear(128, n_output)

        ## average concentration for each species
        self.ode_mean = torch.from_numpy(np.mean(data.conc, axis=0))

    def feature(self, t):
        """
        Extract feature layer from single (scaled) time value.

        :param t: float
        :returns: 7D tensor
        """
        return torch.tensor([
            t,
            math.sin(t),
            math.sin(t*2),
            math.sin(t*3),
            math.sin(t*4),
            math.sin(t*5),
            math.sin(t*6)
            ])

    # Feedforward function
    def forward(self, t, t_max=10):
        in_scal = t / t_max
        f1 = self.feature(in_scal)
        h1 = func.silu(self.fc1(f1))
        h2 = func.silu(self.fc2(h1))
        h3 = func.silu(self.fc3(h2))
        y = func.silu(self.fc4(h3)) * self.ode_mean
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    # Backpropagation function
    def backprop(self, loss, optimizer):
        self.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()
