## Neutral Network for the apoptosis model

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    '''
    5-layer fully connected neural network
    Activation functions are SiLU (Swish)
    parameters:
        n_species = the number of chemical species
        ode_mean = the mean value of the ode solution used in the output
            scaling layer
        t_max = the max time point t in the input scaling layer
    '''
    def __init__(self, n_species, data):
        super(Net, self).__init__()
        self.fc1= nn.Linear(n_species, 256)
        self.fc2= nn.Linear(256, 256)
        self.fc3= nn.Linear(256, 256)
        self.fc4= nn.Linear(256, 256)
        self.fc5= nn.Linear(256, n_species)

        ## average concentration for each species
        self.ode_mean = torch.from_numpy(np.mean(data.conc, axis=0))

    def feature(self, t):
        """
        Extract feature layer from single (scaled) time value.

        :param t: float
        :returns: 2D tensor
        """
        return torch.tensor([
            t,
            math.exp(-t)
            ])

    # Feedforward function
    def forward(self, t, t_max=60):
        in_scal = t / t_max
        f1 = self.feature(in_scal)
        h1 = func.silu(self.fc1(f1))
        h2 = func.silu(self.fc2(h1))
        h3 = func.silu(self.fc3(h2))
        h4 = func.silu(self.fc3(h3))
        y = func.silu(self.fc4(h4)) * self.ode_mean
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc5.reset_parameters()

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
