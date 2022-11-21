## Neutral Network for the glycolysis model

import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import numpy as np


class Net(nn.Module):
    """
    4-layer fully connected neural network
    Activation functions are SiLU (Swish)
    Requires:
        The input data has been pre-processed with input scaling and input feature
    """
    def __init__(self, n_input):
        super(Net, self).__init__()
        self.fc1= nn.Linear(n_input, n_input)
        self.fc2= nn.Linear(n_input, n_input)
        self.fc3= nn.Linear(n_input, n_input)
        self.fc4= nn.Linear(n_input, n_input)

    # Feedforward function
    def forward(self, x):
        x = torch.from_numpy(x)
        h1 = nn.SiLU(self.fc1(x))
        h2 = nn.SiLU(self.fc2(h1))
        h3 = nn.SiLU(self.fc3(h2))
        y = nn.SiLU(self.fc4(h3))
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    # Backpropagation function
    def backprop(self, loss_val, optimizer):
        self.train()
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        # return obj_val.item()

    """def backprop_no_ode(self, data, loss, optimizer):
        self.train()
        loss_data = self.weighted_loss(data.data_inputs, data.data_labels, loss)
        loss_aux = self.weighted_loss(data.aux_inputs, data.aux_labels, loss)
        obj_val = loss_data+loss_aux
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()""" # func doesn't need to exist, since loss is now passed as argument

    # Test function. Avoids calculation of gradients.

    # ode_mean is a the the magnitudes of the mean values of the ODE solution, in a vector
    #   ode_mean should be a 1xn vector
    def test(self, data, loss, epoch, ode_mean):
        self.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(data.x_test)
            targets = torch.from_numpy(data.y_test * ode_mean)
            cross_val = loss(self.forward(inputs), targets)
        return cross_val.item()
