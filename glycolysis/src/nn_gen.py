## Neutral Network for the glycolysis model

import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import numpy as np


def calculate_weights(loss_values):
    loss_values = np.asarray(loss_values)
    if np.any(loss_values < 0):
        raise ValueError("Loss values should be non-negative")

    # find smallest order of magnitude in list of losses
    loss_magnitudes = [math.floor(math.log(loss, 10)) for loss in loss_values]
    smallest_mag = min(loss_magnitudes)
    weights = np.asarray([10**(x - smallest_mag) for x in loss_magnitudes])

    if len(weights) == 1:
        return weights[0]
    else:
        return weights


class Net(nn.Module):
    '''
    4-layer fully connected neural network
    Activation functions are SiLU (Swish)
    Requires:
        The input data has been pre-processed with input scaling and input feature
    '''
    def __init__(self, n_input):
        super(Net, self).__init__()
        self.fc1= nn.Linear(n_input, n_input)
        self.fc2= nn.Linear(n_input, n_input)
        self.fc3= nn.Linear(n_input, n_input)
        self.fc4= nn.Linear(n_input, n_input)
    
    # Feedforward function
    def forward(self, x):
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
     # ode_mean is a the the magnitudes of the mean values of the ODE solution, in a vector
    #   ode_mean should be a 1xn vector
    #!!!!!! the loss function is yet to be defined in main.py !!!!!
    def backprop(self, data, loss, epoch, optimizer, ode_mean):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train * ode_mean)
        obj_val= loss
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
    
    # Test function. Avoids calculation of gradients.
    # ode_mean is a the the magnitudes of the mean values of the ODE solution, in a vector
    #   ode_mean should be a 1xn vector
    def test(self, data, loss, epoch, ode_mean):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test * ode_mean)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()


