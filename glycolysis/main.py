#main file for glycolosis

import json, argparse, torch, sys
import numpy as np
import torch.optim as optim
from src.nn_gen import Net
from src.data_gen import TestData

#all functions taken from Workshop 2... need to be fixed to fit the dataset(s) we are using -Callum
def get_data_and_model(n_training_data, n_test_data):
    """
    Initializes the neural network model and training/test data
    :param
    :return: class Net and class Data
    """

    neural_net = Net(4)
    net_data = TestData(n_training_data, n_test_data)

    return neural_net, net_data


def run_nn(param, model, data):
    """
    Trains and tests the neural network on data provided as parameters
    :param param: (maybe nested) dictionary containing hyper-parameters pulled from a json file
    :param model: the neural network class
    :param data: class containing the training and test data
    :return: ?
    """

    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    loss = torch.nn.MSELoss(reduction= 'mean')



if __name__ == '__main__':
    model, data = get_data_and_model(10, 5)
