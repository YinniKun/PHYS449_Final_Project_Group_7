#main file for glycolosis

import json, argparse, torch, sys
import src.data_gen as get_data
import numpy as np
import torch.optim as optim
from src.nn_gen import Net


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

    data = get_data.Data()
    #model = Net()
    data.save_as_csv()

