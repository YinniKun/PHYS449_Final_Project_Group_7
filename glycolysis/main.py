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
    learning_rate = param['optim']['learning_rate']


    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    mean_square_loss = torch.nn.MSELoss(reduction= 'mean')

    model.reset()  # reset model parameters every time the nn is run



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', help='json parameter file relative path', type=str)
    parser.add_argument('--data_seed', help='Random seed for generating training data', type=int)
    args = parser.parse_args()

    with open(args.param) as json_param_file:
        params = json.load(json_param_file)

    num_data_points = params['data']['num_data_points']

    data = get_data.Data(n_points=args.data_points)
    data.save_as_csv()

    model = Net(1)  # need to figure out the parameters to send to this, using 1 as dummy parameter

    run_nn(params, model, data)

