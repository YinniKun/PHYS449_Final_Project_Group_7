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
    data_loss_train_size = param['data']['num_data_loss']
    ode_loss_train_size = param['data']['num_ode_loss']  # cant find in the paper where they specify this set size


    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mean_square_loss = torch.nn.MSELoss(reduction= 'mean')

    model.reset()  # reset model parameters every time the nn is run

    # get data_loss_train_size random time points for training the data loss
    rand_indexes = np.random.default_rng().choice(data.conc[:, 0].size, data_loss_train_size, replace=False)
    rand_indexes = np.sort(rand_indexes)
    data_loss_train_times = data.conc[rand_indexes, 0]

    # get ode_loss_train_size equispaced time points for training the ode loss
    ode_loss_indexes = np.round(np.linspace(0, len(data.conc[:, 0]) - 1, ode_loss_train_size)).astype(int)
    ode_loss_train_times = data.conc[ode_loss_indexes, 0]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', help='json parameter file relative path', type=str)
    parser.add_argument('--data_seed', help='Random seed for generating training data', type=int)
    args = parser.parse_args()

    with open(args.param) as json_param_file:
        params = json.load(json_param_file)

    num_data_points = params['data']['num_data_points']

    data = get_data.Data(n_points=num_data_points)
    data.save_as_csv()

    model = Net(1)  # need to figure out the parameters to send to this, using 1 as dummy parameter

    run_nn(params, model, data)

