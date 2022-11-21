# main file for glycolysis

import argparse
import json
import math
import torch

import numpy as np
import torch.optim as optim

import src.data_gen as get_data
from src.nn_gen import Net


def calculate_weights(loss_values):
    """
    Returns a weight that scales the input loss to the order of 1 (10^0)
    :param loss_values: 1D numpy array
    :return: scalar value if len(loss_values)=1, 1D numpy array otherwise
    """
    loss_values = np.asarray(loss_values)
    if np.any(loss_values < 0):
        raise ValueError("Loss values should be non-negative")

    # find smallest order of magnitude in list of losses
    loss_magnitudes = [math.floor(math.log(loss, 10)) for loss in loss_values]
    weights = np.asarray([10 ** x for x in loss_magnitudes])

    if len(weights) == 1:
        return weights[0]
    else:
        return weights


def weighted_loss(input, label, loss, model):
    """
    calculates weighted loss for any set of num
    :param input: 1D Pytorch tensor
    :param label: 1D Pytorch tensor
    :param loss: Pytorch tensor
    :return: 1D Pytorch tensor
    """
    loss_value = loss(model.forward(input), label)
    loss_value = loss_value / input.size()
    weight = calculate_weights(loss_value)
    loss_value = loss_value * weight
    return loss_value


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


                                    ############ CALLUM START #############

    s = param['data']['num_species_tot']
    m = param['data']['num_species_measured']

    # should put a flag somewhere (maybe in the Data class) to indicate if the data is noiseless or noisy
    # Statements for noisy/noiseless flag, make the noisy/noiseless part of the .json and include in data param??
    if param['optim']['noisy?']:
        epoch_init_iter = param['optim']['init_iters_noisy']
        epoch_full_iter = param['optim']['full_iters_noisy']

    elif not param['optim']['noisy?']:
        epoch_init_iter = param['optim']['init_iters_noiseless']
        epoch_full_iter = param['optim']['full_iters_noiseless']

    else:
        print("cannot determine if noisy or noiseless")
        exit()


                                        ############ CALLUM FINISH #############


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

    aux_loss_train_times = np.asarray([data.conc[0, 0], data.conc[-1, 0]])


                                    ############ CALLUM START #############

    # loop through initial iterations training just data and aux loss
    for x in range(0, epoch_init_iter-1):
        losses_data = []
        losses_aux = []

        # calculate data loss for measures species
        for y in range(0, m):
            for inc, z in enumerate(data.data_inputs):
                loss_data = weighted_loss(y, data.data_labels[inc], mean_square_loss, model)
                losses_data = losses_data.append(loss_data)

        # calculate auxiliary loss for all species
        for y in range(0, s):
            for inc, z in enumerate(data.aux_inputs):
                loss_aux = weighted_loss(inc, data.aux_labels[y], mean_square_loss, model)
                losses_aux = losses_aux.append(loss_aux)

        loss_tot = losses_data.sum() + losses_aux.sum()
        Net.backprop(loss_tot, optimizer)

        # Print loss 10 times per training session
        if not x % (epoch_init_iter / 10):
            print(f"Initial Data and Aux Training:\t \
            Epoch {x}/{epoch_init_iter}: \t loss: {loss_tot} \n")
        else:
            pass
    print(f"Initial Data and Aux Training:\t Done")

    # then loop through full iterations training all loss
    for x in range(0, epoch_full_iter - 1):
        losses_data = []
        losses_ode = []
        losses_aux = []

        # calculate data loss for measured species
        for y in range(0, m):
            for inc, z in enumerate(data.data_inputs):
                loss_data = weighted_loss(y, data.data_labels[inc], mean_square_loss, model)
                losses_data = losses_data.append(loss_data)

        # calculate ode loss for all species
        for y in range(0, s):
            for inc, z in enumerate(data.ode_state):
                loss_ode = weighted_loss(y, data.state_derivative[inc], mean_square_loss, model)
                losses_ode = losses_ode.append(loss_ode)

        # calculate auxiliary loss for all species
        for y in range(0, s):
            for inc, z in enumerate(data.aux_inputs):
                loss_aux = weighted_loss(y, data.aux_labels[inc], mean_square_loss, model)
                losses_aux = losses_aux.append(loss_aux)

        loss_tot = losses_data.sum() + loss_ode.sum() + loss_aux.sum()
        Net.backprop(loss_tot, optimizer)

        # Print loss 10 times per training session
        if not x % (epoch_full_iter/10):
            print(f"Full Training: \t \
            Epoch {x}/{epoch_full_iter}:\t loss: {loss_tot}\n")
        else:
            pass
    # need to replace data.data_inputs, data.data_labels, data.ode_state, etc. with actual data variables

                                    ############ CALLUM FINISH #############


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

    model = Net(1)  # need to figure out the parameters to send to this, using 1 as dummy parameter.
                    # Should it be 128, because NN width in paper is 128 for glycolysis?
                    # also where do the 7 feature layers fit in?

    run_nn(params, model, data)

