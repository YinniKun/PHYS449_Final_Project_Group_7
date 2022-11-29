# main file for glycolysis

import os
import argparse
import json
import math
import torch
from datetime import datetime

import numpy as np
import torch.optim as optim

import src.data_gen as get_data
from src.nn_gen import Net


def calculate_weights(loss_values):
    """
    Returns a weight that scales the input loss to the order of 1 (10^0)
    :param loss_values: 1D python list of loss scalars
    :return: scalar value if len(loss_values)=1, 1D numpy array otherwise
    """
    loss_values = np.asarray(loss_values)
    # find smallest order of magnitude in list of losses
    loss_magnitudes = [math.floor(math.log(loss, 10)) for loss in loss_values]
    weights = np.asarray([10 ** x for x in loss_magnitudes])
    weights = 1 / weights
    if len(weights) == 1:
        return weights[0]
    else:
        return weights

def update_p_vals(data_class, model, p):
    """
    returns the state vector for use in the ode loss
    :param data_class: Data class instance
    :param model: the nn model class
    :param p: current network estimate of p
    :return: python list specifying the state vectors at the ode loss times
    """

    states_all_t = [model.forward(t) for t in data_class.ode_inputs]
    states_np = np.asarray([state.cpu().detach().numpy() for state in states_all_t])  # shape (n_ode, 7)

    sgd_index = np.random.default_rng().integers(low=0, high=states_np[:, 0].size, size=1)

    dx_dt = np.asarray([np.gradient(states_np[:, i]) for i in range(states_np[0].size)])  # shape (7, 5)


    # Need to find the loss of the inner sum to find the weights w for the gradient descent update rule
    inner_loss = np.zeros(7)  # this holds the inner losses for each s species
    grad_update = np.zeros(14)  # how much to update each value of p
    t = data_class.ode_inputs[sgd_index]
    f = np.asarray(get_data.glycolysis_model([t], p, states_np[sgd_index][0]))
    inner_loss += (dx_dt[:, sgd_index[0]] - f) ** 2

    weights = np.asarray(calculate_weights(inner_loss))

    # calculate gradient descent steps
    grad_f = np.asarray(get_data.grad_glycolysis_model(t, p, states_np[sgd_index][0]))
    grad_update += np.asarray([np.sum(2 * f * df_dp_i * weights) for df_dp_i in grad_f])

    return grad_update, inner_loss * weights, states_np


def weighted_loss(inputs, labels, loss, model, num_conc):
    """
    calculates weighted losses for each time point
    :param inputs: 1D Numpy array
    :param labels: 2D Numpy array
    :param loss: Pytorch tensor
    :param model: Pytorch tensor
    :return: 1D python list of weighted losses
    """
    loss_values = []
    for inc, x in enumerate(inputs):
        label = torch.from_numpy(np.asarray([labels[inc]]))
        loss_value = loss(model.forward(x)[num_conc], label)
        loss_values.append(loss_value)
    loss_sum = sum([x.item() for x in loss_values])
    weight = calculate_weights([loss_sum / inputs.size])
    loss_values = [x * weight for x in loss_values]
    return loss_values


def run_nn(param, model, data):
    """
    Trains and tests the neural network on data provided as parameters
    :param param: (maybe nested) dictionary containing hyper-parameters pulled from a json file
    :param model: the neural network class
    :param data: class containing the training and test data
    :return: ?
    """
    now = datetime.now()
    date_time = now.strftime("%m-%d %H%M%S")
    learning_rate = param['optim']['learning_rate']
    s = param['data']['num_species_tot']
    m = param['data']['num_species_measured']
    p = get_data.guess_p()

    init_loss_path = f'data/init_training_loss_data_{date_time}.txt'
    p_track_path = f'data/p_versus_epochs_{date_time}.txt'
    conc_track_path = f'data/network_conc_{date_time}.txt'
    all_loss_track_path = f'data/all_losses_{date_time}.txt'

    with open(init_loss_path, 'w+') as file:
        pass
    with open(p_track_path, 'w+') as file:
        pass
    with open(conc_track_path, 'w+') as file:
        pass
    with open(all_loss_track_path, 'w+') as file:
        pass

    # should put a flag somewhere (maybe in the Data class) to indicate if the data is noiseless or noisy
    # Statements for noisy/noiseless flag, make the noisy/noiseless part of the .json and include in data param??
    if param['data']['noisy?']:
        epoch_init_iter = param['optim']['init_iters_noisy']
        epoch_full_iter = param['optim']['full_iters_noisy']

    elif not param['data']['noisy?']:
        epoch_init_iter = param['optim']['init_iters_noiseless']
        epoch_full_iter = param['optim']['full_iters_noiseless']

    else:
        print("cannot determine if noisy or noiseless")
        return -1



    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mean_square_loss = torch.nn.MSELoss(reduction= 'mean')

    model.reset()  # reset model parameters every time the nn is run

    # concentration indexes for glycolysis model (0 indexed)
    m = [4, 5]
    # loop through initial iterations training just data and aux loss
    for x in range(0, epoch_init_iter-1):
        summed_data_losses = []
        summed_aux_losses = []

        # calculate data loss for measured species
        for inc, y in enumerate(m):
            # get losses for each time point
            data_losses = weighted_loss(data.data_inputs, data.data_labels[:, y],
                                             mean_square_loss, model, y)
            # inner sum over time points
            data_losses_sum = data_losses[0]
            for loss in data_losses[1:]:
                data_losses_sum = torch.add(data_losses_sum, loss)

            # append inner sums of each concentration to a list
            summed_data_losses.append(data_losses_sum)

        # outer sum of losses for m concentrations
        data_loss_total = summed_data_losses[0]
        for loss in summed_data_losses[1:]:
            data_loss_total = torch.add(data_loss_total, loss)


        # calculate auxiliary loss for all species
        for y in range(0, s):
            # get losses for each time point
            aux_losses = weighted_loss(data.aux_inputs, data.aux_labels[y],
                                            mean_square_loss, model, y)
            # inner sum over time points
            aux_losses_sum = aux_losses[0]
            for loss in aux_losses[1:]:
                aux_losses_sum = torch.add(aux_losses_sum, loss)

            # append inner sums of each concentration to a list
            summed_aux_losses.append(aux_losses_sum)

        # outer sum of loss of all species
        aux_loss_total = summed_aux_losses[0]
        for loss in summed_aux_losses[1:]:
            aux_loss_tot = torch.add(aux_loss_total, loss)

        # sum of total data and aux losses
        loss_tot = data_loss_total + aux_loss_tot

        model.backprop(loss_tot, optimizer)

        # Print loss 10 times per training session
        if not x % (epoch_init_iter / 10):
            print(f"Initial Data and Aux Training:\t \
            Epoch {x}/{epoch_init_iter}: \t loss: {loss_tot} \n")

        if x % 10 == 0:  # track initial losses (just data and auxiliary) at 100 points
            with open(init_loss_path, 'a') as graph_data:
                # format is [data_loss aux_loss epoch] for each line
                graph_data.write(f'{data_loss_total}\t {aux_loss_total}\t {x}\n')

    print(f"Initial Data and Aux Training:\t Done")

    # then loop through full iterations training all loss
    for x in range(int(epoch_full_iter)):
        summed_data_losses = []
        summed_aux_losses = []

        # calculate data loss for measured species
        for inc, y in enumerate(m):
            # get losses for each time point
            data_losses = weighted_loss(data.data_inputs, data.data_labels[:, y],
                                        mean_square_loss, model, y)
            # inner sum over time points
            data_losses_sum = data_losses[0]
            for loss in data_losses[1:]:
                data_losses_sum = torch.add(data_losses_sum, loss)

            # append inner sums of each concentration to a list
            summed_data_losses.append(data_losses_sum)

        # outer sum of losses for m concentrations
        data_loss_total = summed_data_losses[0]
        for loss in summed_data_losses[1:]:
            data_loss_total = torch.add(data_loss_total, loss)


        # calculate auxiliary loss for all species
        for y in range(0, s):
            # get losses for each time point
            aux_losses = weighted_loss(data.aux_inputs, data.aux_labels[y],
                                       mean_square_loss, model, y)
            # inner sum over time points
            aux_losses_sum = aux_losses[0]
            for loss in aux_losses[1:]:
                aux_losses_sum = torch.add(aux_losses_sum, loss)

            # append inner sums of each concentration to a list
            summed_aux_losses.append(aux_losses_sum)

        # outer sum of loss of all species
        aux_loss_total = summed_aux_losses[0]
        for loss in summed_aux_losses[1:]:
            aux_loss_tot = torch.add(aux_loss_total, loss)

        # sum of total data and aux losses
        loss_tot = data_loss_total + aux_loss_tot

        model.backprop(loss_tot, optimizer)

        # update p using ode loss
        d_ode_loss_dp, ode_loss, network_predicted_states = update_p_vals(data, model, p)
        p = p + learning_rate * d_ode_loss_dp
        if x % 5 == 0:
            print(f'loss_tot no ode: {loss_tot.item()}')
            print(f'ode_loss: {ode_loss}')
            print(f'p_after: {p}')

        if x % 1000 == 0:  # track p and losses every 1000 epochs
            with open(all_loss_track_path, 'a') as graph_data:
                # format is [data_loss aux_loss ode loss epoch] for each line
                graph_data.write(f'{data_loss_total}\t {aux_loss_total}\t {ode_loss}\t{x}\n')

            with open(p_track_path, 'a') as graph_data:
                # format is [p0 p1 p2 p3 ..... p14] for each line
                p_string = " ".join([str(x) for x in p])
                graph_data.write(f'{p_string}\t{x}\n')

        if x % 5000 == 0:  # track network concentration predictions every 5000 epochs
            with open(conc_track_path, 'a') as graph_data:
                # format is [t S1 S2 S3 S4 S5 S6 S7] for each line.
                for i, prediction in enumerate(network_predicted_states):
                    line_string = ' '.join([str(data.data_inputs[i]) + " " + ' '.join([str(x) for x in prediction])])
                    graph_data.write(f'{line_string}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="src/param.json", help='json parameter file relative path', type=str)
    parser.add_argument('--data_seed', help='Random seed for generating training data', type=int)
    args = parser.parse_args()

    with open(args.param) as json_param_file:
        params = json.load(json_param_file)

    num_data_points = params['data']['num_data_points']

    data_dir = f'{os.path.dirname(__file__)}{os.sep}data'
    if not os.path.isdir(data_dir):
        print(f'Created directory: {data_dir}')
        os.mkdir(data_dir)
    data = get_data.Data(params['data'], n_points=num_data_points)
    data.save_as_csv()
    model = Net(7, data)
    model.double()
    #print(data.data_labels, data.data_labels[0], data.aux_labels)
    run_nn(params, model, data)
