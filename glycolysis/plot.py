# Plot the glycolysis model.

import json
import argparse
import math

import numpy as np
from matplotlib import pyplot as plt

import src.data_gen as get_data

# ****************************************************************************
# Specify what is to be plotted.
# ****************************************************************************

# File name of predicted p to plot vs true p over epochs.
do_plot_p_vs_epoch = True
p_file = 'p_versus_epochs_11-28 150647.txt'

# File name file of losses to plot over epochs.
do_plot_loss_vs_epoch = False
loss_file = ''

# File name of concentrations to plot vs measured/true data over time.
do_plot_conc_from_file = False
conc_file = 'data/network_conc_11-28 150647.txt'

# Predicted p for generating concentrations to plot vs measured/true data over
# time (can be a list of float or a whitespace separated string).
do_plot_conc_from_p = False
p = '2.5719585301918 106.78695966315719 4.928489845917875 17.309286498015584 86.92947289981659 0.9374354402554003 13.023036803363441 1.6006457800226783 14.598217984302414 3.509660965838129 0.6553798881690627 0.17121275040390832 0.8258334214382487 4.420317267284303'

# ****************************************************************************
# Constants.
# ****************************************************************************

# True p values.
J0 = 2.5
k1 = 100
k2 = 6
k3 = 16
k4 = 100
k5 = 1.28
k6 = 12
k = 1.8
kappa = 13
q = 4
K1 = 0.52
psi = 0.1
N = 1
A = 4
p0 = np.asarray([J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi, N, A])


def plot_loss():
    pass


def plot_p_vs_true_p():
    pass


def plot_conc_from_file(true_conc, pred_conc):
    """Plot predicted concentrations from p vs measured/true data over time.

    Generate one figure with subplots of the concentration of each chemical
    species. Plot is named pred_conc_from_p_vs_true.pdf.
    :param true_conc: (num_data_points, S+1) array of time and true concs
    :param pred_conc: (num_data_points, S+1) array of time and predicted concs

    """
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 24),
                            constrained_layout=True)
    n_species = 7
    for ind in range(n_species):

        row, col = get_row_col(ind)

        # Plot.
        axs[row, col].plot(true_conc[:, 0], true_conc[:, ind + 1],
                           color='b', label='Exact')
        axs[row, col].plot(pred_conc[:, 0], pred_conc[:, ind + 1],
                           color='r', linestyle='dashdot', label='Learned')
        # Configure subplot.
        axs[row, col].set_xlabel('t (min)')
        axs[row, col].set_ylabel(r'S_{0} (mM)'.format(ind+1))
        # Autoscale axes to each subplot.
        axs[row, col].relim()
        axs[row, col].autoscale()

    # Delete unused bottom right subplot and add legend.
    handles, labels = axs[row, col].get_legend_handles_labels()
    col += 1
    fig.delaxes(axs[row, col])
    fig.legend(handles, labels, loc=(0.6, 0.2))
    fig.suptitle('Concentration predicted by network', fontsize=16)

    # Save plot
    plt.savefig('data/pred_conc_from_file_vs_true.pdf', bbox_inches='tight')


def plot_conc_from_p(true_conc, pred_conc):
    """Plot predicted concentrations from p vs measured/true data over time.

    Generate one figure with subplots of the concentration of each chemical
    species. Plot is named pred_conc_from_p_vs_true.pdf.
    :param true_conc: (num_data_points, S+1) array of time and true concs
    :param pred_conc: (num_data_points, S) array of predicted concs

    """
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 24),
                            constrained_layout=True)
    n_species = 7
    for ind in range(n_species):

        row, col = get_row_col(ind)

        # Plot.
        axs[row, col].plot(true_conc[:, 0], true_conc[:, ind + 1],
                           color='b', label='Exact')
        axs[row, col].plot(true_conc[:, 0], pred_conc[:, ind],
                           color='r', linestyle='dashdot', label='Learned')
        # Configure subplot.
        axs[row, col].set_xlabel('t (min)')
        axs[row, col].set_ylabel(r'S_{0} (mM)'.format(ind+1))
        # Autoscale axes to each subplot.
        axs[row, col].relim()
        axs[row, col].autoscale()

    # Delete unused bottom right subplot and add legend.
    handles, labels = axs[row, col].get_legend_handles_labels()
    col += 1
    fig.delaxes(axs[row, col])
    fig.legend(handles, labels, loc=(0.6, 0.2))
    fig.suptitle('Analytical concentration from predicted parameters', fontsize=16)

    # Save plot
    plt.savefig('data/pred_conc_from_file_vs_true.pdf', bbox_inches='tight')


def get_row_col(ind):
    """Find row, col of the given species index (0 to n_species-1)."""
    return math.floor(ind / 2), ind % 2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="src/param.json", help='json parameter file relative path', type=str)
    parser.add_argument('--data_seed', help='Random seed for generating training data', type=int)
    args = parser.parse_args()

    with open(args.param) as json_param_file:
        params = json.load(json_param_file)

    # num_data_points = params['data']['num_data_points']
    # t = np.linspace(start=0, stop=10, num=num_data_points, dtype=np.float64)

    # Only read true concentration once if it is needed.
    # !!!!!!!! Still need to get the measured data with noise somehow !!!!!!!!!!!
    if do_plot_conc_from_file or do_plot_conc_from_p:
        # Shape (num_data_points, S+1) with first column time (min).
        true_conc = np.loadtxt('data/true_conc.txt', delimiter='\t', skiprows=1)

    if do_plot_conc_from_file:
        # Shape (num_data_points, S) with first column time (min).
        pred_conc = np.loadtxt(conc_file)
        # Pass without time axis.
        plot_conc_from_file(true_conc, pred_conc)

    if do_plot_conc_from_p:
        if type(p) == str:
            p = list(float(param) for param in p.split())
        # Shape (num_data_points, S).
        pred_conc = get_data.glycolysis_model(true_conc[:, 0], p)
        plot_conc_from_p(true_conc, pred_conc)
