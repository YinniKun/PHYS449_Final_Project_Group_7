# Plot the glycolysis model.

import os
import math

import numpy as np
from matplotlib import pyplot as plt

import src.data_gen as get_data

# ****************************************************************************
# Specify what is to be plotted.
# ****************************************************************************

# File name of predicted p to plot vs true p over epochs.
do_plot_p_vs_epoch = False
p_file = 'data/p_versus_epochs_11-28 203141.txt'

# File name file of losses to plot over epochs.
do_plot_loss_vs_epoch = False
loss_file = 'data/all_losses_11-28 203141.txt'

# File name of concentrations to plot vs measured/true data over time.
do_plot_conc_from_file = False
conc_file = 'data/network_conc_11-28 203141.txt'
entry = 5  # Index of the entry to plot, since the file may have many epochs.

# Predicted p for generating concentrations to plot vs measured/true data over
# time (can be a list of float or a whitespace separated string).
do_plot_conc_from_p = False
p = '-0.6762044145348005 106.23339162161375 35.28790351907249 57.64332159931993 132.96802797215415 2.2439889854315402 13.39090120928569 2.5826227671061948 40.656343141279045 7.215781778386592 7.345766312761084 38.875775469197265 35.08737720011627 57.81059180122517'
p_name = '11-28 203141'

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

# Legend names.
names = ['J_0', 'k_1', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'k', '\kappa', 'q', 'K1', '\psi', 'N', 'A']

# Indices of parameters in value ranges.
near_0 = [0, 5, 7, 10, 11, 12]
near_5 = [2, 9, 13]
near_10 = [3, 6, 8]
near_100 = [1, 4]

# Plot colours.
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
           'rebeccapurple', 'tab:pink']


def plot_loss(loss, input_name):
    """Plot loss values over epochs.

    Plot is named losses_input_name.pdf.

    :param p: (num_epochss, K+1) array of predicted parameters and time
    :param input_name: name to append to end of plot name

    """
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 12),
                            constrained_layout=True)
    loss_set = ['Data', 'Auxiliary', 'ODE']
    for row, name in enumerate(loss_set):
        axs[row].plot(loss[:, -1], loss[:, row], label=f'{name} loss')
        # Configure subplot.
        axs[row].set_xlabel('Epoch')
        axs[row].set_ylabel('Mean square loss')
        # Autoscale axes to each subplot.
        axs[row].relim()
        axs[row].autoscale()
        axs[row].legend()

    # Add title.
    fig.suptitle('Training loss', fontsize=16)

    # Save plot
    plt.savefig(f'plots/losses_{input_name}.pdf', bbox_inches='tight')


def plot_p_vs_true_p(p, input_name):
    """Plot p values vs true p values over epochs.

    Plot is named pred_p_vs_true_input_name.pdf.

    :param p: (num_epochss, K+1) array of predicted parameters and time
    :param input_name: name to append to end of plot name

    """
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(7, 24),
                            constrained_layout=True)
    row = 0
    for p_set in [near_0, near_5, near_10, near_100]:
        c_num = 0
        for ind in p_set:
            axs[row].axhline(y=p0[ind],
                             color=colours[c_num],
                             label=r'Exact ${0}={1}$'.format(names[ind], p0[ind]))
            axs[row].plot(p[:, -1], p[:, ind],
                          color=colours[c_num], linestyle='dashdot',
                          label=r'Learned ${0}$'.format(names[ind]))
            # Configure subplot.
            axs[row].set_xlabel('Epoch')
            axs[row].set_ylabel('Parameter value (unitless)')
            # Autoscale axes to each subplot.
            axs[row].relim()
            axs[row].autoscale()
            c_num += 1
        axs[row].legend()
        row += 1

    # Add title.
    fig.suptitle('Parameter values predicted by network', fontsize=16)

    # Save plot
    plt.savefig(f'plots/pred_p_vs_true_{input_name}.pdf', bbox_inches='tight')


def plot_conc_from_file(true_conc, meas_conc, pred_conc, input_name, index=-1):
    """Plot predicted concentrations from p vs measured/true data over time.

    Generate one figure with subplots of the concentration of each chemical
    species. Plot is named network_conc_vs_true_input_name_index.pdf.

    :param true_conc: (num_data_points, S+1) array of time and true concs
    :param meas_conc: (num_data_points, S+1) array of time and measured concs
    :param pred_conc: (num_data_points, S+1) array of time and predicted concs
    :param input_name: name to append to end of plot name
    :param index: index of conc_file entry (default is last entry)

    """
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 24),
                            constrained_layout=True)

    measured = [4, 5]  # Measured species (indexed at zero).
    n_species = len(pred_conc[0, :]) - 1   # Infer number of chemical species.

    # Tuple of indices when a new entry of concentrations begin.
    entry_boundaries = np.where(pred_conc[:, 0] == pred_conc[:, 0][0])[0]
    n_entries = len(entry_boundaries)  # Number of entries in entire conc_file.

    # Find where to slice pred_conc to isolate the entry of interest.
    if index >= n_entries:
        msg = f'index is {index} but must be <={n_entries}'
        raise ValueError(f'In plot_conc_from_file, {msg}')
    start = entry_boundaries[index]
    if n_entries > 1:
        stop = start + entry_boundaries[1] - entry_boundaries[0]
    else:
        stop = None  # Slice until end.

    for ind in range(n_species):

        row, col = get_row_col(ind)

        # Plot.
        if ind in measured:
            axs[row, col].plot(meas_conc[:, 0], meas_conc[:, ind + 1],
                               color='b', label='Exact',
                               marker='o', markersize=3, linestyle='None')
            print()
        else:
            axs[row, col].plot(true_conc[:, 0], true_conc[:, ind + 1],
                               color='b', label='Exact')
        axs[row, col].plot(pred_conc[:, 0][start:stop],
                           pred_conc[:, ind + 1][start:stop],
                           color='r', linestyle='dashdot', label='Learned')
        # Configure subplot.
        axs[row, col].set_xlabel('t (min)')
        axs[row, col].set_ylabel(r'$S_{0}$ (mM)'.format(ind+1))
        # Autoscale axes to each subplot.
        axs[row, col].relim()
        axs[row, col].autoscale()

    # Delete unused bottom right subplot and add legend and title.
    handles, labels = axs[row, col].get_legend_handles_labels()
    col += 1
    fig.delaxes(axs[row, col])
    fig.legend(handles, labels, loc=(0.6, 0.2))
    fig.suptitle('Concentration predicted by network', fontsize=16)

    # Save plot
    plt.savefig(f'plots/network_conc_vs_true_{input_name}_entry_{index}.pdf',
                bbox_inches='tight')


def plot_conc_from_p(true_conc, pred_conc, input_name):
    """Plot predicted concentrations from p vs measured/true data over time.

    Generate one figure with subplots of the concentration of each chemical
    species. Plot is named pred_conc_from_p_vs_true_input_name.pdf.

    :param true_conc: (num_data_points, S+1) array of time and true concs
    :param meas_conc: (num_data_points, S+1) array of time and measured concs
    :param pred_conc: (num_data_points, S) array of predicted concs
    :param input_name: name to append to end of plot name

    """
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 24),
                            constrained_layout=True)
    measured = [4, 5]  # Measured species (indexed at zero).
    n_species = len(pred_conc[0, :]) - 1   # Infer number of chemical species.

    for ind in range(n_species):

        row, col = get_row_col(ind)

        # Plot.
        if ind in measured:
            axs[row, col].plot(meas_conc[:, 0], meas_conc[:, ind + 1],
                               color='b', label='Exact')
        else:
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

    # Delete unused bottom right subplot and add legend and title.
    handles, labels = axs[row, col].get_legend_handles_labels()
    col += 1
    fig.delaxes(axs[row, col])
    fig.legend(handles, labels, loc=(0.6, 0.2))
    fig.suptitle('Analytical concentration from predicted parameters', fontsize=16)

    # Save plot
    plt.savefig(f'plots/pred_conc_from_p_vs_true_{input_name}.pdf',
                bbox_inches='tight')


def get_row_col(ind):
    """Find row, col of the given species index (0 to n_species-1)."""
    return math.floor(ind / 2), ind % 2


def get_name(file):
    """Get component of file name that follows the final underscore."""
    return os.path.splitext(os.path.basename(file))[0].split('_')[-1]


if __name__ == '__main__':

    plots_dir = f'{os.path.dirname(__file__)}{os.sep}plots'
    if not os.path.isdir(plots_dir):
        print(f'Created directory: {plots_dir}')
        os.mkdir(plots_dir)

    if do_plot_loss_vs_epoch:
        if not os.path.isfile(f'{loss_file}'):
            raise ValueError(f'Cannot find loss_file {loss_file}')
        # Shape (num_epochs, 4) with last column time (epochs).
        loss = np.loadtxt(loss_file)
        plot_loss(loss, get_name(loss_file))

    if do_plot_p_vs_epoch:
        if not os.path.isfile(f'{p_file}'):
            raise ValueError(f'Cannot find p_file {p_file}')
        # Shape (num_epochs, K+1) with last column time (epochs).
        p = np.loadtxt(p_file)
        plot_p_vs_true_p(p, get_name(p_file))

    # Only read true/measured concentration once if it is needed.
    if do_plot_conc_from_file or do_plot_conc_from_p:
        # Shape (num_data_points, S+1) with first column time (min).
        true_conc = np.loadtxt('data/true_conc.txt',
                               delimiter='\t', skiprows=1)
        # Shape (num_data_points, S+1) with first column time (min).
        meas_conc = np.loadtxt('data/true_conc.txt',
                               delimiter='\t', skiprows=1)

    if do_plot_conc_from_file:
        if not os.path.isfile(f'{conc_file}'):
            raise ValueError(f'Cannot find conc_file {conc_file}')
        # Shape (num_data_points, S) with first column time (min).
        pred_conc = np.loadtxt(conc_file)
        plot_conc_from_file(true_conc, meas_conc, pred_conc,
                            get_name(conc_file), index=entry)

    if do_plot_conc_from_p:
        if type(p) == str:
            p = list(float(param) for param in p.split())
        if len(p) != 14:
            msg = f'p has {len(p)} values but must have 14'
            raise ValueError(f'In do_plot_conc_from_p, {msg}')
        # Shape (num_data_points, S).
        pred_conc = get_data.glycolysis_model(true_conc[:, 0], p)
        plot_conc_from_p(true_conc, meas_conc, pred_conc, get_name(p_name))
