# Plot the apoptosis model.

import os
import math

import numpy as np
from matplotlib import pyplot as plt

import src.data_gen as get_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path

# ****************************************************************************
# Specify what is to be plotted.
# ****************************************************************************

# File name of predicted p to plot vs true p over epochs.
do_plot_p_vs_epoch = True
p_file = max([f for f in Path('data').glob('p_versus_epochs*.txt')], key=lambda item: item.stat().st_ctime)
# p_file = 'data/p_versus_epochs_12-01 082950.txt'

# File name file of losses to plot over epochs.
do_plot_loss_vs_epoch = True
loss_file = max([f for f in Path('data').glob('all_losses*.txt')], key=lambda item: item.stat().st_ctime)
# loss_file = 'data/all_losses_12-01 082950.txt'

# File name of concentrations to plot vs measured/true data over time.
do_plot_conc_from_file = True
conc_file = max([f for f in Path('data').glob('network_conc*.txt')], key=lambda item: item.stat().st_ctime)
# conc_file = 'data/network_conc_12-01 082950.txt'
entry = -1  # Index of the entry to plot, since the file may have many epochs.

# Predicted p for generating concentrations to plot vs measured/true data over
# time (can be a list of float or a whitespace separated string).
do_plot_conc_from_p = True
p = '96.76790453992906 28.33982489795185 29.89611985918972 42.65319430991449 206.11860210324397 20.617080337050393 27958.542711870636 -65.28772413001332 -57.983112928340965'
p_name = '11-30 191939'

# ****************************************************************************
# Constants.
# ****************************************************************************

# True p values.
k1 = 2.67e-9 * 3600 * 1e5
kd1 = 1e-2 * 3600
kd2 = 8e-3 * 3600
k3 = 6.8e-8 * 3600 * 1e5
kd3 = 5e-2 * 3600
kd4 = 1e-3 * 3600
k5 = 7e-5 * 3600 * 1e5
kd5 = 1.67e-5 * 3600
kd6 = 1.67e-4 * 3600
p0 = np.asarray([k1, kd1, kd2, k3, kd3, kd4, k5, kd5, kd6])

# Legend names.
names = ['k_1', 'k_{d1}', 'k_{d2}', 'k_3', 'k_{d3}', 'k_{d4}', 'k_5', 'k_{d5}', 'k_{d6}']

# Indices of parameters in value ranges.
near_0 = [7, 8]
near_2 = [0, 5]
near_25 = [1, 2, 3]
near_200 = [4]
near_1e4 = [6]

# Plot constants.
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
           'rebeccapurple', 'tab:pink']
leg_loc = (0.8, 0.05)

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
    p_sets = [near_0, near_2, near_25, near_200, near_1e4]
    fig, axs = plt.subplots(nrows=len(p_sets), ncols=1, figsize=(7, 24),
                            constrained_layout=True)
    row = 0
    for p_set in p_sets:
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
    measured = [2]  # Measured species (indexed at zero).
    n_species = len(true_conc[0, :]) - 1   # Infer number of chemical species.

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

    # Add legend and title.
    handles, labels = axs[row, col].get_legend_handles_labels()
    fig.legend(handles, labels, loc=leg_loc)
    fig.suptitle('Concentration predicted by network', fontsize=16)

    # Save plot
    plt.savefig(f'plots/network_conc_vs_true_{input_name}_entry_{index}.pdf',
                bbox_inches='tight')


def plot_conc_from_p(true_conc, meas_conc, pred_conc, input_name):
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
    measured = [2]  # Measured species (indexed at zero).
    n_species = len(true_conc[0, :]) - 1   # Infer number of chemical species.

    for ind in range(n_species):

        row, col = get_row_col(ind)

        # Plot.
        if ind in measured:
            axs[row, col].plot(meas_conc[:, 0], meas_conc[:, ind + 1],
                               color='b', label='Exact',
                               marker='o', markersize=3, linestyle='None')
        else:
            axs[row, col].plot(true_conc[:, 0], true_conc[:, ind + 1],
                               color='b', label='Exact')
        axs[row, col].plot(true_conc[:, 0], pred_conc[:, ind],
                           color='r', linestyle='dashdot', label='Learned')
        # Configure subplot.
        axs[row, col].set_xlabel('t (hours)')
        axs[row, col].set_ylabel(r'$S_{0}$ ($10^5$ molecules/cell)'.format(ind+1))
        # Autoscale axes to each subplot.
        axs[row, col].relim()
        axs[row, col].autoscale()

    # Add legend and title.
    handles, labels = axs[row, col].get_legend_handles_labels()
    fig.legend(handles, labels, loc=leg_loc)
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
        p_from_file = np.loadtxt(p_file)
        plot_p_vs_true_p(p_from_file, get_name(p_file))

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
        if len(p) != 9:
            msg = f'p has {len(p)} values but must have 9'
            raise ValueError(f'In do_plot_conc_from_p, {msg}')
        # Shape (num_data_points, S).
        pred_conc = get_data.apoptosis_model(true_conc[:, 0], p)
        plot_conc_from_p(true_conc, meas_conc, pred_conc, get_name(p_name))
