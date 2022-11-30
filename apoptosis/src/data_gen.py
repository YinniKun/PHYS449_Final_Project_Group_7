## Data for the apoptosis model
import numpy as np
import math
from scipy.integrate import odeint
from torch.utils.data import Dataset

def guess_p():
    # actual p values
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
    noise = np.asarray([np.random.default_rng().normal(0, p_i * 0.15) for p_i in p0])
    p = p0 + noise

    return p


def apoptosis_model(t, p, x=None):
    """
    Solve apoptosis ODE for S concentrations over time given parameters in p.

    :param t: array of N time values. If x is provided, N must be 1.
    :param p: list of K parameter values
    :param x: array of S concentrations if not solving model for >1 time point
    :return: NxS array of concentrations (or derivatives if x is provided)

    """

    # Assert dimensions are compatible.
    if x is not None and len(t) > 1:
        msg = 'If not solving apoptosis_model over time, provide one t value'
        raise ValueError(f'{msg}, not {len(t)} values')

    k1, kd1, kd2, k3, kd3, kd4, k5, kd5, kd6 = p

    def f(x, t):
        """Apoptosis ODE model."""
        v4_1 = kd1 * x[4]
        v4_2 = kd2 * x[4]
        v5_3 = kd3 * x[5]
        v5_4 = kd4 * x[5]
        v7_5 = kd5 * x[7]
        v7_6 = kd6 * x[7]
        v03 = k1 * x[3] * x[0]
        v12 = k3 * x[1] * x[2]
        v36 = k5 * x[6] * x[3]
        return [
            -v03 + v4_1,
            v4_2 - v12 + v5_3 + v5_4,
            -v12 + v5_3,
            v5_4 - v03 + v4_1 - v36 + v7_5 + v4_2,
            -v4_2 + v03 - v4_1,
            -v5_4 + v12 - v5_3,
            -v36 + v7_5 + v7_6,
            v36 - v7_5 - v7_6,
        ]

    # Initial non-dimensional concentration of each species. (death)
    x0 = [
        1.34,
        1.0,
        2.67,
        0.0,
        0.0,
        0.0,
        2.9e-2,
        0.0
    ]

    # Solve model for all time points if no x is provided.
    if x is None:
        res = odeint(f, x0, t)
    # Otherwise, find the time derivatives of each species in x at t given p.
    else:
        res = f(x, t)
    return res


def grad_apoptosis_model(t, p, x):
    """
    Find gradient of the apoptosis ODE wrt p for S concentrations at time t.

    :param t: float
    :param p: list of K parameter values
    :param x: array of S concentrations
    :return: KxS array of gradient values

    """
    k1, kd1, kd2, k3, kd3, kd4, k5, kd5, kd6 = p

    d_v03_d_k1 = x[3] * x[0]
    d_v4_1_d_kd1 = x[4]
    d_v4_2_d_kd2 = x[4]
    d_v12_d_k3 = x[1] * x[2]
    d_v5_3_d_kd3 = x[5]
    d_v5_4_d_kd4 = x[5]
    d_v36_d_k5 = x[6] * x[3]
    d_v7_5_d_kd5 = x[7]
    d_v7_6_d_kd6 = x[7]

    d_f_d_k1 = [-d_v03_d_k1, 0, 0, -d_v03_d_k1, d_v03_d_k1, 0, 0, 0]
    d_f_d_kd1 = [d_v4_1_d_kd1, 0, 0, d_v4_1_d_kd1, -d_v4_1_d_kd1, 0, 0, 0]
    d_f_d_kd2 = [0, d_v4_2_d_kd2, 0, d_v4_2_d_kd2, -d_v4_2_d_kd2, 0, 0, 0]
    d_f_d_k3 = [0, -d_v12_d_k3, -d_v12_d_k3, 0, 0, d_v12_d_k3, 0, 0]
    d_f_d_kd3 = [0, d_v5_3_d_kd3, d_v5_3_d_kd3, 0, 0, -d_v5_3_d_kd3, 0, 0]
    d_f_d_kd4 = [0, d_v5_4_d_kd4, 0, d_v5_4_d_kd4, 0, -d_v5_4_d_kd4, 0, 0]
    d_f_d_k5 = [0, 0, 0, -d_v36_d_k5, 0, 0, -d_v36_d_k5, d_v36_d_k5]
    d_f_d_kd5 = [0, 0, 0, d_v7_5_d_kd5, 0, 0, d_v7_5_d_kd5, -d_v7_5_d_kd5]
    d_f_d_k6 = [0, 0, 0, 0, 0, 0, d_v7_6_d_kd6, -d_v7_6_d_kd6]

    return np.asarray([
        d_f_d_k1,
        d_f_d_kd1,
        d_f_d_kd2,
        d_f_d_k3,
        d_f_d_kd3,
        d_f_d_kd4,
        d_f_d_k5,
        d_f_d_kd5,
        d_f_d_k6
        ])


class Data(Dataset):
    """Dataset of concentrations of apoptosis chemical species over time."""

    def __init__(self, params, t_min=0, t_max=60, n_points=601):
        """Solve apoptosis ODE for concentrations over time."""
        # Correct parameter values.
        k1 = 2.67e-9 * 3600 * 1e5
        kd1 = 1e-2 * 3600
        kd2 = 8e-3 * 3600
        k3 = 6.8e-8 * 3600 * 1e5
        kd3 = 5e-2 * 3600
        kd4 = 1e-3 * 3600
        k5 = 7e-5 * 3600 * 1e5
        kd5 = 1.67e-5 * 3600
        kd6 = 1.67e-4 * 3600
        self.p0 = [k1, kd1, kd2, k3, kd3, kd4, k5, kd5, kd6]

        # Find correct solution.
        self.time = np.linspace(t_min, t_max, n_points, dtype=np.float64)  # 1D np array size n_points
        self.conc = apoptosis_model(self.time, self.p0)  # 2D np array size n_points x n_species

        data_loss_train_size = params['num_data_loss']
        ode_loss_train_size = params['num_ode_loss']  # cant find in the paper where they specify this set size
        noisy_bool = params['noisy?']

        # Create noise and apply to correct concentration time points
        if noisy_bool:
            conc_tr = np.transpose(self.conc)
            mu_conc = np.std(conc_tr[3])
            conc_noise = np.empty((2000, 1))
            for i in range(0, conc_tr[0].size):
                conc_noise[i] = np.random.default_rng().normal(0, 0.05 * mu_conc)
            self.conc[:, 3] = self.conc[:, 3] + conc_noise[:, 0]
            print("\nNoisy Concentration\n")

        # get times for training data

        # get data_loss_train_size random time points for training the data loss
        rand_indexes = np.random.default_rng().choice(self.conc[:, 0].size, data_loss_train_size, replace=False)
        rand_indexes = np.sort(rand_indexes)  # random time points

        # get ode_loss_train_size equispaced time points for training the ode loss
        ode_loss_indexes = np.round(np.linspace(0, len(self.conc[:, 0]) - 1, ode_loss_train_size)).astype(int)

        self.data_inputs = self.time[rand_indexes]
        self.data_labels = self.conc[rand_indexes]

        self.aux_inputs = np.asarray([self.time[0], self.time[int(self.time.size / 2)]])
        self.aux_labels = np.asarray([[x, y] for x, y in zip(self.conc[0, :], self.conc[-1, :])])

        self.ode_inputs = self.time[ode_loss_indexes]

    def __len__(self):
        """Get number of samples in dataset."""
        return self.n_points

    def save_as_txt(self):
        """Save time and concentrations to txt files in data folder."""
        head = ['t', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
        np.savetxt('data/true_conc.txt',
                   np.hstack((self.time[:, None], self.conc)),
                   header='\t'.join(head),
                   delimiter='\t',
                   comments='')
        np.savetxt('data/meas_conc.txt',
                   np.hstack((self.data_inputs[:, None], self.data_labels)),
                   header='\t'.join(head),
                   delimiter='\t',
                   comments='')
