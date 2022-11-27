## Data for the glycolysis model
import numpy as np
import math
from scipy.integrate import odeint
from torch.utils.data import Dataset

def guess_p():
    # actual p values
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

    p0 = [J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi, N, A]

    noise = np.random.default_rng().normal(0, 1, len(p0))
    p0 = p0 + noise

    # log is done on K1 in gradient calculation so have to ensure it is not less than 0
    while p0[10] < 0:
        noise = np.random.default_rng().normal(0, 1, len(p0))
        p0 = p0 + noise

    return p0


def glycolysis_model(t, p, x=None):
    """
    Solve glycolysis ODE for S concentrations over time given parameters in p.

    :param t: array of N time values. If x is provided, N must be 1.
    :param p: list of K parameter values
    :param x: array of S concentrations if not solving model for >1 time point
    :return: NxS array of concentrations (or derivatives if x is provided)

    """

    # Assert dimensions are compatible.
    if x is not None and len(t) > 1:
        msg = 'If not solving glycolysis_model over time, provide one t value'
        raise ValueError(f'{msg}, not {len(t)} values')

    J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi, N, A = p

    def f(x, t):
        """Glycolysis ODE model."""
        v1 = k1 * x[0] * x[5] / (1 + max((x[5] / K1), 0.001) ** q)
        v2 = k2 * x[1] * (N - x[4])
        v3 = k3 * x[2] * (A - x[5])
        v4 = k4 * x[3] * x[4]
        v5 = k5 * x[5]
        v6 = k6 * x[1] * x[4]
        v7 = k * x[6]
        J = kappa * (x[3] - x[6])
        return [
            J0 - v1,
            2 * v1 - v2 - v6,
            v2 - v3,
            v3 - v4 - J,
            v2 - v4 - v6,
            -2 * v1 + 2 * v3 - v5,
            psi * J - v7,
        ]

    # Initial concentration of each species.
    x0 = [
        0.50144272,
        1.95478666,
        0.19788759,
        0.14769148,
        0.16059078,
        0.16127341,
        0.06404702,
    ]

    # Solve model for all time points if no x is provided.
    if x is None:
        res = odeint(f, x0, t)
    # Otherwise, find the time derivatives of each species in x at t given p.
    else:
        res = f(x, t)
    return res


def grad_glycolysis_model(t, p, x):
    """
    Find gradient of the glycolysis ODE wrt p for S concentrations at time t.

    :param t: float
    :param p: list of K parameter values
    :param x: array of S concentrations
    :return: KxS array of gradient values

    """
    J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi, N, A = p

    d_v1_comp = ((k1 * x[0] * x[5] / (1 + (max((x[5] / K1), 0.001))**q)**2) *
                 (max((x[5] / K1), 0.001))**q)

    d_v1_d_k1 = x[0] * x[5] / (1 + (max((x[5] / K1), 0.001)) ** q)
    d_v2_d_k2 = x[1] * (N - x[4])
    d_v3_d_k3 = x[2] * (A - x[5])
    d_v4_d_k4 = x[3] * x[4]
    d_v5_d_k5 = x[5]
    d_v6_d_k6 = x[1] * x[4]
    d_v7_d_k = x[6]
    d_J_d_kappa = x[3] - x[6]
    d_v1_d_q = -d_v1_comp * math.log(max((x[5] / K1), 0.001))
    d_v1_d_K1 = d_v1_comp * q / max(K1, 0.001)
    d_v2_d_N = k2 * x[1]
    d_v3_dA = k3 * x[2]

    d_f_d_J0 = [1, 0, 0, 0, 0, 0, 0]
    d_f_d_k1 = [-d_v1_d_k1, 2 * d_v1_d_k1, 0, 0, 0, -2 * d_v1_d_k1, 0]
    d_f_d_k2 = [0, -d_v2_d_k2, d_v2_d_k2, 0, d_v2_d_k2, 0, 0]
    d_f_d_k3 = [0, 0, -d_v3_d_k3, d_v3_d_k3, 0, 2 * d_v3_d_k3, 0]
    d_f_d_k4 = [0, 0, 0, -d_v4_d_k4, -d_v4_d_k4, 0, 0]
    d_f_d_k5 = [0, 0, 0, 0, 0, -d_v5_d_k5, 0]
    d_f_d_k6 = [0, -d_v6_d_k6, 0, 0, -d_v6_d_k6, 0, 0]
    d_f_d_k = [0, 0, 0, 0, 0, 0, -d_v7_d_k]
    d_f_d_kappa = [0, 0, 0, -d_J_d_kappa, 0, 0, psi * d_J_d_kappa]
    d_f_d_q = [-d_v1_d_q, 2 * d_v1_d_q, 0, 0, 0, -2 * d_v1_d_q, 0]
    d_f_d_K1 = [-d_v1_d_K1, 2 * d_v1_d_K1, 0, 0, 0, -2 * d_v1_d_K1, 0]
    d_f_d_psi = [0, 0, 0, 0, 0, 0, kappa * (x[3] - x[6])]
    d_f_d_N = [0, -d_v2_d_N, d_v2_d_N, 0, d_v2_d_N, 0, 0]
    d_f_d_A = [0, 0, -d_v3_dA, d_v3_dA, 0, 2 * d_v3_dA, 0]

    return np.asarray([
        d_f_d_J0,
        d_f_d_k1,
        d_f_d_k2,
        d_f_d_k3,
        d_f_d_k4,
        d_f_d_k5,
        d_f_d_k6,
        d_f_d_k,
        d_f_d_kappa,
        d_f_d_q,
        d_f_d_K1,
        d_f_d_psi,
        d_f_d_N,
        d_f_d_A
        ])


class Data(Dataset):
    """Dataset of concentrations of glycolysis chemical species over time."""

    def __init__(self, params, t_min=0, t_max=10, n_points=2001):
        """Solve glycolysis ODE for concentrations over time."""
        # Correct parameter values.
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
        self.p0 = [J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi, N, A]

        # Find correct solution.
        self.time = np.linspace(t_min, t_max, n_points, dtype=np.float64)  # 1D np array size n_points
        self.conc = glycolysis_model(self.time, self.p0)  # 2D np array size n_points x 7 (7 species)

        data_loss_train_size = params['num_data_loss']
        ode_loss_train_size = params['num_ode_loss']  # cant find in the paper where they specify this set size

        # get times for training data

        # get data_loss_train_size random time points for training the data loss
        rand_indexes = np.random.default_rng().choice(self.conc[:, 0].size, data_loss_train_size, replace=False)
        rand_indexes = np.sort(rand_indexes)  # random time points

        # get ode_loss_train_size equispaced time points for training the ode loss
        ode_loss_indexes = np.round(np.linspace(0, len(self.conc[:, 0]) - 1, ode_loss_train_size)).astype(int)

        self.data_inputs = self.time[rand_indexes]
        self.data_labels = self.conc[rand_indexes]

        self.aux_inputs = np.asarray([self.time[0], self.time[-1]])
        self.aux_labels = np.asarray([[x, y] for x, y in zip(self.conc[0, :], self.conc[-1, :])])

        self.ode_inputs = self.time[ode_loss_indexes]

    def __len__(self):
        """Get number of samples in dataset."""
        return self.n_points

    def save_as_csv(self):
        """Save time and concentrations to data.csv file."""
        head = ['t', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
        np.savetxt('src/data.csv',
                   np.hstack((self.time[:, None], self.conc)),
                   header='\t'.join(head),
                   delimiter='\t',
                   comments='')
