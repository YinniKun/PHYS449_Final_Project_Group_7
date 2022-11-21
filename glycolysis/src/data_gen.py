## Data for the glycolysis model
import numpy as np
from scipy.integrate import odeint
from torch.utils.data import Dataset


def glycolysis_model(t, p):
    """
    Solve glycolysis ODE for S concentrations over time given parameters in p.

    :param t: array of N time values
    :param p: list of K parameter values
    :return: NxS array of concentrations
    """
    J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi, N, A = p

    def f(x, t):
        """Glycolysis ODE model."""
        v1 = k1 * x[0] * x[5] / (1 + (x[5] / K1) ** q)
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
    return odeint(f, x0, t)


class Data(Dataset):
    """Dataset of concentrations of glycolysis chemical species over time."""

    def __init__(self, t_min=0, t_max=10, n_points=2001):
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
        self.time = np.linspace(t_min, t_max, n_points)  # 1D np array size n_points
        self.conc = glycolysis_model(self.time, self.p0)  # 2D np array size n_points x 7 (7 species)

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

