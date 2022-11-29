import numpy as np
from scipy.integrate import odeint
from torch.utils.data import Dataset


def apoptosis_model(t, p):
    """
    Solve apoptosis ODE for S concentrations over time given parameters in p.
    :param t: array of N time values
    :param p: list of K parameter values
    :return: NxS array of concentrations
    """
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
    return odeint(f, x0, t)


class Data(Dataset):
    """Dataset of concentrations of glycolysis chemical species over time."""

    def __init__(self, t_min=0, t_max=60, n_points=601):
        """Solve glycolysis ODE for concentrations over time."""
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
        self.time = np.linspace(t_min, t_max, n_points)
        self.conc = apoptosis_model(self.time, self.p0)

    def __len__(self):
        """Get number of samples in dataset."""
        return self.n_points

    def save_as_csv(self):
        """Save time and concentrations to data.csv file."""
        head = ['t', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
        np.savetxt('data.csv',
                   np.hstack((self.time[:, None], self.conc)),
                   header='\t'.join(head),
                   delimiter='\t',
                   comments='')