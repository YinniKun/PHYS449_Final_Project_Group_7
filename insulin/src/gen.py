import numpy as np
from scipy.integrate import odeint
from torch.utils.data import Dataset


def glucose_insulin_model(t, meal_t, meal_q, p):
    """
    Solve glycolysis ODE for S concentrations over time given parameters in p.
    :param t: array of N time values
    :param p: list of K parameter values
    :return: NxS array of concentrations
    """
    Vp, Vi, Vg, E,  tp, ti, td, k, Rm, a1, C1, C2, C3, C4, C5, Ub, U0, Um, Rg, alpha, beta = p

    def f(y, t):
        """Glucose Insulin ODE model."""
        f1 = Rm / (1 + np.exp(-y[2] / Vg / C1 + a1))
        f2 = Ub * (1 - np.exp(-y[2] / Vg / C2))
        kappa = (1 / Vi + 1 / E / ti) / C4
        f3 = (U0 + Um / (1 + (kappa * y[1]) ** (-beta))) / Vg / C3
        f4 = Rg / (1 + np.exp(alpha * (y[5] / Vp / C5 - 1)))
        IG = np.sum(
            meal_q * k * np.exp(k * (meal_t - t)) * np.heaviside(t - meal_t, 0.5)
        )
        tmp = E * (y[0] / Vp - y[1] / Vi)
        return [
            f1 - tmp - y[0] / tp,
            tmp - y[1] / ti,
            f4 + IG - f2 - f3 * y[2],
            (y[0] - y[3]) / td,
            (y[3] - y[4]) / td,
            (y[4] - y[5]) / td,
        ]

    # Initial concentration of each species.
    Vp0, Vi0, Vg0 = 3, 11, 10
    y0 = [12 * Vp0, 4 * Vi0, 110 * Vg0 ** 2, 0, 0, 0]
    return odeint(f, y0, t)


class Data(Dataset):
    """Dataset of concentrations of glycolysis chemical species over time."""

    def __init__(self, t_min=0, t_max=3000, n_points=3001):
        """Solve glycolysis ODE for concentrations over time."""
        # Correct parameter values.
        Vp = 3
        Vi = 11
        Vg = 10
        E = 0.2
        tp = 6
        ti = 100
        td = 12
        k = 1 / 120
        Rm = 209
        a1 = 6.6
        C1 = 300
        C2 = 144
        C3 = 100
        C4 = 80
        C5 = 26
        Ub = 72
        U0 = 4
        Um = 90
        Rg = 180
        alpha = 7.5
        beta = 1.772
        self.p0 = [Vp, Vi, Vg, E, tp, ti, td, k, Rm, a1, C1, C2, C3, C4, C5, Ub, U0, Um, Rg, alpha, beta]

        # Find correct solution.
        self.time = np.linspace(t_min, t_max, n_points)
        self.meal_t = np.array([300, 650, 1100, 2000])
        self.meal_q = np.array([60e3, 40e3, 50e3, 100e3])
        self.conc = glucose_insulin_model(self.time, self.meal_t, self.meal_q, self.p0)
        self.IG = np.sum(
            self.meal_t * k * np.exp(k * (self.meal_t - self.t)) * np.heaviside(self.t - self.meal_t, 0.5)
        )

    def __len__(self):
        """Get number of samples in dataset."""
        return self.n_points

    def save_as_csv(self):
        """Save time and concentrations to data.csv file."""
        head = ['t', 'Vp0', 'Vi0', 'Vg0', 'IG']
        np.savetxt('data.csv',
                   np.hstack((self.time[:, None], self.conc, self.IG)),
                   header='\t'.join(head),
                   delimiter='\t',
                   comments='')