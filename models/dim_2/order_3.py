from filter.kalman import Kalman
from filter.plotter import Plot2dMixin
import numpy as np


class State3Measurement1(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = np.zeros(6)
        self.x[0:2] = z
        self.P = np.identity(6)
        self.P[0:2, 0:2] = R

    def F(self, dt):
        t_2 = 0.5 * dt**2
        return np.array([[1.0, 0.0, dt,  0.0, t_2, 0.0],
                         [0.0, 1.0, 0.0, dt,  0.0, t_2],
                         [0.0, 0.0, 1.0, 0.0, dt,  0.0],
                         [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def H(self):
        return np.eye(N=2, M=6)

    def Q(self, dt):
        t_5 = 0.05 * dt**5
        t_4 = 0.125 * dt**4
        t_3 = 1.0 / 3.0 * dt**3
        t_2 = 0.5 * dt**2
        # plant noise [m^2/s^5]
        return self.plant_noise * np.array([[t_5, 0.0, t_4, 0.0, t_3, 0.0],
                                            [0.0, t_5, 0.0, t_4, 0.0, t_3],
                                            [t_4, 0.0, t_3, 0.0, t_2, 0.0],
                                            [0.0, t_4, 0.0, t_3, 0.0, t_2],
                                            [t_3, 0.0, t_2, 0.0, dt,  0.0],
                                            [0.0, t_3, 0.0, t_2, 0.0, dt]])
