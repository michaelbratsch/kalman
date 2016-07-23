from filter.kalman import Kalman
from filter.plotter import Plot2dMixin
import numpy as np


class State4Measurement1(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = np.zeros(8)
        self.x[0:2] = z
        self.P = np.identity(8)
        self.P[0:2, 0:2] = R

    def F(self, dt):
        t_2 = 0.5 * dt**2
        t_3 = 1.0 / 6.0 * dt**3
        return np.array([[1.0, 0.0, dt,  0.0, t_2, 0.0, t_3, 0.0],
                         [0.0, 1.0, 0.0, dt,  0.0, t_2, 0.0, t_3],
                         [0.0, 0.0, 1.0, 0.0, dt,  0.0, t_2, 0.0],
                         [0.0, 0.0, 0.0, 1.0, 0.0, dt,  0.0, t_2],
                         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dt,  0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def H(self):
        return np.eye(N=2, M=8)

    def Q(self, dt):
        return self.plant_noise * np.identity(8)
