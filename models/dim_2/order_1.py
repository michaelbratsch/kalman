from kalman.kalman import Kalman
from kalman.plotter import Plot2dMixin
import numpy as np


class State1Mearsurement1(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = z
        self.P = R

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return None

    def F(self, dt):
        return np.identity(2)

    def H(self):
        return np.identity(2)

    def Q(self, dt):
        return self.plant_noise * np.identity(2)
