from filter.kalman import Kalman
from filter.plotter import Plot2dMixin
import numpy as np


class State1Mearsurement1(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = z
        self.P = R

    def get_speed(self):
        return None

    def F(self, dt):
        return np.identity(2)

    def H(self):
        return np.identity(2)

    def Q(self, dt):
        return self.plant_noise * np.identity(2)
