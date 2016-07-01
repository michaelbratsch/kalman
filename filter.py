from kalman import Kalman
import numpy as np
from plotter import Plot2dMixin


class Position2d(Kalman, Plot2dMixin):

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return None

    def F(self, dt):
        return np.array([[1.0, 0.0],
                         [0.0, 1.0]])

    def H(self, dt):
        return np.array([[1.0, 0.0],
                         [0.0, 1.0]])

    def Q(self, dt):
        return np.array([[1.0, 0.0],
                         [0.0, 1.0]])


class LowSpeed2d(Kalman, Plot2dMixin):

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return self.x[2], self.x[3]

    def F(self, dt):
        return np.array([[1.0, 0.0, dt, 0.0],
                         [0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def H(self, dt):
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]])

    def Q(self, dt):
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
