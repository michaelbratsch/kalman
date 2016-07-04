from kalman import Kalman
import numpy as np
from plotter import Plot2dMixin


class State1Mearsurement1_2d(Kalman, Plot2dMixin):

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return None

    def F(self, dt):
        return np.identity(2)

    def H(self, dt):
        return np.identity(2)

    def Q(self, dt):
        return np.identity(2)


class State2Measurement1_2d(Kalman, Plot2dMixin):

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return self.x[2], self.x[3]

    def F(self, dt):
        return np.array([[1.0, 0.0, dt,  0.0],
                         [0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def H(self, dt):
        return np.eye(N=2, M=4)

    def Q(self, dt):
        return np.identity(4)


class State3Measurement1_2d(Kalman, Plot2dMixin):

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return self.x[2], self.x[3]

    def F(self, dt):
        t_2 = 0.5 * dt**2
        return np.array([[1.0, 0.0, dt,  0.0, t_2, 0.0],
                         [0.0, 1.0, 0.0, dt,  0.0, t_2],
                         [0.0, 0.0, 1.0, 0.0, dt,  0.0],
                         [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def H(self, dt):
        return np.eye(N=2, M=6)

    def Q(self, dt):
        return np.identity(6)


class State4Measurement1_2d(Kalman, Plot2dMixin):

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return self.x[2], self.x[3]

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

    def H(self, dt):
        return np.eye(N=2, M=8)

    def Q(self, dt):
        return np.identity(8)
