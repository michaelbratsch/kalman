import math

from kalman import Kalman
import numpy as np
from plotter import Plot2dMixin


class State1Mearsurement1_2d(Kalman, Plot2dMixin):

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


class State2Measurement1_2d(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = np.zeros(4)
        self.x[0:2] = z
        self.P = np.identity(4)
        self.P[0:2, 0:2] = R

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

    def H(self):
        return np.eye(N=2, M=4)

    def Q(self, dt):
        t_3 = 1.0 / 3.0 * dt**3
        t_2 = 0.5 * dt**2
        return self.plant_noise * np.array([[t_3, 0.0, t_2, 0.0],
                                            [0.0, t_3, 0.0, t_2],
                                            [t_2, 0.0, dt,  0.0],
                                            [0.0, t_2, 0.0, dt]])


class State2_With_TurnRate_Measurement1_2d(Kalman, Plot2dMixin):

    def __init__(self, turn_rate, *args, **kwargs):
        super(State2_With_TurnRate_Measurement1_2d, self).__init__(
            *args, **kwargs)
        self.turn_rate = turn_rate

    def initialize_state(self, z, R):
        self.x = np.zeros(4)
        self.x[0:2] = z
        self.P = np.identity(4)
        self.P[0:2, 0:2] = R

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return self.x[2], self.x[3]

    def F(self, dt):
        omega = self.turn_rate
        sinOt = math.sin(omega) * dt
        cosO = math.cos(omega)
        OcosO = 1.0 - cosO
        return np.array([[1.0, 0.0, sinOt / omega, -OcosO / omega],
                         [0.0, 1.0, OcosO / omega, sinOt / omega],
                         [0.0, 0.0, cosO, -sinOt],
                         [0.0, 0.0, sinOt, cosO]])

    def H(self):
        return np.eye(N=2, M=4)

    def Q(self, dt):
        t_3 = 1.0 / 3.0 * dt**3
        t_2 = 0.5 * dt**2
        return self.plant_noise * np.array([[t_3, 0.0, t_2, 0.0],
                                            [0.0, t_3, 0.0, t_2],
                                            [t_2, 0.0, dt,  0.0],
                                            [0.0, t_2, 0.0, dt]])


class State3Measurement1_2d(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = np.zeros(6)
        self.x[0:2] = z
        self.P = np.identity(6)
        self.P[0:2, 0:2] = R

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

    def H(self):
        return np.eye(N=2, M=6)

    def Q(self, dt):
        t_5 = 0.05 * dt**5
        t_4 = 0.125 * dt**4
        t_3 = 1.0 / 3.0 * dt**3
        t_2 = 0.5 * dt**2
        return self.plant_noise * np.array([[t_5, 0.0, t_4, 0.0, t_3, 0.0],
                                            [0.0, t_5, 0.0, t_4, 0.0, t_3],
                                            [t_4, 0.0, t_3, 0.0, t_2, 0.0],
                                            [0.0, t_4, 0.0, t_3, 0.0, t_2],
                                            [t_3, 0.0, t_2, 0.0, dt,  0.0],
                                            [0.0, t_3, 0.0, t_2, 0.0, dt]])


class State4Measurement1_2d(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = np.zeros(8)
        self.x[0:2] = z
        self.P = np.identity(8)
        self.P[0:2, 0:2] = R

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

    def H(self):
        return np.eye(N=2, M=8)

    def Q(self, dt):
        return self.plant_noise * np.identity(8)
