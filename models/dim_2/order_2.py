import math

from filter.kalman import Kalman
from filter.plotter import Plot2dMixin
import numpy as np


class State2Measurement1Base(Kalman, Plot2dMixin):

    def initialize_state(self, z, R):
        self.x = np.zeros(4)
        self.x[0:2] = z
        self.P = np.identity(4)
        self.P[0:2, 0:2] = R

    def H(self):
        return np.eye(N=2, M=4)

    def Q(self, dt):
        t_3 = 1.0 / 3.0 * dt**3
        t_2 = 0.5 * dt**2
        # plant noise [m^2/s^3]
        return self.plant_noise * np.array([[t_3, 0.0, t_2, 0.0],
                                            [0.0, t_3, 0.0, t_2],
                                            [t_2, 0.0, dt,  0.0],
                                            [0.0, t_2, 0.0, dt]])


class State2Measurement1(State2Measurement1Base):

    def F(self, dt):
        return np.array([[1.0, 0.0,  dt, 0.0],
                         [0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])


class State2Measurement1PerfectTurn(State2Measurement1Base):

    def __init__(self, turn_rate, *args, **kwargs):
        super(State2Measurement1PerfectTurn, self).__init__(
            *args, **kwargs)
        assert turn_rate != 0.0, "Turn-rate can not be set to zero."
        # turn rate [rad/s]
        self.turn_rate = turn_rate

    def F(self, dt):
        omega = self.turn_rate
        sinOt = math.sin(omega * dt)
        cosOt = math.cos(omega * dt)
        OcosOt = 1.0 - cosOt
        return np.array([[1.0, 0.0, sinOt / omega, -OcosOt / omega],
                         [0.0, 1.0, OcosOt / omega, sinOt / omega],
                         [0.0, 0.0, cosOt, -sinOt],
                         [0.0, 0.0, sinOt, cosOt]])


class State2Measurement1TwistedTurn(State2Measurement1Base):

    def __init__(self, turn_rate, *args, **kwargs):
        super(State2Measurement1TwistedTurn, self).__init__(
            *args, **kwargs)
        assert turn_rate != 0.0, "Turn-rate can not be set to zero."
        # turn rate [rad/s]
        self.turn_rate = turn_rate

    def F(self, dt):
        omega = self.turn_rate
        sinOt = math.sin(omega) * dt
        cosO = math.cos(omega)
        OcosO = 1.0 - cosO
        return np.array([[1.0, 0.0, sinOt / omega, -OcosO / omega],
                         [0.0, 1.0, OcosO / omega, sinOt / omega],
                         [0.0, 0.0, cosO, -sinOt],
                         [0.0, 0.0, sinOt, cosO]])
