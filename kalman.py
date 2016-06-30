import logging
from operator import pos

import matplotlib.pyplot as plt
import numpy as np


log = logging.getLogger(__name__)


class Kalman(object):

    def __init__(self, x, P):
        self.x = x
        self.P = P

        self.print_state_and_covariance("State initial")

        self.update_plotter()

    def FT(self, dt):
        return np.transpose(self.F(dt))

    def HT(self, dt):
        return np.transpose(self.H(dt))

    def filter(self, dt, z, R):

        log.debug("Filter input dt: %s, z: %s" % (dt, z))
        # self.print_state_and_covariance("State before")

        # PREDICT
        xhatminus = np.dot(self.F(dt), self.x)
        Pminus = np.dot(np.dot(self.F(dt), self.P), self.FT(dt)) + self.Q(dt)

        # UPDATE
        # innovation
        ytilde = z - np.dot(self.H(dt), xhatminus)
        S = np.dot(np.dot(self.H(dt), Pminus), self.HT(dt)) + R
        Sinv = np.linalg.inv(S)

        # state
        KalmanGain = np.dot(np.dot(Pminus, self.HT(dt)), Sinv)
        self.x = xhatminus + np.dot(KalmanGain, ytilde)
        self.P = np.dot(
            np.identity(len(self.x)) - np.dot(KalmanGain, self.H(dt)), Pminus)

        self.update_plotter(z)
        self.print_state_and_covariance("State after")

    def print_state_and_covariance(self, name):
        log.debug("%s x: %s" % (name, self.x))
        log.debug("P: \n%s" % self.P)


class Plot2dMixin(object):

    # plot initial state
    positions = []
    speeds = []
    measurements = []

    def update_plotter(self, measurement=None):
        self.positions.append(self.get_position())
        self.speeds.append(self.get_speed())
        if measurement is not None:
            self.measurements.append(measurement)

    def plot(self):
        fig = plt.figure()
        axes = fig.add_subplot(111)

        x, y = zip(*self.positions)
        axes.plot(x, y, marker='o')

        x, y = zip(*self.measurements)
        axes.plot(x, y, 'ro')

        for pos, speed in zip(self.positions, self.speeds):
            if speed:
                x, y = pos
                dx, dy = speed
                axes.plot([x, x + dx], [y, y + dy], color='red')

        plt.show()


class LowSpeed2dSecondOrder(Kalman, Plot2dMixin):

    def get_position(self):
        return self.x[0], self.x[1]

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
