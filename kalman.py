import logging

import numpy as np


log = logging.getLogger(__name__)


class Kalman(object):

    def __init__(self, x, P):
        super(Kalman, self).__init__()

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
