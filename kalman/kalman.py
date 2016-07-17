import logging

import numpy as np


log = logging.getLogger(__name__)


class Kalman(object):

    def __init__(self, plant_noise):
        # needs to be called to make MixIns work
        super(Kalman, self).__init__()

        # will be initialized on first measurement
        self.x = None
        self.P = None

        self.plant_noise = plant_noise

    def FT(self, dt):
        return np.transpose(self.F(dt))

    def HT(self):
        return np.transpose(self.H())

    def filter(self, dt, z, R):
        log.debug("Filter input dt: %s, z: %s" % (dt, z))

        if self.x is None:
            self.initialize_state(z, R)
            info_text = "State initial"
        else:
            self._filter(dt, z, R)
            info_text = "State after"

        self.print_state_and_covariance(info_text)
        self.update_plotter(z)

    def extrapolate(self, dt):
        self.x = np.dot(self.F(dt), self.x)
        self.P = np.dot(np.dot(self.F(dt), self.P), self.FT(dt)) + self.Q(dt)

        cond_P = np.linalg.cond(self.P)
        if cond_P > 10**10:
            log.warning('Huge condition number: %s' % cond_P)

    def update(self, z, R):
        # innovation
        ytilde = z - np.dot(self.H(), self.x)
        S = np.dot(np.dot(self.H(), self.P), self.HT()) + R
        Sinv = np.linalg.inv(S)

        # state
        KalmanGain = np.dot(np.dot(self.P, self.HT()), Sinv)
        self.x = self.x + np.dot(KalmanGain, ytilde)
        self.P = np.dot(
            np.identity(len(self.x)) - np.dot(KalmanGain, self.H()), self.P)

    def _filter(self, dt, z, R):
        self.extrapolate(dt)
        self.update(z, R)

    def print_state_and_covariance(self, name):
        log.debug("%s x: %s" % (name, self.x))
        log.debug("P: \n%s" % self.P)
