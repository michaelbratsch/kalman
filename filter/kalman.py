import logging
import math

import numpy as np


def gaussian_density(x, S, Sinv):
    return (math.exp(-0.5 * np.dot(np.dot(x, Sinv), x.T)) /
            math.sqrt(np.linalg.det(2.0 * math.pi * S)))


class Kalman(object):
    """
        plant_noise:     noise that is not covered by the defined model
        density_scalings: scale the computed density for an innovation;
                         necessary for probabilities used by IMM
    """

    def __init__(self, plant_noise, density_scaling=1.0):
        # needs to be called to make MixIns work
        super(Kalman, self).__init__()

        # will be initialized on first measurement
        self.x = None
        self.P = None

        self.plant_noise = plant_noise

        # density of the model, used for IMM-PDA
        self.density = 1.0
        self.density_scalings = density_scaling

        self.false_density = 0.0

        self.log = logging.getLogger(self.__class__.__name__)

    def filter(self, dt, z, R):
        self.log.debug("Filter input dt: %s, z: %s" % (dt, z))

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
        self.P = np.dot(np.dot(self.F(dt), self.P), self.F(dt).T) + self.Q(dt)

        cond_P = np.linalg.cond(self.P)
        if cond_P > 10**10:
            self.log.warning('Huge condition number: %.2e' % cond_P)

    def update(self, z, R):
        # innovation
        ytilde = z - np.dot(self.H(), self.x)
        S = np.dot(np.dot(self.H(), self.P), self.H().T) + R
        Sinv = np.linalg.inv(S)

        # compute density function for the innovation
        self.density = self.density_scalings * gaussian_density(
            x=ytilde, S=S, Sinv=Sinv)

        # if false density is zero, the scaling does not change the innovation
        ytilde *= self.density

        sum_densities = self.density + self.false_density
        if sum_densities:
            ytilde /= sum_densities

        # update
        KalmanGain = np.dot(np.dot(self.P, self.H().T), Sinv)
        self.x = self.x + np.dot(KalmanGain, ytilde)
        self.P = np.dot(
            np.identity(len(self.x)) - np.dot(KalmanGain, self.H()), self.P)

    def _filter(self, dt, z, R):
        self.extrapolate(dt)
        self.update(z, R)

    def print_state_and_covariance(self, name):
        self.log.debug("%s x: %s" % (name, self.x))
        self.log.debug("P: \n%s" % self.P)
