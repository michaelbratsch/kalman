import numpy as np
import logging

log = logging.getLogger(__name__)

class Kalman(object):
    def __init__(self, x, P):
        self.x = x
        self.P = P

    def print_state_and_covariance(self, name):
        log.debug("%s x: %s" % (name, self.x))
        log.debug("P: \n%s" % self.P)

    def FT(self, dt):
        return np.transpose(self.F(dt))

    def HT(self, dt):
        return np.transpose(self.H(dt))

    def filter(self, dt, z, R):

        log.debug("Filter input dt: %s, z: %s" % (dt, z))
        #self.print_state_and_covariance("State before")

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
        log.debug("x: %s, xhat: %s" %(self.x, xhatminus))
        if dt < 0:
            self.x = self.x + np.dot(KalmanGain, ytilde)
        else:
            self.x = xhatminus + np.dot(KalmanGain, ytilde)
        self.P = np.dot(np.identity(len(self.x)) - np.dot(KalmanGain,self.H(dt)), Pminus)

        self.print_state_and_covariance("State after")


class LowSpeed2dSecondOrder(Kalman):
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
