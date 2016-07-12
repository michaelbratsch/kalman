#!/usr/bin/python

import logging
import math

from kalman.models import (State1Mearsurement1_2d,
                           State2Measurement1_2d,
                           State3Measurement1_2d,
                           State4Measurement1_2d)
import numpy as np


logging.basicConfig(level=logging.WARN)


def correlation(a, b):
    return math.sqrt(a * b)

position2d = State1Mearsurement1_2d(plant_noise=1.0)

low_speed2d = State2Measurement1_2d(plant_noise=0.05)

acceleration2d = State3Measurement1_2d(plant_noise=0.001)

jerk2d = State4Measurement1_2d(plant_noise=0.000001)


def generate_measurements(n):
    for i in range(1, n):
        r = 10.0
        alpha = 10.0
        x = r * math.sin(i / alpha)
        y = r * math.cos(i / alpha) - r

        # MEASUREMENT
        z = np.array([x, y])

        s_xx_R = 10.0
        s_yy_R = 10.0
        s_xy_R = -0.0 * correlation(s_xx_R, s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

        z += np.random.multivariate_normal(mean=z, cov=R)

        yield z, R

for z, R in generate_measurements(60):
    for filter_2d in [position2d, low_speed2d, acceleration2d, jerk2d]:
        filter_2d.filter(dt=1.0, z=z, R=R)

print "Condition numbers of covariances:"
print "position: ", np.linalg.cond(position2d.P)
print "speed: ", np.linalg.cond(low_speed2d.P)
print "acceleration: ", np.linalg.cond(acceleration2d.P)
print "jerk: ", np.linalg.cond(jerk2d.P)

position2d.plot(221)
low_speed2d.plot(222)
acceleration2d.plot(223)
jerk2d.plot(224)

State1Mearsurement1_2d.show()
