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


x1 = np.zeros(2)
P1 = np.identity(2)
position2d = State1Mearsurement1_2d(x=x1, P=P1)


x2 = np.zeros(4)
P2 = np.identity(4)
low_speed2d = State2Measurement1_2d(x=x2, P=P2)


x3 = np.zeros(6)
P3 = np.identity(6)
acceleration2d = State3Measurement1_2d(x=x3, P=P3)


x4 = np.zeros(8)
P4 = np.identity(8)
jerk2d = State4Measurement1_2d(x=x4, P=P4)


for filter_2d in [position2d, low_speed2d, acceleration2d, jerk2d]:
    for i in range(1, 15):
        r = 2.0
        x = r * math.sin(i / 3.0)
        y = r * math.cos(i / 3.0) - r

        # MEASUREMENT
        z = np.array([x, y])

        s_xx_R = 1.0
        s_yy_R = 1.0
        s_xy_R = -0.0 * correlation(s_xx_R, s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

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
