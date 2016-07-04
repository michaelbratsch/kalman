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

s_xx = 1.0
s_yy = 1.0
s_xy = -0.0 * correlation(s_xx, s_yy)


x1 = np.zeros(2)
P1 = np.array([[s_xx, s_xy],
               [s_xy, s_yy]])

position2d = State1Mearsurement1_2d(x=x1, P=P1)


x2 = np.zeros(4)
P2 = np.array([[s_xx, s_xy, 0.0, 0.0],
               [s_xy, s_yy, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])

low_speed2d = State2Measurement1_2d(x=x2, P=P2)


x3 = np.zeros(6)
P3 = np.array([[s_xx, s_xy, 0.0, 0.0, 0.0, 0.0],
               [s_xy, s_yy, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

acceleration2d = State3Measurement1_2d(x=x3, P=P3)


x4 = np.zeros(8)
P4 = np.array([[s_xx, s_xy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [s_xy, s_yy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

jerk2d = State4Measurement1_2d(x=x4, P=P4)


for filter_2d in [position2d, low_speed2d, acceleration2d, jerk2d]:
    for i in range(1, 10):
        # MEASUREMENT
        z = np.array([i, 0])

        s_xx_R = 1.0
        s_yy_R = 1.0
        s_xy_R = -0.9 * correlation(s_xx_R, s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

        filter_2d.filter(dt=1.0, z=z, R=R)

position2d.plot(221)
low_speed2d.plot(222)
acceleration2d.plot(223)
jerk2d.plot(224)

State1Mearsurement1_2d.show()
