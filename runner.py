#!/usr/bin/python

import logging
import math

from models import (State1Mearsurement1_2d,
                    State2Measurement1_2d,
                    State3Measurement1_2d)
import numpy as np


logging.basicConfig(level=logging.WARN)


def correlation(a, b):
    return math.sqrt(a * b)

s_xx = 1.0
s_yy = 1.0
s_xy = -0.0 * correlation(s_xx, s_yy)


x1 = np.array([0.0, 0.0])
P1 = np.array([[s_xx, s_xy],
               [s_xy, s_yy]])

position2d = State1Mearsurement1_2d(x=x1, P=P1)


x2 = np.array([0.0, 0.0, 0.0, 0.0])
P2 = np.array([[s_xx, s_xy, 0.0, 0.0],
               [s_xy, s_yy, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])

low_speed2d = State2Measurement1_2d(x=x2, P=P2)


x3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P3 = np.array([[s_xx, s_xy, 0.0, 0.0, 0.0, 0.0],
               [s_xy, s_yy, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

acceleration2d = State3Measurement1_2d(x=x3, P=P3)


for filter_2d in [position2d, low_speed2d, acceleration2d]:
    for i in range(1, 10):
        # MEASUREMENT
        z = np.array([i, 0])

        sigma_xx_R = 1.0
        sigma_yy_R = 1.0
        sigma_xy_R = -0.9 * correlation(sigma_xx_R, sigma_yy_R)

        R = np.array([[sigma_xx_R, sigma_xy_R],
                      [sigma_xy_R, sigma_xx_R]])

        filter_2d.filter(dt=1.0, z=z, R=R)

position2d.plot(311)
low_speed2d.plot(312)
acceleration2d.plot(313)
State1Mearsurement1_2d.show()
