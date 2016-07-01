#!/usr/bin/python

import logging
import math

from filter import LowSpeed2d, Position2d
import numpy as np


logging.basicConfig(level=logging.DEBUG)


def correlation(a, b):
    return math.sqrt(a * b)

# INITIAL STATE
x1 = np.array([0.0, 0.0, 0.0, 0.0])

sigma_xx_P = 1.0
sigma_yy_P = 1.0
sigma_xy_P = -0.0 * correlation(sigma_xx_P, sigma_yy_P)

P1 = np.array([[sigma_xx_P, sigma_xy_P, 0.0, 0.0],
               [sigma_xy_P, sigma_yy_P, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])

low_speed2d = LowSpeed2d(x=x1, P=P1)


x2 = np.array([0.0, 0.0])
P2 = np.array([[sigma_xx_P, sigma_xy_P],
               [sigma_xy_P, sigma_yy_P]])

position2d = Position2d(x=x2, P=P2)


for filter_2d in [low_speed2d, position2d]:
    for i in range(1, 10):
        # MEASUREMENT
        z = np.array([i, 0])

        sigma_xx_R = 1.0
        sigma_yy_R = 1.0
        sigma_xy_R = -0.9 * correlation(sigma_xx_R, sigma_yy_R)

        R = np.array([[sigma_xx_R, sigma_xy_R],
                      [sigma_xy_R, sigma_xx_R]])

        filter_2d.filter(dt=1.0, z=z, R=R)

low_speed2d.plot(212)
position2d.plot(211)
Position2d.show()
