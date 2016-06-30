#!/usr/bin/python

from kalman import LowSpeed2dSecondOrder
import numpy as np
import logging
import math

logging.basicConfig(level=logging.DEBUG)


def correlation(a, b):
    return math.sqrt(a*b)

# INITIAL STATE
x = np.array([0.0, 0.0, 0.0, 0.0])

sigma_x_P = 1.0
sigma_y_P = 1.0
sigma_x_y_P = 0.9 * correlation(sigma_x_P, sigma_y_P)

P = np.array([[sigma_x_P, sigma_x_y_P, 0.0, 0.0],
              [sigma_x_y_P, sigma_y_P, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

low_speed = LowSpeed2dSecondOrder(x=x, P=P)


# MEASUREMENT
z = np.array([1, 0])

sigma_x_R = 1.0
sigma_y_R = 3.0
sigma_x_y_R = 0.9 * correlation(sigma_x_R, sigma_y_R)

R = np.array([[sigma_x_R, sigma_x_y_R],
              [sigma_x_y_R, sigma_x_R]])

low_speed.filter(dt=-1.0, z=z, R=R)
