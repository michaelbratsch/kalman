#!/usr/bin/python

import math
from scipy import integrate

from filter.kalman import gaussian_density
import numpy as np


def kernel(*args):
    eye = np.eye(len(args))
    return gaussian_density(np.array(args), eye, eye)


def kernel_accelerated(*args):
    x = np.array(args)
    return (math.exp(-0.5 * np.dot(x, x)) /
            math.sqrt(2.0 * math.pi)**(len(args)))


for dim in range(1, 5):
    print '%dd:' % dim

    for i in range(1, 6):
        bounds = [[-i, i] for _ in range(dim)]

        print i, integrate.nquad(kernel_accelerated, bounds)
