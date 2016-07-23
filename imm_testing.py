#!/usr/bin/python

import math

from filter.imm import IMM, generate_switching_matrix
from models.dim_2.order_2 import State2Measurement1,\
    State2Measurement1PerfectTurn
import numpy as np

filter_models = [State2Measurement1(plant_noise=10**(-4 + 3 * i))
                 for i in range(2)]

filter_models += [State2Measurement1PerfectTurn(plant_noise=10**-2,
                                                turn_rate=0.2),
                  State2Measurement1PerfectTurn(plant_noise=10**-2,
                                                turn_rate=-0.2)]

switching_matrix = generate_switching_matrix(n=len(filter_models), diag=0.95)

print "Switching matrix:\n", switching_matrix

imm2d = IMM(
    filter_models=filter_models,
    switching_matrix=switching_matrix
)


def generate_measurements(n):
    for i in range(1, n + 1):
        # MEASUREMENT
        z = np.array([i, 0.0])

        s_xx_R = 0.7
        s_yy_R = 0.7
        s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

        z = np.random.multivariate_normal(mean=z, cov=R)

        yield z, R

np.random.seed(42)

for z, R in generate_measurements(n=70):
    imm2d.filter(dt=1.0, z=z, R=R)

imm2d.plot_all()
imm2d.show()
