#!/usr/bin/python

import math

from filter.imm import IMM
from models.dim_2.order_2 import State2Measurement1,\
    State2Measurement1PerfectTurn
import numpy as np


n_filters = 3
filter_models = [State2Measurement1(plant_noise=10**(-5 + i))
                 for i in range(n_filters)]
filter_models.append(State2Measurement1PerfectTurn(plant_noise=10**-3,
                                                   turn_rate=0.2))
filter_models.append(State2Measurement1PerfectTurn(plant_noise=10**-3,
                                                   turn_rate=-0.2))
n_filters = len(filter_models)

switching_matrix = np.full((n_filters, n_filters), 0.05 / (n_filters - 1))
np.fill_diagonal(switching_matrix, 0.95)

print "Switching matrix:\n", switching_matrix

imm2d = IMM(
    filter_models=filter_models,
    switching_matrix=switching_matrix
)

low_speed_2d = State2Measurement1(plant_noise=0.0)


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

for z, R in generate_measurements(n=70):
    for filter_2d in [low_speed_2d, imm2d]:
        filter_2d.filter(dt=1.0, z=z, R=R)

low_speed_2d.plot('311')
imm2d.plot('312')
imm2d.plot_probabilities('313')
State2Measurement1.show()
