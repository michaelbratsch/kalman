#!/usr/bin/python

import math

from filter.plotter import plot_all
from models.dim_2.order_1 import State1Mearsurement1
from models.dim_2.order_2 import (State2Measurement1,
                                  State2Measurement1PerfectTurn,
                                  State2Measurement1TwistedTurn)
from models.dim_2.order_3 import State3Measurement1
from models.dim_2.order_4 import State4Measurement1
import numpy as np


plant_noise = 0.001
turn_rate = -0.1

filters = [
    State1Mearsurement1(plant_noise=5.0),
    State2Measurement1(plant_noise=1.0),
    State2Measurement1PerfectTurn(plant_noise=plant_noise,
                                  turn_rate=turn_rate),
    State2Measurement1TwistedTurn(plant_noise=plant_noise,
                                  turn_rate=turn_rate),
    State3Measurement1(plant_noise=0.1),
    State4Measurement1(plant_noise=0.001)
]


def generate_measurements(n):
    for i in range(1, n + 1):
        r = 10.0
        alpha = 10.0
        x = r * math.sin(i / alpha)
        y = r * math.cos(i / alpha) - r

        # MEASUREMENT
        z = np.array([x, y])

        s_xx_R = 1.0
        s_yy_R = 1.0
        s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

        z = np.random.multivariate_normal(mean=z, cov=R)

        yield z, R

np.random.seed(42)

for z, R in generate_measurements(60):
    for filter_2d in filters:
        filter_2d.filter(dt=1.0, z=z, R=R)

print "Condition numbers of covariances:"
for f in filters:
    print "%s: %.1e" % (f.get_title(), np.linalg.cond(f.P))

plot_all(filters, vertical=False, dim=2)

State1Mearsurement1.show()
