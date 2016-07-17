#!/usr/bin/python

import logging
import math

from kalman.models import State2_Measurement1_2d
import numpy as np


logging.basicConfig(level=logging.WARN)


low_speed_turn2d = State2_Measurement1_2d(plant_noise=0.05,
                                          turn_rate=-0.2)


def generate_measurements(n):
    for i in range(n):
        # MEASUREMENT
        z = np.array([i, i], float)

        s_xx_R = 1.0
        s_yy_R = 1.0
        s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

        z += np.random.multivariate_normal(mean=z, cov=R)

        yield z, R

for z, R in generate_measurements(2):
    low_speed_turn2d.filter(dt=1.0, z=z, R=R)

for _ in range(30):
    low_speed_turn2d.extrapolate(dt=2.0)
    low_speed_turn2d.update_plotter()

low_speed_turn2d.plot(111)

State2_Measurement1_2d.show()
