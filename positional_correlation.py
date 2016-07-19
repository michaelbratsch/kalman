#!/usr/bin/python

import logging
import math

from models.dim_2.order_1 import State1Mearsurement1
from models.dim_2.order_2 import State2Measurement1
from models.dim_2.order_3 import State3Measurement1
from models.dim_2.order_4 import State4Measurement1
import numpy as np


logging.basicConfig(level=logging.WARN)


plant_noise = 0.0

position2d = State1Mearsurement1(plant_noise=plant_noise)

low_speed2d = State2Measurement1(plant_noise=plant_noise)

acceleration2d = State3Measurement1(plant_noise=plant_noise)

jerk2d = State4Measurement1(plant_noise=plant_noise)


def generate_measurements(n):
    for i in range(1, n):
        # MEASUREMENT
        z = np.array([i, 0.0])

        s_xx_R = 10.0
        s_yy_R = 10.0
        s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

        z += np.random.multivariate_normal(mean=z, cov=R)

        yield z, R

for z, R in generate_measurements(100):
    for filter_2d in [position2d, low_speed2d, acceleration2d, jerk2d]:
        filter_2d.filter(dt=1.0, z=z, R=R)

position2d.plot(411)
low_speed2d.plot(412)
acceleration2d.plot(413)
jerk2d.plot(414)

State1Mearsurement1.show()
