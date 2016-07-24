#!/usr/bin/python

import math

from data.measurement import DataGenerator
from data.segments import Steady
from filter.plotter import plot_all
from models.dim_2.order_1 import State1Mearsurement1
from models.dim_2.order_2 import State2Measurement1
from models.dim_2.order_3 import State3Measurement1
from models.dim_2.order_4 import State4Measurement1
import numpy as np


plant_noise = 0.0

filters = [
    State1Mearsurement1(plant_noise=plant_noise),
    State2Measurement1(plant_noise=plant_noise),
    State3Measurement1(plant_noise=plant_noise),
    State4Measurement1(plant_noise=plant_noise)
]


s_xx_R = 1.0
s_yy_R = 1.0
s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

R = np.array([[s_xx_R, s_xy_R],
              [s_xy_R, s_xx_R]])

measurement = DataGenerator(
    segments=[Steady(duration=100.0,
                     speed=np.array([1.0, 0.0]),
                     acceleration=np.array([0.0, 0.0]))]
)

for _ in range(100):
    z = measurement.draw(dt=1.0, R=R)
    for filter_2d in filters:
        filter_2d.filter(dt=1.0, z=z, R=R)

plot_all(filters)

State1Mearsurement1.show()
