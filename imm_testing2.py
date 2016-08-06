#!/usr/bin/python

import math

from data.measurement import DataGenerator
from data.segments import Steady, Turn
from filter.imm import IMM
from models.dim_2.order_2 import State2Measurement1,\
    State2Measurement1PerfectTurn
import numpy as np


filter_models = [
    State2Measurement1(plant_noise=0.0),
    State2Measurement1(plant_noise=0.05),
    State2Measurement1PerfectTurn(plant_noise=0.0,
                                  turn_rate=0.2),
    State2Measurement1PerfectTurn(plant_noise=0.0,
                                  turn_rate=-0.2)]


switching_matrix = np.array([[0.84, 0.10, 0.03, 0.03],
                             [0.14, 0.80, 0.03, 0.03],
                             [0.10, 0.10, 0.80, 0.00],
                             [0.10, 0.10, 0.00, 0.80]])

print "Switching matrix:\n", switching_matrix

imm2d = IMM(
    filter_models=filter_models,
    switching_matrix=switching_matrix,
    false_density_to_accuracy=5.0
)

s_xx_R = 1.0
s_yy_R = 1.0
s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

R = np.array([[s_xx_R, s_xy_R],
              [s_xy_R, s_yy_R]])

measurement = DataGenerator(
    segments=[
        Steady(duration=40.0,
               heading=0.5 * math.pi,
               abs_speed=1.0),
        Turn(duration=15.0,
             abs_speed=1.0,
             turnrate=0.2),
        Turn(duration=15.0,
             abs_speed=1.0,
             turnrate=-0.2),
        Steady(duration=20.0,
               abs_speed=1.0,
               acceleration=np.array([0.2, 0.0])),
        Steady(duration=20.0,
               abs_speed=5.0,
               acceleration=np.array([-0.2, 0.0]))
    ]
)


np.random.seed(42)
dt = 1.0

for _ in range(200):
    z = measurement.draw(dt=dt, R=R)
    imm2d.filter(dt=dt, z=z, R=R)

imm2d.plot_all(vertical=False, dim=2)
imm2d.show()
