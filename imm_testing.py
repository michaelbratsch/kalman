#!/usr/bin/python

import math

from data.measurement import DataGenerator
from data.segments import Steady, Turn
from filter.imm import IMM, generate_switching_matrix
from models.dim_2.order_2 import State2Measurement1,\
    State2Measurement1PerfectTurn
import numpy as np


filter_models = [
    State2Measurement1(plant_noise=10**-4,
                       probability_scaling=2.5),
    #    State2Measurement1(plant_noise=10**-1),
    State2Measurement1PerfectTurn(plant_noise=10**-2,
                                  turn_rate=0.2),
    State2Measurement1PerfectTurn(plant_noise=10**-2,
                                  turn_rate=-0.2)]


switching_matrix = generate_switching_matrix(n=len(filter_models), diag=0.95)

print "Switching matrix:\n", switching_matrix

imm2d = IMM(
    filter_models=filter_models,
    switching_matrix=switching_matrix
)

s_xx_R = 0.2
s_yy_R = 0.2
s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

R = np.array([[s_xx_R, s_xy_R],
              [s_xy_R, s_xx_R]])

measurement = DataGenerator(
    segments=[Steady(duration=20.0,
                     heading=0.5 * math.pi,
                     abs_speed=1.0),
              Turn(duration=15.0,
                   abs_speed=1.0,
                   turnrate=0.2),
              Steady(duration=20.0,
                     abs_speed=1.0),
              Turn(duration=15.0,
                   abs_speed=1.0,
                   turnrate=-0.2)]
)


np.random.seed(42)
dt = 1.0

for _ in range(80):
    z = measurement.draw(dt=dt, R=R)
    imm2d.filter(dt=dt, z=z, R=R)

imm2d.plot_all(vertical=False, dim=2)
imm2d.show()
