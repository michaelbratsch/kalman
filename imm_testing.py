#!/usr/bin/python

import math

from data.measurement import DataGenerator
from data.segments import Steady
from filter.imm import IMM, generate_switching_matrix
import matplotlib.pyplot as plt
from models.dim_2.order_2 import State2Measurement1,\
    State2Measurement1PerfectTurn
import numpy as np


filter_models = [
    State2Measurement1(plant_noise=10**-4,
                       probability_scaling=3.0),
    State2Measurement1(plant_noise=10**-1),
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

speed_2d = State2Measurement1(plant_noise=10**-4)

s_xx_R = 0.7
s_yy_R = 0.7
s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

R = np.array([[s_xx_R, s_xy_R],
              [s_xy_R, s_xx_R]])

measurement = DataGenerator(
    segments=[Steady(duration=25.0,
                     speed=np.array([1.0, 0.0]),
                     acceleration=np.array([0.0, 0.0])),
              Steady(duration=25.0,
                     speed=np.array([0.0, 1.0]),
                     acceleration=np.array([0.0, 0.0])),
              Steady(duration=25.0,
                     speed=np.array([1.0, 0.0]),
                     acceleration=np.array([0.0, 0.0])),
              Steady(duration=25.0,
                     speed=np.array([0.0, -1.0]),
                     acceleration=np.array([0.0, 0.0]))]
)


np.random.seed(42)

for _ in range(400):
    z = measurement.draw(dt=1.0, R=R)
    imm2d.filter(dt=1.0, z=z, R=R)
    speed_2d.filter(dt=1.0, z=z, R=R)


imm2d.plot_all()
speed_2d.plot(figure=plt.figure())
imm2d.show()
