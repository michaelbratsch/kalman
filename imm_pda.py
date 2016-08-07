#!/usr/bin/python

import math

from data.measurement import DataGenerator
from data.segments import Steady, Turn
from filter.imm import IMM
from models.dim_2.order_2 import State2Measurement1,\
    State2Measurement1PerfectTurn
import numpy as np
from tracker.tracker import Tracker


def filter_factory():
    filter_models = [
        State2Measurement1(plant_noise=0.0),
        State2Measurement1PerfectTurn(plant_noise=0.0,
                                      turn_rate=0.2),
        State2Measurement1PerfectTurn(plant_noise=0.0,
                                      turn_rate=-0.2)]

    switching_matrix = np.array([[0.90, 0.05, 0.05],
                                 [0.20, 0.80, 0.00],
                                 [0.20, 0.00, 0.80]])

    return IMM(
        filter_models=filter_models,
        switching_matrix=switching_matrix,
        false_density_to_accuracy=3.0
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
             turnrate=-0.2)
    ]
)

tracker = Tracker(filter_factory=filter_factory)

np.random.seed(42)
dt = 1.0

for _ in range(200):
    z = measurement.draw(dt=dt, R=R)
    tracker.filter(dt=dt, z=z, R=R)
    for i in range(1, 2):
        tracker.filter(dt=0.0, z=z + np.array([10.0 * i, 10.0 * i]), R=R)

tracker.plot_all(vertical=False, dim=2)
tracker.show()
