#!/usr/bin/python

import logging

from models.dim_2.order_2 import State2Measurement1
import numpy as np


logging.basicConfig(level=logging.WARN)

plant_noise = 0.0
turn_rate = 0.4

low_speed_old_turn2d = State2Measurement1(plant_noise=plant_noise,
                                          turn_rate=turn_rate,
                                          perfect_turn=False)
low_speed_new_turn2d = State2Measurement1(plant_noise=plant_noise,
                                          turn_rate=turn_rate,
                                          perfect_turn=True)

z = np.array([0.0, 0.0], float)
R = np.array([[1.0, 0.0],
              [0.0, 1.0]])

filters = [low_speed_old_turn2d, low_speed_new_turn2d]

for filter_2d in filters:
    filter_2d.initialize_state(z, R)
    filter_2d.x[2] = 10.0


for _ in range(10):
    for filter_2d in filters:
        filter_2d.extrapolate(dt=2.0)
        filter_2d.update_plotter()
        # print np.diagonal(filter_2d.P), np.linalg.cond(filter_2d.P)
        # print np.linalg.eig(np.dot(filter_2d.FT(0.09), filter_2d.F(0.09)))

low_speed_old_turn2d.plot(121)
low_speed_new_turn2d.plot(122)

State2Measurement1.show()
