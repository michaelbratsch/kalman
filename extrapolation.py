#!/usr/bin/python

import logging

from models.dim_2.order_2 import (State2Measurement1PerfectTurn,
                                  State2Measurement1TwistedTurn)
import numpy as np


logging.basicConfig(level=logging.WARN)

plant_noise = 0.0
turn_rate = 0.4

perfect_turn2d = State2Measurement1PerfectTurn(plant_noise=plant_noise,
                                               turn_rate=turn_rate)
twisted_turn2d = State2Measurement1TwistedTurn(plant_noise=plant_noise,
                                               turn_rate=turn_rate)

z = np.array([0.0, 0.0], float)
R = np.array([[1.0, 0.0],
              [0.0, 1.0]])

filters = [perfect_turn2d, twisted_turn2d]

for filter_2d in filters:
    filter_2d.initialize_state(z, R)
    filter_2d.x[2] = 10.0


for _ in range(10):
    for filter_2d in filters:
        filter_2d.extrapolate(dt=2.0)
        filter_2d.update_plotter()
        # print np.diagonal(filter_2d.P), np.linalg.cond(filter_2d.P)
        # print np.linalg.eig(np.dot(filter_2d.FT(0.09), filter_2d.F(0.09)))

perfect_turn2d.plot(121)
twisted_turn2d.plot(122)

State2Measurement1PerfectTurn.show()
