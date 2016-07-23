#!/usr/bin/python

import logging

from filter.plotter import plot_all
from models.dim_2.order_2 import (State2Measurement1PerfectTurn,
                                  State2Measurement1TwistedTurn)
import numpy as np


logging.basicConfig(level=logging.INFO)

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
    filter_2d.x[2] = 1.0


for _ in range(40):
    for filter_2d in filters:
        filter_2d.extrapolate(dt=0.5)
        filter_2d.update_plotter()

        filter_2d.log.info(
            'product: %s diag: %s' % (
                np.prod(np.linalg.eigvals(filter_2d.P)),
                np.diagonal(filter_2d.P))
        )

plot_all(filters, vertical=False)

State2Measurement1PerfectTurn.show()
