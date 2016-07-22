#!/usr/bin/python

import logging
import math

from filter.imm import IMM
from models.dim_2.order_1 import State1Mearsurement1
from models.dim_2.order_2 import (State2Measurement1,
                                  State2Measurement1PerfectTurn,
                                  State2Measurement1TwistedTurn)
from models.dim_2.order_3 import State3Measurement1
from models.dim_2.order_4 import State4Measurement1
import numpy as np


logging.basicConfig(level=logging.WARN)


position2d = State1Mearsurement1(plant_noise=5.0)

low_speed2d = State2Measurement1(plant_noise=1.0)


plant_noise = 0.001
turn_rate = -0.05

low_speed_perfect_turn2d = State2Measurement1PerfectTurn(
    plant_noise=plant_noise,
    turn_rate=turn_rate)

low_speed_twisted_turn2d = State2Measurement1TwistedTurn(
    plant_noise=plant_noise,
    turn_rate=turn_rate)

acceleration2d = State3Measurement1(plant_noise=0.1)

jerk2d = State4Measurement1(plant_noise=0.001)

imm2d = IMM(
    filter_models=[State2Measurement1PerfectTurn(plant_noise=plant_noise,
                                                 turn_rate=turn_rate)],
    switching_matrix=np.eye(1)
)


def generate_measurements(n):
    for i in range(1, n + 1):
        r = 10.0
        alpha = 10.0
        x = r * math.sin(i / alpha)
        y = r * math.cos(i / alpha) - r

        # MEASUREMENT
        z = np.array([x, y])

        s_xx_R = 10.0
        s_yy_R = 10.0
        s_xy_R = -0.0 * math.sqrt(s_xx_R * s_yy_R)

        R = np.array([[s_xx_R, s_xy_R],
                      [s_xy_R, s_xx_R]])

        z += np.random.multivariate_normal(mean=z, cov=R)

        yield z, R

for z, R in generate_measurements(60):
    for filter_2d in [position2d, low_speed2d, low_speed_perfect_turn2d,
                      low_speed_twisted_turn2d, acceleration2d, jerk2d, imm2d]:
        filter_2d.filter(dt=1.0, z=z, R=R)

print "Condition numbers of covariances:"
print "position: ", np.linalg.cond(position2d.P)
print "speed: ", np.linalg.cond(low_speed2d.P)
print "turnrate perfect: ", np.linalg.cond(low_speed_perfect_turn2d.P)
print "turnrate twisted: ", np.linalg.cond(low_speed_twisted_turn2d.P)
print "acceleration: ", np.linalg.cond(acceleration2d.P)
print "jerk: ", np.linalg.cond(jerk2d.P)

position2d.plot(231)
low_speed2d.plot(232)
low_speed_perfect_turn2d.plot(233)
low_speed_twisted_turn2d.plot(234)
acceleration2d.plot(235)
# jerk2d.plot(236)
imm2d.plot(236)

State1Mearsurement1.show()
