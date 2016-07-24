import numpy as np


class Steady(object):

    def __init__(self, duration, speed, acceleration):
        self.duration = duration
        self.speed = speed
        self.acceleration = acceleration

    def get_value(self, t, position):
        if position is None:
            position = np.zeros(2)
        return 0.5 * t**2 * self.acceleration + self.speed * t + position


class Turn(object):

    def __init__(self, duration, abs_speed, turnrate):
        self.duration = duration
        self.abs_speed = abs_speed
        self.turnrate = turnrate

    def get_value(self, t, position):
        if position is None:
            position = np.zeros(2)
        return position + np.array([0, 0])
