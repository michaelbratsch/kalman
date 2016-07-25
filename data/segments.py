import math
import numpy as np


class Steady(object):

    def __init__(self, duration, abs_speed, heading, acceleration):
        self.duration = duration
        self.speed = np.array([abs_speed * math.sin(heading),
                               abs_speed * math.cos(heading)])
        self.acceleration = acceleration

    def get_value(self, t, position):
        if position is None:
            position = np.zeros(2)
        return 0.5 * t**2 * self.acceleration + self.speed * t + position


class Turn(object):

    def __init__(self, duration, abs_speed, heading, turnrate):
        self.duration = duration

        self.heading = heading

        self.x = -math.cos(heading)
        self.y = math.sin(heading)

        self.radius = abs_speed / turnrate
        self.turnrate = turnrate

    def get_value(self, t, position):
        if position is None:
            position = np.zeros(2)

        rad_covered = t * self.turnrate
        rad_shifted = rad_covered - 0.5 * math.pi + self.heading
        x = self.radius * (math.sin(rad_shifted) - self.x)
        y = self.radius * (math.cos(rad_shifted) - self.y)

        return position + np.array([x, y])
