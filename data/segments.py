import math
import numpy as np


class Steady(object):

    def __init__(self, duration, abs_speed, acceleration=None, heading=None):
        self.duration = duration
        self.heading = heading
        self.abs_speed = abs_speed

        if acceleration is None:
            self.acceleration = np.array([0.0, 0.0])
        else:
            self.acceleration = acceleration

    def get_value(self, t, position, heading):
        if position is None:
            position = np.zeros(2)

        if self.heading is None and heading is None:
            heading_used = 0.0
        elif self.heading is None:
            heading_used = heading
        else:
            heading_used = self.heading

        self.speed = np.array([self.abs_speed * math.sin(heading_used),
                               self.abs_speed * math.cos(heading_used)])

        return (0.5 * t**2 * self.acceleration + self.speed * t + position,
                heading_used)


class Turn(object):

    def __init__(self, duration, abs_speed, turnrate, heading=None):
        self.duration = duration

        self.heading = heading

        self.radius = abs_speed / turnrate
        self.turnrate = turnrate

    def get_value(self, t, position, heading):
        if position is None:
            position = np.zeros(2)

        if self.heading is None and heading is None:
            heading_used = 0.0
        elif self.heading is None:
            heading_used = heading
        else:
            heading_used = self.heading

        self.x = -math.cos(heading_used)
        self.y = math.sin(heading_used)

        rad_covered = t * self.turnrate
        rad_shifted = rad_covered - 0.5 * math.pi + heading_used
        x = self.radius * (math.sin(rad_shifted) - self.x)
        y = self.radius * (math.cos(rad_shifted) - self.y)

        return position + np.array([x, y]), rad_covered + heading_used
