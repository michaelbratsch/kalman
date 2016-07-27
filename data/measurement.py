from itertools import cycle

import numpy as np


class DataGenerator(object):

    def __init__(self, segments):
        self.segments = segments

        self.current_time = 0.0

    def get_value(self):
        position = None
        heading = None
        remaining_time = self.current_time
        # the different segments describing the movement are looped until
        # remaining time is smaller then the duration of the current segment
        for segment in cycle(self.segments):
            if remaining_time > segment.duration:
                remaining_time -= segment.duration
                position, heading = segment.get_value(t=segment.duration,
                                                      position=position,
                                                      heading=heading)
            else:
                position, _ = segment.get_value(t=remaining_time,
                                                position=position,
                                                heading=heading)
                return position

    def draw(self, dt, R):
        self.current_time += dt
        return np.random.multivariate_normal(mean=self.get_value(),
                                             cov=R)
