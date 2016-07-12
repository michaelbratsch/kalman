from matplotlib.patches import Ellipse

import matplotlib.pyplot as plt
import numpy as np


def create_ellipse(pos, cov):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * np.sqrt(vals)

    return Ellipse(xy=pos, width=width, height=height, angle=theta,
                   color='green', fill=False)


class Plot2dMixin(object):

    fig = plt.figure()
    all_axes = []

    def __init__(self):
        # plot initial state
        self.positions = []
        self.position_accuracies = []
        self.speeds = []
        self.measurements = []

    def update_plotter(self, measurement=None):
        self.positions.append(self.get_position())
        self.position_accuracies.append(self.get_position_accuracy())
        self.speeds.append(self.get_speed())
        if measurement is not None:
            self.measurements.append(measurement)

    def plot(self, subplot=111):
        axes = self.fig.add_subplot(subplot)
        axes.set_title(self.__class__.__name__)

        x, y = zip(*self.positions)
        axes.plot(x, y, marker='o')

        x, y = zip(*self.measurements)
        axes.plot(x, y, 'ro')

        for pos, speed in zip(self.positions, self.speeds):
            if speed:
                x, y = pos
                dx, dy = speed
                axes.plot([x, x + dx], [y, y + dy], color='black')

        for pos, cov in zip(self.positions, self.position_accuracies):
            e = create_ellipse(pos, cov)
            axes.add_patch(e)

        axes.set_aspect('equal')
        axes.autoscale()

        self.all_axes.append(axes)

    @classmethod
    def show(cls):
        x_0 = min(axe.get_xlim()[0] for axe in cls.all_axes)
        x_1 = max(axe.get_xlim()[1] for axe in cls.all_axes)
        y_0 = min(axe.get_ylim()[0] for axe in cls.all_axes)
        y_1 = max(axe.get_ylim()[1] for axe in cls.all_axes)
        for axe in cls.all_axes:
            axe.set_xlim((x_0, x_1))
            axe.set_ylim((y_0, y_1))
        plt.show()
