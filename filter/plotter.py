import math
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


def plot_all(filter_models, vertical=True, dim=1):
    items_per_dim = int(math.ceil(len(filter_models) / float(dim)))

    for i, fm in enumerate(filter_models):
        if vertical:
            pattern = '%s%s%s' % (items_per_dim, dim, i + 1)
        else:
            pattern = '%s%s%s' % (dim, items_per_dim, i + 1)

        fm.plot(subplot=pattern)


class Plot2dMixin(object):

    figure = plt.figure()
    pos_axes = []
    plant_noise_format = "%.1e"

    def __init__(self):
        # plot initial state
        self.positions = []
        self.position_accuracies = []
        self.speeds = []
        self.measurements = []

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return self.x[2], self.x[3]

    def update_plotter(self, measurement=None):
        self.positions.append(self.get_position())
        self.position_accuracies.append(self.get_position_accuracy())
        self.speeds.append(self.get_speed())
        if measurement is not None:
            self.measurements.append(measurement)

    def get_axes(self, figure, subplot):
        if figure:
            return figure.add_subplot(subplot)

        return self.figure.add_subplot(subplot)

    def get_title(self):
        return (('%s PN: ' + self.plant_noise_format) %
                (self.__class__.__name__, self.plant_noise))

    def plot(self, figure=None, subplot=111):
        axes = self.get_axes(figure, subplot)

        axes.set_title(self.get_title())

        x, y = zip(*self.positions)
        axes.plot(x, y, marker='o')

        if self.measurements:
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

        self.pos_axes.append(axes)

    @classmethod
    def show(cls):
        x_0 = min(axe.get_xlim()[0] for axe in cls.pos_axes)
        x_1 = max(axe.get_xlim()[1] for axe in cls.pos_axes)
        y_0 = min(axe.get_ylim()[0] for axe in cls.pos_axes)
        y_1 = max(axe.get_ylim()[1] for axe in cls.pos_axes)
        for axe in cls.pos_axes:
            axe.set_xlim((x_0, x_1))
            axe.set_ylim((y_0, y_1))
        plt.show()


class Plot2dIMMMixin(Plot2dMixin):

    def __init__(self):
        super(Plot2dIMMMixin, self).__init__()

        self.probabilities = []

    def get_title(self):
        return self.__class__.__name__

    def update_plotter(self, measurement=None):
        super(Plot2dIMMMixin, self).update_plotter(
            measurement=measurement)

        self.probabilities.append(
            [fm.probability for fm in self.filter_models])

    def plot_probabilities(self, figure=None, subplot=212):
        axes = self.get_axes(figure=figure, subplot=subplot)

        axes.set_title('%s probabilities' % self.get_title())

        for prob, fm in zip(zip(*self.probabilities), self.filter_models):
            axes.plot(prob, label=('%s PN: ' + self.plant_noise_format) %
                      (fm.__class__.__name__, fm.plant_noise))
        axes.legend()

    def plot_all(self, vertical=True, dim=1):
        # plots of different kalman filters
        plot_all(filter_models=self.filter_models, vertical=vertical, dim=dim)

        # plot IMM result
        figure = plt.figure()
        self.plot(figure=figure, subplot='211')
        self.plot_probabilities(figure=figure, subplot='212')
