
from filter.kalman import gaussian_density
from filter.plotter import Plot2dIMMMixin
import numpy as np


def generate_switching_matrix(n, diag):
    if n > 1:
        switching_matrix = np.full((n, n), (1.0 - diag) / (n - 1))
        np.fill_diagonal(switching_matrix, diag)
        return switching_matrix

    return np.eye(1)


class IMM(Plot2dIMMMixin):

    def __init__(self, filter_models, switching_matrix,
                 false_density_to_accuracy=5.0):

        super(IMM, self).__init__()

        # will be initialized on first measurement
        self.x = None
        self.P = None

        self.filter_models = filter_models

        self.false_density = 0.0
        self.false_density_to_accuracy = false_density_to_accuracy

        # the density of the different filters can not be rescaled
        # because the original value is needed for the Kalman filter
        self.sum_densities = sum(
            fm.density for fm in self.filter_models)

        self.switching_matrix = switching_matrix
        assert self.switching_matrix.shape == (len(self.filter_models),
                                               len(self.filter_models))

    def filter(self, dt, z, R):
        if self.x is not None and self.sum_densities:
            # the different Kalman filters are expected to be not initialized
            # so no state interaction can be performed
            self.state_interaction()

        self.update_models(dt, z, R)

        if self.sum_densities:
            self.state_combination()

        if self.sum_densities > 1e-3 * self.false_density:
            self.update_plotter(z)

    def state_interaction(self):

        def mix_states(conditional_model_probabilities):
            for proba_col in conditional_model_probabilities.T:
                yield np.sum(proba * fm.x
                             for fm, proba in zip(self.filter_models,
                                                  proba_col))

        def mix_covariances(conditional_model_probabilities, states_mixed):
            for fm_out, proba_col in zip(self.filter_models,
                                         conditional_model_probabilities.T):
                yield np.sum(proba * (fm.P + np.outer(fm.x - fm_out.x,
                                                      fm.x - fm_out.x))
                             for fm, proba in zip(self.filter_models,
                                                  proba_col))

        def copy_states_and_covs(states_mixed, covariances_mixed):
            for fm, state, covariance in zip(self.filter_models,
                                             states_mixed,
                                             covariances_mixed):
                fm.x = state
                fm.P = covariance

        # scale rows by probability
        conditional_model_probabilities = np.array(np.stack(
            row * fm.density / self.sum_densities
            for row, fm in zip(self.switching_matrix, self.filter_models)
        ))
        # re-scale columns to sum 1
        conditional_model_probabilities = np.array(np.column_stack(
            col / np.sum(col) for col in conditional_model_probabilities.T
        ))

        states_mixed = list(mix_states(conditional_model_probabilities))

        covariances_mixed = mix_covariances(conditional_model_probabilities,
                                            states_mixed)

        copy_states_and_covs(states_mixed, covariances_mixed)

    def set_false_density(self, R):
        Rinv = np.linalg.inv(R)

        x = np.ones(R.shape[0])
        x *= self.false_density_to_accuracy / np.linalg.norm(x)
        self.false_density = gaussian_density(x, R, Rinv)

        for fm in self.filter_models:
            fm.false_density = self.false_density

    def update_models(self, dt, z, R):

        self.set_false_density(R)

        for fm in self.filter_models:
            fm.filter(dt, z, R)

        self.sum_densities = sum(
            fm.density for fm in self.filter_models)

    def state_combination(self):

        self.x = np.sum(fm.density * fm.x / self.sum_densities
                        for fm in self.filter_models)

        self.P = np.sum(
            fm.density / self.sum_densities *
            (fm.P + np.outer(fm.x - self.x, fm.x - self.x))
            for fm in self.filter_models
        )
