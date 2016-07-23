
from filter.plotter import Plot2dMixin
import numpy as np


def generate_switching_matrix(n, diag):
    if n > 1:
        switching_matrix = np.full((n, n), 0.05 / (n - 1))
        np.fill_diagonal(switching_matrix, diag)
        return switching_matrix

    return np.eye(1)


class IMM(Plot2dMixin):

    def __init__(self, filter_models, switching_matrix):
        super(IMM, self).__init__()

        # will be initialized on first measurement
        self.x = None
        self.P = None

        self.filter_models = filter_models
        self.rescale_filter_probabilities()

        self.switching_matrix = switching_matrix
        assert self.switching_matrix.shape == (len(self.filter_models),
                                               len(self.filter_models))

    def rescale_filter_probabilities(self):
        probability_sum = sum(fm.probability
                              for fm in self.filter_models)

        # re-scale probability of different filters
        for fm in self.filter_models:
            fm.probability /= probability_sum

    def filter(self, dt, z, R):
        if self.x is not None:
            # the different Kalman filters are expected to be not initialized
            self.state_interaction()

        self.update_models(dt, z, R)
        self.state_combination()

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
            row * fm.probability
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

    def update_models(self, dt, z, R):

        for fm in self.filter_models:
            fm.filter(dt, z, R)

        # the different filter have updated their probabilities which need to
        # be rescaled
        self.rescale_filter_probabilities()

    def state_combination(self):

        self.x = np.sum(fm.probability * fm.x
                        for fm in self.filter_models)

        self.P = np.sum(
            fm.probability * (fm.P + np.outer(fm.x - self.x, fm.x - self.x))
            for fm in self.filter_models
        )
