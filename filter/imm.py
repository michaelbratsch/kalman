
from filter.plotter import Plot2dMixin
import numpy as np


class IMM(Plot2dMixin):

    def __init__(self, filter_models, switching_matrix):
        super(IMM, self).__init__()

        # will be initialized on first measurement
        self.x = None
        self.P = None

        self.filter_models = filter_models
        self.n_filters = len(self.filter_models)
        self.rescale_filter_probabilities()

        self.switching_matrix = switching_matrix
        assert self.switching_matrix.shape == (self.n_filters, self.n_filters)

    def get_position(self):
        return self.x[0], self.x[1]

    def get_position_accuracy(self):
        return np.array([[self.P[0, 0], self.P[0, 1]],
                         [self.P[1, 0], self.P[1, 1]]])

    def get_speed(self):
        return self.x[2], self.x[3]

    def rescale_filter_probabilities(self):
        probability_sum = sum(fm.probability
                              for fm in self.filter_models)

        # re-scale probability of different filters
        for fm in self.filter_models:
            fm.probability /= probability_sum

    def filter(self, dt, z, R):
        if self.x is None:
            for filter_model in self.filter_models:
                filter_model.initialize_state(z, R)

            # ToDo: put it somewhere else
            self.x = np.zeros(4)
            self.x[0:2] = z
            self.P = np.identity(4)
            self.P[0:2, 0:2] = R
        else:
            self.state_interaction(dt)
            self.model_probability_update(z, R)
            self.state_combination()

        self.update_plotter(z)

    def state_interaction(self, dt):

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

        def extrapolate_models(dt, states_mixed, covariances_mixed):
            for fm, state, covariance in zip(self.filter_models,
                                             states_mixed,
                                             covariances_mixed):
                fm.x = state
                fm.P = covariance
                fm.extrapolate(dt)

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

        extrapolate_models(dt, states_mixed, covariances_mixed)

    def model_probability_update(self, z, R):

        for fm in self.filter_models:
            fm.update(z, R)

        self.rescale_filter_probabilities()

    def state_combination(self):

        self.x = np.sum(fm.probability * fm.x
                        for fm in self.filter_models)

        self.P = np.sum(
            fm.probability * (fm.P + np.outer(fm.x - self.x, fm.x - self.x))
            for fm in self.filter_models
        )
