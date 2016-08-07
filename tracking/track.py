
class Track(object):

    def __init__(self, filter_model):
        self.filter_model = filter_model

    def filter(self, dt, z, R):
        self.filter_model.filter(dt, z, R)

        return self.filter_model.sum_densities > \
            self.filter_model.false_density
