from track import Track


class Tracker(object):

    def __init__(self, filter_factory):
        self.filter_factory = filter_factory

        self.tracks = []

    def filter(self, dt, z, R):

        associated = [track.filter(dt, z, R) for track in self.tracks]

        if not any(associated):

            new_track = Track(filter_model=self.filter_factory())
            new_track.filter(dt, z, R)

            self.tracks.append(new_track)

    def plot_all(self, *args, **kwargs):
        for track in self.tracks:
            track.filter_model.plot_all(*args, **kwargs)

    def show(self, *args, **kwargs):
        for track in self.tracks:
            track.filter_model.show(*args, **kwargs)
