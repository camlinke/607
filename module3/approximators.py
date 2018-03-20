import numpy as np
import tiles3 as tiles

class BinnedApproximator:
    def __init__(self, input_range, num_bins):
        self.num_bins = num_bins
        self.activations = np.linspace(np.min(input_range),
                                np.max(input_range),
                                num_bins)
        self.current_approximation = None

    def approximate(self, state):
        self.current_approximation = np.digitize(state, self.activations)
        return self.current_approximation

    def approximate_no_save(self, state):
        return np.digitize(state, self.activations)

    @property
    def shape(self):
        return self.num_bins

class TiledApproximator:
    def __init__(self):
        pass

    def approximate(self, state):
        pass

    def approximate_no_save(self, state):
        pass

    @property
    def shape(self):
        pass
