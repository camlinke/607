from lib_robotis_hack import *
import numpy as np
import visdom
import tiles3 as tiles

# h = h + alpha_h * (delta * e, - h.T.dot(x)*x)

# tau = (1 - beta_0) * tau + beta_0
# beta = beta_0 / tau

# tornado = (1 - beta) * tornado + beta * delta * e

# rupee = np.sqrt(abs(h.T.dot(tornado)))

# # Alpha is from regular parameter update

# alpha_rupee = 5 * alpha
# beta_not_rupee = (1 - lamb) * alpha_0 / 30

# beta_not_ude = 10 * alpha 
# ude = abs(another_tornado / np.sqrt())

# From Niko Yasui
class IncVariance:
    """Calculates the incremental variance"""
    def __init__(self):
        self.sample_mean = 0
        self.n = 0
        self.ewmv = 0

    def update(self, x):
        old_mean = self.sample_mean

        self.n += 1
        self.sample_mean += (x - self.sample_mean) / self.n
        var_sample = (x - old_mean) * (x - self.sample_mean)
        self.ewmv += (var_sample - self.ewmv) / self.n


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
    def get_shape(self):
        return (self.num_bins, 1)

class TiledApproximator:
    """Uses the tiles3 approximator taken from CMPUT 609. It is a python2
    implementation of Rich Suttons Tile Coding software. This class provides
    an interface on top of that consistent with the other approximators."""
    def __init__(self, shape):
        self.current_approximation = None
    
    def approximate(self, state):
        pass

    @property
    def shape(self):
        pass

class MovingWindowLine:
    def __init__(self name, window_size=100):
        self.name = name
        self.window_size = window_size

    def update(self):
        pass

class Policy:
    def __init__(self):
        pass

class VisdomPlots:
    def __init__(self):
        pass

class VisdomPlot:
    def __init__(self):
        pass

class Learner:
    def __init__(self):
        super().__init__()
    
    def update(self):
        pass

    def predict(self):
        pass

    def plot(self):
        pass

class GTDLambda(Learner):
    """Implementation of the GTD Lambda Algorithm"""
    def __init__(self, alpha, beta, lambd, gamma, function_approximator=None, policy=None,
                 track_rupee=False, track_ude=False, plot_data=False, name=None,
                 graph_info={}, save_data=True):
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.gamma = gamma
        self.function_approximator = function_approximator
        self.theta = np.zeros(self.function_approximator.shape)
        self.e = np.zeros(self.function_approximator.shape)
        self.w = np.zeros(self.function_approximator.shape)
        self.rupee_alpha = 5 * self.alpha
        self.rupee_h = 0
        self.rupee = 0
        self.ude = 0
        self.ude_variance = IncVariance()
        self.track_rupee = track_rupee
        self.track_ude = track_ude
        self.plot_data = plot_data
        self.save_data = save_data
        self.name = name
        self.previous_state = None
        self.previous_action = None
        self.previous_bins = None
        self.current_prediction = None

    def update(self, cumulant, state, action):
        bins = self.function_approximator.current_approximation
        delta = cumulant + self.gamma(state) + self.theta[action][bins].sum() - self.theta[self.previous_action][self.previous_bins].sum()
        
        # TODO calculate Rho
        rho = 1

        self.e *= self.gamma(self.previous_state) * self.lambd
        self.e[previous_action][previous_bins] += 1.0
        self.e *= rho

        temp = np.zeros(self.function_approximator.shape)
        temp[action][bins] += self.gamma(state) * (1 - self.lambd) * (self.e.T.dot(self.w).sum())
        self.theta += self.alpha * (delta * self.e - temp)

        temp = np.zeros(self.function_approximator.shape)
        temp[self.previous_action][self.previous_bins] += self.w[previous_action][previous_bins]
        self.w += self.beta * (delta * self.e - temp)

        self.previous_action = action
        self.previous_bins = np.copy(bins)
        
        if self.track_rupee:
            self.update_rupee(x)
        if self.track_ude:
            self.update_ude(x)
        if self.plot_data:
            self.plot()
        if self.save_data:
            self.save()

    def predict(self, state, action):
        bins = self.approximator.approximate_no_save(state)
        self.current_prediction = theta[action][bins].sum()
        return self.current_prediction

    def update_rupee(self, x):
        tornado = (1 - beta_rupee) * tornado + beta * delta * e
        rupee_h += alpha_h * (delta * e, - rupee_h.T.dot(x)*x)

    def update_ude(self, x):
        tornado = (1 - beta) * tornado + beta * delta
        self.ude_variance.update(x)
        self.ude = np.absolute(tornado / (np.sqrt(self.ude_variance) + 0.00001))

    def plot(self):
        pass

    def save(self):
        if !self.name:
            pass


