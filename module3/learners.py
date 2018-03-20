import numpy as np
import collections

class IncVariance:
    """Calculates the incremental variance.
       Author: Niko Yasui
    """
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

class MovingAverage:
    """Calculates the moving average"""
    def __init__(self, window_size=20):
        self.cumsum = 0
        self.window = collections.deque(window_size*[0], window_size)

    def update(self, x):
        self.window.append(x)
        return np.average(self.window)

    def average(self):
        return np.average(self.window)

class GTDLambda:
    """Implementation of the GTD Lambda Algorithm"""
    def __init__(self, alpha, beta, lambd, gamma, cumulant_function, function_approximator=None, policy=None,
                 track_rupee=False, track_ude=False, plot_data=False, name=None):
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.gamma = gamma
        self.cumulant_function = cumulant_function
        self.function_approximator = function_approximator
        self.policy = policy
        self.theta = np.zeros(self.function_approximator.shape)
        self.e = np.zeros(self.function_approximator.shape)
        self.w = np.zeros(self.function_approximator.shape)
        self.rupee_alpha = 5 * self.alpha
        self.rupee_beta_not = (1 - self.lambd) * self.rupee_alpha / 30
        self.rupee_h = np.zeros(self.function_approximator.shape)
        self.rupee_vec = 0.0 #np.zeros(self.function_approximator.shape)
        self.ude = 0.0
        self.ude_variance = IncVariance()
        self.ude_moving_average = 0
        self.track_rupee = track_rupee
        self.track_ude = track_ude
        self.plot_data = plot_data
        # self.save_data = save_data
        self.name = name
        self.previous_state = None
        self.previous_action = None
        self.previous_bins = None
        self.current_prediction = None
        self.rupee_tornado = 0.0
        self.ude_tornado = 0.0
        self.rupee_tau = 0.0
        self.ude_tornado = 0.0
        self.ude_tau = 0.0
        self.ude_beta_not = 10 * self.alpha
        self.current_gamma = None

    def update(self, state, current_policy):
        cumulant = self.cumulant_function(state)
        bins = self.function_approximator.current_approximation
        self.current_gamma = self.gamma(state)

        delta = cumulant + self.gamma(state) + self.theta[bins].sum() - self.theta[self.previous_bins].sum()
        

        if self.policy.action_probability(state) == 0:
            rho = 0
            return
        else:
            rho = current_policy.action_probability(state) / self.policy.action_probability(state)
            rho *= 0.5

        self.e *= rho * self.gamma(self.previous_state) * self.lambd
        self.e[self.previous_bins] += rho

        temp = np.zeros(self.function_approximator.shape)
        temp[bins] += self.gamma(state) * (1 - self.lambd) * (self.e.T.dot(self.w).sum())
        self.theta += (self.alpha / bins.size) * (delta * self.e - temp)

        temp = np.zeros(self.function_approximator.shape)
        temp[self.previous_bins] += self.w[self.previous_bins]
        self.w += (self.beta / bins.size) * (delta * self.e - temp)

        # self.previous_action = action
        self.previous_bins = np.copy(bins)
        self.previous_state = state.copy()
        
        if self.track_rupee:
            self.update_rupee(bins, delta)
        if self.track_ude:
            self.update_ude(bins, delta)
        # if self.plot_data:
        #     self.plot()
        # if self.save_data:
        #     self.save()

    def predict(self, state):
        bins = self.function_approximator.approximate_no_save(state)
        self.current_prediction = self.theta[bins].sum()
        # print(bins)
        # print(self.theta)
        # print(self.current_prediction)
        return self.current_prediction

    def update_rupee(self, x, delta):
        self.rupee_tau = (1 - self.rupee_beta_not) * self.rupee_tau + self.rupee_beta_not
        beta = self.rupee_beta_not / self.rupee_tau
        self.rupee_tornado = (1 - beta) * self.rupee_tornado + beta * delta * self.e
        self.rupee_h[x] += self.rupee_alpha * (delta * self.e[x] - self.rupee_h[x].sum())
        self.rupee_vec = np.sqrt(np.abs(self.rupee_h.dot(self.rupee_tornado)))

    def update_ude(self, x, delta):
        temp = np.zeros(self.function_approximator.shape)
        temp[x] += 1
        self.ude_tau = (1 - self.ude_beta_not) * self.ude_tau + self.ude_beta_not
        beta = self.ude_beta_not / self.ude_tau
        self.ude_tornado = (1 - beta) * self.ude_tornado + beta * delta
        self.ude_variance.update(self.ude_tornado)
        self.ude_moving_average = (self.ude_tornado + self.ude_moving_average) / self.ude_variance.n
        self.ude = np.absolute(self.ude_tornado / (np.sqrt(self.ude_variance.ewmv) + 0.00001))


class TDLambda:

    def __init__(self, alpha, lambd, gamma, cumulant_function, function_approximator,
                 policy, name=None):
        self.alpha = alpha
        self.lambd = lambd
        self.gamma = gamma
        self.cumulant_function = cumulant_function
        self.function_approximator = function_approximator
        self.policy = policy
        self.weights = np.zeros(self.function_approximator.shape)
        self.z = np.zeros(self.function_approximator.shape)
        self.previous_bins = 0
        self.current_gamma = 0

    def update(self, state, current_policy):
        # if current_policy != self.policy:
        #     return
        cumulant = self.cumulant_function(state)
        bins = np.copy(self.function_approximator.current_approximation)
        gamma = self.gamma(state)

        # temp = np.zeros(self.z.shape)
        # temp[self.previous_bins] = 1
        self.z = gamma * self.lambd * self.z
        self.z[self.previous_bins] += 1

        # temp = np.zeros(self.weights.shape)

        delta = cumulant + (gamma * self.weights[bins].sum()) - (self.weights[self.previous_bins].sum())

        self.weights = self.weights + (self.alpha * delta * self.z)

        # delta = cumulant - self.weights[self.previous_bins]
        # self.z[self.previous_bins] = 1

        # delta += gamma * self.weights[bins]
        # self.weights += self.alpha * delta * self.z
        # z = gamma * self.lambd * self.z

        self.current_gamma = np.copy(gamma)
        self.previous_bins = np.copy(bins)

        # print self.weights

    def predict(self, state):
        bins = self.function_approximator.approximate_no_save(state)
        self.current_prediction = self.weights[bins].sum()
        return self.current_prediction

