import numpy as np


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

class GTDLambda:
    """Implementation of the GTD Lambda Algorithm"""
    def __init__(self, alpha, beta, lambd, gamma, cumulant_function, function_approximator=None, policy=None,
                 track_rupee=False, track_ude=False, plot_data=False, name=None,
                 graph_info={}, save_data=True):
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.gamma = gamma
        self.cumulant_function = cumulant_function
        self.function_approximator = function_approximator
        self.theta = np.zeros(self.function_approximator.shape)
        self.e = np.zeros(self.function_approximator.shape)
        self.w = np.zeros(self.function_approximator.shape)
        self.rupee_alpha = 5 * self.alpha
        self.rupee_beta_not = (1 - self.lambd) * self.rupee_alpha / 30
        self.rupee_h = np.zeros(self.function_approximator.shape)
        self.rupee_vec = np.zeros(self.function_approximator.shape)
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
        self.rupee_tornado = 0
        self.ude_tornado = 0
        self.rupee_tau = 0

    def update(self, state, action):
        cumulant = self.cumulant_function(state)
        bins = self.function_approximator.current_approximation
        delta = cumulant + self.gamma(state) + self.theta[action][bins].sum() - self.theta[self.previous_action][self.previous_bins].sum()
        
        # TODO calculate Rho
        rho = 1

        self.e *= self.gamma(self.previous_state) * self.lambd
        self.e[self.previous_action][self.previous_bins] += 1.0
        self.e *= rho

        temp = np.zeros(self.function_approximator.shape)
        temp[action][bins] += self.gamma(state) * (1 - self.lambd) * (self.e.T.dot(self.w).sum())
        self.theta += self.alpha * (delta * self.e - temp)

        temp = np.zeros(self.function_approximator.shape)
        temp[self.previous_action][self.previous_bins] += self.w[self.previous_action][self.previous_bins]
        self.w += self.beta * (delta * self.e - temp)

        self.previous_action = action
        self.previous_bins = np.copy(bins)

        print(cumulant)
        
        if self.track_rupee:
            self.update_rupee(bins, action, delta)
        # if self.track_ude:
        #     self.update_ude(x)
        # if self.plot_data:
        #     self.plot()
        # if self.save_data:
        #     self.save()

    def predict(self, state, action):
        bins = self.function_approximator.approximate_no_save(state)
        self.current_prediction = self.theta[action][bins].sum()
        # print(bins)
        # print(self.theta)
        # print(self.current_prediction)
        return self.current_prediction

    def update_rupee(self, x, action, delta):
        temp = np.zeros(self.function_approximator.shape)
        temp[action][x] += 1
        self.rupee_tau = (1 - self.rupee_beta_not) * self.rupee_tau + self.rupee_beta_not
        beta = self.rupee_beta_not / self.rupee_tau
        self.rupee_tornado = (1 - beta) * self.rupee_tornado + beta * delta * self.e
        self.rupee_h += self.rupee_alpha * (delta * self.e - self.rupee_h.dot(temp)*temp)
        self.rupee = np.sqrt(np.abs(self.rupee_h.dot(self.rupee_tornado)))

    # def update_ude(self, x):
    #     tornado = (1 - beta) * tornado + beta * delta
    #     self.ude_variance.update(x)
    #     self.ude = np.absolute(tornado / (np.sqrt(self.ude_variance) + 0.00001))

    # def plot(self):
    #     pass

    # def save(self):
    #     if not self.name:
    #         pass
