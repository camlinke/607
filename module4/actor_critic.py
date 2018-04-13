import numpy as np

class DiscreteActorCritic:

    def __init__(self, alpha=0.1, lamda=1.0, gamma_fn=lambda x: 1.0,
                 approximator=None, num_actions=0, state_state=0, 
                 start_action=0):
        self.alpha_w = alpha
        self.alpha_theta = self.alpha_w * 0.1
        self.lamda = lamda
        self.gamma_fn = gamma_fn
        self.approximator = approximator
        self.num_actions = num_actions
        self.w = np.zeros(approximator.shape)
        self.z_w = np.zeros(self.w.shape)
        self.theta = np.zeros((approximator.shape, num_actions))
        self.z_theta = np.zeros((approximator.shape, num_actions))
        self.z = np.zeros((10, 2))
        self.I = 1
        self.a = None

        self.previous_bins = 0
        self.action_probs = [0.5, 0.5]


    def get_action(self, state):
        """Returns action and action probability"""
        bins = self.approximator.approximate(state)
        action_values = self.theta[bins]
        action_probs = np.exp(action_values - np.max(action_values)) / np.sum(np.exp(action_values - np.max(action_values)))
        action = np.random.choice(self.num_actions, p=action_probs)
        self.action_probs = action_probs
        return action

    def learn(self, state, action, reward):
        bins = self.approximator.approximate(state)
        gamma = self.gamma_fn(state)
        delta = reward - gamma * self.w[bins].sum()  - self.w[self.previous_bins].sum()

        # Critic
        self.z_w = gamma * self.lamda * self.z_w
        self.z_w[self.previous_bins] += (self.I * gamma)
        self.w += self.alpha_w * delta * self.z_w


        # Actor
        temp = np.zeros((self.approximator.shape, self.num_actions))
        temp[self.previous_bins][action] += (self.I * self.action_probs[action]) #This would need to change with more than 2 actions
        self.z_theta = (gamma * self.lamda * self.z_theta) + temp
        self.theta = self.theta + (self.alpha_theta * delta * self.z_theta)


        self.I = gamma * self.I
        self.previous_bins = bins #np.copy(bins)


class ContinuousActorCritic:

    def __init__(self, alpha=0.1, lamda=1.0, gamma_fn=lambda x: 1.0,
                 approximator=None, actions=None):
        self.alpha_v = alpha
        self.alpha_mu = self.alpha_v * 0.1
        self.alpha_sigma = self.alpha_v * 0.05
        self.lamda = lamda
        self.gamma_fn = gamma_fn

        self.approximator = approximator

        self.w_v = np.zeros(self.approximator.shape)
        self.e_v = np.zeros(self.w_v.shape)

        self.w_mu = np.zeros(self.approximator.shape)
        self.e_mu = np.zeros(self.w_mu.shape)
        self.w_sigma = np.zeros(self.approximator.shape)
        self.e_sigma = np.zeros(self.w_sigma.shape)

        self.mu = 0
        self.sigma = 1

        self.a = None

        self.previous_bins = 0
        # self.count = 0


    def get_action(self, state):
        bins = self.approximator.approximate(state)
        self.mu = self.w_mu[bins].sum()
        self.sigma = np.exp(self.w_sigma[bins].sum())
        self.a = np.random.normal(self.mu, self.sigma)
        if self.a > 0:
            self.a = min(self.a, 2.0)
        elif self.a < 0:
            self.a = max(self.a, -2.0)
        return self.a

    def learn(self, state, reward):
        bins = self.approximator.approximate(state)
        gamma = self.gamma_fn(state)
        delta = reward + gamma * self.w_v[bins].sum() - self.w_v[self.previous_bins].sum()
        
        # Update Critic
        self.e_v = self.lamda * gamma * self.e_v
        self.e_v[bins] += 1
        self.w_v = self.w_v + self.alpha_v * delta * self.e_v

        # Update Actor

        # mu
        # temp = np.zeros(self.e_sigma.shape)
        # temp[bins] += 1
        self.e_mu = self.lamda * gamma * self.e_mu #+ ((self.a - self.mu) * temp)
        self.e_mu[bins] += (self.a - self.mu)
        self.w_mu = self.w_mu + self.alpha_mu * delta * self.e_mu

        # sigma
        # temp = np.zeros(self.e_sigma.shape)
        # temp[bins] += 1
        self.e_sigma = self.lamda * gamma * self.e_sigma #+ ((np.square((self.a - self.mu)) - 1) * temp)
        self.e_sigma[bins] += (np.square((self.a - self.mu)) - 1)
        self.w_sigma = self.w_sigma + self.alpha_sigma * delta * self.e_sigma

        # self.previous_bins = np.copy#(bins)
        self.previous_bins = bins






































