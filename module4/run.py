import numpy as np
from lib_robotis_hack import *
import serial
import thread
import time
from sarsa import Sarsa
from actor_critic import DiscreteActorCritic, ContinuousActorCritic

# port = "/dev/tty.usbserial-AI03QEMU"

D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU",baudrate=1000000)
s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
# s2 = Robotis_Servo(D,s_list[1])
s2 = None

def log_data(data, filename):
    with open(filename, 'ab') as f:
        np.savetxt(f, data, delimiter=",", newline=" ", footer="\n", comments="")

def observation(s1, s2):
    observation = {
        "s1_load": s1.read_load(),
        "s1_angle": s1.read_angle(),
    }
    return observation

def gamma_fn(state):
    return 0.7

def cumulant(state):
    return state["s1_angle"]
    # if np.isclose([state["s1_angle"]], [1.7], atol=0.3)[0]:
    # # if state["s1_angle"] == 0:
    #     return 1.0
    # else:
    #     return 0
    # if np.isclose([state["s1_angle"]], [0.0], atol=0.2)[0]:
    # # if state["s1_angle"] == 0:
    #     return 1.0
    # else:
    #     return 0

# class Actions:
#     def __init__(self, s1, s2):
#         self.s1 = s1
#         self.s2 = s2

#     def take_action(self, action_number):
#         if action_number == 0:
#             self.s1.move_angle(self.s1.read_angle(), blocking=False)
#         elif action_number == 1:
#             self.s1.move_angle(2, blocking=False)
#         elif action_number == 2:
#             self.s1.move_angle(-2, blocking=False)
#         else:
#             self.s1.move_angle(self.s1.read_angle(), blocking=False)

# a = Actions(s1, s2)

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

class Approximator:
    def __init__(self, input_range, num_bins):
        self.binned_approximator = BinnedApproximator(input_range, num_bins)
        self.current_approximation = None
        self.num_bins = num_bins

    def approximate(self, state):
        bins = self.binned_approximator.approximate(state["s1_angle"])
        # self.current_approximation = np.zeros(self.num_bins * 2)
        self.current_approximation = bins + 0
        if state["s1_angle"] >= 0:
            self.current_approximation = bins + (self.num_bins)
        return self.current_approximation

    def approximate_no_save(self, state):
        bins = self.binned_approximator.approximate(state["s1_angle"])[0]
        # self.current_approximation = np.zeros(self.num_bins * 2)
        if state["s1_angle"] >= 0:
            return bins + (self.num_bins)
        return bins

    @property
    def shape(self):
        return self.num_bins * 2


approximator = Approximator([-2.1, 2.1], 11)
# approximator = BinnedApproximator([-2.1, 2.1], 11)
# class Approximator:
#     def __init__(self, num_bins=11):
#         self.num_bins = num_bins

#     def approximate(self, state):
        

#     @property
#     def shape(self):
#         return self.num_bins

def main():
    ac = ContinuousActorCritic(alpha=0.01, lamda=0.9, gamma_fn=gamma_fn,
                               approximator=approximator)
    for i in range(60000):
        start = time.time()
        state = observation(s1, s2)
        action = ac.get_action(state)
        ac.learn(state, cumulant(state))
        reward = cumulant(state)
        s1.move_angle(action, blocking=False)

        data = [i, reward, ac.mu, ac.sigma]
        log_data(data, "continuous2")

        print("MU: {}".format(ac.mu))
        print("SIGMA: {}".format(ac.sigma))
        sleep = 0.1 - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)

# def main():
#     ac = DiscreteActorCritic(alpha=0.1, lamda=0.9, gamma_fn=gamma_fn,
#                              approximator=approximator, num_actions=2)
#     action_map = [-1.9, 1.9]
#     for i in range(50000):
#         start = time.time()
#         state = observation(s1, s2)
#         a = ac.get_action(state)
#         action = action_map[a]
#         # print(cumulant(state))
#         reward = cumulant(state)
#         ac.learn(state, a, reward)
#         s1.move_angle(action, blocking=False)

#         data = [i, reward, action, ac.action_probs[0], ac.action_probs[1]]
#         print(ac.action_probs)
#         log_data(data, "discrete")
#         sleep = 0.1 - (time.time() - start)
#         if sleep > 0:
#             time.sleep(sleep)


if __name__ == '__main__':
    main()
# def main():
#     angle = s1.read_angle()
#     s = Sarsa(alpha=0.1, gamma_fn=gamma_fn, lambd=1.0, approximator=approximator,
#               numactions=3, initial_state=angle, initial_action=0)
#     for i in range(10000):
#         t = time.time()
#         # D.read_chunk()
#         angle = s1.read_angle()
#         action, action_prob = s.get_action_egreedy(angle, 0.1)
#         a.take_action(action)
#         s.update(cumulant(angle), angle, action)
#         print(action)
#         # print(time.time() - t)

# if __name__ == '__main__':
#     main()