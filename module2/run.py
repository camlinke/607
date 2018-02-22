from lib_robotis_hack import *
import numpy as np
import visdom
import collections
# from learners import TDLambda

D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU",baudrate=1000000)
s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
# s2 = Robotis_Servo(D,s_list[1])

vis = visdom.Visdom()
vis.close(win=None)

actual_results_window = collections.deque(100*[0], 100)
predicted_results_window = collections.deque(100*[0], 100)
x_window = collections.deque([x for x in range(100)], 100)

moving_window = vis.line(
        X=np.column_stack(([0],[0])),
        Y=np.column_stack(([0],[0])),
        opts=dict(
            showlegend=True,
            width=1100,
            height=500,
            xlabel='Time',
            ylabel='Reading',
            title='Servo 2',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            legend=['random', 'prediction'],
        ),
    )



class TDLambda:
    def __init__(self, alpha, gamma, lamb, function_approximator, starting_action):
        self.function_approximator = function_approximator
        self.current_action = None
        self.previous_action = None
        self.weights = self.function_approximator

    def update(state):
        return None

    def predict(state):
        return None

class BinnedApproximator:
    def __init__(self, input_range, num_bins):
        self.activations = np.linspace(np.min(input_range),
                                np.max(input_range),
                                num_bins)

    def approximate(self, state):
        return np.digitize(state, self.activations)


num_bins = 100
# gamma = 0.0
lamb = 0.9
alpha = 0.1
alpha = (1-lamb) * alpha

# bins_count = np.zeros(num_bins)

# actions = stay, clockwise, counterclockwise
actions = np.array([0, 1, 2])
current_action = 1
previous_action = 1
s1.move_angle(2, blocking=False)

w = np.zeros((actions.shape[0], num_bins))

approximator = BinnedApproximator(input_range=[-2.1, 2.1], num_bins=num_bins)
previous_bins = np.array(0)

# z = np.linspace(-np.pi, np.pi, 201)

target = 2

z = np.zeros(w.shape)

e = np.zeros(w.shape)
theta = np.zeros(w.shape)
beta = 0.01 * alpha

# alpha =  (1-lambda)*alpha / Number of active features
# beta = 0.01 * alpha


for i in range(10000):

    if np.isclose([s1.read_angle()], [-2], atol=0.01)[0]:
        current_action = 1
        s1.move_angle(2, blocking=False)

    if np.isclose([s1.read_angle()], [2], atol=0.01)[0]:
        current_action = 2
        s1.move_angle(-2, blocking=False)

    bins = approximator.approximate(s1.read_angle())

    # TD0 (Predicting Load)
    # state_value = w[current_action][bins].sum()
    # previous_state_value = w[previous_action][previous_bins].sum()

    load_cumulant = s1.read_load()
    angle = s1.read_angle()
    # state_value = w[current_action][bins].sum()
    # td_target = load_cumulant + gamma * state_value
    # td_error = td_target - previous_state_value
    # w[current_action][bins] += alpha * td_error

    # previous_bins = np.copy(bins)

    # predicted_value = w[current_action][bins]

    # previous_action = current_action

    # TD(Lambda) Predicting load
    # delta = load_cumulant
    # delta -= w[previous_action][previous_bins]
    # z[previous_action][previous_bins] = 1

    # delta += gamma * w[current_action][bins]
    # w += alpha * delta * z
    # z = gamma * lamb * z

    # TD(lambda) predicting time to zero angle
    # gamma = 1.0
    zero_angle_cumulant = 1
    if np.isclose([s1.read_angle()], [0.0], atol=0.1)[0]:
        # zero_angle_cumulant = 1
        gamma = 0.0
    else:
        # zero_angle_cumulant = 0
        gamma = 1.0

    # delta = zero_angle_cumulant
    # delta -= w[previous_action][previous_bins]
    # z[previous_action][previous_bins] = 1

    # delta += gamma * w[current_action][bins]
    # w += alpha * delta * z
    # z = gamma * lamb * z

    # GTD(Lambda) Learn policy for minimizing distance to 0 angle
    
    delta = zero_angle_cumulant + gamma * theta[current_action][bins].sum() - theta[previous_action][previous_bins].sum()
    # Target Policy is go right if negative, go left if positive (i.e. go back to 0)
    if current_action == 0:
        rho = 0
    elif current_action == 1 and s1.read_angle() > 0:
        rho = 0
        target_action = 2
    elif current_action == 2 and s1.read_angle() < 0:
        rho = 0
        target_action = 1
    else:
        target_action = current_action
        rho = 1.0
    e *= lamb * gamma
    e[previous_action][previous_bins] += 1
    e *= rho

    temp = np.zeros(w.shape)
    temp[current_action][bins] += gamma*(1 - lamb)*(e.T.dot(w).sum())
    theta += alpha * (delta * e - temp)

    temp = np.zeros(w.shape)
    temp[previous_action][previous_bins] += w[previous_action][previous_bins]
    w += beta * (delta * theta - temp)


    previous_bins = np.copy(bins)

    predicted_value = theta[target_action][bins]
    previous_action = current_action


    actual_results_window.append(gamma)
    predicted_results_window.append(predicted_value)
    x_window.append(i)

    #

    vis.line(
        X=np.column_stack((x_window, x_window)),
        Y=np.column_stack((actual_results_window, predicted_results_window)),
        win=moving_window,
        opts=dict(
            showlegend=True,
            legend=['random', 'prediction'])
        )








    # bins = approximator.approximate(raw_state)

    # state_value = w[bins].sum()
    # previous_state_value = w[previous_bins].sum()
    # print state_value
    # print previous_state_value
    
    # td_target = raw_state + gamma * state_value
    # td_error = td_target - previous_state_value
    # w[bins] += alpha * td_error

    # predicted_value = w[bins]

    # actual_results_window.append(raw_state)
    # predicted_results_window.append(predicted_value)
    # x_window.append(i)

    # vis.line(
    #     X=np.column_stack((x_window, x_window)),
    #     Y=np.column_stack((actual_results_window, predicted_results_window)),
    #     win=moving_window,
    #     opts=dict(
    #         showlegend=True,
    #         legend=['random', 'prediction'])
    #     )
    
    # previous_bins = np.copy(bins)