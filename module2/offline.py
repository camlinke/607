from lib_robotis_hack import *
import numpy as np
import visdom
import collections
import time

vis = visdom.Visdom()
vis.close(win=None)

load_window = collections.deque(100*[0], 100)
predicted_results_window_td = collections.deque(100*[0], 100)
predicted_results_window_gtd = collections.deque(100*[0], 100)
gamma_window = collections.deque(100*[0], 100)
predicted_results_window_load = collections.deque(100*[0], 100)
verifier_window_td = collections.deque(100*[0], 100)
verifier_window_gtd = collections.deque(100*[0], 100)
x_window = collections.deque([x for x in range(100)], 100)

moving_window_load = vis.line(
        X=np.column_stack(([0],[0])),
        Y=np.column_stack(([0],[0])),
        opts=dict(
            showlegend=True,
            width=1200,
            height=200,
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            xlabel='Time Steps',
            ylabel='Reading',
            title='Predicted Load vs. Actual Load',
            legend=["Actual Load (N.m)", 'Predicted Load']
        ),
    )

moving_window_on_policy_zero = vis.line(
        X=np.column_stack(([0],[0])),
        Y=np.column_stack(([0],[0])),
        opts=dict(
            showlegend=True,
            width=1200,
            height=200,
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            xlabel='Time Steps',
            ylabel='Reading',
            title='Time to Angle 0 On-Policy',
            legend=['Gamma', 'Prediction (Steps)', 'Verifier']
        ),
    )

moving_window_gtd = vis.line(
        X=np.column_stack(([0],[0])),
        Y=np.column_stack(([0],[0])),
        opts=dict(
            showlegend=True,
            width=1200,
            height=200,
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            xlabel='Current Step',
            ylabel='Reading',
            title='Off-Policy Number of Steps to Zero Angle',
            legend=['Gamma', 'Prediction (Steps)', 'Verifier (Steps)']
        ),
    )


data = np.loadtxt("sample")

class BinnedApproximator:
    def __init__(self, input_range, num_bins):
        self.activations = np.linspace(np.min(input_range),
                                np.max(input_range),
                                num_bins)

    def approximate(self, state):
        return np.digitize(state, self.activations)



# bins_count = np.zeros(num_bins)

# actions = stay, clockwise, counterclockwise
actions = np.array([0, 1, 2])
current_action = 1
previous_action = 1
# s1.move_angle(2, blocking=False)

# w = np.zeros((actions.shape[0], num_bins))

# General Setup
num_bins = 11
approximator = BinnedApproximator(input_range=[-2.1, 2.1], num_bins=num_bins)
previous_bins = np.array(0)



# z = np.linspace(-np.pi, np.pi, 201)

# target = 2

# TD Setup for predicting load
alpha_load = 0.1
lambda_load = 0.99
gamma_load = 0
w_load = np.zeros((actions.shape[0], num_bins))
z_load = np.zeros(w_load.shape)


# TD Setup for predicting zero angle
alpha_zero_angle = 0.1
lambda_zero_angle = 0.99
w_td_zero_angle = np.zeros((actions.shape[0], num_bins))
z_td_zero_angle = np.zeros(w_td_zero_angle.shape)


# GTD Setup
# num_bins = 11
lambda_gtd = 0.9
alpha = 0.1
alpha_gtd = (1-lamb) * alpha
beta = 0.01 * alpha
w_gtd = np.zeros((actions.shape[0], num_bins))
e = np.zeros(w_gtd.shape)
theta = np.zeros(w_gtd.shape)
last_gamma = 0

# Verifier Setup
counter = 0
right_max = 11
left_max = 11
restart = False

# alphas = [0.001, 0.01, 0.1, 0.2, 0.5]
# bins_to_test = [5, 11, 13, 15, 17, 19, 21, 31, 41, 51, 99]
# lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

# for alpha in alphas:
#     for b in bins_to_test:
#         for l in lambdas:
#             counter = 0
#             restart = False
#             errors = []
#             w = np.zeros((actions.shape[0], b))
#             z = np.zeros(w.shape)
#             last_gamma = 0
#             approximator = BinnedApproximator(input_range=[-2.1, 2.1], num_bins=b)
#             current_action = 1
#             previous_action = 1
#             previous_bins = np.array(0)

for i in range(10000):
    # time.sleep(1)
    angle = data[i][2]
    load  = data[i][3]
    bins = approximator.approximate(angle)
    current_action = int(data[i][1])

    # Setup for zero angle cumulant used for verifier 2 and 3
    zero_angle_cumulant = 1
    if np.isclose([angle], [0.0], atol=0.1)[0]:
        gamma_zero_angle = 0.0
    else:
        gamma_zero_angle = 1.0


    # Verifier for TD Lambda
    if np.isclose([angle], [0.0], atol=0.1)[0]:
        counter_td = 0
        restart = True
        # moving_away = True
    elif restart:
        counter_td = 22
        restart = False
    else:
        counter_td -= 1

    # elif np.isclose([angle], [2.0], atol=0.1)[0]:
    #     counter = right_max
    #     moving_away = False
    # elif np.isclose([angle], [-2.0], atol=0.1)[0]:
    #     counter = left_max
    #     moving_away = False
    # else:
    #     if moving_away:
    #         counter += 1
    #     else:
    #         counter -= 1

    # Verifier for GTD Lambda case predicting time to 0 angle under a return to 0 angle policy
    if np.isclose([angle], [0.0], atol=0.1)[0]:
        counter_gtd = 0
        moving_away = True
    elif np.isclose([angle], [2.0], atol=0.1)[0]:
        counter_gtd = right_max
        moving_away = False
    elif np.isclose([angle], [-2.0], atol=0.1)[0]:
        counter_gtd = left_max
        moving_away = False
    else:
        if moving_away:
            counter_gtd += 1
        else:
            counter_gtd -= 1


    # TD Lambda predicting load at each times step
    delta_load = load
    delta_load -= w_load[previous_action][previous_bins]
    z_load[previous_action][previous_bins] = 1

    delta += gamma_load * w_load[current_action][bins]
    w_load += alpha_load * delta_load * z_load
    z_load = gamma_load * lambda_load * z_load

    predicted_load_value = w_load[current_action][bins]

    # TD Lambda predicting steps until 0 under behavior policy (where agent moves back and forth between
    # angle 2 and -2)
    delta_zero_angle = zero_angle_cumulant
    delta_zero_angle -= w_td_zero_angle[previous_action][previous_bins]
    z_td_zero_angle[previous_action][previous_bins] = 1

    deltadelta_zero_angle += gamma_zero_angle * w_td_zero_angle[current_action][bins]
    w_td_zero_angle += alpha_zero_angle * deltadelta_zero_angle * z_td_zero_angle
    z_td_zero_angle = gamma_zero_angle * lambda_zero_angle * z_td_zero_angle

    predicted_td_value = w_td_zero_angle[current_action][bins]


    # GTD Lambda Predicting steps to zero angle undel policy that immediately moves to 0 on every step
    gtd_delta = zero_angle_cumulant + gamma_zero_angle * theta[current_action][bins].sum() - theta[previous_action][previous_bins].sum()
    # Target Policy is go right if negative, go left if positive (i.e. go back to 0)
    if current_action == 0:
        rho = 0
    elif current_action == 1 and angle > 0:
        rho = 0
        target_action = 2
    elif current_action == 2 and angle < 0:
        rho = 0
        target_action = 1
    else:
        target_action = current_action
        rho = 1.0
    e *= lamb * last_gamma_zero_angle
    e[previous_action][previous_bins] += 1.0
    e *= rho

    temp = np.zeros(w_gtd.shape)
    temp[current_action][bins] += gamma_zero_angle*(1 - lambda_gtd)*(e.T.dot(w_gtd).sum())
    theta += alpha_gtd * (gtd_delta * e - temp)

    temp = np.zeros(w_gtd.shape)
    temp[previous_action][previous_bins] += w_gtd[previous_action][previous_bins]
    w_gtd += beta * (gtd_delta * e - temp)

    # previous_GTD_bins = np.copy(gtd_bins)

    predicted_gtd_value = theta[target_action][bins]
    previous_action = current_action
    last_gamma_zero_angle = gamma_zero_angle

    previous_bins = np.copy(bins)

    # predicted_value = w[current_action][bins]
    # previous_action = current_action
    # last_gamma = gamma

    # error = np.square(predicted_value - counter)
    # errors.append(error)

    # actual_results_window.append(gamma)
    # predicted_results_window.append(predicted_value)
    # verifier_window.append(counter)

    load_window.append(load)
    predicted_results_window_load.append(predicted_load_value)
    predicted_results_window_td.append(predicted_td_value)
    predicted_results_window_gtd.append(predicted_gtd_value)
    gamma_window.append(gamma_zero_angle)
    verifier_window_td.append(counter_td)
    verifier_window_gtd.append(counter_gtd)
    x_window.append(i)

    vis.line(
        X=np.column_stack((x_window, x_window)),
        Y=np.column_stack((load_window, predicted_results_window_load)),
        win=moving_window_load,
        opts=dict(
            showlegend=True,
            width=1200,
            height=200,
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            xlabel='Time Steps',
            ylabel='Reading',
            title='Predicted Load vs. Actual Load',
            legend=["Actual Load (N.m)", 'Predicted Load']
        ),
    )

    vis.line(
            X=np.column_stack((x_window, x_window, x_window)),
            Y=np.column_stack((gamma_window, predicted_results_window_td, verifier_window_td)),
            win=moving_window_on_policy_zero,
            opts=dict(
                showlegend=True,
                width=1200,
                height=200,
                marginleft=30,
                marginright=30,
                marginbottom=80,
                margintop=30,
                xlabel='Time Steps',
                ylabel='Reading',
                title='Time to Angle 0 On-Policy',
                legend=['Gamma', 'Prediction (Steps)', 'Verifier']
            ),
        )

    vis.line(
            X=np.column_stack((x_window, x_window, x_window)),
            Y=np.column_stack((gamma_window, predicted_results_window_gtd, verifier_window_gtd)),
            win=moving_window_gtd,
            opts=dict(
                showlegend=True,
                width=1200,
                height=200,
                marginleft=30,
                marginright=30,
                marginbottom=80,
                margintop=30,
                xlabel='Current Step',
                ylabel='Reading',
                title='Off-Policy Number of Steps to Zero Angle',
                legend=['Gamma', 'Prediction (Steps)', 'Verifier (Steps)']
            ),
        )