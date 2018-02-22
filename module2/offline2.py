from lib_robotis_hack import *
import numpy as np
import visdom
import collections
import time

vis = visdom.Visdom()
vis.close(win=None)

actual_results_window = collections.deque(100*[0], 100)
predicted_results_window = collections.deque(100*[0], 100)
verifier_window = collections.deque(100*[0], 100)
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


data = np.loadtxt("sample")

class BinnedApproximator:
    def __init__(self, input_range, num_bins):
        self.activations = np.linspace(np.min(input_range),
                                np.max(input_range),
                                num_bins)

    def approximate(self, state):
        return np.digitize(state, self.activations)


num_bins = 50
# gamma = 0.0
lamb = 0.9
alpha = 0.1
# alpha = (1-lamb) * alpha

# bins_count = np.zeros(num_bins)

# actions = stay, clockwise, counterclockwise
actions = np.array([0, 1, 2])
current_action = 1
previous_action = 1
# s1.move_angle(2, blocking=False)

w = np.zeros((actions.shape[0], num_bins))

# approximator = BinnedApproximator(input_range=[-2.1, 2.1], num_bins=num_bins)
previous_bins = np.array(0)

# z = np.linspace(-np.pi, np.pi, 201)

target = 2

z = np.zeros(w.shape)

e = np.zeros(w.shape)
theta = np.zeros(w.shape)
beta = 0.001 * alpha
last_gamma = 0

counter = 0
right_max = 11
left_max = 11
restart = False

alphas = [0.1]# [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 10.0]
betas = [10.0, 1.0, 0.1, 0.01, 0.001]
bins_to_test = [11]# [5, 11, 13, 15, 17, 19, 21, 31, 41, 51, 99]
lambdas = [0.99]# [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

for a in alphas:
    # for bet in betas:
    for b in bins_to_test:
        for l in lambdas:
            # alpha = (1-l) * a
            # beta = bet * alpha
            counter = 0
            restart = False
            errors = []
            w = np.zeros((actions.shape[0], b))
            z = np.zeros(w.shape)
            # theta = np.zeros(w.shape)
            # e = np.zeros(w.shape)
            last_gamma = 0
            approximator = BinnedApproximator(input_range=[-2.1, 2.1], num_bins=b)
            current_action = 1
            previous_action = 1
            previous_bins = np.array(0)

            for i in range(10000):
                # time.sleep(1)
                angle = data[i][2]
                load = data[i][3]
                bins = approximator.approximate(angle)
                current_action = int(data[i][1])

                zero_angle_cumulant = 1
                if np.isclose([angle], [0.0], atol=0.1)[0]:
                    # zero_angle_cumulant = 1
                    gamma = 0.0
                    # counter = 0
                else:
                    # zero_angle_cumulant = 0
                    gamma = 1.0


                # Verifier for TD Lambda
                if np.isclose([angle], [0.0], atol=0.1)[0]:
                    counter = 0
                    restart = True
                    # moving_away = True
                elif restart:
                    counter = 22
                    restart = False
                else:
                    counter -= 1

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
                # if np.isclose([angle], [0.0], atol=0.1)[0]:
                #     counter = 0
                #     moving_away = True
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


                # TD Lambda predicting load at each times step
                # gamma = 0.0
                # delta = load
                # delta -= w[previous_action][previous_bins]
                # z[previous_action][previous_bins] = 1

                # delta += gamma * w[current_action][bins]
                # w += alpha * delta * z
                # z = gamma * l * z

                # TD Lambda predicting steps until 0 under behavior policy (where agent moves back and forth between
                # angle 2 and -2)
                delta = zero_angle_cumulant
                delta -= w[previous_action][previous_bins]
                z[previous_action][previous_bins] = 1

                delta += gamma * w[current_action][bins]
                w += alpha * delta * z
                z = gamma * l * z


                # # GTD Lambda Predicting steps to zero angle undel policy that immediately moves to 0 on every step
                # delta = zero_angle_cumulant + gamma * theta[current_action][bins].sum() - theta[previous_action][previous_bins].sum()
                # # Target Policy is go right if negative, go left if positive (i.e. go back to 0)
                # if current_action == 0:
                #     rho = 0
                # elif current_action == 1 and angle > 0:
                #     rho = 0
                #     target_action = 2
                # elif current_action == 2 and angle < 0:
                #     rho = 0
                #     target_action = 1
                # else:
                #     target_action = current_action
                #     rho = 1.0
                # e *= l * last_gamma
                # e[previous_action][previous_bins] += 1.0
                # e *= rho

                # temp = np.zeros(w.shape)
                # temp[current_action][bins] += gamma*(1 - l)*(e.T.dot(w).sum())
                # theta += alpha * (delta * e - temp)

                # temp = np.zeros(w.shape)
                # temp[previous_action][previous_bins] += w[previous_action][previous_bins]
                # w += beta * (delta * e - temp)


                previous_bins = np.copy(bins)

                # predicted_value = w[target_action][bins]
                predicted_value = w[current_action][bins]
                previous_action = current_action
                last_gamma = gamma

                # previous_bins = np.copy(bins)

                # predicted_value = w[current_action][bins]
                # previous_action = current_action
                # last_gamma = gamma

                # error = np.square(predicted_value - load)
                # errors.append(error)

                actual_results_window.append(gamma)
                predicted_results_window.append(predicted_value)
                verifier_window.append(counter)
                x_window.append(i)

                vis.line(
                    X=np.column_stack((x_window, x_window, x_window)),
                    Y=np.column_stack((actual_results_window, predicted_results_window, verifier_window)),
                    win=moving_window,
                    opts=dict(
                        showlegend=True,
                        width=1100,
                        height=500,
                        xlabel='Time Steps',
                        ylabel='Reading',
                        title='Time to Angle 0 On-Policy',
                        legend=['Gamma', 'Prediction (Steps)', 'Verifier'])
                    )

                # actual_results_window.append(load)
                # predicted_results_window.append(predicted_value)
                # # verifier_window.append(counter)
                # x_window.append(i)

                # vis.line(
                #     X=np.column_stack((x_window, x_window)),
                #     Y=np.column_stack((actual_results_window, predicted_results_window)),
                #     win=moving_window,
                #     opts=dict(
                #         showlegend=True,
                #         width=1100,
                #         height=500,
                #         xlabel='Time Steps',
                #         ylabel='Reading',
                #         title='Predicted Load vs. Actual Load',
                #         legend=["Actual Load (N.m)", 'Predicted Load'])
                #     )

            # np.save("errors/on-policy-load-{}-{}-{}".format(alpha, b, l), errors)