from lib_robotis_hack import *
import numpy as np
import visdom
import collections
import time

# D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU",baudrate=1000000)
# s_list = find_servos(D)
# s1 = Robotis_Servo(D,s_list[0])


def log_data(data, filename):
    with open(filename, 'ab') as f:
        np.savetxt(f, data, delimiter=",", newline=" ", footer="\n", comments="")


# current_action = 1
# previous_action = 1
# s1.move_angle(2, blocking=False)


# for i in range(25000):
#     if np.isclose([s1.read_angle()], [-2], atol=0.01)[0]:
#         current_action = 1
#         s1.move_angle(2, blocking=False)

#     if np.isclose([s1.read_angle()], [2], atol=0.01)[0]:
#         current_action = 2
#         s1.move_angle(-2, blocking=False)

#     data = [i, current_action, s1.read_angle(), s1.read_load(), s1.read_voltage(), s1.read_encoder(), s1.read_temperature()]

#     log_data(data, "sample")

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


data = np.loadtxt("sample")

count = 0

for i in range(len(data)):
    time.sleep(0.05)
    predicted_value = data[i][2]
    gamma = 1
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
