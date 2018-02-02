"""
Author: Cam Linke
"""

from lib_robotis_hack import *
import numpy as np
import visdom
import collections
import time
import csv

vis = visdom.Visdom()
vis.close(win=None)


# D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU",baudrate=1000000)

# s_list = find_servos(D)
# s1 = Robotis_Servo(D,s_list[0])
# s2 = Robotis_Servo(D,s_list[1])


def logscaled(num):
    if num == 0:
        return 0
    if num < 0:
        return -np.log(-num)
    else:
        return np.log(num)

def log_data(data, filename):
    with open(filename, 'ab') as f:
        np.savetxt(f, data, delimiter=",", newline=" ", footer="\n", comments="")

def plot_offline_data(filename):
    data =  np.loadtxt(filename)
    columns = [data[:,c] for c in range(1, len(data[0]))]
    win = vis.line(
        Y=np.column_stack(columns),
        opts=dict(
            # fillarea=True,
            showlegend=True,
            width=400,
            height=325,
            xlabel='Time',
            ylabel='Reading',
            title='Servo 1',
            marginleft=10,
            marginright=10,
            marginbottom=30,
            margintop=10,
        ),
    )


def run():

    windowed_angle1 = collections.deque(100*[0], 100)
    windowed_load1 = collections.deque(100*[0], 100)
    windowed_volt1 = collections.deque(100*[0], 100)
    windowed_temp1 = collections.deque(100*[0], 100)

    win3 = vis.line(
        X=np.column_stack(([0], [0], [0], [0])),
        Y=np.column_stack(([0], [0], [0], [0])),
        opts=dict(
            # fillarea=True,
            showlegend=True,
            width=1000,
            height=325,
            xlabel='Time',
            ylabel='Reading',
            title='Servo 1',
            marginleft=10,
            marginright=10,
            marginbottom=30,
            margintop=10,
            legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
        ),
    )

    win = vis.line(
        X=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        Y=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        opts=dict(
            # fillarea=True,
            showlegend=True,
            width=400,
            height=325,
            xlabel='Time',
            ylabel='Reading',
            title='Servo 1',
            marginleft=10,
            marginright=10,
            marginbottom=30,
            margintop=10,
            legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
        ),
    )

    windowed_angle2 = collections.deque(100*[0], 100)
    windowed_load2 = collections.deque(100*[0], 100)
    windowed_volt2 = collections.deque(100*[0], 100)
    windowed_temp2 = collections.deque(100*[0], 100)

    win4 = vis.line(
        X=np.column_stack(([0], [0], [0], [0])),
        Y=np.column_stack(([0], [0], [0], [0])),
        opts=dict(
            # fillarea=True,
            showlegend=True,
            width=1000,
            height=325,
            xlabel='Time',
            ylabel='Reading',
            title='Servo 2',
            marginleft=10,
            marginright=10,
            marginbottom=30,
            margintop=10,
            legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
        ),
    )

    win2 = vis.line(
        X=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        Y=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        opts=dict(
            # fillarea=True,
            showlegend=True,
            width=400,
            height=325,
            xlabel='Time',
            ylabel='Reading',
            title='Servo 2',
            marginleft=10,
            marginright=10,
            marginbottom=30,
            margintop=10,
            legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
        ),
    )

    x_window = collections.deque([x for x in range(100)], 100)

    target1 = 1.0
    target2 = 0.75

    for x in range(1, 350):

        # Behavior Policy
        if not s1.is_moving(): 
            target1 = -target1
            s1.move_angle(target1, blocking=False)
        if not s2.is_moving():
            target2 = -target2
            s2.move_angle(target2, blocking=False)

        angle1 = s1.read_angle()
        load1 = logscaled(s1.read_load())
        volt1 = logscaled(s1.read_voltage())
        temp1 = logscaled(s1.read_temperature())

        log_data([x, target1, angle1, load1, volt1, temp1], "servo1.dat")

        angle2 = s2.read_angle()
        load2 = logscaled(s2.read_load())
        volt2 = logscaled(s2.read_voltage())
        temp2 = logscaled(s2.read_temperature())

        log_data([x, target2, angle2, load2, volt2, temp2], "servo2.dat")

        vis.line(
            X=np.column_stack((np.array([x]), np.array([x]), np.array([x]), np.array([x]))),
            Y=np.column_stack((np.array([angle1]), np.array([load1]), np.array([volt1]), np.array([temp1]))),
            win=win,
            update='append',
            opts=dict(
                legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
            )
        )

        vis.line(
            X=np.column_stack((np.array([x]), np.array([x]), np.array([x]), np.array([x]))),
            Y=np.column_stack((np.array([angle2]), np.array([load2]), np.array([volt2]), np.array([temp2]))),
            win=win2,
            update='append',
            opts=dict(
                legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
            )
        )

        windowed_angle1.append(angle1)
        windowed_load1.append(load1)
        windowed_volt1.append(volt1)
        windowed_temp1.append(temp1)

        windowed_angle2.append(angle2)
        windowed_load2.append(load2)
        windowed_volt2.append(volt2)
        windowed_temp2.append(temp2)

        x_window.append(x)

        vis.line(
            X=np.column_stack((x_window, x_window, x_window, x_window)),
            Y=np.column_stack((windowed_angle1, windowed_load1, windowed_volt1, windowed_temp1)),
            win=win3,
            update='replace',
            opts=dict(
                legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
            )
        )

        vis.line(
            X=np.column_stack((x_window, x_window, x_window, x_window)),
            Y=np.column_stack((windowed_angle2, windowed_load2, windowed_volt2, windowed_temp2)),
            win=win4,
            update='replace',
            opts=dict(
                legend=['angle (Rad)', 'load (N.m)', 'voltage (V)', 'temp (C)'],
            )
        )


if __name__ == '__main__':
    run()