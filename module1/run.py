from lib_robotis_hack import *

import visdom
import numpy as np

vis = visdom.Visdom()
vis.close(win=None)
# vis.text('Hello world.')
# vis.image(np.ones((3, 10, 10)))

D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU",baudrate=1000000)

s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
s2 = Robotis_Servo(D,s_list[1])

s1.disable_torque()

# win = vis.line(X=np.array([0]), Y=np.array([0]))
# print(dir(vis))

x = 0
y = 0
pred = 0
pred2 = 0
# while x < 1000:
#     y = s1.read_angle()
#     x += 1

#     pred = pred + 0.1 * (y - pred)
#     vis.line(
#         X=np.array([x]),
#         Y=np.array([y]),
#         win=win,
#         update='append',
#     )

win = vis.line(
        X=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        Y=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        opts=dict(
            # fillarea=True,
            showlegend=False,
            width=500,
            height=500,
            xlabel='Time',
            ylabel='Reading',
            # ytype='log',
            title='Servo 1',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            legend=['angle', 'load', 'voltage', 'temp'],
        ),
    )

win2 = vis.line(
        X=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        Y=np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0]))),
        opts=dict(
            # fillarea=True,
            showlegend=False,
            width=500,
            height=500,
            xlabel='Time',
            ylabel='Reading',
            # ytype='log',
            title='Servo 2',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            legend=['angle', 'load', 'voltage', 'temp'],
        ),
    )

win3 = vis.scatter(
        X=np.column_stack((np.array([0]), np.array([0]))),
        # Y=np.array([0]),
        opts=dict(
            # fillarea=True,
            showlegend=False,
            width=300,
            height=300,
            xlabel='Servo1',
            ylabel='Servo2',
            # ytype='log',
            title='Servo 2',
            # marginleft=30,
            # marginright=30,
            # marginbottom=80,
            # margintop=30,
        ),
    )
# win3 = vis.heatmap(
#     X=np.outer(np.array([0]), np.array([0])),
#     opts=dict(
#         columnnames=['0.1', '0.2', '0.3', '0.4', '0;5', '0.6', '0.7', '0.8', '0.9', '1.0'],
#         rownames=['0.1', '0.2', '0.3', '0.4', '0;5', '0.6', '0.7', '0.8', '0.9', '1.0'],
#         colormap='Electric',
#     )
#     )

# trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
#              marker={'color': 'red', 'symbol': 104, 'size': "10"},
#              text=["one", "two", "three"], name='1st Trace')
# layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

# vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})

import collections

angle = collections.deque(50*[0], 5)
load = collections.deque(50*[0], 5)
volt = collections.deque(50*[0], 5)
temp = collections.deque(50*[0], 5)

current_array = np.column_stack((np.array([0]), np.array([0]), np.array([0]), np.array([0])))

import time

target = 100
while x < 1000:
    start = time.time()
    if s1.is_moving():
        continue
    if s2.is_moving():
        continue
    if target == 100:
        target = 400
    else:
        target = 100
    s1.move_to_encoder(target)
    # angle = angle.append(s1.read_angle())
    # load = angle.append(s1.read_load())
    # volt = angle.append(s1.read_voltage())
    # temp = angle.append(s1.read_temperature())

    angle.append(s1.read_angle())
    load.append(s1.read_load())
    volt.append(s1.read_voltage())
    temp.append(s1.read_temperature())

    current_array = np.column_stack((angle, load, volt, temp))

    s2.move_to_encoder(target)
    angle2 = s2.read_angle()
    load2 = s2.read_load()
    volt2 = s2.read_voltage()
    temp2 = s2.read_temperature()

    # y2 = s2.read_angle()
    # y = np.random.randn()
    # y2 = np.random.randn()
    x += 1
    pred = pred + (0.01) * (y - pred)
    # pred2 = pred2 + (0.01) * (y2 - pred2)
    vis.line(
        # X=np.column_stack((np.array([x]), np.array([x]), np.array([x]), np.array([x]))),
        # Y=np.column_stack((np.array([angle]), np.array([load]), np.array([volt]), np.array([temp]))),
        Y=current_array,
        win=win,
        # update='append'
    )

    vis.line(
        X=np.column_stack((np.array([x]), np.array([x]), np.array([x]), np.array([x]))),
        Y=np.column_stack((np.array([angle2]), np.array([load2]), np.array([volt2]), np.array([temp2]))),
        win=win2,
        update='append'
    )
    print(time.time() - start)

    # vis.scatter(
    #     X=np.column_stack((np.array([angle]), np.array([angle2]))),
    #     win=win3,
    #     # update='append',
    # )

    # vis.line(
    #     X=np.column_stack((np.array([x]), np.array([x]))),
    #     Y=np.column_stack((np.array([y2]), np.array([pred2]))),
    #     win=win2,
    #     update='append'
    # ) 