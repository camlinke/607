import numpy as np
import visdom

vis = visdom.Visdom()
vis.close(win=None)

def moving_average(data):
    N = 500
    cumsum, moving_average = [0], []
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_average.append(moving_ave)
    return moving_average

def plot_offline_data(filename):
    data =  np.loadtxt(filename)
    columns = [data[:,c] for c in range(1, len(data[0])) if c != 2]
    Y = np.column_stack(columns)
    x = np.array([i for i in range(604)])
    x_columns = np.column_stack([x for _ in range(3)])

    print(Y.shape)
    print(x_columns.shape)

    win = vis.line(
        Y=Y,
        X=x_columns,
        opts=dict(
            # fillarea=True,
            showlegend=True,
            width=1400,
            height=650,
            xlabel='Timsteps',
            ylabel='Reading',
            # title='Servo 1',
            marginleft=20,
            marginright=10,
            marginbottom=30,
            margintop=30,
            legend=["Reward", "Left Action", "Right Action"],
            title="Discrete Actor-Critic"

        ),
    )

plot_offline_data("discrete")

# def plot_offline_data(filename):
#     data =  np.loadtxt(filename)
#     columns = [data[:,c] for c in range(1, len(data[0])) if c != 0]
#     x = np.array([i for i in range(59501)])
#     x_columns = np.column_stack([x for _ in range(3)])
#     columns = [moving_average(c) for c in columns]
#     Y = np.column_stack(columns)
#     print(Y.shape)
#     print(x_columns.shape)
#     win = vis.line(
#         Y=Y,
#         X=x_columns,
#         opts=dict(
#             # fillarea=True,
#             showlegend=True,
#             width=1400,
#             height=650,
#             xlabel='Timsteps',
#             ylabel='Reading',
#             # title='Servo 1',
#             marginleft=20,
#             marginright=10,
#             marginbottom=30,
#             margintop=30,
#             legend=["Reward", "Mu (Degrees)", "Sigma (Degrees)"],
#             title="Continuous Actor-Critic"
#         ),
#     )

# plot_offline_data("continuous2")