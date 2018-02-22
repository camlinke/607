import visdom
import numpy as np

vis = visdom.Visdom()
vis.close(win=None)


X = np.zeros(10)
Y = np.zeros(10)

# for _ in range(100):
#     result1 = np.random.random()
#     results.append(result)

mywin = vis.heatmap(
        X=np.outer(X, Y),
        opts=dict(
            # columnnames=np.linspace(0, 1.0, 10),
            # rownames=np.linspace(0, 1.0, 10),
            colormap='Electric',
        ),
    )


for i in range(1000):

    X[np.random.randint(0,10)] += 1
    Y[np.random.randint(0,10)] += 1
    vis.heatmap(
        X=np.outer(X, Y),
        opts=dict(
            # columnnames=np.linspace(0, 1.0, 10),
            # rownames=np.linspace(0, 1.0, 10),
            colormap='Electric',
        ),
        win=mywin,
    )