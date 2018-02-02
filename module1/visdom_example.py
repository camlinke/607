"""
Visdom example for CMPUT 607
Author: Cam Linke

Setup:
1) pip install visdom
2) run python -m visdom.server from the command line
3) navigate to http://localhost:8097 in your browser


for more information visit: https://github.com/facebookresearch/visdom

"""

import visdom
import numpy as np
import collections

vis = visdom.Visdom()

# Uncomment this line to clear the workspace
vis.close(win=None)


static_window = vis.line(
        X=np.column_stack(([0],[0])),
        Y=np.column_stack(([0],[0])),
        opts=dict(
            showlegend=False,
            width=500,
            height=500,
            xlabel='Time',
            ylabel='Reading',
            title='Servo 1',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            legend=['random', 'prediction'],
        ),
    )


# Setup a queue of size 20 for both the actual results and predicted results
# deque will keep keep the queue to size 20 and we use this to update the graph
actual_results_window = collections.deque(20*[0], 20)
predicted_results_window = collections.deque(20*[0], 20)
x_window = collections.deque([x for x in range(20)], 20)

moving_window = vis.line(
        X=np.column_stack(([0],[0])),
        Y=np.column_stack(([0],[0])),
        opts=dict(
            showlegend=False,
            width=500,
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

actual_value = 0
predicte_value = 0

for i in range(100):
    # Use random number as actual value
    # Use moving average as prediction
    actual_value = np.random.randn()
    predicte_value += 0.1 * (actual_value - predicte_value)

    actual_results_window.append(actual_value)
    predicted_results_window.append(predicte_value)
    x_window.append(i)


    # Update line graph
    vis.line(
        X=np.column_stack((i, i)),
        Y=np.column_stack(([actual_value], [predicte_value])),
        win=static_window,
        update='append' # this appends the new data to the existing data
        )

    # Update windowed graph
    vis.line(
        X=np.column_stack((x_window, x_window)),
        Y=np.column_stack((actual_results_window, predicted_results_window)),
        win=moving_window,
        # We don't append since we want to graph our moving window
        )