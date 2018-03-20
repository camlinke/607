import visdom
import numpy as np
import collections

vis = visdom.Visdom()

class VisdomText:
    def __init__(self):
        pass

    def update(self, text):
        pass

class VisdomLine:

    def __init__(self, window_size):
        self.window_size = window_size
        # self.window = collections.deque(self.window_size*[0], self.window_size)
        self.window = [0] * 2
        self.count = 0

    def update(self, x):
        if self.count >= self.window_size:
            if isinstance(self.window, collections.deque):
                self.window.append(x)
            else:
                self.window = collections.deque(self.window, self.window_size)
        else:
            if self.count < 2:
                self.window[self.count] = x
            else:
                self.window.append(x)
        self.count += 1
        return self.window

class VisdomWindowedLine:

    def __init__(self, num_lines, window_size=100, line_data={}, vis=vis):

        self.num_lines = num_lines
        self.window_size = window_size

        self.windows = [VisdomLine(self.window_size) for _ in range(num_lines)]
        self.x_window = VisdomLine(self.window_size)#collections.deque([x for x in range(window_size)], window_size)

        self.line_data = line_data

        self.vis = vis

        # X = np.column_stack([list(self.x_window.window) for _ in range(self.num_lines)])
        # Y = np.column_stack([list(window.window) for window in self.windows])
        # print(X)
        # print(Y.shape)
        # print(self.windows)
        # print(self.x_window)
        # print(self.num_lines)
        self.win = self.vis.line(
            X=np.column_stack([list(self.x_window.window) for _ in range(self.num_lines)]),
            Y=np.column_stack([list(window.window) for window in self.windows]),
            opts=dict(
                showlegend=bool(self.line_data.get('showlegend', False)),
                width=int(self.line_data.get('width', 500)),
                height=int(self.line_data.get('height', 500)),
                marginleft=int(self.line_data.get('marginleft', 30)),
                marginright=int(self.line_data.get('marginright', 30)),
                marginbottom=int(self.line_data.get('marginbottom', 30)),
                margintop=int(self.line_data.get('margintop', 30)),
                xlabel=str(self.line_data.get("xlabel", "")),
                ylabel=str(self.line_data.get("ylabel", "")),
                title=str(self.line_data.get("title", "")),
                legend=list(self.line_data.get("legend", ["" for _ in range(self.num_lines)]))
            ),
        )

        self.counter = 0

    def update(self, data):
        self.counter += 1
        # self.x_window.append(self.counter)
        self.x_window.update(self.counter)
        # X = np.column_stack([self.x_window for _ in range(self.num_lines)])
        # Y = np.column_stack([window.update(data[i]) for i, window in enumerate(self.windows)])

        # print(X)
        # print(Y)
        
        self.vis.line (
            X=np.column_stack([list(self.x_window.window) for _ in range(self.num_lines)]),
            Y=np.column_stack([window.update(data[i]) for i, window in enumerate(self.windows)]),
            win=self.win,
            opts=dict(
                    showlegend=bool(self.line_data.get('showlegend', False)),
                    xlabel=str(self.line_data.get("xlabel", "")),
                    ylabel=str(self.line_data.get("ylabel", "")),
                    title=str(self.line_data.get("title", "")),
                    legend=list(self.line_data.get("legend", ["" for _ in range(self.num_lines)]))
                ),
            )