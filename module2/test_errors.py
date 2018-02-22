import numpy as np
import os
from pprint import pprint


results = []
for file in os.listdir("errors"):
    if file[-4:] == ".npy" and file[:3] == "off": #and file[:2] == "on" and "load" not in file: #and "load" in file: #file[:2] == "on": #file[:3] == "off":
        x = np.load("errors/{}".format(file))
        m = np.average(x)
        results.append((file, m))
#         if m < score:
#             score = m
#             best = file
# print best
# print score

results.sort(key=lambda tup: -tup[1])
pprint(results)
pprint(len(results))