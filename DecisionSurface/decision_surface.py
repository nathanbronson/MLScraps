import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

def decision_surface(clf, x_interval=(-1, 1), y_interval=(-1, 1), resolution=100, pred_fun="predict_proba", proba_map=None, legend=None, title="Decision Surface"):
    if proba_map is None:
        proba_map = lambda x: x.reshape((resolution, resolution, 3))[:, ::-1]
    grid_df = pd.DataFrame({"x": [x_interval[0] + (x_interval[1] - x_interval[0]) * i/resolution for i in range(resolution)] * resolution} | {"y": np.repeat([y_interval[0] + (y_interval[1] - y_interval[0]) * i/resolution for i in range(resolution)], resolution)})
    probs = getattr(clf, pred_fun)(np.concatenate([np.expand_dims(grid_df["x"].values, 1), np.expand_dims(grid_df["y"].values, 1)], axis=-1))
    probs = proba_map(probs)
    im = plt.imshow(probs, extent=(x_interval[0], x_interval[1], y_interval[0], y_interval[1]))
    if legend is not None:
        le = [Line2D([0], [0], marker='o', color=c, label=n) for c, n in legend]
        plt.legend(handles=le, loc="best")
    plt.title(title)
    plt.show()