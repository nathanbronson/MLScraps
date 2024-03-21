import numpy as np
import pandas as pd

from maclaurinregression import MaclaurinRegression
from visualize import plot_reg

TEST = ["sin", "sin2", "sinterm", "add"][-1]

if __name__ == "__main__":
    if TEST == "sin":
        reg = MaclaurinRegression(20)
        x = (np.arange(40000, dtype=np.float32) - 20000)/5000
        y = np.sin(x)
        x = np.expand_dims(x, -1)
        reg = reg.fit(x, y)
    elif TEST == "sin2":
        reg = MaclaurinRegression(20)
        x = 2 * (np.arange(40000, dtype=np.float32) - 20000)/5000
        y = np.sin(x)
        x = np.expand_dims(x, -1)
        reg = reg.fit(x, y)
    elif TEST == "sinterm":
        for i in range(1, 20, 2):
            reg = MaclaurinRegression(i)
            x = 2 * (np.arange(40000, dtype=np.float32) - 20000)/5000
            y = np.sin(x)
            x = np.expand_dims(x, -1)
            reg = reg.fit(x, y)
            plot_reg(reg.predict, (np.array(range(200)) - 100)/15)
            plot_reg(reg.predict, (np.array(range(200)) - 100)/30)
        exit()
    elif TEST == "add":
        reg = MaclaurinRegression(5)
        x = (np.arange(4000, dtype=np.float32) - 2000)/500
        x = np.transpose(np.concatenate([np.expand_dims(t, 0) for t in np.meshgrid(x, x)], axis=0), (1, 2, 0))
        x = np.reshape(x, (-1, 2))
        x = x.astype(np.float32)
        y = np.sum(x, axis=-1)
        reg = reg.fit(x, y)
    plot_reg(reg.predict, pd.DataFrame({str(n): x[:, n] for n in range(x.shape[-1])}))
    plot_reg(reg.predict, pd.DataFrame({str(n): x[:, n] for n in range(x.shape[-1])}))