import matplotlib.pyplot as plt

def plot_reg(f, x):
    plt.plot(x, f(x.reshape((-1, 1))).numpy().reshape(-1))
    plt.show()