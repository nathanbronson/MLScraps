import numpy as np
import tensorflow as tf

from gd import MaclaurinRegressionGD
from visualize import plot_reg

TEST = ["sin", "sin2", "sinterm", "add"][2]

if __name__ == "__main__":
    if TEST == "sin":
        reg = MaclaurinRegressionGD(20)
        x = (tf.range(40000, dtype=tf.float32) - 20000)/5000
        x = tf.random.shuffle(tf.repeat(x, 500))
        y = tf.sin(x)
        x = tf.expand_dims(x, -1)
        reg.compile(optimizer="adam", loss="mse")
        reg.fit(x, y, batch_size=4096, epochs=6)
    elif TEST == "sin2":
        reg = MaclaurinRegressionGD(20)
        x = (tf.range(40000, dtype=tf.float32) - 20000)/5000
        x = tf.random.shuffle(tf.repeat(x, 500)) * 2
        y = tf.sin(x)
        x = tf.expand_dims(x, -1)
        reg.compile(optimizer="adam", loss="mse")
        reg.fit(x, y, batch_size=4096, epochs=20)
    elif TEST == "sinterm":
        for i in range(1, 20, 2):
            reg = MaclaurinRegressionGD(i)
            x = (tf.range(40000, dtype=tf.float32) - 20000)/5000
            x = tf.random.shuffle(tf.repeat(x, 500)) * 2
            y = tf.sin(x)
            x = tf.expand_dims(x, -1)
            reg.compile(optimizer="adam", loss="mse")
            reg.fit(x, y, batch_size=4096, epochs=6)
            plot_reg(reg, (np.array(range(200)) - 100)/15)
            plot_reg(reg, (np.array(range(200)) - 100)/30)
        exit()
    elif TEST == "add":
        reg = MaclaurinRegressionGD(5)
        x = (tf.range(40000, dtype=tf.float32) - 20000)/5000
        print(tf.shape(x))
        x = tf.transpose(tf.concat([tf.expand_dims(t, 0) for t in tf.meshgrid(x, x)], axis=0), perm=(1, 2, 0))
        print(tf.shape(x))
        x = tf.cast(tf.random.shuffle(x), tf.float32)
        y = tf.reduce_sum(x, axis=-1)
        print(tf.shape(y))
        reg.compile(optimizer="adam", loss="mse")
        reg.fit(x, y, batch_size=512, epochs=20)
    plot_reg(reg, (np.array(range(200)) - 100)/15)
    plot_reg(reg, (np.array(range(200)) - 100)/30)