import tensorflow as tf
import statsmodels.api as sm
import pandas as pd

class MaclaurinRegressionGD(tf.keras.Model):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
    
    def build(self, input_shape):
        _d = self.degree + 1
        self.terms = tf.transpose(tf.reshape(tf.concat([tf.expand_dims(t, 0) for t in tf.meshgrid(*tf.split(tf.repeat(tf.expand_dims(tf.range(_d), 0), input_shape[-1], axis=0), num_or_size_splits=input_shape[-1]))], axis=0), (input_shape[-1], -1)), perm=(1, 0))
        self.terms = tf.cast(self.terms[tf.reduce_sum(self.terms, axis=-1) <= _d - 1], tf.float32)
        self.coefs = self.add_weight("coefs", (1, self.terms.shape[0]), dtype=tf.float32)
        self.facs = tf.pow(tf.cast(tf.map_fn(lambda x: tf.math.reduce_prod(tf.range(1,x+1)), tf.math.reduce_prod(self.terms, axis=-1)), tf.float32), -1)
        #self.facs = tf.pow(tf.cast(tf.stack([tf.math.reduce_prod(tf.range(1,x+1)) for x in range(_d)], axis=0), dtype=tf.float32), -1)
        #self.facs = tf.repeat(self.facs, repeats=tf.unique_with_counts(tf.reduce_sum(self.terms, axis=-1))[2])
        return super().build(input_shape)
    
    def call(self, x):
        x = tf.repeat(tf.expand_dims(x, -2), self.terms.shape[-2], axis=-2)
        return tf.reduce_sum(tf.math.multiply(self.facs, tf.math.multiply(self.coefs, tf.reduce_prod(tf.math.pow(x, self.terms), axis=-1))), axis=-1)
    
    def input_transform(self, x):
        x = tf.repeat(tf.expand_dims(x, -2), self.terms.shape[-2], axis=-2)
        return tf.math.multiply(self.facs, tf.reduce_prod(tf.math.pow(x, self.terms), axis=-1))

class MaclaurinRegression(object):
    def __init__(self, degree):
        self.degree = degree
    
    def fit(self, x, y, print_sum=True):
        if x.shape[-1] <= 7:
            varnames = "xyzabcd"[:x.shape[-1]]
        else:
            varnames = ["x" + str(n) for n in range(x.shape[-1])]
        gd = MaclaurinRegressionGD(self.degree)
        gd.build(x.shape)
        x = gd.input_transform(x).numpy()
        reg = sm.OLS(y, pd.DataFrame({"".join([v + "^" + str(int(t)) for v, t in zip(varnames, term)]): x[:, n] for term, n in zip(gd.terms.numpy().tolist(), range(x.shape[-1]))})).fit()
        if print_sum:
            print(reg.summary())
        return reg

if __name__ == "__main__":
    reg = MaclaurinRegressionGD(20)
    x = (tf.range(40000, dtype=tf.float32) - 20000)/5000
    x = tf.random.shuffle(tf.repeat(x, 500))
    y = tf.sin(x)
    x = tf.expand_dims(x, -1)
    reg.compile(optimizer="adam", loss="mse")
    reg.fit(x, y, batch_size=512, epochs=20)