import statsmodels.api as sm
import pandas as pd
import numpy as np

fac = np.vectorize(np.math.factorial)

class MaclaurinTransformation(object):
    def __init__(self, degree):
        self.built = False
        self.degree = degree
    
    def build(self, input_shape):
        _d = self.degree + 1
        self.terms = np.transpose(np.reshape(np.concatenate([np.expand_dims(t, 0) for t in np.meshgrid(*np.split(np.repeat(np.expand_dims(np.arange(_d), 0), input_shape[-1], axis=0), input_shape[-1]))], axis=0), (input_shape[-1], -1)), (1, 0))
        self.terms = self.terms[np.sum(self.terms, axis=-1) <= _d - 1]
        self.facs = np.power(fac(np.prod(self.terms, axis=-1)).astype(np.float32), -1)
        self.terms = self.terms.astype(np.float32)
        self.built = True
    
    def transform(self, x):
        if not self.built:
            self.build(np.shape(x))
        x = np.repeat(np.expand_dims(x, -2), self.terms.shape[-2], axis=-2)
        return np.multiply(self.facs, np.prod(np.power(x, self.terms), axis=-1))

class MaclaurinRegression(object):
    def __init__(self, degree):
        self.degree = degree
        self.transform = MaclaurinTransformation(self.degree)
    
    def fit(self, x, y, print_sum=True):
        if x.shape[-1] <= 7:
            varnames = "xyzabcd"[:x.shape[-1]]
        else:
            varnames = ["x" + str(n) for n in range(x.shape[-1])]
        self.transform.build(x.shape)
        x = self.transform.transform(x)
        reg = sm.OLS(y, pd.DataFrame({"".join([v + "^" + str(int(t)) for v, t in zip(varnames, term)]): x[:, n] for term, n in zip(self.transform.terms.astype(np.int32).tolist(), range(x.shape[-1]))})).fit()
        if print_sum:
            print(reg.summary())
        return reg