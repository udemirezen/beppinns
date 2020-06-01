from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DirichletIC(object):
    """Initial conditions: y([x, t0]) = func([x, t0]).
    """

    def __init__(self, geom, func, on_initial, component=0):
        self.geom = geom
        self.func = func
        self.on_initial = on_initial
        self.component = component

    def filter(self, X):
        X = np.array([x for x in X if self.on_initial(x, self.geom.on_initial(x))])
        return X if len(X) > 0 else np.empty((0, self.geom.dim))

    def collocation_points(self, X):
        return self.filter(X)

    def error(self, X, inputs, outputs, beg, end):
        return outputs[beg:end, self.component : self.component + 1] - self.func(
            X[beg:end]
        )

class NeumannIC(object):
    """Initial conditions: dy/dt([x, t0]) = func([x, t0]).
    """

    def __init__(self, geom, func, on_initial, component=0):
        self.geom = geom
        self.func = func
        self.on_initial = on_initial
        self.component = component
        
    def normal_derivative(self, X, inputs, outputs, beg, end):
        outputs = outputs[:, self.component : self.component + 1]
        dydt = tf.gradients(outputs, inputs)[0][:,1:2]           #<== tried a lot of combinations, nothing seems to work
#        n = np.array(list(map(self.geom.boundary_normal, X[beg:end])))
#        print(n)
#        return tf.reduce_sum(dydt * n, axis=1, keepdims=True)
        return dydt

    def filter(self, X):
        X = np.array([x for x in X if self.on_initial(x, self.geom.on_initial(x))])
        return X if len(X) > 0 else np.empty((0, self.geom.dim))

    def collocation_points(self, X):
        return self.filter(X)

    def error(self, X, inputs, outputs, beg, end):
        return 100*(self.normal_derivative(X, inputs, outputs, beg, end)[beg:end, self.component : self.component + 1] - self.func(
            X[beg:end]))
