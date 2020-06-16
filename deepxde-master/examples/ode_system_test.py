from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde


def main():
    def ode_system(x, y):
        """ODE system.
        dy1/dx = y2
        dy2/dx = -y1
        """
        y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
        dy1_x = tf.gradients(y1, x)[0]
        dy2_x = tf.gradients(y2, x)[0]
        dy3_x = tf.gradients(y3, x)[0]
        return [dy1_x - y2, dy2_x + y1, dy3_x + y2]

    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    def func(x):
        """
        y1 = sin(x)
        y2 = cos(x)
        """
        return np.hstack((np.sin(x), np.cos(x), -np.sin(x)))

    geom = dde.geometry.Cuboid([0,0,0], [10,10,10])
    bc1 = dde.DirichletBC(geom, lambda x: np.sin(x[:, 0:1]), boundary, component=0)
    bc2 = dde.DirichletBC(geom, lambda x: np.cos(x[:, 0:1]), boundary, component=1)
    bc3 = dde.DirichletBC(geom, lambda x: np.sin(x[:, 0:1]), boundary, component=2)
    data = dde.data.PDE(geom, 3, ode_system, [bc1, bc2, bc3], 350, 64, num_test=100)

    layer_size = [3] + [50] * 3 + [3]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
