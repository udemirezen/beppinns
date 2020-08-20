from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde



def mhd(x, y):
    y = y[:, 0:]
    t1 = y
    x1 = y
    dt1 = tf.gradients(t1, x)[0][:,1:2]
    dx1 = tf.gradients(x1, x)[0][:,0:1]
    ddx1 = tf.gradients(dx1, x)[0][:,0:1]
    
    return [dt1 + y*dx1 - 0.01/np.pi * ddx1
            ]

def boundary_space_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],-1) and not np.isclose(x[1], 0)

def boundary_space_right(x, on_boundary):
    return on_boundary and np.isclose(x[0],1) and not np.isclose(x[1], 0)

def boundary_time_left(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0) and not (np.isclose(x[0], -1) or np.isclose(x[0], 1)) and x[0] <= 0

def boundary_time_right(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0) and not (np.isclose(x[0], -1) or np.isclose(x[0], 1)) and x[0] > 0

geom = dde.geometry.Rectangle([-1,0], [1,0.99])
bc1l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_left, component=0)

bc1r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_right, component=0)

ic1l = dde.DirichletBC(geom, lambda x: -np.sin(np.pi * x[:, 0:1]), boundary_time_left, component=0)

ic1r = dde.DirichletBC(geom, lambda x: -np.sin(np.pi * x[:, 0:1]), boundary_time_right, component=0)


bic = [bc1l, bc1r, ic1l, ic1r]

data = dde.data.PDE(geom, 1, mhd, bic, 1000, 600)

layer_size = [2] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
model.train(epochs=20000)
model.compile("L-BFGS-B")
losshistory, train_state = model.train(epochs=5000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()
import matplotlib.pyplot as plt
fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
from matplotlib import cm
surf = ax.plot_trisurf(X_test[:,0], X_test[:,1], best_y[:,0], cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('z')
plt.show()