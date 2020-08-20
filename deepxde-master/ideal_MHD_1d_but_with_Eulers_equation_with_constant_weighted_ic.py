from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde1 as dde



def mhd(x, y):
    rho, v, E = y[:, 0:1], y[:, 1:2], y[:, 2:]
    p = 0.4*(rho*E - 0.5*rho*v*v)
    t1 = rho
    t2 = rho*v
    t3 = rho*E
    x1 = rho*v
    x2 = rho*v*v+p
    x3 = v*(rho*E+p)
    dt1 = tf.gradients(t1, x)[0][:,1:2]
    dt2 = tf.gradients(t2, x)[0][:,1:2]
    dt3 = tf.gradients(t3, x)[0][:,1:2]
    dx1 = tf.gradients(x1, x)[0][:,0:1]
    dx2 = tf.gradients(x2, x)[0][:,0:1]
    dx3 = tf.gradients(x3, x)[0][:,0:1]
    return [dt1 + dx1,
            dt2 + dx2,
            dt3 + dx3
            ]

def boundary_space_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],0) and not np.isclose(x[1], 0)

def boundary_space_right(x, on_boundary):
    return on_boundary and np.isclose(x[0],1) and not np.isclose(x[1], 0)

def boundary_time_left(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0) and not (np.isclose(x[0], 0) or np.isclose(x[0], 1)) and x[0] <= 0.5

def boundary_time_right(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0) and not (np.isclose(x[0], 0) or np.isclose(x[0], 1)) and x[0] > 0.5

geom = dde.geometry.Rectangle([0,0], [1,1.99])
bc1l = dde.DirichletBC(geom, lambda x: 1.4*np.ones((len(x),1)), boundary_space_left, component=0)
bc2l = dde.DirichletBC(geom, lambda x: 0.1*np.ones((len(x),1)), boundary_space_left, component=1)
bc3l = dde.DirichletBC(geom, lambda x: 1.79071429*np.ones((len(x),1)), boundary_space_left, component=2)

bc1r = dde.DirichletBC(geom, lambda x: 1.0*np.ones((len(x),1)), boundary_space_right, component=0)
bc2r = dde.DirichletBC(geom, lambda x: 0.1*np.ones((len(x),1)), boundary_space_right, component=1)
bc3r = dde.DirichletBC(geom, lambda x: 2.505*np.ones((len(x),1)), boundary_space_right, component=2)

ic1l = dde.ConstantWeightDirichletBC(geom, lambda x: 1.4*np.ones((len(x),1)), boundary_time_left, component=0)
ic2l = dde.ConstantWeightDirichletBC(geom, lambda x: 0.1*np.ones((len(x),1)), boundary_time_left, component=1)
ic3l = dde.ConstantWeightDirichletBC(geom, lambda x: 1.79071429*np.ones((len(x),1)), boundary_time_left, component=2)

ic1r = dde.ConstantWeightDirichletBC(geom, lambda x: 1.0*np.ones((len(x),1)), boundary_time_right, component=0)
ic2r = dde.ConstantWeightDirichletBC(geom, lambda x: 0.1*np.ones((len(x),1)), boundary_time_right, component=1)
ic3r = dde.ConstantWeightDirichletBC(geom, lambda x: 2.505*np.ones((len(x),1)), boundary_time_right, component=2)


bic = [bc1l, bc2l, bc3l, bc1r, bc2r, bc3r, ic1l, ic2l, ic3l, ic1r, ic2r, ic3r]

data = dde.data.PDE(geom, 3, mhd, bic, 1000, 600)

layer_size = [2] + [50] * 3 + [3]
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