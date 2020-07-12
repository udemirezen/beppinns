from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde



def mhd(x, y):
    rho, vx, vy, vz, By, Bz, p = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6], y[:, 6:]
    Bx = 0.75
    E = p+0.5*rho*(vx*vx+vy*vy+vz*vz)+0.5*(Bx*Bx+By*By+Bz*Bz)
    pstar = p + 0.5*(Bx*Bx+By*By+Bz*Bz)
    t1 = rho
    t2 = rho*vx
    t3 = rho*vy
    t4 = rho*vz
    t5 = By
    t6 = Bz
    t7 = E
    x1 = rho*vx
    x2 = rho*vx*vx + pstar - Bx*Bx
    x3 = rho*vx*vy - Bx*By
    x4 = rho*vx*vz - Bx*Bz
    x5 = By*vx - Bx*vy
    x6 = Bz*vx - Bx*vz
    x7 = (E+pstar)*vx - Bx*(Bx*vx+By*vy+Bz*vz)
    dt1 = tf.gradients(t1, x)[0][1:2]
    dt2 = tf.gradients(t2, x)[0][1:2]
    dt3 = tf.gradients(t3, x)[0][1:2]
    dt4 = tf.gradients(t4, x)[0][1:2]
    dt5 = tf.gradients(t5, x)[0][1:2]
    dt6 = tf.gradients(t6, x)[0][1:2]
    dt7 = tf.gradients(t7, x)[0][1:2]
    dx1 = tf.gradients(x1, x)[0][0:1]
    dx2 = tf.gradients(x2, x)[0][0:1]
    dx3 = tf.gradients(x3, x)[0][0:1]
    dx4 = tf.gradients(x4, x)[0][0:1]
    dx5 = tf.gradients(x5, x)[0][0:1]
    dx6 = tf.gradients(x6, x)[0][0:1]
    dx7 = tf.gradients(x7, x)[0][0:1]
    
    return [dt1 + dx1,
            dt2 + dx2,
            dt3 + dx3,
            dt4 + dx4,
            dt5 + dx5,
            dt6 + dx6,
            dt7 + dx7
            ]

def func(x):
    """
    y1 = sin(x)
    y2 = cos(x)
    """
    return np.hstack((np.sin(x), np.cos(x), -np.sin(x)))

def boundary_space_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],-1) and not np.isclose(x[1], 0)

def boundary_space_right(x, on_boundary):
    return on_boundary and np.isclose(x[0],1) and not np.isclose(x[1], 0)

def boundary_time_left(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0) and not (np.isclose(x[0], -1) or np.isclose(x[0], 1)) and x[0] <= 0

def boundary_time_right(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0) and not (np.isclose(x[0], -1) or np.isclose(x[0], 1)) and x[0] > 0

geom = dde.geometry.Rectangle([-1,0], [1,0.2])
bc1l = dde.DirichletBC(geom, lambda x: np.ones((len(x),1)), boundary_space_left, component=0)
bc2l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_left, component=1)
bc3l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_left, component=2)
bc4l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_left, component=3)
bc5l = dde.DirichletBC(geom, lambda x: np.ones((len(x),1)), boundary_space_left, component=4)
bc6l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_left, component=5)
bc7l = dde.DirichletBC(geom, lambda x: np.ones((len(x),1)), boundary_space_left, component=6)
bc1r = dde.DirichletBC(geom, lambda x: 0.125*np.ones((len(x),1)), boundary_space_right, component=0)
bc2r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_right, component=1)
bc3r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_right, component=2)
bc4r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_right, component=3)
bc5r = dde.DirichletBC(geom, lambda x: -1*np.ones((len(x),1)), boundary_space_right, component=4)
bc6r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_space_right, component=5)
bc7r = dde.DirichletBC(geom, lambda x: 0.1*np.ones((len(x),1)), boundary_space_right, component=6)
ic1l = dde.DirichletBC(geom, lambda x: np.ones((len(x),1)), boundary_time_left, component=0)
ic2l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_left, component=1)
ic3l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_left, component=2)
ic4l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_left, component=3)
ic5l = dde.DirichletBC(geom, lambda x: np.ones((len(x),1)), boundary_time_left, component=4)
ic6l = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_left, component=5)
ic7l = dde.DirichletBC(geom, lambda x: np.ones((len(x),1)), boundary_time_left, component=6)
ic1r = dde.DirichletBC(geom, lambda x: 0.125*np.ones((len(x),1)), boundary_time_right, component=0)
ic2r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_right, component=1)
ic3r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_right, component=2)
ic4r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_right, component=3)
ic5r = dde.DirichletBC(geom, lambda x: -1*np.ones((len(x),1)), boundary_time_right, component=4)
ic6r = dde.DirichletBC(geom, lambda x: np.zeros((len(x),1)), boundary_time_right, component=5)
ic7r = dde.DirichletBC(geom, lambda x: 0.1*np.ones((len(x),1)), boundary_time_right, component=6)

bic = [bc1l, bc2l, bc3l, bc4l, bc5l, bc6l, bc7l, bc1r, bc2r, bc3r, bc4r, bc5r, bc6r, bc7r, ic1l, ic2l, ic3l, ic4l, ic5l, ic6l, ic7l, ic1r, ic2r, ic3r, ic4r, ic5r, ic6r, ic7r]

data = dde.data.PDE(geom, 7, mhd, bic, 5080, 640)

layer_size = [2] + [50] * 3 + [7]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=12000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()
import matplotlib.pyplot as plt
fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
from matplotlib import cm
surf = ax.plot_trisurf(X_test[:,0], X_test[:,1], best_y[:,0], cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()