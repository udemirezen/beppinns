from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

import deepxde as dde

def saveplot(losshistory, train_state, issave=True, isplot=True):
    if isplot:
        plot_loss_history(losshistory)
        plot_best_state(train_state)
        plt.show()

    if issave:
        save_loss_history(losshistory, "loss.dat")
        save_best_state(train_state, "train.dat", "test.dat")


def plot_loss_history(losshistory):
    loss_train = np.sum(
        np.array(losshistory.loss_train) * losshistory.loss_weights, axis=1
    )
    loss_test = np.sum(
        np.array(losshistory.loss_test) * losshistory.loss_weights, axis=1
    )

    plt.figure()
    plt.semilogy(losshistory.steps, loss_train, label="Train loss")
    plt.semilogy(losshistory.steps, loss_test, label="Test loss")
    for i in range(len(losshistory.metrics_test[0])):
        plt.semilogy(
            losshistory.steps,
            np.array(losshistory.metrics_test)[:, i],
            label="Test metric",
        )
    plt.xlabel("# Steps")
    plt.legend()


def save_loss_history(losshistory, fname):
    print("Saving loss history to {} ...".format(fname))
    loss = np.hstack(
        (
            np.array(losshistory.steps)[:, None],
            np.array(losshistory.loss_train),
            np.array(losshistory.loss_test),
            np.array(losshistory.metrics_test),
        )
    )
    np.savetxt(fname, loss, header="step, loss_train, loss_test, metrics_test")


def plot_best_state(train_state):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

    y_dim = y_train.shape[1]

    # Regression plot
    plt.figure()
    X = X_test[:, 0]
    T = X_test[:,1]
        
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(X, T, best_y)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(X, T, best_y[:,0], cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Residual plot
    plt.figure()
    residual = y_test[:, 0] - best_y[:, 0]
    plt.plot(best_y[:, 0], residual, zorder=1) #<===
    plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual = Observed - Predicted")
    plt.tight_layout()

    if best_ystd is not None:
        plt.figure()
        for i in range(y_dim):
            plt.plot(X_test[:, 0], best_ystd[:, i])  #<===
            plt.plot(
                X_train[:, 0],
                np.interp(X_train[:, 0], X_test[:, 0], best_ystd[:, i]),
                "ok",
            )
        plt.xlabel("x")
        plt.ylabel("std(y)")


def save_best_state(train_state, fname_train, fname_test):
    print("Saving training data to {} ...".format(fname_train))
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()
    train = np.hstack((X_train, y_train))
    np.savetxt(fname_train, train, header="x, y")

    print("Saving test data to {} ...".format(fname_test))
    test = np.hstack((X_test, y_test, best_y))
    if best_ystd is not None:
        test = np.hstack((test, best_ystd))
    np.savetxt(fname_test, test, header="x, y_true, y_pred, y_std")





def pde(x, y):
    y1, y2 = y[:,0:1],y[:,1:]
    dy1_x = tf.gradients(y1, x)[0]
    dy1_x, dy1_t = dy1_x[:, 0:1], dy1_x[:, 1:2]
    dy1_xx = tf.gradients(dy1_x, x)[0][:, 0:1]
    dy1_tt = tf.gradients(dy1_t, x)[0][:, 1:2]
    dy2_x = tf.gradients(y2, x)[0]
    dy2_x, dy2_t = dy2_x[:, 0:1], dy2_x[:, 1:2]
    dy2_xx = tf.gradients(dy2_x, x)[0][:, 0:1]
    dy2_tt = tf.gradients(dy2_t, x)[0][:, 1:2]

    return [dy1_tt - dy1_xx, dy2_tt - dy2_xx]

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 7.49)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc1 = dde.DirichletBC(
    geomtime, lambda x: np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0
)
bc2 = dde.DirichletBC(
    geomtime, lambda x: np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=1
)
ic1 = dde.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial, component=0
)
ic2 = dde.IC(
    geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial, component=1
)

data = dde.data.TimePDE(
    geomtime, 2, pde, [bc1, bc2, ic1, ic2], num_domain=5080, num_boundary=320, num_initial=160
)
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=5000) #was 15000 epochs
model.compile("L-BFGS-B")
losshistory, train_state = model.train()
saveplot(losshistory, train_state, issave=True, isplot=True)
