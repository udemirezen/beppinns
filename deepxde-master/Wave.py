from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

#import deepxde1 as dde

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


def gen_testdata():
    data = np.load("dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y



def pde(x, y):
    dy_x = tf.gradients(y, x)[0]
    dy_x, dy_t = dy_x[:, 0:1], dy_x[:, 1:2]
    dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
    dy_tt = tf.gradients(dy_t, x)[0][:, 1:2]
    return dy_tt - dy_xx

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 2.49)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(
    geomtime, lambda x: np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary
)
ic1 = dde.DirichletIC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)
ic2=dde.NeumannIC(geomtime, lambda x: np.zeros((len(x),1)), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, 1, pde, [bc, ic1, ic2], num_domain=1000, num_boundary=160, num_initial=160
)
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=20000) #was 15000 epochs
model.compile("L-BFGS-B")
losshistory, train_state = model.train()
saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))


