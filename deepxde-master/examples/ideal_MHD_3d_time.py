from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde


def main():
    def mhd(x, y):
        rho, p, vx, vy, vz, Bx, By, Bz = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6], y[:, 6:7], y[:, 7:]
        drho = tf.gradients(rho, x)[0]
        dp = tf.gradients(p, x)[0]
        dvx = tf.gradients(vx, x)[0]
        dvy = tf.gradients(vy, x)[0]
        dvz = tf.gradients(vz, x)[0]
        dBx = tf.gradients(Bx, x)[0]
        dBy = tf.gradients(By, x)[0]
        dBz = tf.gradients(Bz, x)[0]
        drho_dt = drho[3:4]
        nabdotrhov = tf.gradients(rho*vx, x)[0][:,0:1] + tf.gradients(rho*vy, x)[0][1:2] + tf.gradients(rho*vz, x)[0][2:3]
        dvx_dt, dvy_dt, dvz_dt = dvx[3:4], dvy[3:4], dvz[3:4]
        vdotnabvx = vx*dvx[0:1] + vy*dvx[1:2] + vz*dvx[2:3]
        vdotnabvy = vx*dvy[0:1] + vy*dvy[1:2] + vz*dvy[2:3]
        vdotnabvz = vx*dvz[0:1] + vy*dvz[1:2] + vz*dvz[2:3]
        dp_dx = dp[0:1]
        dp_dy = dp[1:2]
        dp_dz = dp[2:3]
        dp_dt = dp[3:4]
        gx = 9.81
        gy = 0
        gz = 0
        mu0 = 4*np.pi*10**(-7)
        gamma = 5/3
        nabcrossBx = dBz[1:2] - dBy[2:3]
        nabcrossBy = dBx[2:3] - dBz[0:1]
        nabcrossBz = dBy[0:1] - dBx[1:2]
        nabcrossBcrossBx = nabcrossBy*Bz - nabcrossBz*By
        nabcrossBcrossBy = nabcrossBz*Bx - nabcrossBx*Bz
        nabcrossBcrossBz = nabcrossBx*By - nabcrossBy*Bx
        vdotnabp = vx*dp_dx + vy*dp_dy + vz*dp_dz
        nabdotv = dvx[0:1] + dvy[1:2] + dvz[2:3]
        dBx_dt = dBx[3:4]
        dBy_dt = dBy[3:4]
        dBz_dt = dBz[3:4]
        vcrossBx = vy*Bz - vz*By
        vcrossBy = vz*Bx - vx*Bz
        vcrossBz = vx*By - vy*Bx
        dvcrossBx = tf.gradients(vcrossBx, x)[0]
        dvcrossBy = tf.gradients(vcrossBy, x)[0]
        dvcrossBz = tf.gradients(vcrossBz, x)[0]
        nabcrossvcrossBx = dvcrossBz[1:2] - dvcrossBy[2:3]
        nabcrossvcrossBy = dvcrossBx[2:3] - dvcrossBz[0:1]
        nabcrossvcrossBz = dvcrossBy[0:1] - dvcrossBx[1:2]
        
        
        return [drho_dt + nabdotrhov,
                rho*(dvx_dt+vdotnabvx)+dp_dx-rho*gx-1/mu0*nabcrossBcrossBx,
                rho*(dvy_dt+vdotnabvy)+dp_dy-rho*gy-1/mu0*nabcrossBcrossBy,
                rho*(dvz_dt+vdotnabvz)+dp_dz-rho*gz-1/mu0*nabcrossBcrossBz,
                dp_dt+vdotnabp+gamma*p*nabdotv,
                dBx_dt-nabcrossvcrossBx,
                dBy_dt-nabcrossvcrossBy,
                dBz_dt-nabcrossvcrossBz
                ]

    def func(x):
        """
        y1 = sin(x)
        y2 = cos(x)
        """
        return np.hstack((np.sin(x), np.cos(x), -np.sin(x)))
    
    def boundary(x, on_boundary):
        return on_boundary
    
    def initial(x, on_initial):
        return on_initial

    geom = dde.geometry.Cuboid([0,0,0], [10,10,10])
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    bc1 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=0)
    bc2 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=1)
    bc3 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=2)
    bc4 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=3)
    bc5 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=4)
    bc6 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=5)
    bc7 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=6)
    bc8 = dde.DirichletBC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=7)
    
    ic1 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=0)
    ic2 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=1)
    ic3 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=2)
    ic4 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=3)
    ic5 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=4)
    ic6 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=5)
    ic7 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=6)
    ic8 = dde.IC(geomtime, lambda x: np.zeros((len(x),1)), boundary, component=7)



    data = dde.data.TimePDE(geomtime, 8, mhd, [bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8, ic1, ic2, ic3, ic4, ic5, ic6, ic7, ic8])#, num_domain=160, num_boundary=160, num_initial=160)

    layer_size = [4] + [50] * 3 + [8]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
