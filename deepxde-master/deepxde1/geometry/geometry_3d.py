from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from SALib.sample import sobol_sequence

from .geometry import Geometry
from .geometry_2d import Rectangle
#from .geometry_nd import Hypersphere#, Hypercube


class Cuboid(Geometry):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")
        if len(xmin) != 3:
            raise ValueError("xmin and xmax are not 3D")
        if np.any(np.array(xmin) >= np.array(xmax)):
            raise ValueError("xmin >= xmax")

        self.xmin, self.xmax = np.array(xmin), np.array(xmax)
        super(Cuboid, self).__init__(
            len(xmin), (self.xmin, self.xmax), np.linalg.norm(self.xmax - self.xmin)
        )

        dx = self.xmax - self.xmin
        self.area = 2 * np.sum(dx * np.roll(dx, 2))
        
#from here
    def inside(self, x):
        return np.all(x >= self.xmin) and np.all(x <= self.xmax)

    def on_boundary(self, x):
        return self.inside(x) and (
            np.any(np.isclose(x, self.xmin)) or np.any(np.isclose(x, self.xmax))
        )

    def boundary_normal(self, x):
        n = np.zeros(self.dim)
        for i, xi in enumerate(x):
            if np.isclose(xi, self.xmin[i]):
                n[i] = -1
                break
            if np.isclose(xi, self.xmax[i]):
                n[i] = 1
                break
        return n

    def uniform_points(self, n, boundary=True):
        n1 = int(np.ceil(n ** (1 / self.dim)))
        xi = []
        for i in range(self.dim):
            if boundary:
                xi.append(np.linspace(self.xmin[i], self.xmax[i], num=n1))
            else:
                xi.append(
                    np.linspace(self.xmin[i], self.xmax[i], num=n1 + 1, endpoint=False)[
                        1:
                    ]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        if random == "pseudo":
            x = np.random.rand(n, self.dim)
        elif random == "sobol":
            x = sobol_sequence.sample(n + 1, self.dim)[1:]
        return (self.xmax - self.xmin) * x + self.xmin

    def periodic_point(self, x, component):
        y = np.copy(x)
        if np.isclose(y[component], self.xmin[component]):
            y[component] += self.xmax[component] - self.xmin[component]
        elif np.isclose(y[component], self.xmax[component]):
            y[component] -= self.xmax[component] - self.xmin[component]
        return y
#to here is a copy from geometry_nd.py

    def random_boundary_points(self, n, random="pseudo"):
        x_corner = np.vstack(
            (
                self.xmin,
                [self.xmin[0], self.xmax[1], self.xmin[2]],
                [self.xmax[0], self.xmax[1], self.xmin[2]],
                [self.xmax[0], self.xmin[1], self.xmin[2]],
                self.xmax,
                [self.xmin[0], self.xmax[1], self.xmax[2]],
                [self.xmin[0], self.xmin[1], self.xmax[2]],
                [self.xmax[0], self.xmin[1], self.xmax[2]],
            )
        )
        n -= 8
        if n <= 0:
            return x_corner

        pts = [x_corner]
        density = n / self.area
        rect = Rectangle(self.xmin[:-1], self.xmax[:-1])
        for z in [self.xmin[-1], self.xmax[-1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u, np.full((len(u), 1), z))))
        rect = Rectangle(self.xmin[::2], self.xmax[::2])
        for y in [self.xmin[1], self.xmax[1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), y), u[:, 1:])))
        rect = Rectangle(self.xmin[1:], self.xmax[1:])
        for x in [self.xmin[0], self.xmax[0]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((np.full((len(u), 1), x), u)))
        return np.vstack(pts)

    def uniform_boundary_points(self, n):
        h = (self.area / n) ** 0.5
        nx, ny, nz = np.ceil((self.xmax - self.xmin) / h).astype(int) + 1
        x = np.linspace(self.xmin[0], self.xmax[0], num=nx)
        y = np.linspace(self.xmin[1], self.xmax[1], num=ny)
        z = np.linspace(self.xmin[2], self.xmax[2], num=nz)

        pts = []
        for v in [self.xmin[-1], self.xmax[-1]]:
            u = list(itertools.product(x, y))
            pts.append(np.hstack((u, np.full((len(u), 1), v))))
        if nz > 2:
            for v in [self.xmin[1], self.xmax[1]]:
                u = np.array(list(itertools.product(x, z[1:-1])))
                pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), v), u[:, 1:])))
        if ny > 2 and nz > 2:
            for v in [self.xmin[0], self.xmax[0]]:
                u = list(itertools.product(y[1:-1], z[1:-1]))
                pts.append(np.hstack((np.full((len(u), 1), v), u)))
        pts = np.vstack(pts)
        if n != len(pts):
            print(
                "Warning: {} points required, but {} points sampled.".format(
                    n, len(pts)
                )
            )
        return pts


#class Sphere(Hypersphere):
#    """
#    Args:
#        center: Center of the sphere.
#        radius: Radius of the sphere.
#    """
#
#    def __init__(self, center, radius):
#        super(Sphere, self).__init__(center, radius)
