import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

class Plane:
    def __init__(self, d, P):
        d = d/math.sqrt(np.sum(d**2))
        self._d = d 
        self._a = d[0]
        self._b = d[1]
        self._c = d[2]
        self._e = -np.dot(d,P)
    
    @property
    def d(self):
        return self._d

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def e(self):
        return self._e

    def dist2Point(self, P):
        d = np.dot(self.d, P) + self.e
        return d

    def Plot(self, ax, xrange=[0,3], yrange=[0,3]):
        if self.c != 0:
            x = np.arange(xrange[0], xrange[1], 0.1)
            y = np.arange(yrange[0], yrange[1], 0.1)
            xx, yy = np.meshgrid(x, y)
            zz = -(self.e + self.a*xx + self.b*yy) / self.c
        else:
            x = np.arange(xrange[0], xrange[1], 0.1)
            y = np.arange(yrange[0], yrange[1], 0.1)
            xx, yy = np.meshgrid(x, y)
            zz = -(self.e + self.a*xx + self.b*yy) / self.c
        
        ax.plot_wireframe(xx, yy, zz, linewidths = 0.3)


        