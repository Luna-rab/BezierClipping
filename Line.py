import numpy as np
import matplotlib.pyplot as plt

# ax+by+c=0
class Line2D:
    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c
    
    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    def Point(self, x):
        return (-1*self.a*x-self.c)/self.b

    def Plot(self, x_min=0., x_max=1.):
        if self.b == 0:
            plt.axvline(x=-1*(self.c/self.a), c='b')
        else :
            x = np.linspace(x_min,x_max,101)
            y = self.Point(x)
            plt.plot(x, y, c='b')

# ax+by+cz+d=0
class Line3D:
    def __init__(self, a, b, c, d):
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    
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
    def d(self):
        return self._d

    def Point(self, x, y):
        return -1*(self.a*x + self.b*y + self.d)/self.c

    def Plot(self, x_min=0., x_max=1., y_min=0., y_max=1.):
        if self.c == 0:
            plt.axvline(x=-1*(self.c/self.a), c='b')
        else :
            x = np.linspace(x_min,x_max,101)
            y = np.linspace(y_min,y_max,101)
            z = self.Point(x, y)
            plt.plot(x, y, z, c='b')
