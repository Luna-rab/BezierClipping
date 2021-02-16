import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import Plane

# ax+by+c=0
class Line2D:
    def __init__(self, start, end):
        self._start = start
        self._end = end
        l = math.sqrt(np.sum((start-end)**2))
        self._a = (start[1] - end[1])/l
        self._b = -(start[0] - end[0])/l
        self._c = (start[0]*end[1] - end[0]*start[1])/l
    
    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end
    
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

    def Plot(self, c='b', x_min=None, x_max=None):
        if x_min == None:
            x_min = min(self.start[0], self.end[0])
        if x_max == None:
            x_max = max(self.start[0], self.end[0])
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        
        if self.b == 0:
            plt.axvline(x=-1*(self.c/self.a), c=c)
        else :
            x = np.linspace(x_min,x_max,101)
            y = self.Point(x)
            plt.plot(x, y, c='b')

    def dist2Point(self, P):
        return self.a*P[0] + self.b*P[1] + self.c

class Line3D:
    def __init__(self, start, end):
        self._start = start
        self._end = end
        self._d = (end - start)/math.sqrt(np.sum((end - start)**2))
    
    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def d(self):
        return self._d
    
    def Point(self, t):
        return self.start + t*self.d
 
    def dist_Point2Line(self, P1):
        P = self.start + np.dot((P1-self.start), self.d) * self.d
        return math.sqrt(np.dot((P-P1), (P-P1)))

    def intersectionPlane(self):
        h1 = np.cross(self.d, np.array([1+1e-3,1,1]))
        h1 = h1/math.sqrt(np.sum(h1)**2)

        h2 = np.cross(self.d, h1)
        h2 = h2/math.sqrt(np.sum(h2)**2)

        return Plane.Plane(h1, self.start), Plane.Plane(h2, self.start)

    def Plot(self, ax, x_min=None, x_max=None, y_min=None, y_max=None):
        if x_min == None:
            x_min = min(self.start[0], self.end[0])
        if x_max == None:
            x_max = max(self.start[0], self.end[0])
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min == None:
            y_min = min(self.start[1], self.end[1])
        if y_max == None:
            y_max = max(self.start[1], self.end[1])
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        if self.d[0] == 0 or self.d[1] == 0:
            t_max = max(self.start[2], self.end[2])/self.d[2]
            t_min = min(self.start[2], self.end[2])/self.d[2]
        else:
            t_min = max((x_min - self.start[0])/self.d[0], (y_min - self.start[1])/self.d[1])
            t_max = min((x_max - self.start[0])/self.d[0], (y_max - self.start[1])/self.d[1])
        
        x = []
        y = []
        z = []
        for t in np.linspace(t_min, t_max, 101):
            x.append(self.Point(t)[0])
            y.append(self.Point(t)[1])
            z.append(self.Point(t)[2])

        ax.plot(x, y, z, color='red',linewidth=1)

