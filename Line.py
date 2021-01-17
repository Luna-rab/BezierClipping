import numpy as np
import matplotlib.pyplot as plt

# ax+by+c=0
class Line2D:
    def __init__(self, start, end):
        self._start = start
        self._end = end
        self._a = start[1] - end[1]
        self._b = -1*(start[0] - end[0])
        self._c = start[0]*end[1] - end[0]*start[1]
    
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

    def Plot(self, x_min=None, x_max=None):
        if x_min == None:
            x_min = min(self.start[0], self.end[0])
        if x_max == None:
            x_max = max(self.start[0], self.end[0])
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        
        if self.b == 0:
            plt.axvline(x=-1*(self.c/self.a), c='b')
        else :
            x = np.linspace(x_min,x_max,101)
            y = self.Point(x)
            plt.plot(x, y, c='b')
    
