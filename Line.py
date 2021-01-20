import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

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
    
    
    def getLocalMat(self):
        mat0 = np.array([
            [1,0,0,-self.start[0]],
            [0,1,0,-self.start[1]],
            [0,0,1,-self.start[2]],
            [0,0,0,1]
        ])

        cos1 = (self.end - self.start)[0]/math.sqrt((self.end[0] - self.start[0])**2 + (self.end[1] - self.start[1])**2)
        sin1 = (self.end - self.start)[1]/math.sqrt((self.end[0] - self.start[0])**2 + (self.end[1] - self.start[1])**2)

        mat1 = np.array([
            [cos1,sin1,0,0],
            [-sin1,cos1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])

        cos2 = (self.end - self.start)[2]/math.sqrt(np.sum((self.end - self.start)**2))
        sin2 = math.sqrt((self.end[0] - self.start[0])**2 + (self.end[1] - self.start[1])**2)/math.sqrt(np.sum((self.end - self.start)**2))

        mat2 = np.array([
            [cos2,0,-sin2,0],
            [0,1,0,0],
            [sin2,0,cos2,0],
            [0,0,0,1]
        ])
        return np.dot(mat2, np.dot(mat1, mat0))

    def Affine(self, mat):
        startVec = np.array([np.append(self.start, 1)]).T
        endVec = np.array([np.append(self.end, 1)]).T

        new_start = np.dot(mat, startVec)[0:3, 0]
        new_end = np.dot(mat, endVec)[0:3, 0]

        return Line3D(new_start, new_end)

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

        ax.plot(x, y, z, color='blue',linewidth=0.3)

def main():
    line = Line3D(np.array([0.1,0.2,0.3]), np.array([1,1,1]))
    line2 = line.Affine(line.getLocalMat())

    fig = plt.figure()
    ax = Axes3D(fig)

    line.Plot(ax)
    line2.Plot(ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    
