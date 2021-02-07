import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import Line
import BezierSurface

class BezierPatch:
    def __init__(self, d):
        for row in d:
            if len(row) != len(d[0]):
                raise Exception('DimensionError')
        
        self._d = d
        self._norder = len(d) - 1
        self._morder = len(d[0]) - 1

    @property
    def d(self):
        return self._d

    @property
    def morder(self):
        return self._morder

    @property
    def norder(self):
        return self._norder

    def getColumnArray(self, P, i):
        col = []
        for p in P:
            col.append(p[i])
        return np.array(col)

    def Bernstein(self,n,i,t):
        return comb(n,i) * (1-t)**(n-i) * t**i

    def Point(self,u,v):
        Puv = np.array([0.,0.])
        for i in range(self.norder + 1):
            for j in range(self.morder + 1):
                Puv += self.Bernstein(self.norder,i,u) * self.Bernstein(self.morder,j,v) * self.d[i][j]
        return Puv

    def Plot(self):
        for u in np.linspace(0,1,11):
            P = []
            for v in np.linspace(0,1,101):
                P.append(self.Point(u,v))
            plt.plot(self.getColumnArray(P,0), self.getColumnArray(P,1), color='blue')

        for v in np.linspace(0,1,11):
            P = []
            for u in np.linspace(0,1,101):
                P.append(self.Point(u,v))
            plt.plot(self.getColumnArray(P,0), self.getColumnArray(P,1), color='blue')

    def Clip(self):
        V0 = self.d[0][0] - self.d[0][self.morder]
        V1 = self.d[self.norder][0] - self.d[self.norder][self.morder]
        L = Line.Line2D(np.array([0,0]), V0+V1)

        D = []
        for i in range(self.norder+1):
            row = []
            for j in range(self.morder+1):
                row.append(np.array([float(i)/self.norder, float(j)/self.morder, L.dist2Point(self.d[i][j])]))
            D.append(row)

        return BezierSurface.BezierSurface(D).ClipU(1, 0) #D=0となるu,vを返す
        
    


