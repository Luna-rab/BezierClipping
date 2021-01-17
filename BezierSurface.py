import sys
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BezierSurface :
    #Pは3次元制御点の行列
    def __init__(self,P):
        for row in P:
            if len(row) != len(P[0]):
                raise Exception('DimensionError')
        
        self._P = P   
        self._norder = len(P) - 1
        self._morder = len(P[0]) - 1

    @property
    def P(self):
        return self._P
    
    @property
    def morder(self):
        return self._morder

    @property
    def norder(self):
        return self._norder

    def getMatrix(self, P, k):
        for row in P:
            if len(row) != len(P[0]):
                raise Exception('DimensionError')
        
        mat = np.empty([len(P), len(P[0])], float)
        for i in range(0, len(P)):
            for j in range(0, len(P[0])):
                mat[i,j] = P[i][j][k]
        return mat

    def Bernstein(self,n,i,t):
        return comb(n,i) * (1-t)**(n-i) * t**i

    def Point(self,u,v):
        Suv = np.array([0.,0.,0.])
        for i in range(self.norder + 1):
            for j in range(self.morder + 1):
                Suv += self.Bernstein(self.norder,i,u) * self.Bernstein(self.morder,j,v) * self.P[i][j]
        return Suv
    
    def Plot(self):
        Suv = []
        for u in np.linspace(0.,1.,101):
            row = []
            for v in np. linspace(0.,1.,101):
                row.append(self.Point(u,v))
            Suv.append(row)
        
        x = self.getMatrix(Suv, 0)
        y = self.getMatrix(Suv, 1)
        z = self.getMatrix(Suv, 2)

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.plot_wireframe(x, y, z, color='blue',linewidth=0.3)
        plt.legend()
        plt.show()

def main():
    
    P = [
        [np.array([0.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,2.,-1.]), np.array([0.,3.,0.])],
        [np.array([1.,0.,1.]), np.array([1.,1.,0.]), np.array([1.,2.,0.]), np.array([1.,3.,0.])],
        [np.array([2.,0.,0.]), np.array([2.,1.,0.]), np.array([2.,2.,0.]), np.array([2.,3.,1.])],
        [np.array([3.,0.,0.]), np.array([3.,1.,-1.]), np.array([3.,2.,0.]), np.array([3.,3.,0.])]
    ]
    
    BS = BezierSurface(P)
    BS.Plot()

if __name__ == "__main__":
    main()
