import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BezierSurface :
    #Pは3次元制御点の行列
    def __init__(self,P):
        self._P = P   
        self._norder = P.shape[0] - 1
        self._morder = P.shape[1] - 1

    @property
    def P(self):
        return self._P
    
    @property
    def morder(self):
        return self._morder

    @property
    def norder(self):
        return self._norder

    def Bernstein(self,n,i,t):
        return comb(n,i) * (1-t)**(n-i) * t**i

    def Point(self,u,v):
        Suv = np.array([0.,0.,0.])
        for i in range(self.norder + 1):
            for j in range(self.morder + 1):
                Suv += self.Bernstein(self.norder,i,u) * self.Bernstein(self.morder,j,v) * self.P[i][j]
        return Suv
    
    def Plot(self):
        Suv = np.empty([0,101,3],float)
        for u in np.linspace(0.,1.,101):
            row = np.empty([0,3],float)
            for v in np. linspace(0.,1.,101):
                row = np.append(row,np.array([self.Point(u,v)]),axis=0)
            Suv = np.append(Suv,np.array([row]),axis=0)
        
        x = Suv[:,:,0]
        y = Suv[:,:,1]
        z = Suv[:,:,2]

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.plot_wireframe(x, y, z, color='blue',linewidth=0.3)
        plt.legend()
        plt.show()

def main():
    
    P = np.array([
        [[0.,0.,0.], [0.,1.,0.], [0.,2.,-1.], [0.,3.,0.]],
        [[1.,0.,1.], [1.,1.,0.], [1.,2.,0.], [1.,3.,0.]],
        [[2.,0.,0.], [2.,1.,0.], [2.,2.,0.], [2.,3.,1.]],
        [[3.,0.,0.], [3.,1.,-1.], [3.,2.,0.], [3.,3.,0.]]
    ])
    
    BS = BezierSurface(P)
    BS.Plot()

if __name__ == "__main__":
    main()
