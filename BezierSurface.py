import sys
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import pprint
import BezierPatch
import Line

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

    def getRowArray(self, P, i):
        return np.array(P[i])
    
    def getColumnArray(self, P, i):
        col = []
        for row in P:
            col.append(row[i])
        return np.array(col)

    def Bernstein(self,n,i,t):
        return comb(n,i) * (1-t)**(n-i) * t**i

    def Point(self,u,v):
        Suv = np.array([0.,0.,0.])
        for i in range(self.norder + 1):
            for j in range(self.morder + 1):
                Suv += self.Bernstein(self.norder,i,u) * self.Bernstein(self.morder,j,v) * self.P[i][j]
        return Suv
    
    def Plot(self, ax):
        Suv = []
        for u in np.linspace(0.,1.,101):
            row = []
            for v in np.linspace(0.,1.,101):
                row.append(self.Point(u,v))
            Suv.append(row)
        
        x = self.getMatrix(Suv, 0)
        y = self.getMatrix(Suv, 1)
        z = self.getMatrix(Suv, 2)

        ax.plot_wireframe(x, y, z, color='blue',linewidth=0.3)

    def getBezierPatch(self,line):
        pl1,pl2 = line.intersectionPlane()
        d = [[None for i in range(self.morder+1)] for j in range(self.norder+1)]

        for i in range(self.norder+1):
            for j in range(self.morder+1):
                d[i][j] = np.array([pl1.dist2Point(self.P[i][j]), pl2.dist2Point(self.P[i][j])])
        return BezierPatch.BezierPatch(d)

    def Clip(self, line):
        return self.getBezierPatch(line).Clip()

    def ClipU(self, v_max, v_min):
        ud = np.empty([0,2],float)
        for row in self.P:
            for d in row:
                ud = np.append(ud, np.array([[d[0], d[2]]]), axis=0)
        pprint.pprint(self.P)
        hull = ConvexHull(ud)
        hull_points = hull.points[hull.vertices]
        hull_points = np.append(hull_points, np.array([hull_points[0]]), axis=0)

        prev_hp = None
        u_max = 0.
        u_min = 0.
        for hp in hull_points:
            if not prev_hp is None:
                x1 = prev_hp[0]
                y1 = prev_hp[1]
                x2 = hp[0]
                y2 = hp[1]
                
                if y1 == 0:
                    x = x1
                    if u_max < x:
                        u_max, u_min = u_min, u_max
                        u_max = x
                    else:
                        u_min = x
                elif y1*y2 < 0:
                    x = (x1*y2 - x2*y1)/(y2 - y1)
                    if u_max < x:
                        u_max, u_min = u_min, u_max
                        u_max = x
                    else:
                        u_min = x
            prev_hp = hp
        if u_max < u_min:
            u_max, u_min = u_min, u_max
        t_max = (u_max-self.P[0][0][0])/(self.P[-1][-1][0]-self.P[0][0][0])
        t_min = (u_min-self.P[0][0][0])/(self.P[-1][-1][0]-self.P[0][0][0])
        print([u_max, u_min])

        #再帰を行う
        x0 = np.empty([0,2],float)
        if u_max-u_min < 1e-6 and v_max-v_min < 1e-6:
            x0 = np.append(x0, np.array([[(u_max+u_min)/2,(v_max+v_min)/2]]))
        elif 0.99 < t_max-t_min:
            div_bezier1, div_bezier2 = self.divideU((t_max+t_min)/2)
            x0 = np.append(x0, div_bezier1.ClipU(v_max, v_min))
            x0 = np.append(x0, div_bezier2.ClipU(v_max, v_min))
        else :
            div_bezier, _ = self.divideU(t_max)
            _, div_bezier = div_bezier.divideU(t_min/t_max)
            x0 = np.append(x0, div_bezier.ClipV(u_max, u_min))
        return x0

    def ClipV(self, u_max, u_min):
        vd = np.empty([0,2],float)
        for row in self.P:
            for d in row:
                vd = np.append(vd, np.array([[d[0], d[2]]]), axis=0)
        pprint.pprint(self.P)
        hull = ConvexHull(vd)
        hull_points = hull.points[hull.vertices]
        hull_points = np.append(hull_points, np.array([hull_points[0]]), axis=0)

        prev_hp = None
        v_max = 0.
        v_min = 0.
        for hp in hull_points:
            if not prev_hp is None:
                x1 = prev_hp[0]
                y1 = prev_hp[1]
                x2 = hp[0]
                y2 = hp[1]
                
                if y1 == 0:
                    x = x1
                    if v_max < x:
                        v_max, v_min = v_min, v_max
                        v_max = x
                    else:
                        v_min = x
                elif y1*y2 < 0:
                    x = (x1*y2 - x2*y1)/(y2 - y1)
                    if v_max < x:
                        v_max, v_min = v_min, v_max
                        v_max = x
                    else:
                        v_min = x
            prev_hp = hp
        if v_max < v_min:
            v_max, v_min = v_min, v_max
        t_max = (v_max-self.P[0][0][1])/(self.P[-1][-1][1]-self.P[0][0][1])
        t_min = (v_min-self.P[0][0][1])/(self.P[-1][-1][1]-self.P[0][0][1])
        print([v_max, v_min])

        #再帰を行う
        x0 = np.empty([0,2],float)
        if u_max-u_min < 1e-6 and v_max-v_min < 1e-6:
            x0 = np.append(x0, np.array([[(u_max+u_min)/2,(v_max+v_min)/2]]))
        elif 0.99 < t_max-t_min:
            div_bezier1, div_bezier2 = self.divideV((t_max+t_min)/2)
            x0 = np.append(x0, div_bezier1.ClipV(v_max, v_min))
            x0 = np.append(x0, div_bezier2.ClipV(v_max, v_min))
        else :
            div_bezier, _ = self.divideV(t_max)
            _, div_bezier = div_bezier.divideV(t_min/t_max)
            x0 = np.append(x0, div_bezier.ClipU(v_max, v_min))
        return x0

    def divideU(self, t):
        P1 = []
        P2 = []
        for i in range(self.morder+1):
            P = self.getColumnArray(self.P,i)
            Ps = [P] + self._de_casteljau_algorithm(P, t)
            row1 = []
            row2 = []
            for lst in Ps:
                row1.append(lst[0])
                row2.append(lst[-1])
            row2.reverse()
            P1.append(row1)
            P2.append(row2)
        P1 = list(map(list, (zip(*P1))))
        P2 = list(map(list, (zip(*P2))))
        return BezierSurface(P1), BezierSurface(P2)
    
    def divideV(self, t):
        P1 = []
        P2 = []
        for i in range(self.norder+1):
            P = self.getRowArray(self.P,i)
            Ps = [P] + self._de_casteljau_algorithm(P, t)
            row1 = []
            row2 = []
            for lst in Ps:
                row1.append(lst[0])
                row2.append(lst[-1])
            row2.reverse()
            P1.append(row1)
            P2.append(row2)
        return BezierSurface(P1), BezierSurface(P2)

    def _de_casteljau_algorithm(self, P, t):
        prev_p = None
        Q = []
        for p in P:
            if not prev_p is None:
                Q.append(np.array((1-t)*prev_p + t*p))
            prev_p = p
        if len(Q) == 1:
            return [Q]
        return [Q] + self._de_casteljau_algorithm(Q, t)
      
def main():
    
    P = [
        [np.array([0.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,2.,-1.]), np.array([0.,3.,0.])],
        [np.array([1.,0.,1.]), np.array([1.,1.,0.]), np.array([1.,2.,0.]), np.array([1.,3.,0.])],
        [np.array([2.,0.,0.]), np.array([2.,1.,0.]), np.array([2.,2.,0.]), np.array([2.,3.,1.])],
        [np.array([3.,0.,0.]), np.array([3.,1.,-1.]), np.array([3.,2.,0.]), np.array([3.,3.,0.])]
    ]
    
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    BS = BezierSurface(P)
    div1,div2 = BS.divideU(0.7)
    div1.Plot(ax)
    div2.Plot(ax)
    pprint.pprint(div1.P)
    pprint.pprint(div2.P)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
