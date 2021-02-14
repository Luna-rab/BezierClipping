import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pprint
import Line
import scipy
#import BezierSurface

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
        Puv = np.array([0.,0.])
        for j in range(self.morder + 1):
            for i in range(self.norder + 1):
                Puv += self.Bernstein(self.norder,i,u) * self.Bernstein(self.morder,j,v) * self.d[i][j]
        return Puv

    def Plot(self, color='blue'):
        for u in np.linspace(0,1,11):
            P = []
            for v in np.linspace(0,1,101):
                P.append(self.Point(u,v))
            plt.plot(self.getColumnArray(P,0), self.getColumnArray(P,1), color=color)

        for v in np.linspace(0,1,11):
            P = []
            for u in np.linspace(0,1,101):
                P.append(self.Point(u,v))
            plt.plot(self.getColumnArray(P,0), self.getColumnArray(P,1), color=color)

    def Clip(self):
        return self.ClipU(1,0)

    def ClipU(self, v_max, v_min):
        V0 = self.d[0][self.morder] - self.d[0][0]
        V1 = self.d[self.norder][self.morder] - self.d[self.norder][0]
        L = Line.Line2D(np.array([0,0]), V0+V1)
        #pprint.pprint(self.d)

        ud = np.empty([0,2],float)
        for i in range(self.norder+1):
            for j in range(self.morder+1):
                ud = np.append(ud, np.array([[float(i)/self.norder, L.dist2Point(self.d[i][j])]]), axis=0)
                
        try:
            hull = ConvexHull(ud)
            hull_points = hull.points[hull.vertices]
            hull_points = np.append(hull_points, np.array([hull_points[0]]), axis=0)
            #pprint.pprint(ud)

            prev_hp = None
            u_max = 0.
            u_min = 0.
            for hp in hull_points:
                if not prev_hp is None:
                    u1 = prev_hp[0]
                    d1 = prev_hp[1]
                    u2 = hp[0]
                    d2 = hp[1]
                    
                    if d1 == 0:
                        u = u1
                        if u_max < u:
                            u_max, u_min = u_min, u_max
                            u_max = u
                        else:
                            u_min = u
                    elif d1*d2 < 0:
                        u = (u1*d2 - u2*d1)/(d2 - d1)
                        if u_max < u:
                            u_max, u_min = u_min, u_max
                            u_max = u
                        else:
                            u_min = u
                prev_hp = hp
            if u_max < u_min:
                u_max, u_min = u_min, u_max
        except scipy.spatial.qhull.QhullError:
            u1 = ud[0][0]
            d1 = ud[0][1]
            u2 = ud[-1][0]
            d2 = ud[-1][1]
            u_max = u_min = (u1*d2 - u2*d1)/(d2 - d1)

        #再帰を行う
        x0 = np.empty([0,2],float)
        if u_max-u_min < 1e-4 and v_max-v_min < 1e-4:
            x0 = np.append(x0, np.array([[(u_max+u_min)/2,(v_max+v_min)/2]]))
            '''
        elif 0.99 < u_max-u_min:
            div_patch1, div_patch2 = self.divideU((u_max+u_min)/2)
            x0 = np.append(x0, div_patch1.ClipU(v_max, v_min))
            x0 = np.append(x0, div_patch2.ClipU(v_max, v_min))
            '''
        else:
            div_patch, _ = self.divideU(u_max)
            _, div_patch = div_patch.divideU(u_min/u_max)
            x0 = np.append(x0, div_patch.ClipV(u_max, u_min))
        #self.Plot([(u_max-u_min)**0.3,(u_max-u_min)**0.3,(u_max-u_min)**0.3])
        x1 = np.array([[u_max-u_min, 0],[0,1]])
        x0 = x0.dot(x1) + np.array([u_min, 0])
        return x0


    def ClipV(self, u_max, u_min):
        V0 = self.d[self.norder][0] - self.d[0][0]
        V1 = self.d[self.norder][self.morder] - self.d[0][self.morder]
        L = Line.Line2D(np.array([0,0]), V0+V1)
        #pprint.pprint(self.d)

        vd = np.empty([0,2],float)
        for i in range(self.norder+1):
            for j in range(self.morder+1):
                vd = np.append(vd, np.array([[float(j)/self.morder, L.dist2Point(self.d[i][j])]]), axis=0)
        
        try:
            hull = ConvexHull(vd)
            hull_points = hull.points[hull.vertices]
            hull_points = np.append(hull_points, np.array([hull_points[0]]), axis=0)
            #pprint.pprint(vd)

            prev_hp = None
            v_max = 0.
            v_min = 0.
            for hp in hull_points:
                if not prev_hp is None:
                    v1 = prev_hp[0]
                    d1 = prev_hp[1]
                    v2 = hp[0]
                    d2 = hp[1]
                    
                    if d1 == 0:
                        v = v1
                        if v_max < v:
                            v_max, v_min = v_min, v_max
                            v_max = v
                        else:
                            v_min = v
                    elif d1*d2 < 0:
                        v = (v1*d2 - v2*d1)/(d2 - d1)
                        if v_max < v:
                            v_max, v_min = v_min, v_max
                            v_max = v
                        else:
                            v_min = v
                prev_hp = hp
            if v_max < v_min:
                v_max, v_min = v_min, v_max
        except scipy.spatial.qhull.QhullError:
            v1 = vd[0][0]
            d1 = vd[0][1]
            v2 = vd[-1][0]
            d2 = vd[-1][1]
            v_max = v_min = (v1*d2 - v2*d1)/(d2 - d1)

        #再帰を行う
        x0 = np.empty([0,2],float)
        if u_max-u_min < 1e-4 and v_max-v_min < 1e-4:
            x0 = np.append(x0, np.array([[(u_max+u_min)/2,(v_max+v_min)/2]]))
            '''
        elif 0.99 < v_max-v_min:
            div_patch1, div_patch2 = self.divideV((v_max+v_min)/2)
            x0 = np.append(x0, div_patch1.ClipV(u_max, u_min))
            x0 = np.append(x0, div_patch2.ClipV(u_max, u_min))
            '''
        else:
            div_patch, _ = self.divideV(v_max)
            _, div_patch = div_patch.divideV(v_min/v_max)
            x0 = np.append(x0, div_patch.ClipU(v_max, v_min))
        #self.Plot([(v_max-v_min)**0.3,(v_max-v_min)**0.3,(v_max-v_min)**0.3])
        x1 = np.array([[1,0],[0, v_max-v_min]])
        x0 = x0.dot(x1) + np.array([0, v_min])
        return x0

    def divideV(self, t):
        if t==0:
            return self, self
        elif t==1:
            return self ,self
        else:
            P1 = []
            P2 = []
            for i in range(self.norder+1):
                P = self.getRowArray(self.d,i)
                Ps = [P] + self._de_casteljau_algorithm(P, t)
                row1 = []
                row2 = []
                for lst in Ps:
                    row1.append(lst[0])
                    row2.append(lst[-1])
                row2.reverse()
                P1.append(row1)
                P2.append(row2)
            return BezierPatch(P1), BezierPatch(P2)

    def divideU(self, t):
        if t==0:
            return self, self
        elif t==1:
            return self ,self
        else:
            P1 = []
            P2 = []
            for j in range(self.morder+1):
                P = self.getColumnArray(self.d,j)
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
            return BezierPatch(P1), BezierPatch(P2)

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

    P=[
        [np.array([0.89442719, 0.99654576]), np.array([0.       , 1.2456822]), np.array([-0.4472136 ,  1.99309152]), np.array([-1.78885438,  1.74395508])], 
        [np.array([ 0.4472136 , -0.33218192]), np.array([0.       , 0.4152274]), np.array([-0.89442719,  0.66436384]), np.array([-1.78885438,  0.91350028])], 
        [np.array([ 0.89442719, -0.66436384]), np.array([ 0.       , -0.4152274]), np.array([-0.89442719, -0.16609096]), np.array([-2.23606798, -0.4152274 ])], 
        [np.array([ 0.89442719, -1.49481864]), np.array([ 0.4472136 , -0.74740932]), np.array([-0.89442719, -0.99654576]), np.array([-1.78885438, -0.74740932])]
    ]
    bp = BezierPatch(P)
    #bp.Plot()
    plt.scatter(bp.getMatrix(P,0),bp.getMatrix(P,1))

    plt.show()

if __name__ == "__main__":
    main()
        
    


