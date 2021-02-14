import numpy as np
from scipy.special import comb
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import scipy
import Line

class BezierCurve :
    #Pは2次元制御点の行列
    def __init__(self,P):
        self._P = P   
        self._order = len(P) - 1
    
    @property
    def P(self):
        return self._P
    
    @property
    def order(self):
        return self._order
    
    def getColumnArray(self, P, i):
        col = []
        for p in P:
            col.append(p[i])
        return np.array(col)
    
    def getMatrix(self, P):
        mat = np.empty([0,2],float)
        for p in P:
            mat = np.append(mat, np.array([p]), axis=0)
        return mat


    def Bernstein(self,n,i,t):
        return comb(n,i) * (1-t)**(n-i) * t**i

    def Point(self,t):
        Pt = np.array([0.,0.])
        for i in range(self.order + 1):
            Pt += self.Bernstein(self.order,i,t) * self.P[i]
        return Pt
    
    def Plot(self, c='b'):
        Pt = []
        for t in np.linspace(0.,1.,101):
            Pt.append(self.Point(t))
        
        x = self.getColumnArray(Pt,0)
        y = self.getColumnArray(Pt,1)

        plt.plot(x, y, c=c)

    def Clip(self, line):
        di = (line.a*self.getColumnArray(self.P,0) + line.b*self.getColumnArray(self.P,1) + line.c)/np.sqrt(line.a**2 + line.b**2)
        ni = np.linspace(0, 1, self.order+1)
        Pi = []
        for (n, d) in zip(ni, di):
            Pi.append(np.array([n,d]))

        td_curve = BezierCurve(Pi)
        td = td_curve._zeroPoint()

        Pt = []
        for t in td:
            Pt.append(self.Point(t))
        return Pt

    def _zeroPoint(self):
        #凸包とx軸との交点を求め、小さいほうからt_min, t_maxとするコード
        try:
            hull = ConvexHull(self.getMatrix(self.P))
        except scipy.spatial.qhull.QhullError:
            x1 = self.P[0][0]
            y1 = self.P[0][1]
            x2 = self.P[-1][0]
            y2 = self.P[-1][1]
            return (x1*y2 - x2*y1)/(y2 - y1)
        hull_points = hull.points[hull.vertices]
        hull_points = np.append(hull_points, np.array([hull_points[0]]), axis=0)
        #print(hull_points)
        prev_hp = None
        x_max = 0.
        x_min = 0.
        for hp in hull_points:
            if not prev_hp is None:
                x1 = prev_hp[0]
                y1 = prev_hp[1]
                x2 = hp[0]
                y2 = hp[1]
                
                if y1 == 0:
                    x = x1
                    if x_max < x:
                        x_max, x_min = x_min, x_max
                        x_max = x
                    else:
                        x_min = x
                elif y1*y2 < 0:
                    x = (x1*y2 - x2*y1)/(y2 - y1)
                    if x_max < x:
                        x_max, x_min = x_min, x_max
                        x_max = x
                    else:
                        x_min = x
            prev_hp = hp
        if x_max < x_min:
            x_max, x_min = x_min, x_max
        t_max = (x_max-self.P[0][0])/(self.P[-1][0]-self.P[0][0])
        t_min = (x_min-self.P[0][0])/(self.P[-1][0]-self.P[0][0])

        #再帰を行う
        x0 = np.empty(0)
        if x_max-x_min < 1e-12:
            x0 = np.append(x0, np.array([(x_max+x_min)/2]))
        elif 0.99 < t_max-t_min:
            div_bezier1, div_bezier2 = self.divide((t_max+t_min)/2)
            x0 = np.append(x0, div_bezier1._zeroPoint())
            x0 = np.append(x0, div_bezier2._zeroPoint())
        else :
            div_bezier, _ = self.divide(t_max)
            _, div_bezier = div_bezier.divide(t_min/t_max)
            x0 = np.append(x0, div_bezier._zeroPoint())
        return x0

    def divide(self, t):
        Ps = [self.P] + self._de_casteljau_algorithm(self.P, t)
        P1 = []
        P2 = []
        for lst in Ps:
            P1.append(lst[0])
            P2.append(lst[-1])
        P2.reverse()
        return BezierCurve(P1), BezierCurve(P2)

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
        np.array([0.,0]),
        np.array([1.,2]),
        np.array([2.,2]),
        np.array([3.,1])
    ]
    
    bc = BezierCurve(P)
    div1,div2 = bc.divide(0.6)
    div1.Plot(c='r')
    div2.Plot(c='g')
    plt.plot(bc.getColumnArray(bc.P,0),bc.getColumnArray(bc.P,1),c='b')
    plt.plot(div1.getColumnArray(div1.P,0),div1.getColumnArray(div1.P,1),c='orange')
    plt.plot(div2.getColumnArray(div2.P,0),div2.getColumnArray(div2.P,1),c='lightgreen')
    plt.show()

if __name__ == "__main__":
    main()