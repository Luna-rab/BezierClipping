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
        self._order = P.shape[0] - 1
    
    @property
    def P(self):
        return self._P
    
    @property
    def order(self):
        return self._order

    def Bernstein(self,n,i,t):
        return comb(n,i) * (1-t)**(n-i) * t**i

    def Point(self,t):
        Pt = np.array([0.,0.])
        for i in range(self.order + 1):
            Pt += self.Bernstein(self.order,i,t) * self.P[i]
        return Pt
    
    def Plot(self):
        Pt = np.empty([0,2],float)
        for t in np.linspace(0.,1.,101):
            Pt = np.append(Pt,np.array([self.Point(t)]),axis=0)
        
        x = Pt[:,0]
        y = Pt[:,1]

        plt.plot(x, y, c='b')

    def Clip(self, line):
        di = (line.a*self.P[:,0] + line.b*self.P[:,1] + line.c)/np.sqrt(line.a**2 + line.b**2)
        ni = np.linspace(0, 1, self.order+1)
        Pi = np.array([ni, di]).T
        
        td_curve = BezierCurve(Pi)
        td = td_curve._zeroPoint()

        Pt = np.empty([0,2],float)
        for t in td:
            Pt = np.append(Pt, np.array([self.Point(t)]), axis=0)
        return Pt
    
    def _zeroPoint(self):
        #凸包とx軸との交点を求め、小さいほうからt_min, t_maxとするコード
        try:
            hull = ConvexHull(self.P)
        except scipy.spatial.qhull.QhullError:
            x1 = self.P[0,0]
            y1 = self.P[0,1]
            x2 = self.P[-1,0]
            y2 = self.P[-1,1]
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
        t_max = (x_max-self.P[0,0])/(self.P[-1,0]-self.P[0,0])
        t_min = (x_min-self.P[0,0])/(self.P[-1,0]-self.P[0,0])

        #再帰を行う
        x0 = np.empty(0)
        if x_max-x_min < 1e-6:
            x0 = np.append(x0, np.array([(x_max+x_min)/2]))
        elif 1 - 1e-6 < t_max-t_min:
            div_bezier1, div_bezier2 = self.divide((t_max+t_min)/2)
            x0 = np.append(x0, div_bezier1._zeroPoint())
            x0 = np.append(x0, div_bezier2._zeroPoint())
        else :
            div_bezier, _ = self.divide(t_max)
            _, div_bezier = div_bezier.divide(t_min/t_max)
            x0 = np.append(x0, div_bezier._zeroPoint())
        return x0

    def divide(self, t):
        Ps = np.append(self.P, self._de_casteljau_algorithm(self.P, t), axis=0)
        P1 = np.empty([0,2])
        P2 = np.empty([0,2])
        index1 = 0
        index2 = self.order
        i = 0
        while i <= self.order:
            P1 = np.append(P1, np.array([Ps[index1,:]]), axis=0)
            P2 = np.append(P2, np.array([Ps[index2,:]]), axis=0)
            index1 += (self.order+1) - i
            index2 += self.order - i
            i += 1
        return BezierCurve(P1), BezierCurve(np.flipud(P2))

    def _de_casteljau_algorithm(self, P, t):
        prev_p = None
        Q = np.empty([0,2])
        for p in P:
            if not prev_p is None:
                Q = np.append(Q, np.array([(1-t)*prev_p + t*p]), axis=0)
            prev_p = p
        if Q.shape[0] == 1:
            return Q
        return np.append(Q, self._de_casteljau_algorithm(Q, t), axis=0)   
    
def main():
    P = np.array([
        [0.,-1],
        [1.,2],
        [1.,-2],
        [0.,1]
    ])
    bc = BezierCurve(P)
    line = Line.Line(a=-1, b=0, c=0.5)
    clip_xy = bc.Clip(line)
    print(clip_xy)
    plt.scatter(clip_xy[:,0], clip_xy[:,1], c='r')
    bc.Plot()
    line.Plot(x_min=0, x_max=3)
    plt.show()

if __name__ == "__main__":
    main()