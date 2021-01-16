import BezierCurve
import Line
import numpy as np
import matplotlib.pyplot as plt

#bezierの制御点
P = np.array([
    [0.,-1],
    [1.,2],
    [1.,-2],
    [0.,1]
])
bc = BezierCurve.BezierCurve(P)
#ax+by+c=0
line = Line.Line(a=-1, b=1, c=0.5)
clip_xy = bc.Clip(line)
print(clip_xy)
plt.scatter(clip_xy[:,0], clip_xy[:,1], c='r')
bc.Plot()
line.Plot(x_min=0, x_max=1)
plt.show()
