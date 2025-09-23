import math
import numpy as np


# ---------------- CURVATURE CALCULATION ---------------- #

def curvature(p1, p2, p3):
    """Circumradius method for a single point"""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    a = math.dist(p2, p3)
    b = math.dist(p1, p3)
    c = math.dist(p1, p2)
    if a == 0 or b == 0 or c == 0:
        return 0
    s = (a + b + c) / 2
    area_squared = s * (s - a) * (s - b) * (s - c)
    if area_squared <= 0:
        return 0
    A = math.sqrt(area_squared)
    return (4 * A) / (a * b * c)


def curvature_vectorized(points):
    """Vectorized curvature calculation"""
    if len(points) < 3:
        return np.array([])
    points = np.array(points)
    x, y = points[:,0], points[:,1]
    a = np.sqrt((x[2:]-x[1:-1])**2 + (y[2:]-y[1:-1])**2)
    b = np.sqrt((x[2:]-x[:-2])**2 + (y[2:]-y[:-2])**2)
    c = np.sqrt((x[1:-1]-x[:-2])**2 + (y[1:-1]-y[:-2])**2)
    s = (a+b+c)/2
    area_squared = s*(s-a)*(s-b)*(s-c)
    area_squared[area_squared<=0]=0
    A = np.sqrt(area_squared)
    curvatures = np.zeros(len(points))
    curvatures[1:-1] = (4*A)/(a*b*c)
    curvatures[np.isnan(curvatures)] = 0
    curvatures[np.isinf(curvatures)] = 0
    return curvatures


