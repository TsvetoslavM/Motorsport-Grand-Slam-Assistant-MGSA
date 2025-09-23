import numpy as np
from .curvature import curvature_vectorized


def smooth_curvatures_vectorized(points, window_size=3):
    """Smooth curvature values using moving average"""
    if len(points)<3:
        return np.array([])
    curvatures = curvature_vectorized(points)
    middle = curvatures[1:-1]
    if len(middle)==0:
        return np.array([])
    kernel = np.ones(window_size)/window_size
    smoothed = np.convolve(middle, kernel, mode='same')
    half = window_size//2
    for i in range(half):
        smoothed[i] = np.mean(middle[:i+half+1])
    for i in range(len(smoothed)-half, len(smoothed)):
        smoothed[i] = np.mean(middle[i-half:])
    return smoothed


def smooth_curvatures_multi_scale(points, window_sizes=[3,5,7]):
    """Compute curvatures at multiple scales"""
    return {w: smooth_curvatures_vectorized(points, window_size=w) for w in window_sizes}


