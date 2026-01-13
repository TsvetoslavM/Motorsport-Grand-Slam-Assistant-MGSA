from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_track_edges(
    x: np.ndarray,
    y: np.ndarray,
    w_tr_left: np.ndarray,
    w_tr_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute left/right track edges and normals from centerline and widths.

    Parameters
    ----------
    x, y:
        1D arrays with centerline coordinates.
    w_tr_left, w_tr_right:
        1D arrays with left / right track widths at each centerline point.

    Returns
    -------
    x_left, y_left, x_right, y_right, nx, ny
        Left/right edge coordinates and unit normal (pointing to the left)
        at each centerline point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w_tr_left = np.asarray(w_tr_left, dtype=float)
    w_tr_right = np.asarray(w_tr_right, dtype=float)

    # Tangent via gradient
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.hypot(dx, dy)
    norm[norm == 0.0] = 1.0
    tx = dx / norm
    ty = dy / norm

    # Left-pointing unit normal
    nx = -ty
    ny = tx

    x_left = x + nx * w_tr_left
    y_left = y + ny * w_tr_left
    x_right = x - nx * w_tr_right
    y_right = y - ny * w_tr_right

    return x_left, y_left, x_right, y_right, nx, ny

