"""Exact solution for dusty shock.

See the following references:

- Laibe G., Price D. J., 2011, MNRAS, 418, 1491
- Laibe G., Price D. J., 2014, MNRAS, 444, 1940
"""

import numpy as np
from numpy import ndarray
from scipy.linalg import expm


def delta_vx(
    time: float, t_s: ndarray, eps: ndarray, delta_vx_init: ndarray
) -> ndarray:
    """Differential velocity for linear drag on multiple dust species.

    Parameters
    ----------
    time
        Time at which to evaluate the expression.
    t_s
        Stopping time for each dust species.
    eps
        The dust fraction for each dust species.
    delta_vx_init
        Initial differential velocity between the dust and gas.

    Returns
    -------
    delta_vx : (N,) ndarray
        The differential velocity between the gas and dust.

    References
    ----------
    See Table (1) in Laibe and Price (2011) MNRAS, 418, 1491, and
    see Equation (64) in Laibe and Price (2014) MNRAS, 444, 1940.
    """
    omega = _drag_matrix(t_s, eps)

    if omega.size > 1:
        return expm(-omega * time) @ delta_vx_init
    return np.exp(-omega[0, 0] * time) * delta_vx_init


def _drag_matrix(t_s: ndarray, eps: ndarray) -> ndarray:
    """Drag matrix for multiple dust species.

    Parameters
    ----------
    t_s
        Stopping time for each dust species.
    eps
        The dust fraction for each dust species.

    Returns
    -------
    omega : (N, N) ndarray
        The drag matrix.

    References
    ----------
    See Equation (65) in Laibe and Price (2014) MNRAS, 444, 1940.
    """
    t_s = np.array(t_s)
    eps = np.array(eps)

    N = t_s.size
    omega = np.zeros((N, N))

    if N > 1:
        for idx in range(N):
            omega[idx, :] = 1.0 / (t_s * (1.0 - np.sum(eps)))
        omega += np.diag(1.0 / (t_s * eps))

    else:
        omega[:] = 1.0 / (t_s * eps * (1 - eps))

    return omega
