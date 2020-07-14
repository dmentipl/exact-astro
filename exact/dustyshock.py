"""Exact solution for dusty shock.

See the following references:

- Lehmann A., Wardle M., 2018, MNRAS, 476, 3185
"""

import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def velocity(
    x_shock: float,
    x_width: float,
    n_dust: int,
    dust_to_gas: float,
    drag_coefficient: ndarray,
    density_left: float,
    velocity_left: float,
    mach_number: float,
    n_points: int = 500,
):
    """Exact solution for dusty shock velocity.

    Parameters
    ----------
    x_shock
        The x-position of the shock.
    x_width
        The width in x over which to compute the solution.
    n_dust
        The number of dust species.
    dust_to_gas
        The dust-to-gas ratio. Same for all species.
    drag_coefficient
        An array of drag-coefficients, one per dust species.
    density_left
        The density on the left. Same for gas and all dust species.
    velocity_left
        The velocity on the left. Same for gas and all dust species.
    mach_number
        The mach number.
    n_points
        The number of points to use in computing the solution. Default
        is 500.

    Returns
    -------
    velocity_gas
        A scipy interp1d function that returns the gas velocity.
    velocity_dust
        A list of scipy interp1d functions that returns the dust
        velocity per species.
    """
    x_L = x_shock - x_width / 2
    x_R = x_shock + x_width / 2

    _dust_to_gas = dust_to_gas * np.ones(n_dust)
    _mach_number = mach_number * np.ones(n_dust)
    _drag_coefficient = drag_coefficient * np.ones(n_dust)
    _density_left = density_left * np.ones(n_dust)
    _velocity_left = velocity_left * np.ones(n_dust)

    position = np.linspace(x_L, x_R, n_points)

    wd_R = _velocity_dust_normalized(
        position[n_points // 2 :],
        _dust_to_gas,
        _mach_number,
        _drag_coefficient,
        _density_left,
        _velocity_left,
    )
    wg_R = np.array(
        [_velocity_gas_normalized(_dust_to_gas, _mach_number, _wd_R) for _wd_R in wd_R]
    )

    wg = np.ones((2 * wg_R.shape[0], wg_R.shape[1]))
    wd = np.ones((2 * wd_R.shape[0], wd_R.shape[1]))
    wg[n_points // 2 :, :] = wg_R
    wd[n_points // 2 :, :] = wd_R

    _velocity_gas = velocity_left * wg[:, 0]
    _velocity_dust = velocity_left * wd

    velocity_gas = interp1d(position, _velocity_gas)
    velocity_dust = [interp1d(position, vd) for vd in _velocity_dust.T]

    return velocity_gas, velocity_dust


def density(
    x_shock: float,
    x_width: float,
    n_dust: int,
    dust_to_gas: float,
    drag_coefficient: ndarray,
    density_left: float,
    velocity_left: float,
    mach_number: float,
    n_points: int = 500,
):
    """Exact solution for dusty shock density.

    Parameters
    ----------
    x_shock
        The x-position of the shock.
    x_width
        The width in x over which to compute the solution.
    n_dust
        The number of dust species.
    dust_to_gas
        The dust-to-gas ratio. Same for all species.
    drag_coefficient
        An array of drag-coefficients, one per dust species.
    density_left
        The density on the left. Same for gas and all dust species.
    velocity_left
        The velocity on the left. Same for gas and all dust species.
    mach_number
        The mach number.
    n_points
        The number of points to use in computing the solution. Default
        is 500.

    Returns
    -------
    density_gas
        A scipy interp1d function that returns the gas density.
    density_dust
        A list of scipy interp1d functions that returns the dust
        density per species.
    """
    v_gas, v_dusts = velocity(
        x_shock,
        x_width,
        n_dust,
        dust_to_gas,
        drag_coefficient,
        density_left,
        velocity_left,
        mach_number,
        n_points,
    )

    def density_gas(x):
        return density_left * velocity_left / v_gas(x)

    def outer_fn(v_dust):
        def inner_fn(x):
            return density_left * velocity_left / v_dust(x)

        return inner_fn

    density_dust = [outer_fn(v_dust) for v_dust in v_dusts]

    return density_gas, density_dust


def _derivative(
    t, y, dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left
):
    w_g = _velocity_gas_normalized(dust_to_gas, mach_number, y)
    K, rho_0, v_0 = drag_coefficient, density_left, velocity_left
    return K / (rho_0 * v_0) * (w_g - y)


def _velocity_gas_normalized(dust_to_gas, mach_number, velocity_dust):
    eps, M, v_d = dust_to_gas, mach_number, velocity_dust
    a, c = 1, M ** -2
    b = np.sum(eps * (v_d - 1)) - M ** -2 - 1
    return (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def _velocity_dust_normalized(
    x, dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left,
):
    t_eval = x
    t_span = x[0], x[-1]
    y0 = velocity_left / velocity_left
    args = (dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left)
    sol = solve_ivp(fun=_derivative, t_span=t_span, y0=y0, t_eval=t_eval, args=args)
    return sol.y.T
