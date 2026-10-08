# mocaxextendpy - Python implementation of Tensor Extension Algorithms
# Copyright 2021 iRuiz Technologies, ltd.
# See LICENSE.txt for details on the End User License Agreement.

import numpy as np


def cheb_points(num_pts):
    """
    Check if number is not greater than one return error.

    Parameters
    ----------
    num_pts : int
        it must be greater than 1.

    Returns
    -------
    value : val
        It returns a value.

    Raises
    ------
    ValueError : size mismatch
        Number of Chebyshev points passed, must be greater than 1
    """
    if int(num_pts) > 1:

        # mocaxpts = cos(num_pts*pi/(num_pts))
        # np.sin( (np.pi*(num_pts - 2 * np.arange(num_pts + 1))) / (2 * num_pts) )
        return np.cos((np.arange(num_pts + 1) * np.pi) / num_pts)
    else:
        raise ValueError(
            "Number of Chebyshev points passed {}, must be a greater than 1".format(num_pts))


def parametrisation(a, b, t):
    """
    Parametrisation used to linearly transform interval [0,1] to [a,b].

    Parameters
    ----------
    a : int
        Range interval value a (a, b)

    b : int
        Range interval value b (a, b)

    t : int
        it must be greater than 1.

    Returns
    -------
    value : val
        It returns a value.

    Raises
    ------
    ValueError :  interval
         Parameter t must be greater zero and less than one
    """

    if (0 <= t).all() and (t <= 1).all():
        value = t * b + (1 - t) * a
        return value
    else:
        raise ValueError(
            "Parameter t must be greater than or equal to zero and less than or equal to one")


def cheb_pts_interval(n, a, b):
    """
    Check if entry a in interval is less than entry b in interval, else return error.

    Parameters
    ----------
    n : int
        Number of cheb points to generate.

    a : int
        Lower end of interval [a, b]

    b : int
        Upper end of interval [a,b]

    Returns
    -------
    cheb_anchor_pts : int
        It returns anchor points.

    Raises
    ------
    ValueError :
         Interval boundary point a must be a less than boundary pointinterval b.

    """

    if a < b:
        cheb_pts = cheb_points(n)
        t = (abs(cheb_pts - 1)) / 2
        cheb_anchor_pts = parametrisation(b, a, t)
        return cheb_anchor_pts
    else:
        raise ValueError(
            'Interval boundary point {} must be strictly less than {}'.format(a, b))
