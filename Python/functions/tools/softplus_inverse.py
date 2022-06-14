# -*- coding: utf-8 -*-

import numpy  as np


def softplus_inverse(y, location = 0.0, dispersion = 1.0):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=softplus_inverse).

    Parameters
    ----------
    y: array, (arbitrary shape)
    location: float, optional
    dispersion: float, optional

    Returns
    ----------
    x : array (same shape as y)
    """

    # Step 1: apply the definition of softplus inverse using vectorization in numpy
    x = np.log(np.exp((y - location) / dispersion) - 1.0)

    return x
