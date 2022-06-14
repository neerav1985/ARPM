# -*- coding: utf-8 -*-

import numpy  as np


def softplus(x, location = 0.0, dispersion=1.0):

    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=softplus).

        Parameters
        ----------
        x: array, (arbitrary shape)
        location: float, optional
        dispersion: float, optional (must be > 0.0)

        Returns
        ----------
        y : array (same shape as x)
        """

    # Step 1: apply the definition of softplus using vectorization
    y = location + dispersion * np.log(1 + np.exp(x))

    return y
