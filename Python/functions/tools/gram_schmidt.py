#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def gram_schmidt(sigma2, v=None):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-exer-cpca-copy-6).

    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        v      : array, shape (n_,n_), optional

    Returns
    -------
        w : array, shape (n_,n_)

    """
    n_ = sigma2.shape[0]

    # Step 0. Initialization
    w = np.empty_like(sigma2)
    p = np.zeros((n_, n_-1))
    if v is None:
        v = np.eye(n_)
      
    for n in range(n_):
        v_n = v[:, [n]]
        for m in range(n):

        # Step 1. Projection
            p[:, [m]] = (w[:, [m]].T @ sigma2 @ v_n) * w[:, [m]]

        # Step 2. Orthogonalization
        u_n = v_n - p[:, :n].sum(axis=1).reshape(-1, 1)

        # Step 3. Normalization
        w[:, [n]] = u_n/np.sqrt(u_n.T @ sigma2 @ u_n)

    return w
