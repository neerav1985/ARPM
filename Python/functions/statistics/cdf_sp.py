# -*- coding: utf-8 -*-

import numpy as np
from bisect import bisect_right


def cdf_sp(x_, x, p=None, method=None):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=cdf_sp).

    Parameters
    ----------
        x_: scalar, array, shape(k_,)
        x : array, shape (j_,)
        p : array, shape (j_,), optional
        method: string, optional

    Returns
    -------
        cdf : array, shape (k_,)

    """

    x_ = np.atleast_1d(x_)
    j_ = x.shape[0]
    k_ = x_.shape[0]

    # Step 1: Sorted scenarios-probabilities

    if p is None:
        p = np.ones(j_) / j_  # equal probabilities as default value

    sort_x = np.argsort(x)
    x_sort = x[sort_x]
    p_sort = p[sort_x]

    # Step 2: Cumulative sums of sorted probabilities

    u_sort = np.zeros(j_ + 1)
    for j in range(1, j_ + 1):
        u_sort[j] = np.sum(p_sort[:j])

    # Step 3: Output cdf

    cdf = np.zeros(k_)

    if method == "linear_interp":
        x_0 = x_sort[0]-(x_sort[1]-x_sort[0])*u_sort[1]/(u_sort[2]-u_sort[1])
        x_sort = np.append(x_0, x_sort)
        cindx = [0]*k_
        for k in range(k_):
            cindx[k] = bisect_right(x_sort, x_[k])
        for k in range(k_):
            if cindx[k] == 0:
                cdf[k] = 0
            elif cindx[k] == j_+1:
                cdf[k] = 1
            else:
                cdf[k] = u_sort[cindx[k]-1] + \
                       (u_sort[cindx[k]] - u_sort[cindx[k]-1]) *\
                       (x_[k]-x_sort[cindx[k]-1]) /\
                       (x_sort[cindx[k]]-x_sort[cindx[k]-1])

    else:
        cindx = [0]*k_
        for k in range(k_):
            cindx[k] = bisect_right(x_sort, x_[k])
        cdf = u_sort[cindx]

    return np.squeeze(cdf)
