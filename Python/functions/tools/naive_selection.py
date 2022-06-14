# -*- coding: utf-8 -*-

import numpy as np


def naive_selection(optim, n_):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-naive-selection).

    Parameters
    ----------
        optim : function
        n_ : int

    Returns
    -------
        x_naive : array, shape(n_, n_)
        f_x_naive : array, shape(n_,)
        s_k_naive_list : list
    """

    # Step 1: Optimize over 1-element selections
    all = np.array([optim(np.array([n])) for n in range(1, n_+1)])
    x1, f_x1 = all[:, 0], all[:, 1]

    # Step 2: Naive sorting
    n_sort = np.argsort(f_x1)

    # Step 3: Initialize selection
    s_k_naive = []
    s_k_naive_list = []

    x_naive = []
    f_x_naive = np.ones(n_)*np.nan
    for k in range(n_):
        # Step 4: Build naive selection (set)
        s_k_naive = n_sort[:k+1] + 1
        s_k_naive_list.append(s_k_naive)

        # Step 5: Optimize over k-element selections
        all = optim(s_k_naive)
        x_naive.append(all[0]),
        f_x_naive[k] = all[1]

    return np.array(x_naive), f_x_naive, s_k_naive_list
