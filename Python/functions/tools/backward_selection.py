# -*- coding: utf-8 -*-

import numpy as np


def backward_selection(optim, n_):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-backward-selection).

    Parameters
    ----------
        optim : function
        n_ : int

    Returns
    -------
        x_bwd : array, shape(n_, n_)
        f_x_bwd : array, shape(n_,)
        s_k_bwd_list : list
    """

    x_bwd = []
    f_x_bwd = np.ones(n_)*np.nan

    # Step 0: Initialize
    s_k_bwd = np.arange(1, n_+1)
    s_k_bwd_list = []
    i_k_bwd = [s_k_bwd]

    for k in range(n_, 0, -1):
        x_k = []
        f_x_k = []
        for s_k_i in i_k_bwd:
            # Step 1: Optimize over constraint set
            all = optim(s_k_i)
            x_k.append(all[0])
            f_x_k.append(all[1])

        # Step 2: Perform light-touch search
        opt_indices = np.argmin(f_x_k)
        x_bwd.append(x_k[opt_indices])
        f_x_bwd[k-1] = f_x_k[opt_indices]
        s_k_bwd = i_k_bwd[opt_indices]
        s_k_bwd_list.append(s_k_bwd)

        # Step 3: Build (k-1)-element set of selections
        i_k_bwd = []
        for n in s_k_bwd:
            i_k_bwd.append(np.setdiff1d(s_k_bwd, n).astype(int))

    x_bwd = np.array(x_bwd)
    return np.array(np.squeeze(x_bwd)[::-1]), f_x_bwd, s_k_bwd_list
