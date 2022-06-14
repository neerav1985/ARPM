# -*- coding: utf-8 -*-

import numpy as np


def forward_selection(optim, n_):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-forward-selection).

    Parameters
    ----------
        optim : function
        n_ : int

    Returns
    -------
        x_fwd : array, shape(n_, n_)
        f_x_fwd : array, shape(n_,)
        s_k_fwd_list : list
    """

    i_1 = np.arange(1, n_+1)
    x_fwd = []
    f_x_fwd = np.ones(n_)*np.nan

    # Step 0: Initialize
    s_k_fwd = []
    s_k_fwd_list = []

    for k in range(n_):
        # Step 1: Build k-element set of selections
        s_prev_fwd = s_k_fwd
        i_k_fwd = []
        x_k = []
        f_x_k = []
        for n in np.setdiff1d(i_1, s_prev_fwd):
            i_k_fwd.append(np.union1d(s_prev_fwd, n).astype(int))

            # Step 2: Optimize over constraint set
            all = optim(i_k_fwd[-1])
            x_k.append(all[0])
            f_x_k.append(all[1])

        # Step 3: Perform light-touch search
        opt_indices = np.argmin(f_x_k)
        x_fwd.append(x_k[opt_indices])
        f_x_fwd[k] = f_x_k[opt_indices]
        s_k_fwd = i_k_fwd[opt_indices]
        s_k_fwd_list.append(s_k_fwd)

    return np.array(x_fwd), f_x_fwd, s_k_fwd_list
