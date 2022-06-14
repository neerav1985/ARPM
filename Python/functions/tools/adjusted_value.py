#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def adjusted_value(v, dates, cf_r, r, fwd=True):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=AdjusedValuesFunctionEB).

    Parameters
    ----------
    v : array, shape(t_,)
    dates : array, shape (t_,)
    cf_r : shape (k_,) optional
    r : shape (k_,) optional
    fwd : int, optional

    Returns
    -------
    v_tilde : array, shape (t_,)

    """

    v_tilde = np.copy(v)
    ind_r = []
    [ind_r.append(np.where(dates == x)[0][0]) for x in r]
    if fwd:
        # forward cash-flow-adjusted values
        if len(ind_r) > 0:
            for r_k, cf_rk in zip(ind_r, cf_r):
                v_tilde[r_k:] = v_tilde[r_k:] *\
                    (1 + cf_rk / (v[r_k] - cf_rk))
    else:
        # backward cash-flow-adjusted values
        if len(ind_r) > 0:
            for r_k, cf_rk in zip(ind_r, cf_r):
                v_tilde[:r_k] = v_tilde[:r_k] *\
                    (1 - cf_rk / v[r_k])

    return np.squeeze(v_tilde)
