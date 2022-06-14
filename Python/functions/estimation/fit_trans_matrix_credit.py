#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import expm

from arpym.views.min_rel_entropy_sp import min_rel_entropy_sp

def fit_trans_matrix_credit(dates, n_oblig, n_cum, tau_hl=None):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=fit_trans_matrix_credit).

    Parameters
    ----------
        dates : array, shape(t_,)
        n_oblig : array, shape (t_, c_)
        n_cum : array, shape (t_, c_, c_)
        tau_hl : scalar, optional

    Returns
    -------
        p : array, shape (c_, c_)

    """
    t_ = len(dates)
    c_ = n_oblig[-1].shape[0]
    delta_t = np.zeros(t_-1)

    num = np.zeros((c_, c_))
    den = np.zeros((c_, c_))
    g = np.zeros((c_, c_))
    
    # Step 1: Compute number of transitions at each time t
    
    m_num = np.zeros(n_cum.shape)
    m_num[0, :, :] = n_cum[0, :, :]
    m_num[1:, :, :] = np.diff(n_cum, axis=0)    # number of transitions at each time t≤t
    
    # Step 2: Esimate prior transition matrix
    
    for i in range(c_):
        for j in range(c_):
            if i != j:
                if tau_hl is None:
                    num[i, j] = n_cum[-1, i, j]
                    for t in range(1, t_):
                        den[i, j] = den[i, j] + n_oblig[t, i]*(np.busday_count(dates[t-1], dates[t]))/252
                    g[i, j] = num[i, j]/den[i, j]      # off-diagonal elements of g given tau_hl=None
                else:
                    for t in range(t_):
                        num[i, j] = num[i, j] + m_num[t, i, j]*np.exp(-(np.log(2)/tau_hl)*(np.busday_count(dates[t], dates[-1]))/252)
                    for t in range(1, t_):
                        den[i, j] = den[i, j] + n_oblig[t-1, i]*(np.exp(-(np.log(2)/tau_hl)*(np.busday_count(dates[t], dates[-1]))/252)
                                                                -np.exp(-(np.log(2)/tau_hl)*(np.busday_count(dates[t-1], dates[-1]))/252))
                    g[i, j] = (np.log(2)/tau_hl)*num[i, j]/den[i, j]    # off-diagonal elements of g given tau_hl

    for i in range(c_):
        g[i, i] = -np.sum(g[i, :])  # diagonal elements of g
    
    p_ = expm(g)    # prior transition matrix
    
    # Step 3: Minimize relative entropy
    
    # probability constraint
    a_eq = np.ones((1, c_))   # 1×c_ dimensional vector of ones
    b_eq = np.array([1])

    # initialize monotonicity constraint
    a_ineq = {}
    a_ineq[0] = np.diagflat(np.ones((1, c_-1)), 1) -\
                   np.diagflat(np.ones((1, c_)), 0)     # (c_-1)×c_ upper triangular matrix
    a_ineq[0] = a_ineq[0][:-1]
    b_ineq = np.zeros((c_-1))    # 1×(c_-1) dimensional vector of ones
    
    p = np.zeros((c_-1, c_))
    for c in range(c_-1):
        p[c, :] = min_rel_entropy_sp(p_[c, :],    # minimize relative entropy
                                 a_ineq[c], b_ineq, a_eq, b_eq, False)
        a_temp = a_ineq.get(c).copy()    # update monotonicity constraint
        a_temp[c, :] = -a_temp[c, :]
        a_ineq[c+1] = a_temp.copy()
    
    p = np.r_[p, np.array([np.r_[np.zeros(c_-1), 1]])]    # default constraint

    return p