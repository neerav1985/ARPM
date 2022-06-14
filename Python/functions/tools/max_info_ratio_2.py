# -*- coding: utf-8 -*-
import numpy as np

from arpym.statistics.simulate_normal import simulate_normal


def max_info_ratio_2(mu_pi, mu_s, sig2):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=max_info_ratio_2).

    Parameters
    ----------
        mu_x : array, shape(n_,)
        mu_s : array, shape(m_)
        sig2 : array, shape(n_+m_, n_+m_ )

    Returns
    -------
        max_ir2_mean : scalar

    """

    n_ = len(mu_pi)
    sig2_pi = sig2[:n_, :n_]
    sig2_s = sig2[n_:, n_:]
    sig_pi_s = sig2[:n_, n_:]
    
    # Step 1: Monte Carlo scenarios for the signals
    
    j_ = 1000
    s_j = simulate_normal(mu_s, sig2_s, j_)
    
    # Step 2: Maximum conditional information ratio
    
    max_ir2_j = np.zeros(j_)
    for j in range(j_):
        # conditional moments
        mu_pi_cond_j = mu_pi + sig_pi_s@np.linalg.solve(sig2_s,(s_j[j] - mu_s).T)
        sig2_pi_cond_j = sig2_pi - sig_pi_s @np.linalg.solve(sig2_s, sig_pi_s.T)
        # maximum conditional information ratio
        max_ir2_j[j] = mu_pi_cond_j.T @ np.linalg.solve(sig2_pi_cond_j, mu_pi_cond_j)
    
    # Step 3: Maximum (l2-mean unconditional) information ratio
    
    max_ir2_mean = np.sum(max_ir2_j)/j_
    
    return max_ir2_mean  
