#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from arpym.estimation.fit_lfm_ols import fit_lfm_ols
from arpym.statistics.meancov_sp import meancov_sp
from arpym.estimation.factor_analysis_mlf import factor_analysis_mlf
from arpym.statistics.kalman_filter import kalman_filter


def fit_state_space(x, k_, p=None):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-dyn-pcfun).

    Parameters
    ----------
        x : array, shape (t_, n_) if n_>1 or (t_, ) for n_=1
        p : array, shape (t_,)
        k_ : scalar

    Returns
    -------
        h : array, shape (t_, k_) if k_>1 or (t_, ) for k_=1
        alpha_hat : array, shape (n_,)
        beta_hat : array, shape (n_, k_) if k_>1 or (n_, ) for k_=1
        delta2_hat : array, shape(n_, n_)
        alpha_hat_h : array, shape(k_,)
        beta_hat_h : array, shape(k_, k_)
        sigma2_hat_h : array, shape(k_, k_)

    """
    
    t_= x.shape[0]
    if len(x.shape) == 1:
        x = x.reshape((t_, 1))
    
    if p is None:
        p = np.ones(t_) / t_  # equal probabilities as default value

    # Step 1: Compute target HFP mean and covariance
    
    m_x_hat_hfp, s2_x_hat_hfp = meancov_sp(x, p)  # FP mean and covariance
    
    # Step 2: Compute shifts
    
    alpha_hat = m_x_hat_hfp  # shift
    
    # Step 3: Perform FA
    
    beta_hat, delta2_hat = factor_analysis_mlf(s2_x_hat_hfp, k_)  # loadings and idiosyncratic variances
    
    if len(beta_hat.shape) == 0:
        beta_hat = beta_hat.reshape(-1,1)

    # Step 4: Compute regression factors
    
    h_fa = beta_hat.T@np.linalg.inv(s2_x_hat_hfp)@(x-m_x_hat_hfp).T  # regression factors
        
    # Step 5: Estimate regression LFM for H_t

    alpha_hat_h, beta_hat_h, sigma2_hat_h, _ = fit_lfm_ols(h_fa.T[1:, ], h_fa.T[:-1, ], p[:-1])
    alpha_hat_h, beta_hat_h, sigma2_hat_h =\
        np.atleast_1d(alpha_hat_h), np.atleast_2d(beta_hat_h), np.atleast_2d(sigma2_hat_h)  # parameters of the transition equation

    # Step 6: Unearth factors

    h = kalman_filter(x, alpha_hat, beta_hat, np.diagflat(delta2_hat),
                      alpha_hat_h, beta_hat_h, sigma2_hat_h)  # hidden factors
                      
    return h, np.squeeze(alpha_hat), np.squeeze(beta_hat), np.squeeze(delta2_hat),\
        np.squeeze(alpha_hat_h), np.squeeze(beta_hat_h), np.squeeze(sigma2_hat_h)
