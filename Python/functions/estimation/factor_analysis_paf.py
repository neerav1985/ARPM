#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# +
import numpy as np
from scipy import linalg
from random import gauss

from arpym.tools.pca_cov import pca_cov


# -

def factor_analysis_paf(cv_x, k_, sigma=None, beta_0=None, maxiter=100, eta=1e-2):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-estim-assess-copy-1).

    Parameters
    ----------
        cv_x : array, shape (n_, n_)
        k_ : integer
        sigma: array, shape (n_, n_)
        beta_0: array, shape (n_, k_)
        maxiter : integer, optional
        eta : float, optional

    Returns
    -------
        beta : array, shape (n_, k_) for k_ > 1 or (n_, ) for k_ = 1
        delta2 : array, shape (n_,)

    """
    n_ = cv_x.shape[0]
    
    if (beta_0 is None):
        beta_0 = np.zeros((n_, k_))
        for i in range(n_):
            for j in range(k_):
                beta_0[i, j] = np.random.normal(0, 1)
                
    if (sigma is None):
        sigma = np.eye(n_)
                   
    # Step 0: Intitalize matrix b
        
    beta = beta_0   # initialized loadings

    for i in range(maxiter):
    
        # Step 1: Compute scale
        
        gamma = np.zeros(n_)
        for n in range(n_):
            gamma[n] = min(1, np.sqrt(cv_x[n, n] / (beta @ beta.T)[n, n]))    # scale

        # Step 2: Rescale loadings
        
        beta = np.diag(gamma)@beta   # rescaled loadings
        
        # Step 3: Update variances
        
        delta2 = np.diag(cv_x - beta @ beta.T)    # updated idiosyncratic variances
        
        # Step 4: Perform PCA
        
        e_k, lambda2_k = pca_cov(np.linalg.lstsq(sigma, (np.linalg.lstsq(sigma, cv_x, rcond=None)[0] - np.diag(delta2)).T , rcond=None)[0], k_)    # eigenvalues and eigenvectors 

        # Step 5: Update loadings

        beta = sigma @ (e_k @ np.diag(np.sqrt(lambda2_k)))    # updated loadings

        # Step 6: Check convergence condition
        
        if np.trace(np.linalg.lstsq(sigma, (np.linalg.lstsq(sigma, cv_x, rcond=None)[0] - beta @ beta.T - np.diag(delta2)).T,  rcond=None)[0]) <= eta:    # convergence criteria
            break
            
    beta = beta    # optimal loadings
    delta2 = delta2    # optimal idiosyncratic variances
    
    return beta, delta2
