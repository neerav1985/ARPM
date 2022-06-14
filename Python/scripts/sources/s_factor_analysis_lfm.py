#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_factor_analysis_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_factor_analysis_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmsys-id).

# +
import numpy as np

from arpym.estimation.cov_2_corr import cov_2_corr
from arpym.estimation.factor_analysis_paf import factor_analysis_paf
from arpym.tools.pca_cov import pca_cov
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-parameters)

mu_x = np.zeros(3)
sigma2_x  = np.array([[1.7,0.89,0.25],[0.89,2.22,0.48],[0.25,0.48,1.4]])    # target covariance
beta0 = np.array([1.54, 3.15, 3.62]).reshape(-1,1)    # initial loadings
k_ = 1    # dimension of hidden factor
o = 1    # rotatation parameter

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-implementation-step01): Compute scale matrix

alpha = mu_x    # shift

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-implementation-step02): Compute scale matrix

s_vol = cov_2_corr(sigma2_x)[1]    # vector of standard deviations
sigma = np.diag(s_vol)   # scale matrix

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-implementation-step03): Compute the factor loadings and idiosyncratic variances

beta, delta2 = factor_analysis_paf(sigma2_x, k_, sigma, beta0)    # loadings and idiosyncratic variances

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-implementation-step04): Rotate factor loadings and covariances of regression factor and fitted model

beta_ = - beta    # rotated loadings
c = beta_.T @ np.linalg.inv(sigma2_x)
sigma2_z = c @ sigma2_x @ c.T    # regression factor covariances
sigma2_x_si = beta @ beta.T + np.diag(delta2)    # fitted model covariance

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-implementation-step05): Compute r-squared

r2 = np.trace(np.linalg.lstsq(sigma, (np.linalg.lstsq(sigma,  beta_ @ beta_.T, rcond=None)[0] + np.diag(delta2)),  rcond=None)[0]) / np.trace(np.linalg.lstsq(sigma, (np.linalg.lstsq(sigma,  sigma2_x, rcond=None)[0]),  rcond=None)[0])    # r-sqaured

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-implementation-step06): Compute covariance of residuals

sigma2_e = sigma2_x - sigma2_x_si    # covariance of residuals

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_factor_analysis_lfm-implementation-step07): Compute the factor loadings and idiosyncratic variances in the setup of isotropic variances

e_k, lambda2_k = pca_cov(sigma2_x)    # eigenvalues and eigenvectors
delta2_epsi = lambda2_k[-1]    # idiosyncratic variances
beta_epsi = e_k[:,1].reshape(-1,1) @ np.sqrt(lambda2_k[0] -delta2_epsi).reshape(1,1)  # loadings  
