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

# # s_pca_truncated_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pca_truncated_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-trunc-statistical).

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc, rcParams
from matplotlib.ticker import MaxNLocator

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from arpym.statistics.meancov_sp import meancov_sp
from arpym.estimation.cov_2_corr import cov_2_corr
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.pca_cov import pca_cov
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-parameters)

k_ = 10  # number of factors
n_plus = 10  # long position index
n_minus = 200  # short position index

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step00): Load the weekly time series of the stock values

path = '~/databases/global-databases/equities/db_stocks_SP500/'
data = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0, header=[0, 1],
                   parse_dates=True)
n_ = len(data.columns)-1
v = data.iloc[:, 1:n_+1].values

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step01): Compute linear returns of stocks

x = v[1:, :] / v[:-1, :] - 1  # linear returns
t_ = x.shape[0]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step02): Estimate expectation and covariance of X and define sigma matrix

m_x_hat, s2_x_hat = meancov_sp(x)  # HFP moments
sigma2 = np.diag(np.diag(s2_x_hat))  # scale matrix
sigma = np.sqrt(sigma2)
sigma_inv = np.diag(1/np.diag(sigma))

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step03): Compute principal component decomposition

e_hat, lambda2_hat = pca_cov(sigma_inv@s2_x_hat@sigma_inv)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step04): Estimate the loadings, the factor extraction matrix and shift

alpha_hat_pc = m_x_hat  # shift
beta_hat_pc = sigma@e_hat[:, :k_]  # loadings
gamma_hat_pc = e_hat[:, :k_].T@sigma_inv  # construction matrix

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step05): Compute the factor realizations and their expectation and covariance

z_hat_pc = (x - m_x_hat)@gamma_hat_pc.T  # factors
m_z_hat, s2_z_hat = meancov_sp(z_hat_pc)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step06): Compute the residuals and the joint sample covariance of residuals and factors

u = x - (alpha_hat_pc + z_hat_pc@beta_hat_pc.T)  # residuals
_, s2_uz_hat = meancov_sp((np.c_[u, z_hat_pc]))

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step07): Compute correlations among  residuals

c2_uz_hat, _ = cov_2_corr(s2_uz_hat)
c2_u_hat = c2_uz_hat[:n_, :n_]  # correlation among residuals

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step08): Compute the truncated covariance of the returns

s2_u_hat = s2_uz_hat[:n_, :n_]
s_u_hat = np.sqrt(np.diag(s2_u_hat))
s2_x_trunc = beta_hat_pc@s2_z_hat@beta_hat_pc.T +\
                    np.diag(np.diag(s_u_hat))  # truncated covariance

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step09): Estimate the standard deviations of the portfolio returns using the sample covariance and the truncated covariance

# +
w1 = 1 / n_*np.ones((n_, 1))  # equal-weights portfolio

w2 = np.zeros((n_, 1))  # long-short portfolio
w2[n_plus] = 2
w2[n_minus] = -1

# HFP std of equal-weights portfolio
s_1_hat = np.sqrt(w1.T@s2_x_hat@w1)
# truncated std of equal-weights portfolio
s_1_trunc = np.sqrt(w1.T@s2_x_trunc@w1)

# HFP std of long-short portfolio
s_2_hat = np.sqrt(w2.T@s2_x_hat@w2)
# truncated std of long-short portfolio
s_2_trunc = np.sqrt(w2.T@s2_x_trunc@w2)
# -

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_pca_truncated_lfm-implementation-step10): Define data used for ploting of the histogram

[f_l, xi_l] = histogram_sp(c2_u_hat[np.triu_indices(c2_u_hat.shape[0],
                           1)])

# ## Plots

# +
# Figure specifications
plt.style.use('arpm')

# Histogram: correlations among residuals

mydpi = 72.0
fig = plt.figure(figsize=(1280.0/mydpi,720.0/mydpi),dpi=mydpi)
ax0 = plt.axes([0.595, 0.83, 0.92, 0.45])
ax0.plot(c2_u_hat.mean(),0,'ro')
plt.xlim(-0.6, 1.6)
plt.ylim(0, 9)
h = plt.bar(xi_l, f_l, width=xi_l[1]-xi_l[0],
            facecolor=[.7, .7, .7],
            edgecolor='k')
plt.text(0.3, 6.5, r'$\mathbb{C}$' + r'$r$' + r'$\{U_m, U_n\}$',
         fontsize=20)
plt.xlabel(r'Correlation values', fontsize=17)
plt.ylabel(r'Frequencies', fontsize=17)
ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

c2_x, _ = cov_2_corr(s2_x_hat)
c2_x = np.tril(c2_x[:n_, :n_], -1)
corr_x = c2_x[np.nonzero(c2_x)]  # reshape the correlations
n, xout = histogram_sp(corr_x)

ax1 = plt.axes([0.595, 0.3, 0.92, 0.45])
plt.xlim(-0.6, 1.6)
plt.ylim(0, 4)
ax1.plot(corr_x.mean(),0,'ro')
ax1.axes.get_xaxis().set_ticks([])
h1 = ax1.bar(xout, n, width=xout[1]-xout[0], facecolor=[.7, .7, .7], edgecolor='k')
plt.ylabel(r'Frequencies', fontsize=17)
plt.gca().invert_yaxis()
plt.text(0.6, 3.8, r'$\mathbb{C}$' + r'$r$' + r'$\{X_{m,t}, X_{n,t}\}$',
         fontsize=20)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

add_logo(fig, location=4)
plt.tight_layout()
# -


