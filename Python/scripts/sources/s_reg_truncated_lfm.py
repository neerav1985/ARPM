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

# # s_reg_truncated_lfm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_reg_truncated_lfm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-trunc-time).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

from arpym.statistics.meancov_sp import meancov_sp
from arpym.estimation.fit_lfm_ols import fit_lfm_ols
from arpym.estimation.cov_2_corr import cov_2_corr
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-parameters)

spot = np.array([0, 1, 9])  # targets and factors to spot
n_long = 61  # long index
n_short = np.array([366, 244])  # short indices

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step00): Load data

# +
path = '~/databases/global-databases/equities/db_stocks_SP500/'
data = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0, header=[0, 1],
                   parse_dates=True)
idx_sector = pd.read_csv(path + 'db_sector_idx.csv', index_col=0,
                         parse_dates=True)
idx_sector = idx_sector.drop("RealEstate", axis=1)  # delete RealEstate

dates = np.intersect1d(data.index, idx_sector.index)
data = data.loc[dates]
idx_sector = idx_sector.loc[dates]

t_ = len(data.index) - 1
n_ = len(data.columns)
k_ = len(idx_sector.columns)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step01): Compute linear returns of X and Z

v_stock = data.values  # stock values
x = (v_stock[1:, :] - v_stock[:-1, :]) / v_stock[:-1, :]  # linear return of the stock values
v_sector = idx_sector.values  # sector indices
z = (v_sector[1:, :] - v_sector[:-1, :]) / v_sector[:-1, :]  # linear return of the sector indices

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step02): Compute OLSFP estimates and residuals

alpha, beta, s2, eps = fit_lfm_ols(x, z) #  compute OLSFP estimates and residuals

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step03): Compute the joint covariance and correlation

# +
# compute covariance
[mu_epsz, sig2_epsz] = meancov_sp(np.hstack((eps, z)))  # compute covariance between ε and Z
sig2_eps = sig2_epsz[:n_, :n_]  # variance of ε
sig2_z = sig2_epsz[n_:, n_:]  # variance of Z

# compute correlation
c2_epsz, _ = cov_2_corr(sig2_epsz)  #  compute correlation between ε and Z
c_epsz = c2_epsz[:n_, n_:]  
c2_eps = np.tril(c2_epsz[:n_, :n_], -1)  # correlation among residuals
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_reg_truncated_lfm-implementation-step04): Compute standard deviations of two portfolios

# +
w_1 = np.ones(n_) / n_  # equal weight portfolio
w_2 = np.zeros(n_)  # long/short weight portfolio
w_2[n_long] = 0.69158715  # long weight portfolio
w_2[n_short] = np.array([-0.67752045, -0.01406671])  # short weight portfolio

_, sig2_x = meancov_sp(x)  # compute historical covariance of Xt
sig2_x_trunc = beta @ sig2_z @ beta.T + np.diag(np.diag(sig2_eps))  # truncated target covariance of Xt

std_1 = np.sqrt(w_1.T @ sig2_x @ w_1)  # standard deviation of the equal weight portfolio from sig2_x
std_trunc_1 = np.sqrt(w_1.T @ sig2_x_trunc @ w_1)  # standard deviation of the euqal weight portfolio from sig2_x_trunc

std_2 = np.sqrt(w_2.T @ sig2_x @ w_2)  # standard deviation of the long/short weight portfolio from sig2_x
std_trunc_2 = np.sqrt(w_2.T @ sig2_x_trunc @ w_2)  # standard deviation of the long/short weight portfolio from sig2_x_trunc
# -

# ## Plots

# +
# (untruncated) correlations among residuals
corr_eps = c2_eps[np.nonzero(c2_eps)]  # reshape the correlations
n, xout = histogram_sp(corr_eps)

mydpi = 72.0
fig = plt.figure(figsize=(1280.0/mydpi,720.0/mydpi),dpi=mydpi)
ax0 = plt.axes([0.595, 0.83, 0.92, 0.45])
ax0.plot(corr_eps.mean(),0,'ro')
plt.xlim(-0.6, 1.6)
plt.ylim(0, 7)
h = ax0.bar(xout, n, width=xout[1]-xout[0], facecolor=[.7, .7, .7], edgecolor='k')
plt.text(0.24, 6.2, r'$\mathbb{C}$' + r'$r$' + r'$\{\.ε_m\, \.ε_n\}$',
         fontsize=20)
plt.xlabel(r'Correlation values', fontsize=17)
plt.ylabel(r'Frequencies', fontsize=17)
ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Cross correlations in regression LFM')

c2_x, _ = cov_2_corr(sig2_x)
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
# -


