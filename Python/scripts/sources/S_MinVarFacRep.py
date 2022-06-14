#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # S_MinVarFacRep [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_MinVarFacRep&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-optim-pseudo-inv-lo).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, ones, zeros
from numpy.linalg import solve
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot

# input parameters
n_ = 100  # max market dimension
nstep = arange(5,n_+1)  # grid of market dimensions
s2_z_ = array([[1]])  # variance of factor

stepsize = len(nstep)
s2_p_z_mv = zeros((stepsize, 1))
s2_p_z = zeros((stepsize, 1))

for n in range(stepsize):  # set covariance of the residuals
    d = rand(nstep[n], 1)
    s2_u = np.diagflat(d * d)

    # ## Compute the low-rank-diagonal covariance of the market
    beta = rand(nstep[n], 1)  # loadings
    s2_p = beta@s2_z_@beta.T + s2_u

    # ## Compute the pseudo inverse of beta associated with the inverse covariance of the P&L's
    sig2_mv = np.diagflat(1 / (d * d))
    betap_mv = solve(beta.T@sig2_mv@beta,beta.T@sig2_mv)
    # NOTE: betap_mv does not change if we set sig2_mv = inv(s2_p)

    # ## Compute an arbitrary pseudo inverse of beta
    sig = rand(nstep[n], nstep[n])
    sig2 = sig@sig.T
    betap = solve(beta.T@sig2@beta,beta.T@sig2)

    # ## Compute the variances of the factor-replicating portfolio P&L
    s2_p_z_mv[n] = betap_mv@s2_p@betap_mv.T
    s2_p_z[n] = betap@s2_p@betap.T  # ## Plot the variances for each market dimension

figure()

plot(nstep, s2_p_z_mv, 'b', linewidth=1.5, markersize=2)
plot(nstep, s2_p_z, color= [.9, .3, 0], lw= 1.5, markersize=2)
plot(nstep, s2_z_[0]*ones(stepsize), color= [.5, .5, .5], lw= 1.5, markersize=2)
plt.tight_layout()
xlabel(r'$\bar{n}$')
ylabel('variance')
title('Minimum variance factor-replicating portfolio')
h = legend(['$\sigma^2_{\Pi^{MV}_Z}$', '$\sigma^2_{\Pi_Z}$', '$\sigma^2_{Z}$']);
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
