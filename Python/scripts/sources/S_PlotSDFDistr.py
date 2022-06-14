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

# # S_PlotSDFDistr [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PlotSDFDistr&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sdfcomparison).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, ones, zeros, diag, eye, exp, sqrt, tile, diagflat
from numpy import sum as npsum, min as npmin, max as npmax
from numpy.linalg import solve
from numpy.random import multivariate_normal as mvnrnd

from scipy.stats import norm, uniform

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim

plt.style.use('seaborn')

from ARPM_utils import save_plot
from SDFkern import SDFkern

# parameter

# parameters
n_ = 250
j_ = 500
r = 0.05
a_p = 0.7
b_p = 1
a_sdf = 0
b_sdf = 0.9
rho = 0.7
# -

# ## Generate the payoff matrix

# +
# Generate the normal vector
c2 = rho*ones((n_, n_)) + (1 - rho)*eye(n_)  # correlation matrix
x = mvnrnd(zeros(n_), c2, j_).T

# Generate the payoffs
v_pay = ones((n_, j_))
v_pay[1] = exp(x[1]) / (sqrt(exp(1) - 1)*exp(0.5))
v_pay[2::2,:] = (exp(x[2::2,:])-exp(0.5) / (sqrt(exp(1) - 1))*exp(0.5))
v_pay[3::2,:] = (-exp(-x[3::2,:])+exp(0.5) / (sqrt(exp(1) - 1))*exp(0.5))
v_pay[2:,:] = diagflat(uniform.rvs(loc=0.8, scale=0.2, size=(n_ - 2, 1)))@v_pay[2:,:]  # rescaling
v_pay[2:,:] = v_pay[2:,:]+tile(uniform.rvs(loc=-0.3, scale=1, size=(n_ - 2, 1)), (1, j_))  # shift
# -

# ## Compute the probabilities

p = uniform.rvs(loc=a_p, scale=b_p-a_p, size=(j_, 1))
p = p /npsum(p)

# ## Compute the "true" Stochastic Discount Factor vector of scenarios

sdf_true = uniform.rvs(loc=a_sdf, scale=b_sdf-a_sdf, size=(1, j_))
c = 1 / ((sdf_true@p)*(1 + r))
sdf_true = sdf_true*c  # constraint on the expectation of SDF

# ## Compute the current values vector

v = v_pay@diagflat(p)@sdf_true.T

# ## Compute the projection Stochastic Discount Factor

sdf_proj = v.T@(solve(v_pay@diagflat(p)@v_pay.T,v_pay))

# ## Compute the Kernel Stochastic Discount Factor

sdf_ker = SDFkern(v_pay, v, p)

# ## Generate the figure

# +
# Compute the gaussian smoothed histograms
bw = 0.1  # band-width
x = arange(npmin(sdf_true) - 5*bw,npmax(sdf_true) + 5*bw,0.01)

# Gaussian smoothings
Y = tile(x, (len(sdf_true), 1)) - tile(sdf_true.T, (1, len(x)))
sdf_true = p.T@norm.pdf(Y, 0, bw)
Y = tile(x, (len(sdf_proj), 1)) - tile(sdf_proj.T, (1, len(x)))
sdf_proj = p.T@norm.pdf(Y, 0, bw)
Y = tile(x, (len(sdf_ker), 1)) - tile(sdf_ker.T, (1, len(x)))
sdf_ker = p.T@norm.pdf(Y, 0, bw)

figure()
plot(x, sdf_true[0])
plot(x, sdf_proj[0], 'g')
plot(x, sdf_ker[0], 'm')
yl = ylim()
plot([v[0], v[0]], [0, yl[1]], 'k--')
ylim(yl)
xlim([x[0], x[-1]])
legend(['True SDF','Proj SDF','Kern SDF','Risk Free']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
