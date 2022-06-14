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

# # S_FundTheoAPriskneuNum [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FundTheoAPriskneuNum&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ftheoasrnm).

# ## Prepare the environment

# +
import os.path as path
import sys, os

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import ones, zeros, diag, eye, exp, sqrt, tile, r_
from numpy import sum as npsum, min as npmin, max as npmax
from numpy.random import multivariate_normal as mvnrnd

from scipy.stats import uniform

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, ylabel, \
    xlabel, scatter

plt.style.use('seaborn')

from MREprobandSDF import MREprobandSDF
from SDFkern import SDFkern
from ARPM_utils import save_plot

# parameters
n_ = 250
j_ = 1000
r = 0.05
a_p = 0.7
b_p = 1
a_sdf = 0
b_sdf = 0.9
rho = 0.7
# -

# ## Simulate the payoff matrix

# +
# Generate the normal vector
c2 = rho*ones((n_, n_)) + (1 - rho)*eye(n_)  # correlation matrix
X = mvnrnd(zeros(n_), c2, j_).T

# Generate the payoffs
v_pay = ones((n_, j_))
v_pay[1] = exp(X[1]) / (sqrt(exp(1) - 1)*exp(0.5))
v_pay[2::2,:] = (exp(X[2::2,:])-exp(0.5)) / (sqrt(exp(1) - 1)*exp(0.5))
v_pay[3::2,:] = (-exp(-X[3::2,:])+exp(0.5))/(sqrt(exp(1) - 1)*exp(0.5))
v_pay[2:,:] = np.diagflat(uniform.rvs(loc=0.8, scale=0.2, size=(n_ - 2, 1)))@v_pay[2:,:]  # rescaling
v_pay[2:,:] = v_pay[2:,:]+tile(uniform.rvs(loc=-0.3, scale=1, size=(n_ - 2, 1)), (1, j_))  # shift
# -

# ## Compute the probabilities

p = uniform.rvs(loc=a_p, scale=b_p-a_p, size=(j_, 1))
p = p /npsum(p)

# ## Simulate the "true" Stochastic Discount Factor vector of Scenarios

sdf_true = uniform.rvs(loc=a_sdf, scale=b_sdf-a_sdf, size=(j_, 1))
c = 1 / ((sdf_true.T@p)*(1 + r))
sdf_true = sdf_true@c  # constraint on sdf expectation

# ## Compute the current values vector

v = v_pay@(sdf_true * p)

# ## Compute the kernel Stochastic Discount Factor

sdf_ker = SDFkern(v_pay, v, p)

# ## Compute the minimum relative entropy Stochastic Discount Factor

sdf_mre, p_mre = MREprobandSDF(v_pay, v, p.T, 1)

# ## Compute the risk neutral probabilities using the Stochastic Discount Factors found at the previous steps

q_true = sdf_true.T@np.diagflat(p) / v[0]
q_ker = sdf_ker@np.diagflat(p) / v[0]
q_mre = sdf_mre@np.diagflat(p) / v[0]

# ## For each instrument in the market and for each risk neutral probability found at the previous step, compute the left-hand side and the right-hand side of the fundamental theorem of asset pricing

y = v / v[0]
x = r_['-1', v_pay@q_true.T, v_pay@q_ker.T, v_pay@q_mre.T]

# ## Generate the figure

pick = range(50)  # We just pick first 50 dots to make the figure more
figure()
plot([npmin(y[pick]), npmax(y[pick])], [npmin(y[pick]), npmax(y[pick])], lw=1)
scatter(np.array(y[pick]), np.array(x[pick, 0]), marker='x',
        s=50, color=[1, 0.3, 0], lw=1)
scatter(np.array(y[pick]), np.array(x[pick, 1]), marker='o',
        s=70, color=[0.4, 0.4, 0], facecolor="none")
scatter(np.array(y[pick]), np.array(x[pick, 2]), marker='.',
        s=30, color=[0.5, 0, 1])
xlabel('r. h. side')
ylabel('l. h. side')
legend(['45$^o$ line','True prob.','Kern prob.','MRE prob.']);
# # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

