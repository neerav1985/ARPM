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

# # s_current_values [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_current_values&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-simcurval).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_current_values-parameters)

n_ = 250  # number of instruments
j_ = 1000  # number of scenarios
r = 0.05  # risk-free interest rate
a_sdf, b_sdf = 0, 0.9  # left/right boundaries of uniform distr for SDF
a_mu, b_mu = -0.3, 0.7  # left/right bounds of uniform distr. for payoff exp
a_sd, b_sd = 0.8, 1  # left/right boundaries of uniform distr. for payoff std
rho = 0.7  # parameter for correlation matrix

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_current_values-implementation-step01): generate the normal vector

# compute correlation matrix
c2 = (1 - rho)*np.eye(n_) + rho*np.ones((n_, n_))  
# simulations from normal distribution
x = simulate_normal(np.zeros(n_), c2, j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_current_values-implementation-step02): Generate the payoffs matrix

# +
mu_xn = np.exp(0.5)  # expectation of std lognormal
sd_xn = mu_xn * np.sqrt(np.exp(1) - 1)  # stdev of std lognormal

v_pay = (1 + r) * np.ones((j_, n_))
v_pay[:, 1] = np.exp(x[:, 1]) / sd_xn
v_pay[:, 2::2] = (np.exp(x[:, 2::2]) - mu_xn) / sd_xn
v_pay[:, 3::2] = (-np.exp(-x[:, 3::2]) + mu_xn) / sd_xn
v_pay[:, 2:] = v_pay[:, 2:] * \
                  np.random.uniform(a_sd, b_sd, n_ - 2)  # scale
v_pay[:, 2:] = v_pay[:, 2:] + \
                  np.random.uniform(a_mu, b_mu, n_ - 2)  # shift
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_current_values-implementation-step03): Compute the probabilities

p = np.random.uniform(0, 1, j_)
p = p / np.sum(p)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_current_values-implementation-step04): Simulate the Monte Carlo scenarios for the Stochastic Discount Factor

sdf_omega_j = np.random.uniform(a_sdf, b_sdf, j_)
c = 1 / ((1 + r)*(sdf_omega_j@p))
sdf_omega_j = c * sdf_omega_j  # constraint on the expectation of SDF

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_current_values-implementation-step05): Compute the current values

v = np.zeros(n_)
for n in range(n_):
    v[n] = np.sum(p*sdf_omega_j*v_pay[:, n])

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_current_values-implementation-step06): Save databases

out = pd.DataFrame(v_pay)
out.to_csv('~/databases/temporary-databases/db_valuation_vpay.csv')
del out
out = pd.DataFrame(v)
out.to_csv('~/databases/temporary-databases/db_valuation_v.csv')
del out
out = pd.DataFrame(p)
out.to_csv('~/databases/temporary-databases/db_valuation_prob.csv')
del out

# ## Plots

# +
# rearrange rows by v
ind_row = np.argsort(v)[::-1]

# rearrange columns by average values
ind_col = np.argsort(np.mean(v_pay, axis=1))

# heatmaps of V and v
xstep = 200  # step of the x-axis
ystep = 50  # step of the y-axis

plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax = plt.subplot2grid((1, 11), (0, 0), colspan=7)
plt.imshow(v_pay[np.ix_(ind_col, ind_row)].T, vmin=-0.5, vmax=1.5,
           cmap=plt.get_cmap('jet'), aspect='auto')
ax.xaxis.get_major_ticks()[0].set_visible(False)
ax.yaxis.get_major_ticks()[0].set_visible(False)
plt.grid(False)
plt.xlabel('scenario')
plt.ylabel('instrument')
plt.title('Future payoff')

ax = plt.subplot2grid((1, 11), (0, 8))
plt.imshow(v[ind_row].reshape(-1, 1), vmin=-0.5, vmax=1.5,
           cmap=plt.get_cmap('jet'), aspect='auto')
plt.xticks([])

ax.yaxis.get_major_ticks()[0].set_visible(False)
plt.grid(False)
plt.ylabel('instrument')
plt.title('Current value')

cbar = np.arange(1.5, -0.51, -0.01).reshape(-1, 1)
ax = plt.subplot2grid((1, 11), (0, 10))
plt.imshow(cbar, cmap=plt.get_cmap('jet'), aspect='auto')
plt.xticks([])
plt.yticks(np.arange(0, 220, 20),
           np.array([1.5, 1.3, 1.1, 0.9, 0.7, 0.5,
                     0.3, 0.1, -0.1, -0.3, -0.5]))
plt.grid(False)
plt.title('Scale')
add_logo(fig, size_frac_x=3/4, location=1, alpha=0.8, set_fig_size=False)
