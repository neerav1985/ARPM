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

# # s_evaluation_certainty_equiv [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_evaluation_certainty_equiv&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBcertequivexputilfun).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools.histogram_sp import histogram_sp
from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.transpose_square_root import transpose_square_root
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-parameters)

j_ = 10**5  # number of scenarios
v_tnow = np.array([1, 1])  # current values
mu = np.array([0, 0])  # instruments P&L's expectations
h = np.array([45, 55])  # portfolio holdings
lambda_ = np.array([1/150, 1/200, 1/300])  # risk aversion parameters
rho = -0.5  # correlation parameter
# standard deviations appearing in the P&L's distributions
sig_11, sig_22 = 0.1, 0.3

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-implementation-step01): Scenarios for the instruments P&L's

# covariance parameter
sig2 = np.array([[(sig_11) ** 2, rho*sig_11*sig_22],
                [rho*sig_11*sig_22, (sig_22) ** 2]])
sig = transpose_square_root(sig2)
n_ = len(h)  # number of the instruments
# scenarios for standard normal random variable
z = simulate_normal(np.zeros(n_), np.eye(n_), j_)
pi = np.exp(np.array([mu]*j_) + z@sig) -  v_tnow  # P&L's scenarios
p = np.ones(j_)/j_   # flat scenario-probabilities

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-implementation-step02): Ex-ante performance scenarios

y_h = h@pi.T  # ex-ante performance scenarios
# number of bins for the ex-ante performance histogram
bins = np.round(150 * np.log(j_))
# centers and heights of the bins
heights, centers = histogram_sp(y_h, p=p, k_=bins)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-implementation-step03): Certainty-equivalent

# +


def ut(y, lam):  # exponential utility function
    return -np.exp(-lam * y)


def ut_inv(z, lam):  # inverse of exponential utility function 
    return - np.log(-z) / lam

# expected utility
expected_ut_y = np.array([])
for lam in lambda_:
    expected_ut_y = np.append(expected_ut_y, p@ut(y_h, lam))

# certainty-equivalent
ceq_y = ut_inv(expected_ut_y, lambda_)
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
# colors
gray = [.9, .9, .9]
color1 = [0.95, 0.35, 0]
color2 = [.3, .8, .8]
color3 = [.9, .7, .5]

heights_ = np.r_[heights[np.newaxis, :],
                 heights[np.newaxis, :]] / np.max(heights)
heights_[0, centers <= 0] = 0
heights_[1, centers > 0] = 0
width = centers[1] - centers[0]

# histograms of ex-ante performances
b = plt.bar(centers, heights_[0], width=width,
            facecolor=gray, edgecolor=color2)
b = plt.bar(centers, heights_[1], width=width,
            facecolor=gray, edgecolor=color3)
p1 = plt.plot([ceq_y[0], ceq_y[0]], [0, 0], color=color1, marker='.',
              markersize=8)
p2 = plt.plot([ceq_y[1], ceq_y[1]], [0, 0], color='b', marker='.',
              markersize=8)
p3 = plt.plot([ceq_y[2], ceq_y[2]], [0, 0], color='k', marker='.',
              markersize=8)
plt.legend(['$\lambda$ = ' +
            str(round(lambda_[0], 4)) +
            ' high risk aversion ', '$\lambda$ = ' +
            str(round(lambda_[1], 4)) +
            ' medium risk aversion ', '$\lambda$ = ' +
            str(round(lambda_[2], 4)) +
            ' low risk aversion '])
plt.ylim([-0.05, 1.05])
plt.ylabel('Certainty-equivalent ($)')
plt.xlabel('Portfolio P&L ($)')
plt.title(r'Market ex-ante P&L distribution ($\rho$=' +
          str(rho) + ', $\sigma$=' + str(sig_11) + ', '
          + str(sig_22) + ')')
add_logo(fig, location=4, alpha=0.8, set_fig_size=False)
