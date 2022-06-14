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

# # s_uncorr_no_indep [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_uncorr_no_indep&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-uncorr-vs-no-indep).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics.meancov_sp import meancov_sp
from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.logo import add_logo
from arpym.tools.pca_cov import pca_cov
from arpym.tools.plot_ellipse import plot_ellipse
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_uncorr_no_indep-parameters)

j_ = 5*10**4  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_uncorr_no_indep-implementation-step01): Generate simulations

# simulations of X
x = simulate_normal(0, 1, j_)
# simulations of Y
y = x ** 2

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_uncorr_no_indep-implementation-step02): Compute sample mean and covariance

# +
# sample expectation and covariance
mu_xy, sig2_xy = meancov_sp(np.c_[x, y])

sig2_x_y = sig2_xy[0,1]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_uncorr_no_indep-implementation-step03): Expectation-covariance ellipsoid computations

# generate ellipsoid
ellipse_muxy_sig2xy = plot_ellipse(mu_xy, sig2_xy, r=1, display_ellipse=False)

# ## Plots

plt.style.use('arpm')
dark_grey = [100/256, 100/256, 100/256]
fig = plt.figure()
# ellipse and scatter
plt.plot(ellipse_muxy_sig2xy[:, 0], ellipse_muxy_sig2xy[:, 1], c='k', lw=1, label='Expactation/covariance ellipsoid')
plt.scatter(x, y, facecolor='none', color=dark_grey, label='Scatter plot')
# axes of ellipse
e, lambda2 = pca_cov(sig2_xy)
diag_lambda = np.diagflat(np.sqrt(np.maximum(lambda2, 0)))
plt.plot([mu_xy[0] - diag_lambda[0][0]*e[0][0], mu_xy[0] + diag_lambda[0][0]*e[0][0]],
         [mu_xy[1] - diag_lambda[0][0]*e[0][1], mu_xy[1] + diag_lambda[0][0]*e[0][1]],
         'k--', linewidth=0.75, label='Axes of ellipsoid')
plt.plot([mu_xy[0] - diag_lambda[1][1]*e[1][0], mu_xy[0] + diag_lambda[1][1]*e[1][0]],
         [mu_xy[1] - diag_lambda[1][1]*e[1][1], mu_xy[1] + diag_lambda[1][1]*e[1][1]],
         'k--', linewidth=0.75)
# setup figure axes
r = 1
xmin = mu_xy[0] - 2 * r
xmax = mu_xy[0] + 2 * r
ymin = mu_xy[1] - 2 * r
ymax = mu_xy[1] + 2 * r
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('$X$', fontsize=20)
plt.ylabel('$Y$', fontsize=20)
# figure legend
plt.legend()
#
add_logo(fig, size_frac_x=1/9)
