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

# # s_lognormal_l2_geometry [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_lognormal_l2_geometry&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBRandGeomLogN).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.plot_ellipse import plot_ellipse
from arpym.tools.transpose_square_root import transpose_square_root
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_lognormal_l2_geometry-parameters)

j_ = 10000  # number of scenarios
mu = np.array([0, 0.1])  # location
svec = np.array([0.9, 0.7])  # standard deviations
rho = 0.2  # correlation coefficient

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_lognormal_l2_geometry-implementation-step01): Generate lognormal scenarios and compute expectation and covariance

sig2 = np.diag(svec)@np.array([[1, rho], [rho, 1]])@np.diag(svec)  # normal covariance
x_n = simulate_normal(mu, sig2, j_) # normal scenarios
x = np.exp(x_n)  # lognormal scenarios
mu_x = np.exp(mu + 0.5*np.diag(sig2))  # lognormal expectation
sig2_x = np.diag(mu_x)@(np.exp(sig2) - np.ones((2, 1)))@np.diag(mu_x)  # lognormal covariance

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_lognormal_l2_geometry-implementation-step02): Compute expectation inner product, lengths, distance, angle

e_inn_prod = mu_x[0]*mu_x[1] + sig2_x[0, 1]  # expectation inner product
e_len_x1 = np.sqrt(mu_x[0]**2 + sig2_x[0, 0])  # expectation length of X1
e_len_x2 = np.sqrt(mu_x[1]**2 + sig2_x[1, 1])  # expectation length of X2
e_dist = np.sqrt(e_len_x1**2 + e_len_x2**2 - 2*e_inn_prod)  # expectation distance
e_ang = e_inn_prod/(e_len_x1*e_len_x2)  # expectation angle

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_lognormal_l2_geometry-implementation-step03): Compute covarance inner product, lengths, distance, angle

cov_inn_prod = sig2_x[0, 1]  # covariance inner product
cov_len_x1 = np.sqrt(np.diag(sig2_x)[0])  # covariance length of X1 
cov_len_x2 = np.sqrt(np.diag(sig2_x)[1])  # covariance length of X2
cov_dist = np.sqrt(cov_len_x1**2 + cov_len_x2**2 - 2*cov_inn_prod)  # covariance distance
cov_ang = cov_inn_prod/(cov_len_x1*cov_len_x2)  # covariance angle

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_lognormal_l2_geometry-implementation-step04): Compute visualisation map

c2 = np.diag(1/np.sqrt(np.diag(sig2_x)))@sig2_x@np.diag(1/np.sqrt(np.diag(sig2_x)))  # correlation matrix
c = transpose_square_root(c2, method='Riccati')  # Ricatti root of c2
x_visual = c.T@np.diag(np.sqrt(np.diag(sig2_x)))  # visualisation vectors

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_lognormal_l2_geometry-implementation-step05): Expectation-covariance ellipsoid computations

ellipse_mux_sig2x = plot_ellipse(mu_x, sig2_x, display_ellipse=False, plot_axes=True, plot_tang_box=True)

# ## Plot

# +
plt.style.use('arpm')

# Colors
gray = [150/255, 150/255, 150/255]
light_gray = [230/255, 230/255, 230/255]
light_blue = [181/255, 225/255, 223/255]

# Figure specifications
plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)

x1 = max(abs((x_visual[0])))
x2 = max(abs((x_visual[1])))

ax1 = plt.axes([0.14, 0.12, 0.25, 0.35])
ax1.scatter(x[:, 0], x[:, 1], 2, marker='*', linewidths=1, color=gray)
ax1.tick_params(axis='x', colors='None')
ax1.tick_params(axis='y', colors='None')
ellipse_mux_sig2x = plot_ellipse(mu_x, sig2_x, display_ellipse=True)
plt.xlabel('$X_1$', labelpad=-5)
plt.ylabel('$X_2$', labelpad=-5)
plt.xlim([-0.1, 4])
plt.ylim([-0.1, 2.8])

ax2 = plt.axes([0.14, -0.01, 0.25, 0.08])
plt.hist(np.sort(x[:, 0]), bins=int(100*np.log(j_)), density=True, bottom=0, color=light_gray)
plt.xlim([-0.1, 4])
plt.ylim([0, 1])
plt.gca().invert_yaxis()

ax3 = plt.axes([0.05, 0.12, 0.05, 0.35])
plt.hist(np.sort(x[:, 1]), bins=int(100*np.log(j_)), density=True,
         color=light_blue, bottom=0, orientation='horizontal')
plt.xlim([0, 1])
plt.gca().invert_xaxis()
plt.ylim([-0.1, 2.8])

ax4 = plt.axes([0.46, 0.12, 0.25, 0.35])
plt.quiver(0, 0, x_visual[0, 0], x_visual[1, 0], color = light_gray, lw= 2, angles='xy',scale_units='xy',scale=1)
plt.quiver(0, 0, x_visual[0, 1], x_visual[1, 1], color = light_blue, lw= 2, angles='xy',scale_units='xy',scale=1)
quiv1 = plt.plot(0, 0, color=light_gray, lw= 2, marker=None)
quiv2 = plt.plot(0, 0, color=light_blue, lw= 2, marker=None)
plt.plot(0, 0, 'o',markeredgecolor='k',markerfacecolor='w')
plt.grid(True)
plt.ylim([-0.1, 2.8])
plt.xlim([-0.1, 4])
plt.legend(['$X_1$','$X_2$'])

add_logo(f, axis=ax1, location=4, set_fig_size=False)
# -


