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

# # s_ncop_nmarg [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_ncop_nmarg&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-norm-cop-giv-norm-marg).

# +
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_ncop_nmarg-parameters)

j_ = 10**5  # number of scenarios
mu = np.zeros(2)  # location parameter
rho = -0.8  # correlation coefficient
sigma = np.array([1, 1])  # standard deviations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_ncop_nmarg-implementation-step01): Generate a sample from the bivariate normal distribution

sigma2 = np.diag(sigma) @ np.array([[1, rho], [rho, 1]]) @ np.diag(sigma) # covariance
x = simulate_normal(mu, sigma2, j_).reshape((j_, -1)) # normal scenarios 
x1 = x[:, 0]
x2 = x[:, 1]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_ncop_nmarg-implementation-step02): Evaluate cdf's of the marginal variables

llim = np.floor(
        min(mu[0]-5*np.sqrt(sigma2[0, 0]), mu[1]-5*np.sqrt(sigma2[1, 1])))
ulim = np.ceil(
        max(mu[0]+5*np.sqrt(sigma2[0, 0]), mu[1]+5*np.sqrt(sigma2[1, 1])))
x_grid = np.linspace(llim, ulim, 100) # evenly spaced numbers over the given interval
cdf_x1 = stats.norm.cdf(x_grid, mu[0], np.sqrt(sigma2[0, 0])) # cdf of the marginal variable X₁
cdf_x2 = stats.norm.cdf(x_grid, mu[1], np.sqrt(sigma2[1, 1])) # cdf of the marginal variable X₂

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_ncop_nmarg-implementation-step03): Obtain scenarios for the grades

u = stats.norm.cdf(x, mu, sigma) # grade scenarios
u_1 = u[:, 0]
u_2 = u[:, 1]

# ## Plot

# +
plt.style.use('arpm')

# Colors
y_color = [153/255, 205/255, 129/255]
u_color = [60/255, 149/255, 145/255]
m_color = [63/255, 0/255, 102/255]

xlim = [np.percentile(x1, 0.5), np.percentile(x1, 99.5)]
ylim = [np.percentile(x2, 0.5), np.percentile(x2, 99.5)]

# Figure specifications
plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
gs0 = gridspec.GridSpec(2, 2)

# Marginal X1
gs00 = gridspec.GridSpecFromSubplotSpec(23, 20,
                  subplot_spec=gs0[0], wspace=2, hspace=2.5)
ax1 = plt.Subplot(f, gs00[:-5, 4:-4])
f.add_subplot(ax1)
ax1.tick_params(labelsize=14)
plt.plot(x_grid, cdf_x1, lw=2, color='C3', label=r'$F_{X_{1}}(x)$')
plt.ylabel('$F_{X_1}$', fontsize=17)

# Copula scenarios
gs01 = gridspec.GridSpecFromSubplotSpec(46, 18, subplot_spec=gs0[1],
                                        wspace=0, hspace=0.6)
ax2 = plt.Subplot(f, gs01[:-10, 4:-5], ylim=[0, 1], xlim=[0, 1])
f.add_subplot(ax2)
plt.scatter(u_2, u_1, s=5, color=u_color)
ax2.tick_params(labelsize=14)
plt.xlabel('$U_2$', fontsize=17, labelpad=-5)
plt.ylabel('$U_1$', fontsize=17, labelpad=-11)

# Grade U1
ax3 = plt.Subplot(f, gs01[:-10, 2])
f.add_subplot(ax3)
ax3.tick_params(labelsize=14)
plt.xlim([0, 2])
plt.ylim([0, 1])
ax3.tick_params(axis='y', colors='None')
plt.hist(np.sort(u_1), bins=int(10*np.log(j_)), density=True,
         color=u_color, orientation='horizontal')
plt.xlabel('$f_{U_1}$', fontsize=17)
ax3.xaxis.tick_top()

# Grade U2
ax4 = plt.Subplot(f, gs01[41:46, 4:-5], sharex=ax2)
f.add_subplot(ax4)
plt.hist(np.sort(u_2), bins=int(10*np.log(j_)),
         density=True, color=u_color)
ax4.tick_params(labelsize=14)
ax4.tick_params(axis='x', colors='white')
ax4.yaxis.tick_right()
plt.ylabel('$f_{U_2}$', fontsize=17)
plt.ylim([0, 2])
plt.xlim([0, 1])

# Joint scenarios
gs02 = gridspec.GridSpecFromSubplotSpec(2*25, 2*20,
            subplot_spec=gs0[2], wspace=0.6, hspace=1)
ax5 = plt.Subplot(f, gs02[2*7:, 2*4:-8], ylim=ylim, xlim=xlim)
f.add_subplot(ax5)
plt.scatter(x1, x2, s=5, color=y_color, label=r'$F_{X_{1}}(x)$')
ax5.tick_params(labelsize=14)
plt.xlabel('$X_1$', fontsize=17)
plt.ylabel('$X_2$', fontsize=17)

# Histogram X1
ax7 = plt.Subplot(f, gs02[0:12, 2*4:-8], sharex=ax5)
f.add_subplot(ax7)
plt.hist(x1, bins=int(80*np.log(j_)),
         density=True, color=y_color)
ax7.tick_params(labelsize=14)
ax7.set_ylim([0, 0.45])
ax7.set_xlim(xlim)
ax7.tick_params(axis='x', colors='None')
plt.ylabel('$f_{X_1}$', fontsize=17)

# Histogram X2
ax8 = plt.Subplot(f, gs02[2*7:, -7:-2], sharey=ax5)
f.add_subplot(ax8)
plt.hist(x2, bins=int(80*np.log(j_)), density=True,
         orientation='horizontal', color=y_color)
ax8.tick_params(labelsize=14)
ax8.set_xlim([0, 0.4])
ax8.set_ylim(ylim)
ax8.tick_params(axis='y', colors='None')
plt.xlabel('$f_{X_2}$', fontsize=17)

# Marginal X2
gs03 = gridspec.GridSpecFromSubplotSpec(25, 18, subplot_spec=gs0[3])
ax6 = plt.Subplot(f, gs03[7:, 4:-5])
f.add_subplot(ax6)
plt.plot(x_grid, cdf_x2, lw=2, color='C3', label=r'$F_{X_{2}}(x)$')
plt.xlabel('$F_{X_2}$', fontsize=17)
ax6.tick_params(labelsize=14)

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
