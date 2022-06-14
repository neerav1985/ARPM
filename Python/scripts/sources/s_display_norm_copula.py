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

# # s_display_norm_copula [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_display_norm_copula&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=Frechet-HoeffBoundCop).

# +
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arpym.statistics.simulate_normal import simulate_normal
from arpym.statistics.norm_cop_pdf import norm_cop_pdf
from arpym.tools.logo import add_logo


# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_display_norm_copula-parameters)

j_ = 5000  # number of scenarios
mu = np.array([0, 0])  # expectations
rho = -0.5  # correlation coefficient
svec = np.array([1, 1])  # standard deviations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_display_norm_copula-implementation-step01): Generate normal scenarios and scenarios for the grades

# +
sigma2 = np.diag(svec) @ np.array([[1, rho], [rho, 1]]) @ np.diag(svec)
x = simulate_normal(mu, sigma2, j_)  # normal scenarios

u1 = stats.norm.cdf(x[:, 0], mu[0], svec[0])
u2 = stats.norm.cdf(x[:, 1], mu[1], svec[1])
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_display_norm_copula-implementation-step02): Compute pdf and cdf surfaces

# +
# grid in the unit square
grid = np.arange(0.01, 1, 0.01)
n_grid = len(grid)

pdf_u = np.zeros((n_grid, n_grid))
cdf_u = np.zeros((n_grid, n_grid))
for n in range(n_grid):
    for m in range(n_grid):
        u = np.r_[grid[n], grid[m]].reshape(-1, 1)
        pdf_u[n, m] = norm_cop_pdf(u, mu, sigma2)  # copula pdf
        x = stats.norm.ppf(u.flatten(), mu.flatten(), svec)
        cdf_u[n, m], _ = stats.mvn.mvnun(np.array([-100, -100]), x.flatten(), mu.flatten(), sigma2)

u_1, u_2 = np.meshgrid(grid, grid)
# -

# ## Plots

# +
plt.style.use('arpm')
u_color = [60/255, 149/255, 145/255]

# set figure specification
f = plt.figure(1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax1 = plt.axes([0.10, 0.5, 0.35, 0.35], projection='3d')
ax1.plot_surface(u_1, u_2, pdf_u.T, facecolor='k', edgecolor=u_color)
ax1.view_init(30, -120)
plt.xlabel('Grade $U_1$', labelpad=7)
plt.ylabel('Grade $U_2$', labelpad=7)
ax1.set_zlabel('Normal copula pdf')
plt.title("Pdf of normal copula", fontsize=20, fontweight='bold')

ax2 = plt.axes([0.55, 0.5, 0.35, 0.35], projection='3d')
ax2.plot_surface(u_1, u_2, cdf_u.T, facecolor='k', edgecolor=u_color)
ax2.view_init(30, -120)
plt.xlabel('Grade $U_1$', labelpad=7)
plt.ylabel('Grade $U_2$', labelpad=7)
ax2.set_zlabel('Normal copula cdf')
plt.title("Cdf of normal copula", fontsize=20, fontweight='bold')

ax3 = plt.axes([0.3, 0.12, 0.38, 0.35])
plt.gca().set_aspect('equal', adjustable='box')
ax3.scatter(u1, u2, s=10, color=u_color, marker='*')
ax3.tick_params(axis='x', colors='None')
ax3.tick_params(axis='y', colors='None')
plt.xlabel('Grade $U_1$', labelpad=-5)
plt.ylabel('Grade $U_2$', labelpad=-5)
plt.title("Normal copula", fontsize=20, fontweight='bold')

ax4 = plt.axes([0.395, 0.001, 0.195, 0.07])
plt.hist(np.sort(u1), bins=int(10*np.log(j_)), density=True, bottom=0)
plt.gca().invert_yaxis()

ax5 = plt.axes([0.305, 0.12, 0.05, 0.35])
plt.hist(np.sort(u2), bins=int(10*np.log(j_)), density=True,
         color=u_color, bottom=0, orientation='horizontal')
plt.gca().invert_xaxis()

add_logo(f, axis=ax1, location=4, set_fig_size=False)
