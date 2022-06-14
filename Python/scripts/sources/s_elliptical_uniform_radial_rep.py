#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_elliptical_uniform_radial_rep [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_elliptical_uniform_radial_rep&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_elliptical_uniform_radial_rep).

# +
import numpy as np
from scipy.stats import chi, multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.collections import PolyCollection

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_elliptical_uniform_radial_rep-parameters)

z0 = np.array([1.2, 0.64])  # point to examine

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_elliptical_uniform_radial_rep-implementation-step01): Calculate normal pdf

# +
# define grid for evaluation of bivariate normal pdf
z1_grid = np.linspace(-2.5, 2.5, 100)
z2_grid = np.linspace(-2.5, 2.5, 100)
z1_grid, z2_grid = np.meshgrid(z1_grid, z2_grid)
z_grid = np.stack([z1_grid, z2_grid], axis=2)

# calculate standard normal pdf on grid
f_Z = multivariate_normal.pdf(z_grid, np.zeros(2), np.eye(2))

# calculate value of standard normal pdf at z0
f_Z_z0 = multivariate_normal.pdf(z0, np.zeros(2), np.eye(2))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_elliptical_uniform_radial_rep-implementation-step02): Calculate radial pdf

# +
# define grid for evaluation of radial pdf
r_grid = np.linspace(0.01, 2.5, 50)

# calculate pdf of radial component on grid
f_R = chi.pdf(r_grid, 2)

# calculate radial component of z0
r0 = np.sqrt(z0.T@z0)

# calculate value of pdf of radial component at r0
f_R_r0 = chi.pdf(r0, 2)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_elliptical_uniform_radial_rep-implementation-step03): Calculate uniform on circle pdf values

# +
# define grid for evaluation of uniform pdf
y1_grid = np.linspace(-1, 1, 50)
y2_grid = np.sqrt(1-y1_grid**2)
y1_grid = np.append(y1_grid, np.flip(y1_grid))
y2_grid = np.append(y2_grid, -y2_grid)

# calculate pdf of uniform component on grid
f_Y = np.full(100, 1/(2*np.pi))

# calculate uniform component of z0
y0 = z0/r0

# calculate value of pdf of uniform component at y0
f_Y_y0 = 1/(2*np.pi)
# -

# ## Plots

# +
plt.style.use('arpm')

# pdf
fig = plt.figure(facecolor='white')
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

# axes
ax1.plot([0, 2.5], [0, 0], 0, color='black', linewidth=0.5)
ax1.plot([0, 0], [0, 2.5], 0, color='black', linewidth=0.5)
ax1.plot([0, 0], [0, 0], [0, np.max(f_Z)*1.1], color='black', linewidth=0.5)

# bivariate density
ax1.plot_wireframe(z1_grid, z2_grid, f_Z,
                   rcount=100, ccount=100, alpha=0.02)
ax1.text(-0.5, 0.5, 0.15, r'$f_Z$', color='black',
        fontsize=17)

# radial density
scale_rad = 4*np.pi
ax1.plot(np.full(50, 0), r_grid, f_R/scale_rad, color='red')
verts = list(zip(r_grid, f_R/scale_rad))
verts.append((max(r_grid), 0.0))
verts = [verts]
poly = PolyCollection(verts, facecolors='red')
poly.set_alpha(0.1)
ax1.add_collection3d(poly, zs=0, zdir='x')
ax1.text(0, 2.5, 0.015, r'$f_R$', color='red', fontsize=17)

# uniform on unit circle
ax1.plot(y1_grid, y2_grid, 0, zdir='z', color='skyblue',
         linewidth=1)
# density
scale_unif = 1/(0.03*np.pi)
f_Y_d = np.linspace(0, f_Y_y0/scale_unif, 100)
y1_grid_d, f_Y_d = np.meshgrid(y1_grid[:50], f_Y_d)
y2_grid_d = np.sqrt(1-y1_grid_d**2)
ax1.plot_surface(y1_grid_d, y2_grid_d, f_Y_d, alpha=0.2,
                 rstride=20, cstride=10, color='skyblue',
                 shade=False)
ax1.plot_surface(y1_grid_d, -y2_grid_d, f_Y_d, alpha=0.2,
                 rstride=20, cstride=10, color='skyblue',
                 shade=False)
ax1.text(0.9, -0.44, 0.005, r'$f_{Y}$', color='steelblue',
        fontsize=17)
# label unit circle
ax1.text(-0.3, -0.5, 0, r'$\mathcal{S}^{1}$', color='steelblue',
        fontsize=17)

# annotate chosen point
# z0
ax1.scatter(z0[0], z0[1], 0, color='green')
ax1.text(z0[0], z0[1], 0.002, r'$z_0$', color='green', fontsize=17)
# r0
ax1.scatter(0, r0, 0, color='red')
ax1.text(0, r0+0.05, 0.002, r'$r_0$', color='red', fontsize=17)
# y0
ax1.scatter(y0[0], y0[1], 0, color='skyblue')
ax1.text(y0[0], y0[1], 0.002, r'$y_0$', color='steelblue', fontsize=17)

# connecting lines
ax1.plot([0, z0[0]], [0, z0[1]], 0, zdir='z',
         color='green', linestyle='--')  # 0 to x0
# cylinder for chosen point
ax1.plot(y1_grid*r0, y2_grid*r0, 0, zdir='z', color='green')
ax1.plot(y1_grid*r0, y2_grid*r0, f_Z_z0, zdir='z', color='green',
         linestyle='--')
# density
f_Z_z0_d = np.linspace(0, f_Z_z0, 50)
z1_grid_d, f_Z_z0_d = np.meshgrid(r0*y1_grid[:50], f_Z_z0_d)
z2_grid_d = np.sqrt(r0**2-z1_grid_d**2)
ax1.plot_surface(z1_grid_d, z2_grid_d, f_Z_z0_d, alpha=0.05,
                 rstride=20, cstride=10, color='green',
                 shade=False)
ax1.plot_surface(z1_grid_d, -z2_grid_d, f_Z_z0_d, alpha=0.05,
                 rstride=20, cstride=10, color='green',
                 shade=False)

ax1.set_xlim(-2.5, 2.5)
plt.xticks([])
ax1.set_ylim(-2.5, 2.5)
plt.yticks([])
ax1.set_zlim(0.02, np.max(f_Z)*0.78)
ax1.set_zticks([])

plt.axis('off')

ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.view_init(33, 12)
ax1.grid(False)

add_logo(fig)
plt.tight_layout()
plt.show()
