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

# # s_uniform_inside_circle [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_uniform_inside_circle&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExUnifCircleBivariate).

# +
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.collections import PolyCollection

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_inside_circle-parameters)

k_ = 200  # number evaluation points for each axis
x1_cond = 0.9  # conditioning value of X1 used to define conditional pdf

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_inside_circle-implementation-step01): Compute joint pdf

# +
# define points at which to evaluate pdfs
x_1 = np.linspace(-1.1, 1.1, k_)
x_2 = np.linspace(-1.1, 1.1, k_)
x1_grid, x2_grid = np.meshgrid(x_1, x_2)
x_grid = np.stack([x1_grid, x2_grid], axis=2)

# indicator function
def indicator_joint(x_1, x_2):
    return (x_1**2 + x_2**2 <= 1)

# compute joint pdf
f_x1x2 = (1/np.pi)*indicator_joint(x_grid[:, :, 0], x_grid[:, :, 1])


# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_inside_circle-implementation-step02): Compute conditional pdf

# +
# indicator function
def indicator_cond(x1_cond, x_2):
    return (x_2**2 <= 1-x1_cond**2)

# compute conditional pdf
f_x2_given_x1 = (1/(2*np.sqrt(1-x1_cond**2)))*indicator_cond(x1_cond, x_2)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_uniform_inside_circle-implementation-step03): Compute marginal pdf

# compute marginal pdf
f_x1 = np.zeros(k_)
for k in range(k_):
    if x_1[k]**2 <= 1:
        f_x1[k] = (2/np.pi)*np.sqrt(1-x_1[k]**2)

# ## Plots

# +
plt.style.use('arpm')

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')

# joint pdf
ax1.plot_surface(x1_grid, x2_grid, f_x1x2,
                 rcount=200, ccount=200,
                 color='white', linewidth=0, alpha=0.5)

# intersection with plane x_1=x1_cond
verts2 = [[(-np.sqrt(1-x1_cond**2), 0),
           (-np.sqrt(1-x1_cond**2), np.max(f_x1x2)),
           (np.sqrt(1-x1_cond**2), np.max(f_x1x2)),
           (np.sqrt(1-x1_cond**2), 0)
          ]]
poly2 = PolyCollection(verts2)
poly2.set_alpha(0.5)
ax1.add_collection3d(poly2, zs=x1_cond, zdir='x')

ax1.plot([x1_cond, x1_cond], [-1.3, 1.3], 0, zdir='z')
ax1.plot([x1_cond, x1_cond], [-np.sqrt(1-x1_cond**2), np.sqrt(1-x1_cond**2)],
         np.max(f_x1x2), zdir='z')

ax1.set_xlim(-1.3, 1.3)
ax1.set_xlabel(r'$x_1$', fontsize=17, labelpad=10)
ax1.set_ylim(-1.3, 1.3)
ax1.set_ylabel(r'$x_2$', fontsize=17, labelpad=8)
ax1.set_zlim(0, np.max(f_x1x2)*1.3)

# add plane of intersection defining conditional pdf
add_logo(fig1)
plt.tight_layout()

# conditional pdf
fig2 = plt.figure()
ax2 = plt.gca()
ax2.fill_between(x_2, 0, f_x2_given_x1, where=f_x2_given_x1>0,
                 alpha=0.5)
plt.vlines([x_2[np.argmax(f_x2_given_x1)], x_2[-np.argmax(f_x2_given_x1)-1]],
           0, np.max(f_x2_given_x1),
           color='C0', linewidth=2)
plt.hlines(np.max(f_x2_given_x1), x_2[np.argmax(f_x2_given_x1)],
           x_2[-np.argmax(f_x2_given_x1)-1],
           color='C0', linewidth=2)
plt.vlines(0, 0, np.max(f_x2_given_x1)*1.1, color='black', linewidth=0.5)

plt.xlim(-1.3, 1.3)
plt.ylim(0, np.max(f_x2_given_x1)*1.1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$x_2$', fontsize=17)
plt.ylabel(r'$f_{X_2|x_1}(x_2)$',
          fontsize=17)
plt.title(r'Conditional pdf of $X_2|X_1='+str(x1_cond)+r'}$',
          fontsize=20, fontweight='bold')
add_logo(fig2, location=1)
plt.tight_layout()

# marginal pdf
fig3 = plt.figure()
ax3 = plt.gca()
ax3.plot(x_1, f_x1, color='C0', linewidth=2)
ax3.fill_between(x_1, 0, f_x1, interpolate=True,
                 alpha=0.5)

plt.xlim(-1.3, 1.3)
plt.ylim(0, 1.0)
plt.vlines(0, 0, 1.1, color='black', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$x_1$', fontsize=17)
plt.ylabel(r'$f_{X_1}(x_1)$', fontsize=17)
plt.title(r'Marginal pdf of $X_1}$',
          fontsize=20, fontweight='bold')

add_logo(fig3, location=1)
plt.tight_layout()
