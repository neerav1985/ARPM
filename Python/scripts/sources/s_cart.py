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

# # s_cart [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cart&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_cart).

# +
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cart-parameters)

l_max = 80  # maximum number of leaves

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_cart-implementation-step00): Load input and target scenarios

data = pd.read_csv('~/databases/temporary-databases/db_ml_variables.csv')
j_bar = int(data['j_in_sample'][0])
z = data['z'].values.reshape(j_bar, 2)
x = data['x'].values[:j_bar]


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cart-implementation-step01): Build CART models and compute in-sample errors

tree_delta = []
x_bar = np.zeros((j_bar, l_max-1))
error = np.zeros(l_max-1)
for leaves in np.arange(2, l_max+1):
    # fit CART model
    tree_delta_l_bar = tree.DecisionTreeRegressor(max_depth=80,
                                                  max_leaf_nodes=leaves)
    tree_delta.append(tree_delta_l_bar)
    # predicted simulations
    x_bar_l = tree_delta_l_bar.fit(z, x).predict(z)
    x_bar[:, leaves-2] = x_bar_l
    # in-sample error
    error[leaves-2] = np.mean((x-x_bar_l)**2)

# ## Plots

# +
plt.style.use('arpm')

idxx0 = np.where(np.abs(z[:, 0]) <= 2)[0]
idxx1 = np.where(np.abs(z[:, 1]) <= 2)[0]
idxx = np.intersect1d(idxx0, idxx1)
lightblue = [0.2, 0.6, 1]
lightgreen = [0.6, 0.8, 0]

# Auxiliary functions

def muf(z1, z2):
    return z1 - np.tanh(10*z1*z2)

def sigf(z1, z2):
    return np.sqrt(np.minimum(z1**2, 1/(10*np.pi)))

plt.figure()
mydpi = 72.0
fig = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)

# Parameters
n_classes = 2
plot_colors = "rb"
plot_step = 0.06

z_1_min = z[:, 0].min()
z_1_max = z[:, 0].max()
z_2_min = z[:, 1].min()
z_2_max = z[:, 1].max()
zz1, zz2 = np.meshgrid(np.arange(z_1_min, z_1_max, plot_step),
                       np.arange(z_2_min, z_2_max, plot_step))

# Error
ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
insamplot = ax1.plot(np.arange(2, l_max+1), error, color='k')
ax1.set_ylabel('In-sample error', color='k')
ax1.tick_params(axis='y', colors='k')
ax1.set_xlabel('Number of leaves')
plt.xlim([2, l_max])
ax1.set_title('In-sample error as function \n of number of leaves',
              fontweight='bold', y=1.10)
ax1.grid(False)


# Error
ax4 = plt.subplot2grid((2, 4), (0,2))
tree.plot_tree(tree_delta[-1].fit(z, x))
ax4.set_title('Tree', fontweight='bold', y=1.17)

# Conditional expectation surface
ax2 = plt.subplot2grid((2, 4), (1, 0), rowspan=2, colspan=2,
                       projection='3d')
step = 0.01
zz1, zz2 = np.meshgrid(np.arange(-2, 2, step), np.arange(-2, 2, step))
ax2.plot_surface(zz1, zz2, muf(zz1, zz2), color=lightblue, alpha=0.7,
                 label='$\mu(z_1, z_2)$')

ax2.scatter3D(z[idxx, 0], z[idxx, 1],
              x[idxx], s=10, color=lightblue, alpha=1,
              label='$(Z_1, Z_2, X)$')
ax2.set_xlabel('$Z_1$')
ax2.set_ylabel('$Z_2$')
ax2.set_zlabel('$X$')
ax2.set_title('Conditional expectation surface', fontweight='bold')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
# ax.legend()

# Fitted surface
ax3 = plt.subplot2grid((2, 4), (1, 2), rowspan=2, colspan=2, projection='3d')
x_plot = tree_delta[-1].predict(np.c_[zz1.ravel(), zz2.ravel()])
x_plot = x_plot.reshape(zz1.shape)
ax3.plot_surface(zz1, zz2, x_plot, alpha=0.5, color=lightgreen)
ax3.scatter3D(z[idxx, 0], z[idxx, 1],
              tree_delta[-1].predict(z[idxx, :]), s=10,
              alpha=1, color=lightgreen)
ax3.set_xlabel('$Z_1$')
ax3.set_ylabel('$Z_2$')
ax3.set_zlabel('$\overline{X}$')
plt.title('Fitted surface, leaves = %1i; ' % l_max, fontweight='bold')

add_logo(fig, size_frac_x=1/8)
plt.tight_layout()
