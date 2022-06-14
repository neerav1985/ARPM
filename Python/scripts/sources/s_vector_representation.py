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

# # s_vector_representation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_vector_representation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_vector_representation).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.gridspec as gridspec

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_vector_representation-parameters)

v = np.array([1.0, 1.5, 0.8])  # vector

# ## Plots

# +
plt.style.use('arpm')

# arrow in 3D plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# representations of a vector
fig1 = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0,
                  facecolor = 'white')
gs = fig1.add_gridspec(10, 2)

# arrow in 3D Cartesian plane
ax1 = fig1.add_subplot(gs[:, 0], projection='3d')

# vector v
a = Arrow3D([0, v[0]], [0, v[1]], 
            [0, v[2]], mutation_scale=20, 
            arrowstyle="-|>", color='C0')
ax1.add_artist(a)
ax1.text(v[0], v[1], v[2],
         '('+str(v[0])+', '+str(v[1])+', '+str(v[2])+')',
         fontsize=17, color='C0')

# bottom rectangle
plt.plot([0, v[0]], [v[1], v[1]], [0, 0], ls='--', color='lightgrey')
plt.plot([v[0], v[0]], [0, v[1]], [0, 0], ls='--', color='lightgrey')

# top rectangle
plt.plot([0, v[0]], [0, 0], [v[2], v[2]], ls='--', color='lightgrey')
plt.plot([0, 0], [0, v[1]], [v[2], v[2]], ls='--', color='lightgrey')
plt.plot([0, v[0]], [v[1], v[1]], [v[2], v[2]], ls='--', color='lightgrey')
plt.plot([v[0], v[0]], [0, v[1]], [v[2], v[2]], ls='--', color='lightgrey')

# vertical lines
plt.plot([v[0], v[0]], [v[1], v[1]], [0, v[2]], ls='--', color='lightgrey')
plt.plot([v[0], v[0]], [0, 0], [0, v[2]], ls='--', color='lightgrey')
plt.plot([0, 0], [v[1], v[1]], [0, v[2]], ls='--', color='lightgrey')

# axes
ax1.axis('off')
ax1.set_xlim([0, np.ceil(max(v))*1.2])
ax1.set_ylim([0, np.ceil(max(v))*1.2])
ax1.set_zlim([0, np.ceil(max(v))*1.2])

plt.title('Geometrical representation', fontsize=20, fontweight='bold')

x_axis = Arrow3D([-0.03, np.ceil(max(v))*1.2], [0, 0], 
            [0, 0], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax1.add_artist(x_axis)
ax1.text(np.ceil(max(v))*1.1, -0.1, 0.1, r'$\mathrm{\mathbb{R}}^{(1)}$',
         fontsize=17, color='black')
ax1.text(v[0], 0, -0.2, v[0], fontsize=17, color='C0')

y_axis = Arrow3D([0, 0], [-0.03, np.ceil(max(v))*1.2], 
            [0, 0], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax1.add_artist(y_axis)
ax1.text(0, np.ceil(max(v))*1.1, 0.1, r'$\mathrm{\mathbb{R}}^{(2)}$',
         fontsize=17, color='black')
ax1.text(0, v[1], -0.21, v[1], fontsize=17, color='C0')

z_axis = Arrow3D([0, 0], [0, 0], 
            [-0.01, np.ceil(max(v))*1.2], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax1.add_artist(z_axis)
ax1.text(0, 0.1, np.ceil(max(v))*1.1, r'$\mathrm{\mathbb{R}}^{(3)}$',
         fontsize=17, color='black')
ax1.text(0, 0.1, v[2]*1.05, v[2], fontsize=17, color='C0')

# formatting
ax1.view_init(20, 30)
ax1.grid(False)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# coordinate representation
ax2 = fig1.add_subplot(gs[1:8, 1])
ax2.scatter([1, 2, 3], [v[0], v[1], v[2]])

plt.title('Analytical representation', fontsize=20, fontweight='bold',
          pad=25)

plt.xticks(np.arange(1, 4), ('(1)', '(2)', '(3)'), fontsize=14)
plt.xlabel(r'$\mathrm{\mathbb{N}}$', fontsize=17, labelpad=10)

ax2.set_ylim([0, np.ceil(max(v))*1.2])
plt.yticks(fontsize=14)
plt.ylabel(r'$\mathrm{\mathbb{R}}$', fontsize=17, labelpad=20,
           rotation=0)

add_logo(fig1, set_fig_size=False)
plt.tight_layout()
