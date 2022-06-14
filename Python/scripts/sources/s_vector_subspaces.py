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

# # s_vector_subspaces [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_vector_subspaces&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_vector_subspaces).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_vector_subspaces-parameters)

v_1 = np.array([1.0, 1.5, 0.8])
v_2 = np.array([0.3, -1.0, 1.0])
v_3 = np.array([1.0, 2.0, 1.5])
v_4 = np.array([0.46, 0.4, 0.52])

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

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0,
                  facecolor = 'white')

# non-degenerate parallelotope
ax1 = fig.add_subplot(122, projection='3d')

# vector v_1
a1 = Arrow3D([0, v_1[0]], [0, v_1[1]], 
            [0, v_1[2]], mutation_scale=20, 
            arrowstyle="-|>", color='C0')
ax1.add_artist(a1)
ax1.text(v_1[0], v_1[1], v_1[2], r'$v_1$',
         fontsize=17, color='C0')

# vector v_2
a2 = Arrow3D([0, v_2[0]], [0, v_2[1]], 
            [0, v_2[2]], mutation_scale=20, 
            arrowstyle="-|>", color='C0')
ax1.add_artist(a2)
ax1.text(v_2[0], v_2[1], v_2[2], r'$v_2$',
         fontsize=17, color='C0', horizontalalignment='right')

# vector v_3
a3 = Arrow3D([0, v_3[0]], [0, v_3[1]], 
            [0, v_3[2]], mutation_scale=20, 
            arrowstyle="-|>", color='C0')
ax1.add_artist(a3)
ax1.text(v_3[0], v_3[1], v_3[2], r'$v_3$',
         fontsize=17, color='C0')

# vertices of parallelotope
verts = [np.zeros(3), v_1, v_2, v_3,
         v_1+v_2, v_1+v_3, v_2+v_3, v_1+v_2+v_3]

# faces of parallelotope
faces = [[verts[0],verts[1],verts[4],verts[2]],
         [verts[0],verts[1],verts[5],verts[3]],
         [verts[0],verts[2],verts[6],verts[3]],
         [verts[6],verts[2],verts[4],verts[7]],
         [verts[6],verts[3],verts[5],verts[7]],
         [verts[1],verts[5],verts[7],verts[4]]]

# plot sides
ax1.add_collection3d(Poly3DCollection(faces, 
 facecolors='lightgrey', linewidths=0.2, edgecolors='darkgrey', alpha=0.1))

# axes
ax1.axis('off')
ax1.set_xlim([0, np.ceil(max(v_1))*1.2])
ax1.set_ylim([0, np.ceil(max(v_1))*1.2])
ax1.set_zlim([0, np.ceil(max(v_1))*1.2])

plt.title('Parallelotope for 3-dimensional subspace', fontsize=20, fontweight='bold')

x_axis = Arrow3D([-0.03, np.ceil(max(v_1))*1.2], [0, 0], 
            [0, 0], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax1.add_artist(x_axis)
y_axis = Arrow3D([0, 0], [-0.03, np.ceil(max(v_1))*1.2], 
            [0, 0], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax1.add_artist(y_axis)
z_axis = Arrow3D([0, 0], [0, 0], 
            [-0.01, np.ceil(max(v_1))*1.2], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax1.add_artist(z_axis)

# formatting
ax1.view_init(20, 30)
ax1.grid(False)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# degenerate parallelotope
ax2 = fig.add_subplot(121, projection='3d')

# vector v_1
a1 = Arrow3D([0, v_1[0]], [0, v_1[1]], 
            [0, v_1[2]], mutation_scale=20, 
            arrowstyle="-|>", color='C0')
ax2.add_artist(a1)
ax2.text(v_1[0], v_1[1], v_1[2], r'$v_1$',
         fontsize=17, color='C0')

# vector v_2
a2 = Arrow3D([0, v_2[0]], [0, v_2[1]], 
            [0, v_2[2]], mutation_scale=20, 
            arrowstyle="-|>", color='C0')
ax2.add_artist(a2)
ax2.text(v_2[0], v_2[1], v_2[2], r'$v_2$',
         fontsize=17, color='C0', horizontalalignment='right')

# vector v_4
a4 = Arrow3D([0, v_4[0]], [0, v_4[1]], 
            [0, v_4[2]], mutation_scale=20, 
            arrowstyle="-|>", color='darkorange')
ax2.add_artist(a4)
ax2.text(v_4[0], v_4[1], v_4[2], r'$v_4$',
         fontsize=17, color='darkorange')

# vertices of parallelotope
verts2 = [np.zeros(3), v_1, v_2, v_1+v_2]

# faces of parallelotope
faces2 = [[verts2[0],verts2[1],verts2[3],verts2[2]]]

# plot sides
ax2.add_collection3d(Poly3DCollection(faces2, 
 facecolors='lightgrey', linewidths=0.2, edgecolors='darkgrey', alpha=0.1))

# axes
ax2.axis('off')
ax2.set_xlim([0, np.ceil(max(v_1))*1.2])
ax2.set_ylim([0, np.ceil(max(v_1))*1.2])
ax2.set_zlim([0, np.ceil(max(v_1))*1.2])

plt.title('Parallelotope for 2-dimensional subspace', fontsize=20, fontweight='bold')

x_axis = Arrow3D([-0.03, np.ceil(max(v_1))*1.2], [0, 0], 
            [0, 0], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax2.add_artist(x_axis)
y_axis = Arrow3D([0, 0], [-0.03, np.ceil(max(v_1))*1.2], 
            [0, 0], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax2.add_artist(y_axis)
z_axis = Arrow3D([0, 0], [0, 0], 
            [-0.01, np.ceil(max(v_1))*1.2], mutation_scale=20, 
            arrowstyle="-|>", color='black')
ax2.add_artist(z_axis)

# formatting
ax2.view_init(20, 30)
ax2.grid(False)
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

add_logo(fig, set_fig_size=False)
plt.tight_layout()
