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

# # s_riemann_integration_multivariate [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_riemann_integration_multivariate&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_multivariate).

# +
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_multivariate-parameters)

# +
# function
x1, x2 = sym.symbols('x1, x2')
f = x1**3 + x1**2 + x2**2 +x1*x2
display(f)
f_ = sym.lambdify((x1, x2), f, 'numpy')

# domain
a1 = -1
b1 = 1
a2 = -1
b2 = 1

# number of intervals in partition of each edge
k_ = 20
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_multivariate-implementation-step01): Define partition

# +
# partition of sides
endpoints_1 = np.linspace(a1, b1, k_+1)
endpoints_2 = np.linspace(a2, b2, k_+1)
Delta_1 = np.c_[endpoints_1[:-1], endpoints_1[1:]]
Delta_2 = np.c_[endpoints_2[:-1], endpoints_2[1:]]

# partition of domain
Delta = np.zeros((k_**2,2,2))
for k in range(k_):
    for j in range(k_):
        Delta[j+k_*k,:,:] = np.array([Delta_1[k,:], Delta_2[j,:]])
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_multivariate-implementation-step02): Choose points in partition

c = np.c_[(Delta[:, 0, 0] + Delta[:, 0, 1])/2,
          (Delta[:, 1, 0] + Delta[:, 1, 1])/2]
f_c = f_(c[:, 0], c[:, 1])

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_multivariate-implementation-step03): Calculate Riemann sum

volume = (Delta[:, 0, 1]-Delta[:, 0, 0])*(Delta[:, 1, 1]-Delta[:, 1, 0])
s_f_Delta = np.sum(f_c*volume)
s_f_Delta

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_multivariate-implementation-step04): Calculate integral

integral = sym.integrate(f, (x1, a1, b1), (x2, a2, b2))
float(integral)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_multivariate-implementation-step05): Define grid for plotting

x_grid1 = np.linspace(a1, b1, 100)
x_grid2 = np.linspace(a2, b2, 100)
x_grid1, x_grid2 = np.meshgrid(x_grid1, x_grid2)
f_x = f_(x_grid1, x_grid2)

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)

ax1 = fig.add_subplot(121, projection='3d')
# function
ax1.plot_surface(x_grid1, x_grid2, f_x,
                 rcount=100, ccount=100,
                 linewidth=0, color='C0',
                 shade=True, alpha=0.7)
# integral
ax1.text(b1, b2, np.max(f_x)*1.35,
         'Integral: '+str(round(integral,2)),
         fontsize=17, bbox=dict(facecolor='white', edgecolor='darkgray'))

ax1.set_title('Function', fontsize=20, fontweight='bold')
ax1.set_zlim([0, np.max(f_x)*1.3])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for tick in ax1.zaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.grid(False)
ax1.view_init(20, 135)

ax2 = fig.add_subplot(122, projection='3d')
# step function
dx = Delta[0, 0, 1] - Delta[0, 0, 0]
dy = Delta[0, 1, 1] - Delta[0, 1, 0]
ax2.bar3d(Delta[:, 0, 0], Delta[:, 1, 0], np.zeros_like(f_c),
          dx, dy, f_c,
          color='white', linewidth=0.5, alpha=1,
          shade=True, edgecolor='darkgray')
# Riemann sum
ax2.text(b1, b2, np.max(f_x)*1.35,
         'Riemann sum: '+str(round(s_f_Delta,2)),
         fontsize=17, bbox=dict(facecolor='white', edgecolor='darkgray'))

ax2.set_title('Step function', fontsize=20, fontweight='bold')
ax2.set_zlim([0, np.max(f_x)*1.3])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for tick in ax2.zaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.grid(False)
ax2.view_init(20, 135)

add_logo(fig, set_fig_size=False)
plt.tight_layout()
