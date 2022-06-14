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

# # s_saddlepoint [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_saddlepoint&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_saddlepoint).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from arpym.tools.logo import add_logo


# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_saddlepoint-parameters)

# +
# function
def f(x):
    return x[0]**2 - x[1]**2

# points to test convexity/concavity criteria
x_1 = np.array([0, 1])
y_1 = np.array([0, -1])

x_2 = np.array([1, 0])
y_2 = np.array([-1, 0])
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_saddlepoint-implementation-step01): Convexity/concavity test

t = np.linspace(0, 1, 100, endpoint=True)
z_1 = np.array([t_*x_1+(1-t_)*y_1 for t_ in t])
f_line1 = np.array([t_*f(x_1)+(1-t_)*f(y_1) for t_ in t])
z_2 = np.array([t_*x_2+(1-t_)*y_2 for t_ in t])
f_line2 = np.array([t_*f(x_2)+(1-t_)*f(y_2) for t_ in t])

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_saddlepoint-implementation-step02): Values for plots

points = np.linspace(-1, 1, 100)
x1_grid, x2_grid = np.meshgrid(points, points)
f_x = []
for x2 in points:
    for x1 in points:
        x = np.array([x1, x2])
        f_x.append(f(x))
f_x = np.array(f_x).reshape(100, 100)

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0,
                  facecolor = 'white')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, f_x,
                rcount=100, ccount=100,
                alpha=0.5, linewidth=0)
ax.plot(z_1[:, 0], z_1[:, 1], f_line1,
        color='darkorange', lw=2)
ax.scatter([x_1[0], y_1[0]], [x_1[1], y_1[1]], [f_line1[0], f_line1[1]],
           color='darkorange', s=40, depthshade=False)
ax.plot(z_2[:, 0], z_2[:, 1], f_line2,
        color='darkorange', lw=2)
ax.scatter([x_2[0], y_2[0]], [x_2[1], y_2[1]], [f_line2[0], f_line2[1]],
           color='darkorange', s=40, depthshade=False)
ax.view_init(40, 125)

add_logo(fig)
plt.tight_layout()
