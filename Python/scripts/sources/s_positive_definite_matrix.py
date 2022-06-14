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

# # s_positive_definite_matrix [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_positive_definite_matrix&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_positive_definite_matrix).

# +
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_positive_definite_matrix-parameters)

# positive definite matrix
q2 = np.array([[1 , -1],
               [-1,  2]])
if np.linalg.det(q2) <= 0:
    print('Choose q2 to be positive definite')

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_positive_definite_matrix-implementation-step01): Define grid of input values

v1 = np.array([0])
v2 = np.array([0])
delta = (q2[0,0]*q2[1,1]-q2[0,1]**2)/q2[0,0]**2
for c in np.linspace(0.5, 5, num=10):
    v2_half = np.linspace(-c, c, 200)
    v1_plus = -q2[0,1]/q2[0,0]*v2_half + np.sqrt(c**2/q2[0,0]-delta*v2_half**2)
    v1_minus = -q2[0,1]/q2[0,0]*v2_half - np.sqrt(c**2/q2[0,0]-delta*v2_half**2)
    v2_ = np.append(v2_half, np.flip(v2_half))
    v2 = np.append(v2, v2_)
    v1_ = np.append(v1_plus, np.flip(v1_minus))
    v1 = np.append(v1, v1_)


# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_positive_definite_matrix-implementation-step02): Calculate quadratic form values

f_v = np.array([])
for i in range(len(v1)):
    v = np.array([v1[i], v2[i]])
    quad = v.T@q2@v
    f_v = np.append(f_v, quad)

# ## Plots

# +
plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0,
                 facecolor = 'white')
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_trisurf(v1, v2, f_v, linewidth=0, antialiased=True,
               color='C1', alpha=0.5, shade=True)

ax1.set_title('Values of the positive definite quadratic form',
              fontsize=20, fontweight='bold', pad=55)
ax1.set_xlabel(r'$v_1$', fontsize=17)
plt.xticks(fontsize=14)
ax1.set_ylabel(r'$v_2$', fontsize=17)
plt.yticks(fontsize=14)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.view_init(30, -75)

ax2 = fig.add_subplot(1, 2, 2)
for c in np.linspace(1, 5, num=5):
    ax2.plot(v1[f_v==c**2], v2[f_v==c**2], color='C0')

ax2.set_title('Iso-contours of the positive definite quadratic form',
              fontsize=20, fontweight='bold')
ax2.set_xlabel(r'$v_1$', fontsize=17)
plt.xticks(fontsize=14)
ax2.set_ylabel(r'$v_2$', fontsize=17)
plt.yticks(fontsize=14)

add_logo(fig, set_fig_size=False)
plt.tight_layout()
