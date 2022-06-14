# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # s_bivariate_normal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bivariate_normal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_normal).

# +
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_normal-parameters)

# location parameter
mu = np.array([0, 0])
# dispersion parameter
sigma2 = np.array([[1, 0],
                   [0, 1]])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_normal-implementation-step01): Define grid of input values

x1 = np.linspace(mu[0] - 3*np.sqrt(sigma2[0,0]),
            mu[0] + 3*np.sqrt(sigma2[0,0]),
            100)
x2 = np.linspace(mu[1] - 3*np.sqrt(sigma2[1,1]),
            mu[1] + 3*np.sqrt(sigma2[1,1]),
            100)
x1_grid, x2_grid = np.meshgrid(x1, x2)
x_grid = np.stack([x1_grid, x2_grid], axis=2)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bivariate_normal-implementation-step02): Calculate pdf values

pdf = multivariate_normal.pdf(x_grid, mu, sigma2)

# ## Plots

# +
plt.style.use('arpm')

# axis limits
delta = np.sqrt(max(sigma2[0,0], sigma2[1,1]))

# pdf
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0,
                 facecolor = 'white')
fig.suptitle('Normal pdf iso-contours',
            fontsize=20, fontweight='bold')
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.contour(x1_grid, x2_grid, pdf, 15, cmap=cm.coolwarm)

ax1.set_xlabel(r'$x_1$', fontsize=17)
ax1.set_xlim(mu[0]-3*delta, mu[0]+3*delta)
plt.xticks(fontsize=14)
ax1.set_ylabel(r'$x_2$', fontsize=17)
ax1.set_ylim(mu[1]-3*delta, mu[1]+3*delta)
plt.yticks(fontsize=14)
ax1.set_zlim(0, np.max(pdf)*1.05)
ax1.set_zticks(np.arange(0, np.max(pdf)*1.05, 0.05))
for tick in ax1.zaxis.get_major_ticks():
    tick.label.set_fontsize(14)

ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.view_init(20, -125)

# iso-contours
ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(x1_grid, x2_grid, pdf, 15, cmap=cm.coolwarm)

ax2.set_aspect('equal')
ax2.set_xlabel(r'$x_1$', fontsize=17)
ax2.set_xlim(mu[0]-3*delta, mu[0]+3*delta)
plt.xticks(fontsize=14)
ax2.set_ylabel(r'$x_2$', fontsize=17)
ax2.set_ylim(mu[1]-3*delta, mu[1]+3*delta)
plt.yticks(fontsize=14)

add_logo(fig, set_fig_size=False, location=1)
plt.tight_layout()
