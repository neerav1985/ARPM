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

# # s_clt_student_t [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_clt_student_t&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-ind-vs-no-corr).

# +
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt

from arpym.statistics.simulate_t import simulate_t
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_clt_student_t-parameters)

n_ = 100  # dimension of random variables
nu = 5  # degrees of freedom
j_ = 10000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_clt_student_t-implementation-step01): Generate independent Student t scenarios

x_tilde = np.zeros((j_, n_))
for n in range(n_):
    x_tilde[:, n] = simulate_t(0, 1, nu, j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_clt_student_t-implementation-step02): Generate joint Student t scenarios

x = simulate_t(np.zeros(n_), np.eye(n_), nu, j_)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_clt_student_t-implementation-step03): Define sums $Y$ and $\tilde{Y}$

# +
# sum of i.i.d Student t random variables
y_tilde = np.sum(x_tilde, axis=1)

# sum of jointly t random variables
y = np.sum(x, axis=1)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_clt_student_t-implementation-step04): Calculate histograms

# +
# determine appropriate bin centers
_, xis = histogram_sp(np.append(y, y_tilde), k_=100)

# histogram of sum of i.i.d t random variables
f_Y_tilde, _ = histogram_sp(y_tilde, xi=xis)

# histogram of sum of jointly t random variables
f_Y, _ = histogram_sp(y, xi=xis)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_clt_student_t-implementation-step05): Calculate theoretical pdfs

# +
# define a grid of points
y_grid = np.linspace(-8*np.sqrt(n_*nu/(nu-2)), 8*np.sqrt(n_*nu/(nu-2)), 200)

# normal pdf
f_N = norm.pdf(y_grid, loc=0, scale = np.sqrt(n_*nu/(nu-2)))

# Student t pdf 
f_t = t.pdf(y_grid, df=nu, loc=0, scale=np.sqrt(n_))
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

# iid Student t scenarios
ax1 = fig.add_subplot(2, 1, 1)
# histogram of sums
ax1.bar(xis, f_Y_tilde, width=xis[1]-xis[0], color='gray',
        label=r'$\{\tilde{y}^{(j)}\}_{j=1}^{\bar{\jmath}}$')
# theoretical pdf (Student t)
ax1.plot(y_grid, f_t, color='red', linewidth=1.5)
# theoretical pdf (normal)
ax1.plot(y_grid, f_N, color='C0', linewidth=1.5)

plt.title(r'Distribution of $\tilde{Y}$, the sum of i.i.d. $t$ random variables',
          fontsize=20, fontweight='bold')
plt.xticks(fontsize=14)
ax1.set_ylim([0, norm.pdf(0, loc=0, scale=np.sqrt(n_*nu/(nu-2)))*1.5])
ax1.set_yticks([])
ax1.grid(False)

# jointly Student t distributed scenarios
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
# histogram of sums
ax2.bar(xis, f_Y, width=xis[1]-xis[0], color='lightgray',
        label=r'$\{y^{(j)}\}_{j=1}^{\bar{\jmath}}$')
# theoretical pdf (Student t)
ax2.plot(y_grid, f_t, color='red', linewidth=1.5,
         label=r'$f_{0, \bar{n}, \nu}^{\mathit{t}}$')
# theoretical pdf (normal)
ax2.plot(y_grid, f_N, color='C0', linewidth=1.5,
         label=r'$f_{0, \frac{\nu}{\nu -2} \bar{n}}^{\mathit{N}}$')

plt.title(r'Distribution of $Y$, the sum of jointly $t$ random variables',
          fontsize=20, fontweight='bold')
ax2.set_xlim([-8*np.sqrt(n_*nu/(nu-2)), 8*np.sqrt(n_*nu/(nu-2))])
ax2.set_ylim([0, norm.pdf(0, loc=0, scale=np.sqrt(n_*nu/(nu-2)))*1.5])
ax2.set_yticks([])
ax2.grid(False)

plt.figlegend(fontsize=17, loc='upper right', bbox_to_anchor=(0.98, 0.93),
              borderpad=0.4)
add_logo(fig, set_fig_size=False)
plt.tight_layout()
