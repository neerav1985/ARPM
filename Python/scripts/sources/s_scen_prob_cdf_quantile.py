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

# # s_scen_prob_cdf_quantile [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_scen_prob_cdf_quantile&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-univ-fpcase-stud).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics.cdf_sp import cdf_sp
from arpym.statistics.quantile_sp import quantile_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_cdf_quantile-parameters)

# +
l_ = 500  # number points to evaluate cdf
k_ = 99  # number of confidence levels
h = 0.01  # bandwidth for Gaussian kernel

x = np.array([1, 2, 0])  # scenarios
p = np.array([0.31, 0.07, 0.62])  # probabilities
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_cdf_quantile-implementation-step01): Compute cdf

x_grid = np.linspace(min(x)-3.5, max(x)+1, l_)  # values to compute cdf
cdf = cdf_sp(x_grid, x, p)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_cdf_quantile-implementation-step02): Compute linearly interpolated cdf

cdf_linint = cdf_sp(x_grid, x, p, method='linear_interp')

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_cdf_quantile-implementation-step03): Compute quantile

c_ = np.linspace(0.01, 0.99, k_)  # confidence levels
q_x_c = quantile_sp(c_, x, p)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_cdf_quantile-implementation-step04): Compute the median

med_x = quantile_sp(0.5, x, p)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_cdf_quantile-implementation-step05): Compute linearly interpolated quantile

q_x_c_linint = quantile_sp(c_, x, p, method='linear_interp')

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_cdf_quantile-implementation-step06): Compute smooth quantile

q_x_c_smooth = quantile_sp(c_, x, p, method='kernel_smoothing', h=h)

# ## Plots

# +
plt.style.use('arpm')

# cdf plot
fig1 = plt.figure()

# plot (discontinuous) cdf
levels = np.unique(cdf)
for level in levels:
    if level == np.min(levels):
        plt.plot(x_grid[cdf == level], cdf[cdf == level],
                 label='cdf', color='dimgray', lw=1.5)
        plt.plot(np.min(x_grid[cdf == level]), level,
                    color='dimgray', marker='<')
        plt.plot(np.max(x_grid[cdf == level]), level,
                    color='white', marker='o')
        plt.plot(np.max(x_grid[cdf == level]), level,
                    color='dimgray', marker='o', fillstyle='none')
    elif level == np.max(levels):
        plt.plot(x_grid[cdf == level], cdf[cdf == level],
                 color='dimgray', lw=1.5)
        plt.plot(np.min(x_grid[cdf == level]), level,
                    color='dimgray', marker='o')
        plt.plot(np.max(x_grid[cdf == level]), level,
                    color='dimgray', marker='>')
    else:
        plt.plot(x_grid[cdf == level], cdf[cdf == level],
                 color='dimgray', lw=1.5)
        plt.plot(np.min(x_grid[cdf == level]), level,
                    color='dimgray', marker='o')
        plt.plot(np.max(x_grid[cdf == level]), level,
                    color='white', marker='o')
        plt.plot(np.max(x_grid[cdf == level]), level,
                    color='dimgray', marker='o', fillstyle='none')

# plot linearly interpolated cdf
plt.plot(x_grid, cdf_linint, label='linearly interpolated cdf',
         color='C1', lw=1.5, linestyle='--')

# style
plt.xlabel('$x$', fontsize=17)
plt.ylabel('cdf', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=17)
add_logo(fig1)
plt.tight_layout()

# quantile plot
fig2 = plt.figure()

# plot (discontinuous) quantile
for scenario in x:
    if scenario == np.max(x):
        plt.plot(c_[q_x_c == scenario], q_x_c[q_x_c == scenario],
                 label='quantile', color='dimgray', lw=1.5)
        plt.plot(np.min(c_[q_x_c == scenario]), scenario,
                    color='white', marker='o')
        plt.plot(np.min(c_[q_x_c == scenario]), scenario,
                    color='dimgray', marker='o', fillstyle='none')
        plt.plot(np.max(c_[q_x_c == scenario]), scenario,
                    color='white', marker='o')
        plt.plot(np.max(c_[q_x_c == scenario]), scenario,
                    color='dimgray', marker='o', fillstyle='none')
    else:
        plt.plot(c_[q_x_c == scenario], q_x_c[q_x_c == scenario],
                 color='dimgray', lw=1.5)
        plt.plot(np.min(c_[q_x_c == scenario]), scenario,
                    color='white', marker='o')
        plt.plot(np.min(c_[q_x_c == scenario]), scenario,
                    color='dimgray', marker='o', fillstyle='none')
        plt.plot(np.max(c_[q_x_c == scenario]), scenario,
                    color='dimgray', marker='o')

# plot linearly interpolated quantile
plt.plot(c_, q_x_c_linint, label='linearly interpolated quantile',
         color='C1', lw=1.5)

# plot smooth quantile
plt.plot(c_, q_x_c_smooth, label='smooth quantile',
         color='orange', lw=1.5, linestyle='--')

# style
plt.xlabel('confidence level $c$', fontsize=17)
plt.ylabel('quantile', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=17)
add_logo(fig2)
plt.tight_layout()
