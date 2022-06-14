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

# #  s_binary_margin_losses
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s-binary-margin-losses).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_binary_margin_losses-parameters)

s_grid = np.linspace(-4, 4, 200)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_binary_margin_losses-implementation-step01): Compute 0-1 loss, hinge, exp, logistic, square and tang losses

loss_01 = np.zeros(200)
loss_01[s_grid<=0] = 1
loss_01[s_grid>0] = 0
loss_hinge = np.maximum(1-s_grid, 0)
loss_exp = np.exp(-s_grid)
loss_square = (1-s_grid)**2
loss_logistic = (1/np.log(2))*np.log(1+np.exp(-s_grid))
loss_tang = (2*np.arctan(s_grid)-1)**2

# ## Plots

# +
dark_gray = [33/255, 37/255, 41/255]
light_teal = [71/255, 180/255, 175/255]
pink = [199/255, 21/255, 133/255]
purple = [58/255, 13/255, 68/255]
orange = [255/255, 153/255, 0/255]
blue = [13/255, 94/255, 148/255]

plt.style.use('arpm')
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
pos = np.where(np.abs(np.diff(loss_01)) >= 0.5)[0]+1
s_ax = s_grid
s_ax = np.insert(s_grid, pos, np.nan)
loss_01 = np.insert(loss_01, pos, np.nan)
plt.plot(s_ax, loss_01, lw=2.3, label=r'$0-1$', color=light_teal)
plt.plot(s_grid, loss_hinge, lw=2.3, label=r'Hinge', color=dark_gray)
plt.plot(s_grid, loss_exp, lw=2.3, label=r'Exponential', color=pink)
plt.plot(s_grid, loss_logistic, lw=2.5, label=r'Logistic', color=orange)
plt.plot(s_grid, loss_square, lw=2.3, label=r'Square', color=purple)
plt.plot(s_grid, loss_tang, lw=2.3, label=r'Tangent', color=blue)
plt.legend(loc=3)
plt.xlabel(r'$s$')
plt.ylabel('Loss')
plt.ylim([-1, 5])
plt.xlim([-4, 4])
add_logo(f, location=4, set_fig_size=False)
