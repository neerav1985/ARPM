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

# # s_univariate_derivatives [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_univariate_derivatives&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_univariate_derivatives).

# +
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_univariate_derivatives-parameters)

# +
# variable
x = sym.symbols('x')

# function
f = 3*x**3 - 8*x**2 - 5*x+6
f
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_univariate_derivatives-implementation-step01): Find first derivative

df_dx = sym.diff(f, x)
df_dx  # first derivative

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_univariate_derivatives-implementation-step02): Find second derivative

d2f_dx2 = sym.diff(f, x, 2)
sym.expand(d2f_dx2)  # second derivative

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_univariate_derivatives-implementation-step03): Calculate values for plotting

# +
# variable values
x_grid = np.linspace(-2, 4, 100)

# function
f_ = sym.lambdify(x, f, 'numpy')
f_x = f_(x_grid)

# first derivative
df_dx_ = sym.lambdify(x, df_dx, 'numpy')
df_dx_x = df_dx_(x_grid)

# second derivative
d2f_dx2_ = sym.lambdify(x, d2f_dx2, 'numpy')
d2f_dx2_x = d2f_dx2_(x_grid)
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(720.0/72.0, 960.0/72.0), dpi = 72.0)

# input function
ax1 = plt.subplot(311)
ax1.plot(x_grid, f_x)
ax1.set_title('Function', fontsize=20, fontweight='bold')
ax1.axhline()
plt.xlabel(r'$x$', fontsize=17)
plt.ylabel(r'$f(x)$', fontsize=17, rotation=0, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# first derivative
ax2 = plt.subplot(312)
ax2.plot(x_grid, df_dx_x)
ax2.set_title('First derivative', fontsize=20, fontweight='bold')
ax2.axhline()
plt.xlabel(r'$x$', fontsize=17)
plt.ylabel(r'$\frac{df(x)}{dx}$', fontsize=17, rotation=0, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# second derivative
ax3 = plt.subplot(313)
ax3.plot(x_grid, d2f_dx2_x)
ax3.set_title('Second derivative', fontsize=20, fontweight='bold')
ax3.axhline()
plt.xlabel(r'$x$', fontsize=17)
plt.ylabel(r'$\frac{d^2f(x)}{dx^2}$', fontsize=17, rotation=0, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

add_logo(fig, set_fig_size=False)
plt.tight_layout()
