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

# # s_riemann_integration_univariate [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_riemann_integration_univariate&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_univariate).

# +
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_univariate-parameters)

# +
# function
x = sym.symbols('x')
f = -x**3 + 5*x**2 + x - 5
display(f)
f_ = sym.lambdify(x, f, 'numpy')

# interval
a = 0
b = 4

# number of intervals in partition
k_ = 20
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_univariate-implementation-step01): Define partition

endpoints = np.linspace(a, b, k_+1)
Delta = np.c_[endpoints[:-1], endpoints[1:]]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_univariate-implementation-step02): Choose points in partition

c = (Delta[:, 1]+Delta[:,0])/2
f_c = f_(c)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_univariate-implementation-step03): Calculate Riemann sum

s_f_Delta = np.sum((Delta[:, 1]-Delta[:,0])*f_c)
s_f_Delta

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_univariate-implementation-step04): Calculate integral

int_f = sym.integrate(f, (x, a, b))
float(int_f)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_riemann_integration_univariate-implementation-step05): Define grid for plotting

x_grid = np.linspace(a, b, 100)
f_x = f_(x_grid)

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)
ax = plt.gca()

# function
plt.plot(x_grid, f_x)

# step function
w = Delta[0,1]-Delta[0,0]
plt.bar(Delta[:, 0], f_c, width=w, align='edge',
        color='lightgrey', edgecolor='darkgray',
        linewidth=0.5, alpha=0.5)

# Riemann sum and integral
plt.text(0.25, 0.9,
         'Integral: '+str('{:3.2f}'.format(round(float(int_f),2)))+'\n'+
         r'Riemann sum ($\bar{k}=$'+str('{:3d}'.format(k_))+'): '+
         str('{:3.2f}'.format(round(s_f_Delta,2))),
         fontsize=17, transform=ax.transAxes,
         linespacing=1.2, verticalalignment='center',
         horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='darkgray', boxstyle='square,pad=0.5'))

plt.axhline()
plt.ylim([min(min(f_x)*1.3, 0), max(f_x)*1.3])
plt.xlabel(r'$x$', fontsize=17)
plt.ylabel(r'$f(x)$', fontsize=17, rotation=0, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

add_logo(fig, set_fig_size=False, location=4)
plt.tight_layout()
