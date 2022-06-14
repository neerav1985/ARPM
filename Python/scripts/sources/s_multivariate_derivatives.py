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

# # s_multivariate_derivatives [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_multivariate_derivatives&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives).

# +
import numpy as np
import sympy as sym
from IPython.display import display, Math
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives-parameters)

# +
# coefficients of quadratic form
a = 3
b = 2
c = 4

# point on iso-contour to examine
x1_0 = 0.4
x2_0 = -(b/(2*c))*x1_0 + np.sqrt(1/c + ((b**2-4*a*c)/(4*c**2))*x1_0**2)
x_0 = np.array([x1_0, x2_0])
display(Math(r'x_0:'))
print(x_0)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives-implementation-step01): Define function

# +
# variable
x1, x2 = sym.symbols('x1, x2')

# function (quadratic form)
f = a*x1**2 + b*x1*x2 + c*x2**2
display(Math(r'f(x):'))
display(f)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives-implementation-step02): Calculate the partial derivatives

# first order partial derivatives
df_dx1 = sym.diff(f, x1)
df_dx2 = sym.diff(f, x2)
display(Math(r'\frac{\partial f}{\partial x_1}:'))
display(df_dx1)
print('-------------')
display(Math(r'\frac{\partial f}{\partial x_2}:'))
display(df_dx2)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives-implementation-step03): Gradient vector

# +
# gradient
grad_f = sym.Matrix([df_dx1, df_dx2])
display(Math(r'\nabla_{\boldsymbol{x}}\, f:'))
display(grad_f)

# evaluated at point x_0
grad_f_ = sym.lambdify((x1, x2), grad_f, 'numpy')
grad_f_x0 = grad_f_(x_0[0],x_0[1]).squeeze()
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives-implementation-step04): Calculate the second order partial derivatives

# second order partial derivatives
d2f_dx1dx1 = sym.diff(df_dx1, x1)
d2f_dx2dx2 = sym.diff(df_dx2, x2)
d2f_dx2dx1 = sym.diff(df_dx1, x2)
d2f_dx1dx2 = sym.diff(df_dx2, x1)
display(Math(r'\frac{\partial^2 f}{\partial x_1^2}:'))
display(d2f_dx1dx1)
print('-------------')
display(Math(r'\frac{\partial^2 f}{\partial x_2^2}:'))
display(d2f_dx2dx2)
print('-------------')
display(Math(r'\frac{\partial^2 f}{\partial x_2 \partial x_1}:'))
display(d2f_dx2dx1)
print('-------------')
display(Math(r'\frac{\partial^2 f}{\partial x_1 \partial x_2}:'))
display(d2f_dx1dx2)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives-implementation-step05): Hessian matrix

# Hessian
hessian_f = sym.Matrix([[d2f_dx1dx1, d2f_dx1dx2],
                     [d2f_dx2dx1, d2f_dx2dx2]])
display(Math(r'\nabla_{\boldsymbol{x}, \boldsymbol{x}}\, f:'))
display(hessian_f)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_multivariate_derivatives-implementation-step06): Calculate points on the iso-contour and tangent

# +
# iso-contour
theta = np.linspace(0, 2*np.pi, 100)
x2_iso = np.sqrt(4*a/(4*a*c-b**2))*np.cos(theta)
x1_iso = 1/np.sqrt(a)*np.sin(theta)-b/(2*a)*x2_iso

# tangent
x2_tan = [x_0[1]-0.5, x_0[1]+0.5]
x1_tan = -grad_f_x0[1]/grad_f_x0[0]*(x2_tan-x_0[1])+x_0[0] 
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)
ax = plt.gca()

# iso-contour
plt.plot(x1_iso, x2_iso, label='iso-contour')

# scaled gradient vector
norm_grad_f_x0 = grad_f_x0/(2*np.linalg.norm(grad_f_x0))
ax.plot([x1_0, x1_0+norm_grad_f_x0[0]], [x2_0, x2_0+norm_grad_f_x0[1]],
        color='#043F72', label='gradient')
ax.arrow(x1_0, x2_0, norm_grad_f_x0[0], norm_grad_f_x0[1],
         color='#043F72', length_includes_head=True,
         head_width=0.04, head_length=0.02,
         lw=0.5)

# tangent
plt.plot(x1_tan, x2_tan, color='darkorange', ls='--',
         label='tangent')

# annotations
ax.text(x1_0, x2_0, r'$\boldsymbol{x}$', fontsize=17, color='black',
        verticalalignment='top', horizontalalignment='right')
ax.text(x1_tan[0], x1_tan[1], r'$df(\boldsymbol{x})$', fontsize=17, color='black',
        verticalalignment='top', horizontalalignment='right')
ax.text(x1_0+norm_grad_f_x0[0]-0.08, x2_0+norm_grad_f_x0[1]-0.04,
        r'$\nabla_{\boldsymbol{x}}f(\boldsymbol{x})$', fontsize=17, color='black',
        verticalalignment='top', horizontalalignment='right')
ranglegrad = tuple(x_0 + 0.08*norm_grad_f_x0)
rangletan = tuple(x_0 - 0.08*norm_grad_f_x0 + np.array([2*(0.08*norm_grad_f_x0)[0],0]))
anga = np.rad2deg(np.arccos(np.abs(grad_f_x0[1])/np.linalg.norm(grad_f_x0)))
angb = 180-anga
consty = 'angle,angleA='+str(round(anga,2))+',angleB='+str(round(angb,2))+',rad=0'
ax.annotate('', xy=ranglegrad, xycoords='data',
            xytext=rangletan, textcoords='data',
            arrowprops=dict(arrowstyle='-', color='#043F72', lw=0.5,
                            connectionstyle=consty)
            )

plt.legend(fontsize=17, loc='upper left')
plt.axhline()
plt.axvline()
lim = max(max(np.append(x1_iso, x2_iso)),1)*1.2
plt.xlim([-lim, lim])
ax.set_aspect('equal')
plt.xlabel(r'$x_1$', fontsize=17)
plt.ylabel(r'$x_2$', fontsize=17, rotation=0, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

add_logo(fig, set_fig_size=False)
plt.tight_layout()
